"""
train.py (VoxelViTUNet3D: 3D Voxel-wise Regression, A-SIM 128^3)
Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Created: 2025-07-30 | Last-Modified: 2025-12-23

Optuna Patch (2025-12-23):
- Add scheduler choices for Optuna: cosine_warmup vs constant_warmup
- Tune max_lr, warmup_ratio, min_lr_ratio (cosine only)
- Switch scheduler stepping to optimizer-step granularity (accum aware)
- Add --out_metrics JSON for Optuna driver to read (best/final val loss + config)
- Keep existing augmentation/normalization behavior (data_loader handles it)
"""

from __future__ import annotations

import sys, os, argparse, random
import json
import math
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR, CosineAnnealingLR
from tqdm import tqdm
import pandas as pd
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.data_loader import get_dataloader
from src.logger import get_logger
from src.model import VoxelViTUNet3D as ViT3D


# ----------------------------
# Utilities
# ----------------------------
class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def str2bool(v):
    """Utility to parse boolean CLI args like --validate_keys False"""
    return str(v).lower() in ("1", "true", "t", "yes", "y")


def _resolve_list_for_split(
    split: str,
    common_path: str | None,
    train_path: str | None,
    val_path: str | None,
) -> str | None:
    """Select include/exclude list for a split. Priority: split-specific > common > None."""
    if split == "train":
        return train_path or common_path
    elif split == "val":
        return val_path or common_path
    else:
        return common_path


# ----------------------------
# Input selection helper
# ----------------------------
def select_inputs(x: torch.Tensor, case: str, keep_two: bool) -> torch.Tensor:
    """
    x: [B,2,D,H,W], channels=[ngal, vpec]
    case: "both" | "ch1" | "ch2"
    keep_two=True  -> 항상 2채널 반환(결측 채널은 0으로 패딩)
    keep_two=False -> 단일 채널 반환
    """
    assert x.ndim == 5 and x.size(1) == 2, f"Expected [B,2,D,H,W], got {tuple(x.shape)}"
    if case == "both":
        return x
    if case == "ch1":
        if keep_two:
            ch1 = x[:, 0:1]
            z = torch.zeros_like(ch1)
            return torch.cat([ch1, z], dim=1)
        else:
            return x[:, 0:1]
    if case == "ch2":
        if keep_two:
            ch2 = x[:, 1:2]
            z = torch.zeros_like(ch2)
            return torch.cat([z, ch2], dim=1)
        else:
            return x[:, 1:2]
    raise ValueError(f"Unknown input case: {case}")


# ----------------------------
# Scheduler builders (Optuna)
# ----------------------------
def build_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    total_updates: int,          # number of optimizer.step() calls
    warmup_ratio: float,
    max_lr: float,
    min_lr_ratio: float = 1e-2,  # cosine only
):
    """
    Step per optimizer update (NOT per epoch).
    - constant_warmup: linear warmup -> constant(max_lr)
    - cosine_warmup  : linear warmup -> cosine decay to eta_min=max_lr*min_lr_ratio
    """
    total_updates = int(total_updates)
    if total_updates <= 0:
        raise ValueError(f"total_updates must be > 0, got {total_updates}")

    warmup_ratio = float(warmup_ratio)
    warmup_updates = int(total_updates * warmup_ratio)
    warmup_updates = max(0, min(warmup_updates, total_updates))

    # base lr = max_lr; schedule via multipliers
    for pg in optimizer.param_groups:
        pg["lr"] = float(max_lr)

    if warmup_updates == 0:
        if scheduler_type == "constant_warmup":
            return ConstantLR(optimizer, factor=1.0, total_iters=total_updates)
        if scheduler_type == "cosine_warmup":
            eta_min = float(max_lr) * float(min_lr_ratio)
            return CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=eta_min)
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    warmup = LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=warmup_updates,
    )
    remain = max(1, total_updates - warmup_updates)

    if scheduler_type == "constant_warmup":
        main = ConstantLR(optimizer, factor=1.0, total_iters=remain)
    elif scheduler_type == "cosine_warmup":
        eta_min = float(max_lr) * float(min_lr_ratio)
        main = CosineAnnealingLR(optimizer, T_max=remain, eta_min=eta_min)
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    return SequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_updates])


def maybe_write_metrics(path: Optional[str], payload: Dict[str, Any], logger):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"🧾 Wrote metrics JSON: {path}")


# ----------------------------
# Train
# ----------------------------
def train(args):
    logger = get_logger("train_vit_unet3d")
    set_seed(args.seed, deterministic=args.deterministic)
    logger.info("🚀 Starting VoxelViTUNet3D training for 3D voxel-wise regression")
    logger.info(f"Args: {vars(args)}")
    start_time = time.time()

    # ---- Resolve include/exclude lists per split ----
    train_include = _resolve_list_for_split("train", args.include_list, args.train_include_list, args.val_include_list)
    val_include   = _resolve_list_for_split("val",   args.include_list, args.train_include_list, args.val_include_list)
    train_exclude = _resolve_list_for_split("train", args.exclude_list, args.train_exclude_list, args.val_exclude_list)
    val_exclude   = _resolve_list_for_split("val",   args.exclude_list, args.train_exclude_list, args.val_exclude_list)

    # ---- Normalization config (data_loader) ----
    normalization_cfg = {
        "mode": "custom",
        "normalize_input": True,
        "normalize_target": (args.target_field == "rho"),
        "eps": 1e-12,
    }
    logger.info(f"🧮 Normalization config: {normalization_cfg}")

    # ---- Augmentation config (data_loader) ----
    augmentation_cfg = {
        "enable": bool(args.use_augmentation),
        "flip": True,
        "mirror": True,
        "permute_axes": True,
    }
    logger.info(f"🧊 Augmentation config: {augmentation_cfg}")

    # ---- Data ----
    train_loader = get_dataloader(
        yaml_path=args.yaml_path,
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        target_field=args.target_field,
        train_val_split=args.train_val_split,
        sample_fraction=args.sample_fraction,
        dtype=torch.float32,
        seed=args.seed,
        validate_keys=args.validate_keys,
        strict=False,
        include_list_path=train_include,
        exclude_list_path=train_exclude,
        augmentation=augmentation_cfg,
        normalization=normalization_cfg,
        apply_augmentation_in=("train",),
    )
    val_loader = get_dataloader(
        yaml_path=args.yaml_path,
        split="val",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        target_field=args.target_field,
        train_val_split=args.train_val_split,
        sample_fraction=1.0,
        dtype=torch.float32,
        seed=args.seed,
        validate_keys=args.validate_keys,
        strict=False,
        include_list_path=val_include,
        exclude_list_path=val_exclude,
        augmentation=augmentation_cfg,
        normalization=normalization_cfg,
        apply_augmentation_in=("train",),
    )

    logger.info(f"📊 Train samples (files): {len(train_loader.dataset)}")
    logger.info(f"📊 Validation samples (files): {len(val_loader.dataset)}")

    # ---- Model ----
    image_size_3d = (args.frames, args.image_size, args.image_size)
    patch_size_3d = (args.frame_patch_size, args.image_patch_size, args.image_patch_size)
    frame_stride = args.frame_patch_stride or args.frame_patch_size
    image_stride = args.image_patch_stride or args.image_patch_size
    patch_stride_3d = (frame_stride, image_stride, image_stride)
    for s, p in zip(patch_stride_3d, patch_size_3d):
        if s > p:
            raise ValueError(f"Stride {s}>{p} is invalid.")

    if args.input_case == "both":
        in_ch = 2
    else:
        in_ch = 2 if args.keep_two_channels else 1

    model = ViT3D(
        image_size=image_size_3d,
        in_channels=in_ch,
        out_channels=1,
        patch_size=patch_size_3d,
        patch_stride=patch_stride_3d,
        dim=args.emb_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dim_head=args.vit_dim_head,
        encoder_channels=tuple(args.vit_encoder_channels),
        decoder_channels=tuple(args.vit_decoder_channels),
        dropout=args.vit_dropout,
    ).to(args.device)

    logger.info(
        f"🧱 Model created: image_size={image_size_3d}, patch={patch_size_3d}, "
        f"stride={patch_stride_3d}, dim={args.emb_dim}, depth={args.depth}, heads={args.heads}, "
        f"in_channels={in_ch}, input_case={args.input_case}, keep_two={args.keep_two_channels}"
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"🔢 Trainable params: {num_params/1e6:.2f}M")

    # ---- Optimizer / AMP ----
    use_amp = args.amp and str(args.device).startswith("cuda")
    optimizer = Adam(model.parameters(), lr=args.max_lr)

    # ✅ AMP Compatibility Wrapper
    try:
        import torch.amp as amp
        scaler = amp.GradScaler("cuda") if use_amp else amp.GradScaler(enabled=False)

        def amp_autocast():
            if not use_amp:
                from contextlib import nullcontext
                return nullcontext()
            return amp.autocast("cuda", dtype=torch.float16)

    except Exception:
        from torch.cuda.amp import GradScaler as OldScaler, autocast as old_autocast
        scaler = OldScaler(enabled=use_amp)

        def amp_autocast():
            return old_autocast(enabled=use_amp)

    # ---- Gradient Accumulation ----
    accum = max(1, int(getattr(args, "grad_accum_steps", 1)))
    eff_bs = args.batch_size * accum
    if accum > 1:
        logger.info(
            f"🧮 Using gradient accumulation: grad_accum_steps={accum} "
            f"(effective_batch = {args.batch_size} * {accum} = {eff_bs})"
        )

    # ---- Scheduler (Optuna-friendly; step per optimizer update) ----
    updates_per_epoch = math.ceil(len(train_loader) / accum)
    total_updates = updates_per_epoch * args.epochs

    scheduler = build_warmup_scheduler(
        optimizer=optimizer,
        scheduler_type=args.scheduler_type,
        total_updates=total_updates,
        warmup_ratio=args.warmup_ratio,
        max_lr=args.max_lr,
        min_lr_ratio=args.min_lr_ratio,
    )
    logger.info(
        f"🗓️ Scheduler: {args.scheduler_type} | total_updates={total_updates} "
        f"| warmup_ratio={args.warmup_ratio} | max_lr={args.max_lr:.2e} | min_lr_ratio={args.min_lr_ratio:.2e}"
    )

    early_stopper = EarlyStopping(patience=args.patience, delta=args.es_delta)

    # ---- Paths ----
    os.makedirs(args.ckpt_dir, exist_ok=True)
    sample_percent = int(args.sample_fraction * 100)
    case_tag = f"icase-{args.input_case}{'-keep2' if args.keep_two_channels else ''}"
    aug_tag = "augON" if args.use_augmentation else "augOFF"
    norm_tag = "normCUSTOM" if normalization_cfg.get("mode", "none") != "none" else "normNONE"
    sched_tag = f"{args.scheduler_type}_wu{args.warmup_ratio:.3f}_minr{args.min_lr_ratio:.2e}"

    model_prefix = (
        f"{case_tag}_vitunet3d_{aug_tag}_{norm_tag}_tgt-{args.target_field}_D{args.frames}_ps"
        f"{args.frame_patch_size}x{args.image_patch_size}_"
        f"st{frame_stride}x{image_stride}_dim{args.emb_dim}_depth{args.depth}_heads{args.heads}_"
        f"bs{args.batch_size}_acc{accum}_eff{eff_bs}_{sched_tag}_maxlr{args.max_lr:.2e}_"
        f"s{args.seed}_smp{sample_percent}"
    )
    best_model_path = os.path.join(args.ckpt_dir, f"{model_prefix}_best.pt")
    final_model_path = os.path.join(args.ckpt_dir, f"{model_prefix}_final.pt")
    log_path = os.path.join(args.ckpt_dir, f"{model_prefix}_log.csv")

    # ---- Loop ----
    log_records, best_val_loss = [], float("inf")
    global_update = 0  # optimizer update counter

    for epoch in range(args.epochs):
        logger.info(f"🔁 Epoch {epoch+1}/{args.epochs} started.")
        model.train()
        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")

        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(loop):
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            x = select_inputs(x, args.input_case, args.keep_two_channels)

            with amp_autocast():
                pred = model(x)
                loss = F.mse_loss(pred, y)
                loss = loss / accum

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % accum == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                scheduler.step()
                global_update += 1

            epoch_loss += (loss.item() * accum) * x.size(0)

            if step % max(1, args.log_interval) == 0:
                loop.set_postfix(loss=f"{(loss.item() * accum):.5f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        # leftover grads
        if (step + 1) % accum != 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            scheduler.step()
            global_update += 1

        avg_train_loss = epoch_loss / len(train_loader.dataset)
        logger.info(f"📊 Avg Train Loss: {avg_train_loss:.6f}")

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(args.device, non_blocking=True)
                y_val = y_val.to(args.device, non_blocking=True)

                x_val = select_inputs(x_val, args.input_case, args.keep_two_channels)

                with amp_autocast():
                    pred_val = model(x_val)
                    loss_val = F.mse_loss(pred_val, y_val)

                val_loss += loss_val.item() * x_val.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"📉 Epoch {epoch+1:03d} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")

        log_records.append(
            {"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "lr": current_lr, "global_update": global_update}
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"✅ New best model saved (epoch {epoch+1})")

        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            logger.warning(f"🛑 Early stopping at epoch {epoch+1}")
            break

    # ---- Save ----
    torch.save(model.state_dict(), final_model_path)
    pd.DataFrame(log_records).to_csv(log_path, index=False)
    logger.info(f"📦 Final model saved: {final_model_path}")
    logger.info(f"📝 Training log saved: {log_path}")

    end_time = time.time()
    elapsed = end_time - start_time

    # 시/분/초 변환
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    logger.info(f"⏱️ Total training time: {hours:02d}:{minutes:02d}:{seconds:02d} (HH:MM:SS)")

    # ---- Optuna metrics JSON (minimal) ----
    metrics_payload = {
        "model": "VoxelViTUNet3D",
        "best_val_loss": float(best_val_loss),
        "final_val_loss": float(log_records[-1]["val_loss"]) if log_records else None,
        "scheduler_type": args.scheduler_type,
        "max_lr": float(args.max_lr),
        "warmup_ratio": float(args.warmup_ratio),
        "min_lr_ratio": float(args.min_lr_ratio),
        "batch_size": int(args.batch_size),
        "grad_accum_steps": int(accum),
        "effective_batch": int(eff_bs),
        "seed": int(args.seed),
        "epochs_ran": int(log_records[-1]["epoch"]) if log_records else 0,
        "global_updates": int(global_update),
    }
    maybe_write_metrics(args.out_metrics, metrics_payload, logger)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VoxelViTUNet3D (A-SIM).")
    parser.add_argument("--yaml_path", type=str, required=True)
    parser.add_argument("--target_field", type=str, choices=["rho", "tscphi"], default="rho")
    parser.add_argument("--train_val_split", type=float, default=0.8)
    parser.add_argument("--sample_fraction", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", type=str2bool, default=True)

    # ViT-specific geometry
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--image_patch_size", type=int, default=16)
    parser.add_argument("--frame_patch_size", type=int, default=16)
    parser.add_argument("--image_patch_stride", type=int, default=None)
    parser.add_argument("--frame_patch_stride", type=int, default=None)

    # ViT-specific capacity
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp_dim", type=int, default=512)

    # Training
    parser.add_argument("--epochs", type=int, default=200)

    # ---- Optuna LR + Scheduler knobs ----
    parser.add_argument(
        "--scheduler_type",
        type=str,
        choices=["cosine_warmup", "constant_warmup"],
        default="cosine_warmup",
        help="Optuna searches this: cosine_warmup vs constant_warmup",
    )
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr_ratio", type=float, default=1e-2)

    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--es_delta", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt_dir", type=str, default="results/vit/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_accum_steps", type=int, default=1)

    # Input ablations
    parser.add_argument("--input_case", type=str, choices=["both", "ch1", "ch2"], default="both")
    parser.add_argument("--keep_two_channels", action="store_true")

    # Dataset validation
    parser.add_argument("--validate_keys", type=str2bool, default=True)

    # Augmentation on/off
    parser.add_argument("--use_augmentation", action="store_true")

    # Include/Exclude lists (common + split-specific)
    parser.add_argument("--include_list", type=str, default=None)
    parser.add_argument("--exclude_list", type=str, default=None)
    parser.add_argument("--train_include_list", type=str, default=None)
    parser.add_argument("--val_include_list", type=str, default=None)
    parser.add_argument("--train_exclude_list", type=str, default=None)
    parser.add_argument("--val_exclude_list", type=str, default=None)

    # Optuna driver output
    parser.add_argument("--out_metrics", type=str, default=None)

    parser.add_argument("--vit_encoder_channels", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--vit_decoder_channels", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--vit_dropout", type=float, default=0.1)
    parser.add_argument("--vit_dim_head", type=int, default=64)

    args = parser.parse_args()

    try:
        train(args)
    except Exception:
        import traceback
        print("🔥 Training failed due to exception:")
        traceback.print_exc()
        sys.exit(1)
