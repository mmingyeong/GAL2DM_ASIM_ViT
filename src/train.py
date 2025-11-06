"""
train.py (VoxelViTUNet3D: 3D Voxel-wise Regression, A-SIM 128^3)
Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Created: 2025-07-30 | Last-Modified: 2025-11-03
"""

from __future__ import annotations
import sys, os, argparse, random, numpy as np, torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import pandas as pd

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


def get_clr_scheduler(optimizer, min_lr: float, max_lr: float, cycle_length: int = 8):
    """Epoch-wise triangular cyclical LR"""
    assert max_lr >= min_lr > 0
    assert cycle_length >= 2
    def triangular_clr(epoch: int):
        mid = cycle_length // 2
        ep = epoch % cycle_length
        scale = ep / max(1, mid) if ep <= mid else (cycle_length - ep) / max(1, mid)
        return (min_lr / max_lr) + (1.0 - (min_lr / max_lr)) * scale
    for pg in optimizer.param_groups:
        pg["lr"] = max_lr
    return LambdaLR(optimizer, lr_lambda=triangular_clr)


def str2bool(v):
    """Utility to parse boolean CLI args like --validate_keys False"""
    return str(v).lower() in ("1", "true", "t", "yes", "y")


def _resolve_list_for_split(split: str, common_path: str | None, train_path: str | None, val_path: str | None) -> str | None:
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
    Select input channels according to the experiment case.

    Args:
        x:    [B,2,D,H,W] tensor (original dataloader yields two channels)
        case: "both" | "ch1" | "ch2"
        keep_two: if True, always return [B,2,...] by zero-padding the missing channel
                  if False, return [B,1,...] for single-channel cases
    """
    assert x.ndim == 5 and x.size(1) == 2, f"Expected [B,2,D,H,W], got {tuple(x.shape)}"
    if case == "both":
        return x
    if case == "ch1":
        if keep_two:
            ch1 = x[:, 0:1]
            z   = torch.zeros_like(ch1)
            return torch.cat([ch1, z], dim=1)
        else:
            return x[:, 0:1]
    if case == "ch2":
        if keep_two:
            ch2 = x[:, 1:2]
            z   = torch.zeros_like(ch2)
            return torch.cat([z, ch2], dim=1)
        else:
            return x[:, 1:2]
    raise ValueError(f"Unknown input case: {case}")


# ----------------------------
# Train
# ----------------------------
def train(args):
    logger = get_logger("train_vit_unet3d")
    set_seed(args.seed, deterministic=args.deterministic)
    logger.info("🚀 Starting VoxelViTUNet3D training for 3D voxel-wise regression")
    logger.info(f"Args: {vars(args)}")

    # ---- Resolve include/exclude lists per split ----
    train_include = _resolve_list_for_split("train", args.include_list, args.train_include_list, args.val_include_list)
    val_include   = _resolve_list_for_split("val",   args.include_list, args.train_include_list, args.val_include_list)
    train_exclude = _resolve_list_for_split("train", args.exclude_list, args.train_exclude_list, args.val_exclude_list)
    val_exclude   = _resolve_list_for_split("val",   args.exclude_list, args.train_exclude_list, args.val_exclude_list)

    # ---- Data ----
    train_loader = get_dataloader(
        yaml_path=args.yaml_path, split="train",
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        target_field=args.target_field, train_val_split=args.train_val_split,
        sample_fraction=args.sample_fraction, dtype=torch.float32, seed=args.seed,
        validate_keys=args.validate_keys, strict=False,
        include_list_path=train_include, exclude_list_path=train_exclude
    )
    val_loader = get_dataloader(
        yaml_path=args.yaml_path, split="val",
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        target_field=args.target_field, train_val_split=args.train_val_split,
        sample_fraction=1.0, dtype=torch.float32, seed=args.seed,
        validate_keys=args.validate_keys, strict=False,
        include_list_path=val_include, exclude_list_path=val_exclude
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
        image_size=image_size_3d, in_channels=in_ch, out_channels=1,
        patch_size=patch_size_3d, patch_stride=patch_stride_3d,
        dim=args.emb_dim, depth=args.depth, heads=args.heads,
        mlp_dim=args.mlp_dim, encoder_channels=(32, 64, 128),
        decoder_channels=(256, 128, 64), dropout=0.1
    ).to(args.device)

    logger.info(
        f"🧱 Model created: image_size={image_size_3d}, patch={patch_size_3d}, "
        f"stride={patch_stride_3d}, dim={args.emb_dim}, depth={args.depth}, heads={args.heads}, "
        f"in_channels={in_ch}, input_case={args.input_case}, keep_two={args.keep_two_channels}"
    )

    # ---- Optimizer / Scheduler / AMP ----
    use_amp = args.amp and str(args.device).startswith("cuda")
    optimizer = Adam(model.parameters(), lr=args.max_lr)
    scheduler = get_clr_scheduler(optimizer, args.min_lr, args.max_lr, args.cycle_length)
    early_stopper = EarlyStopping(patience=args.patience, delta=args.es_delta)

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

    # ---- Paths ----
    os.makedirs(args.ckpt_dir, exist_ok=True)
    sample_percent = int(args.sample_fraction * 100)
    case_tag = f"icase-{args.input_case}{'-keep2' if args.keep_two_channels else ''}"
    model_prefix = (
        f"{case_tag}_vitunet3d_tgt-{args.target_field}_D{args.frames}_ps"
        f"{args.frame_patch_size}x{args.image_patch_size}_"
        f"st{frame_stride}x{image_stride}_dim{args.emb_dim}_depth{args.depth}_heads{args.heads}_"
        f"bs{args.batch_size}_clr[{args.min_lr:.0e}-{args.max_lr:.0e}]_s{args.seed}_smp{sample_percent}"
    )
    best_model_path = os.path.join(args.ckpt_dir, f"{model_prefix}_best.pt")
    final_model_path = os.path.join(args.ckpt_dir, f"{model_prefix}_final.pt")
    log_path = os.path.join(args.ckpt_dir, f"{model_prefix}_log.csv")

    # ---- Loop ----
    log_records, best_val_loss = [], float("inf")
    for epoch in range(args.epochs):
        logger.info(f"🔁 Epoch {epoch+1}/{args.epochs} started.")
        model.train(); epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")

        for step, (x, y) in enumerate(loop):
            x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
            x = select_inputs(x, args.input_case, args.keep_two_channels)

            optimizer.zero_grad(set_to_none=True)
            with amp_autocast():
                pred = model(x)
                loss = F.mse_loss(pred, y)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward(); optimizer.step()
            epoch_loss += loss.item() * x.size(0)
            if step % max(1, args.log_interval) == 0:
                loop.set_postfix(loss=f"{loss.item():.5f}")

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader.dataset)
        logger.info(f"📊 Avg Train Loss: {avg_train_loss:.6f}")

        # ---- Validation ----
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(args.device, non_blocking=True), y_val.to(args.device, non_blocking=True)
                x_val = select_inputs(x_val, args.input_case, args.keep_two_channels)
                with amp_autocast():
                    pred_val = model(x_val)
                    loss_val = F.mse_loss(pred_val, y_val)
                val_loss += loss_val.item() * x_val.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"📉 Epoch {epoch+1:03d} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")
        log_records.append({"epoch":epoch+1,"train_loss":avg_train_loss,"val_loss":avg_val_loss,"lr":current_lr})

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


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VoxelViTUNet3D (A-SIM).")
    parser.add_argument("--yaml_path", type=str, required=True)
    parser.add_argument("--target_field", type=str, choices=["rho","tscphi"], default="rho")
    parser.add_argument("--train_val_split", type=float, default=0.8)
    parser.add_argument("--sample_fraction", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", type=bool, default=True)

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
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--cycle_length", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--es_delta", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt_dir", type=str, default="results/vit/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--amp", action="store_true")

    # Input ablations
    parser.add_argument("--input_case", type=str, choices=["both", "ch1", "ch2"], default="both",
                        help="Select which input channels are provided to the model.")
    parser.add_argument("--keep_two_channels", action="store_true",
                        help="If set, keep in_channels=2 and zero-pad the missing channel for single-channel cases.")

    # Dataset validation
    parser.add_argument("--validate_keys", type=str2bool, default=True,
                        help="Pre-scan HDF5 to check required keys (input/output_*). Set False to skip (faster).")

    # Include/Exclude lists (common + split-specific)
    parser.add_argument("--include_list", type=str, default=None,
                        help="(Optional) Path to a text file listing files to INCLUDE for both train/val.")
    parser.add_argument("--exclude_list", type=str, default=None,
                        help="(Optional) Path to a text file listing files to EXCLUDE for both train/val.")
    parser.add_argument("--train_include_list", type=str, default=None,
                        help="(Optional) Train-only include list path.")
    parser.add_argument("--val_include_list", type=str, default=None,
                        help="(Optional) Val-only include list path.")
    parser.add_argument("--train_exclude_list", type=str, default=None,
                        help="(Optional) Train-only exclude list path.")
    parser.add_argument("--val_exclude_list", type=str, default=None,
                        help="(Optional) Val-only exclude list path.")

    args = parser.parse_args()

    try:
        train(args)
    except Exception:
        import traceback; print("🔥 Training failed due to exception:"); traceback.print_exc(); sys.exit(1)
