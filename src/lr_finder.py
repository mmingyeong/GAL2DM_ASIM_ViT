#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import traceback
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch_lr_finder import LRFinder

from src.data_loader import get_dataloader
from src.logger import get_logger
from src.model import VoxelViTUNet3D as ViT3D
from src.train import select_inputs


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "t", "yes", "y")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def ensure_parent_dir(path: Optional[str]):
    if path:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)


def save_history_csv(history: Dict[str, Any], csv_path: str):
    ensure_parent_dir(csv_path)
    lrs = history.get("lr", [])
    losses = history.get("loss", [])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "lr", "loss"])
        for i, (lr, loss) in enumerate(zip(lrs, losses), start=1):
            writer.writerow([i, lr, loss])


def summarize_history(history: Dict[str, Any]) -> Dict[str, Any]:
    lrs = np.array(history.get("lr", []), dtype=float)
    losses = np.array(history.get("loss", []), dtype=float)

    summary: Dict[str, Any] = {
        "num_points": int(len(lrs)),
        "start_lr": float(lrs[0]) if len(lrs) else None,
        "end_lr": float(lrs[-1]) if len(lrs) else None,
        "min_loss": None,
        "min_loss_lr": None,
        "steepest_lr": None,
        "steepest_loss": None,
    }

    if len(losses) == 0:
        return summary

    min_idx = int(np.argmin(losses))
    summary["min_loss"] = float(losses[min_idx])
    summary["min_loss_lr"] = float(lrs[min_idx])

    if len(losses) >= 2:
        x = np.log10(np.clip(lrs, 1e-300, None))
        dx = np.diff(x)
        dy = np.diff(losses)
        grad = dy / np.clip(dx, 1e-12, None)
        steepest_idx = int(np.argmin(grad)) + 1
        summary["steepest_lr"] = float(lrs[steepest_idx])
        summary["steepest_loss"] = float(losses[steepest_idx])

    return summary


class InputSelectDataset(Dataset):
    """
    Wrap a dataset and apply select_inputs() at dataset level
    so the final object is still a real PyTorch Dataset/DataLoader.
    """

    def __init__(self, base_dataset, input_case: str, keep_two_channels: bool):
        self.base_dataset = base_dataset
        self.input_case = input_case
        self.keep_two_channels = keep_two_channels

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        x = x.unsqueeze(0)  # [C,D,H,W] -> [1,C,D,H,W]
        x = select_inputs(x, self.input_case, self.keep_two_channels)
        x = x.squeeze(0)
        return x, y


def build_default_output_paths(args):
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.plot_path is None:
        args.plot_path = os.path.join(out_dir, "lr_finder_curve.png")
    if args.history_path is None:
        args.history_path = os.path.join(out_dir, "lr_finder_history.json")
    if args.csv_path is None:
        args.csv_path = os.path.join(out_dir, "lr_finder_history.csv")
    if args.summary_path is None:
        args.summary_path = os.path.join(out_dir, "lr_finder_summary.json")


def run_lr_finder(args):
    build_default_output_paths(args)
    logger = get_logger("lr_finder_vit", log_dir=args.log_dir)

    logger.info("🚀 Starting LR finder for VoxelViTUNet3D")
    logger.info(f"Args: {vars(args)}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device='{args.device}' but CUDA is not available.")

    if args.device.startswith("cuda"):
        gpu_idx = torch.cuda.current_device()
        logger.info(
            f"🖥️ CUDA available | device={args.device} | "
            f"name={torch.cuda.get_device_name(gpu_idx)} | "
            f"count={torch.cuda.device_count()}"
        )
    else:
        logger.info(f"🖥️ Using device={args.device}")

    normalization_cfg = {
        "mode": "custom",
        "normalize_input": True,
        "normalize_target": (args.target_field == "rho"),
        "eps": 1e-12,
    }

    augmentation_cfg = {
        "enable": bool(args.use_augmentation),
        "flip": True,
        "mirror": True,
        "permute_axes": True,
    }

    logger.info(f"🧮 Normalization config: {normalization_cfg}")
    logger.info(f"🧊 Augmentation config: {augmentation_cfg}")

    base_loader = get_dataloader(
        yaml_path=args.yaml_path,
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
        sample_fraction=args.sample_fraction,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        target_field=args.target_field,
        dtype=torch.float32,
        seed=args.seed,
        train_val_split=args.train_val_split,
        validate_keys=args.validate_keys,
        strict=False,
        exclude_list_path=args.exclude_list,
        include_list_path=args.include_list,
        augmentation=augmentation_cfg,
        normalization=normalization_cfg,
        apply_augmentation_in=("train",) if args.use_augmentation else (),
    )

    logger.info(
        f"📦 Base DataLoader ready | dataset_size={len(base_loader.dataset)} | "
        f"num_batches={len(base_loader)} | batch_size={args.batch_size} | "
        f"num_workers={args.num_workers} | pin_memory={args.pin_memory}"
    )

    selected_dataset = InputSelectDataset(
        base_loader.dataset,
        input_case=args.input_case,
        keep_two_channels=args.keep_two_channels,
    )

    train_loader = DataLoader(
        selected_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=getattr(base_loader, "worker_init_fn", None),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    logger.info(
        f"📦 LR-finder DataLoader rebuilt | dataset_size={len(train_loader.dataset)} | "
        f"num_batches={len(train_loader)}"
    )

    x0, y0 = next(iter(train_loader))
    logger.info(
        f"🔎 First batch | x.shape={tuple(x0.shape)} | y.shape={tuple(y0.shape)} | "
        f"x.dtype={x0.dtype} | y.dtype={y0.dtype} | "
        f"x.min={x0.min().item():.6g} | x.max={x0.max().item():.6g} | "
        f"y.min={y0.min().item():.6g} | y.max={y0.max().item():.6g}"
    )

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
        encoder_channels=(32, 64, 128),
        decoder_channels=(256, 128, 64),
        dropout=args.dropout,
    ).to(args.device)

    param_info = count_parameters(model)
    logger.info(
        f"🧱 Model created: VoxelViTUNet3D | image_size={image_size_3d} | "
        f"patch={patch_size_3d} | stride={patch_stride_3d} | "
        f"dim={args.emb_dim} | depth={args.depth} | heads={args.heads} | "
        f"mlp_dim={args.mlp_dim} | in_ch={in_ch} | "
        f"params_total={param_info['total']:,} | trainable={param_info['trainable']:,}"
    )

    # ---- Materialize lazy parameters before LRFinder snapshots state_dict ----
    # VoxelViTUNet3D creates pos_embedding lazily on first forward, so we force one
    # dummy forward here to ensure reset() can restore a complete state_dict.
    with torch.no_grad():
        _x, _y = next(iter(train_loader))
        _x = _x.to(args.device, non_blocking=True)
        _ = model(_x)
    logger.info("🧩 Lazy model parameters materialized with one dummy forward.")

    optimizer = Adam(model.parameters(), lr=args.start_lr)
    criterion = torch.nn.MSELoss()
    logger.info(
        f"⚙️ Optimizer/Loss ready | optimizer=Adam | start_lr={args.start_lr:.3e} | loss=MSELoss"
    )

    lr_finder = LRFinder(model, optimizer, criterion, device=args.device)

    num_iter = min(args.num_iter, len(train_loader))
    if num_iter <= 1:
        raise ValueError(
            f"num_iter became {num_iter}. Check dataset size / batch_size / sample_fraction."
        )

    logger.info(
        f"📈 Running range_test | step_mode={args.step_mode} | "
        f"start_lr={args.start_lr:.3e} | end_lr={args.end_lr:.3e} | num_iter={num_iter}"
    )

    lr_finder.range_test(
        train_loader,
        end_lr=args.end_lr,
        num_iter=num_iter,
        step_mode=args.step_mode,
    )

    history = lr_finder.history
    summary = summarize_history(history)

    logger.info(
        f"✅ LR finder finished | points={summary['num_points']} | "
        f"min_loss={summary['min_loss']} @ lr={summary['min_loss_lr']} | "
        f"steepest_lr={summary['steepest_lr']}"
    )

    ensure_parent_dir(args.plot_path)
    plot_ret = lr_finder.plot(suggest_lr=True)

    if isinstance(plot_ret, tuple):
        ax = plot_ret[0]
        suggested_lr = plot_ret[1] if len(plot_ret) > 1 else None
    else:
        ax = plot_ret
        suggested_lr = None

    fig = ax.figure
    fig.savefig(args.plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"🖼️ Saved LR-Loss plot: {args.plot_path}")

    if suggested_lr is not None:
        logger.info(f"💡 Suggested LR from plot(): {suggested_lr:.6e}")

    ensure_parent_dir(args.history_path)
    with open(args.history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info(f"🧾 Saved LR finder history JSON: {args.history_path}")

    save_history_csv(history, args.csv_path)
    logger.info(f"📄 Saved LR finder history CSV: {args.csv_path}")

    summary_payload = {
        "model": "VoxelViTUNet3D",
        "target_field": args.target_field,
        "input_case": args.input_case,
        "keep_two_channels": bool(args.keep_two_channels),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "sample_fraction": float(args.sample_fraction),
        "start_lr": float(args.start_lr),
        "end_lr": float(args.end_lr),
        "requested_num_iter": int(args.num_iter),
        "actual_num_iter": int(num_iter),
        "step_mode": args.step_mode,
        "pin_memory": bool(args.pin_memory),
        "validate_keys": bool(args.validate_keys),
        "use_augmentation": bool(args.use_augmentation),
        "dataset_size": int(len(train_loader.dataset)),
        "num_batches": int(len(train_loader)),
        "image_size": list(image_size_3d),
        "patch_size": list(patch_size_3d),
        "patch_stride": list(patch_stride_3d),
        "emb_dim": int(args.emb_dim),
        "depth": int(args.depth),
        "heads": int(args.heads),
        "mlp_dim": int(args.mlp_dim),
        "dropout": float(args.dropout),
        "model_in_ch": int(in_ch),
        "model_out_ch": 1,
        "model_params_total": int(param_info["total"]),
        "model_params_trainable": int(param_info["trainable"]),
        "suggested_lr": float(suggested_lr) if suggested_lr is not None else None,
        "history_summary": summary,
    }
    ensure_parent_dir(args.summary_path)
    with open(args.summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    logger.info(f"📌 Saved LR finder summary JSON: {args.summary_path}")

    try:
        lr_finder.reset()
        logger.info("♻️ LR finder state reset complete.")
    except Exception as e:
        logger.warning(f"⚠️ lr_finder.reset() failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LR finder for VoxelViTUNet3D (A-SIM).")

    # Data
    parser.add_argument("--yaml_path", type=str, required=True)
    parser.add_argument("--target_field", type=str, choices=["rho", "tscphi"], default="rho")
    parser.add_argument("--train_val_split", type=float, default=0.8)
    parser.add_argument("--sample_fraction", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--validate_keys", type=str2bool, default=True)
    parser.add_argument("--exclude_list", type=str, default=None)
    parser.add_argument("--include_list", type=str, default=None)

    # Device / reproducibility
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    # Input ablations
    parser.add_argument("--input_case", type=str, choices=["both", "ch1", "ch2"], default="both")
    parser.add_argument("--keep_two_channels", action="store_true")

    # ViT geometry
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--image_patch_size", type=int, default=16)
    parser.add_argument("--frame_patch_size", type=int, default=16)
    parser.add_argument("--image_patch_stride", type=int, default=None)
    parser.add_argument("--frame_patch_stride", type=int, default=None)

    # ViT capacity
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    # LR finder controls
    parser.add_argument("--start_lr", type=float, default=1e-7)
    parser.add_argument("--end_lr", type=float, default=1e-1)
    parser.add_argument("--num_iter", type=int, default=100)
    parser.add_argument("--step_mode", type=str, choices=["exp", "linear"], default="exp")

    # Usually OFF for LR finder
    parser.add_argument("--use_augmentation", action="store_true")

    # Outputs
    parser.add_argument("--out_dir", type=str, default="results/lr_finder_vit/manual")
    parser.add_argument("--plot_path", type=str, default=None)
    parser.add_argument("--history_path", type=str, default=None)
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--summary_path", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="logs")

    args = parser.parse_args()

    try:
        run_lr_finder(args)
    except Exception:
        traceback.print_exc()
        raise