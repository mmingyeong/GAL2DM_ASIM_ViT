"""
train.py (ViT3D with Full 3D Volume Regression)

Description:
    Training script for ViT3D to perform full 3D volume regression (voxel-to-voxel)
    using Mean Squared Error (MSE) loss. Targets may be 'rho' or 'tscphi'.

Author:
    Mingyeong Yang (mmingyeong@kasi.re.kr)
Created:
    2025-07-30
"""

from __future__ import annotations

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import pandas as pd

from src.data_loader import get_dataloader
from src.logger import get_logger
from src.model import ViT3D


# ----------------------------
# Utils
# ----------------------------
class EarlyStopping:
    def __init__(self, patience=10, delta=0.0):
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


def get_clr_scheduler(optimizer, min_lr, max_lr, cycle_length=8):
    # triangular cyclical LR (epoch-wise)
    def triangular_clr(epoch):
        mid = cycle_length // 2
        ep = epoch % cycle_length
        if ep <= mid:
            scale = ep / max(1, mid)
        else:
            scale = (cycle_length - ep) / max(1, mid)
        # returns LR multiplier relative to optimizer's base LR
        # we will set base LR = max_lr, multiplier in [min_lr/max_lr, 1]
        return (min_lr / max_lr) + (1.0 - (min_lr / max_lr)) * scale

    # set optimizer base LR as max_lr
    for pg in optimizer.param_groups:
        pg["lr"] = max_lr
    return LambdaLR(optimizer, lr_lambda=triangular_clr)


# ----------------------------
# Train
# ----------------------------
def train(args):
    logger = get_logger("train_vit_3dvolume")
    set_seed(args.seed, deterministic=args.deterministic)

    logger.info("🚀 Starting ViT3D training for 3D volume regression:")
    logger.info(vars(args))

    # -------- Dataloaders (A-SIM spec via YAML) --------
    train_loader = get_dataloader(
        yaml_path=args.yaml_path,
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        target_field=args.target_field,     # 'rho' or 'tscphi'
        train_val_split=args.train_val_split,
        sample_fraction=args.sample_fraction,
        dtype=torch.float32,
        seed=args.seed
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
        sample_fraction=1.0,  # 검증은 전체 권장
        dtype=torch.float32,
        seed=args.seed
    )

    logger.info(f"📊 Train samples (files): {len(train_loader.dataset)}")
    logger.info(f"📊 Validation samples (files): {len(val_loader.dataset)}")

    # -------- Model --------
    model = ViT3D(
        image_size=args.image_size,         # 128
        frames=args.frames,                 # 128
        image_patch_size=args.image_patch_size,  # 16 (divides 128)
        frame_patch_size=args.frame_patch_size,  # 16 (divides 128)
        dim=args.emb_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        in_channels=2,                      # A-SIM: ngal, vpec
        out_channels=1                      # predict one field
    ).to(args.device)

    # -------- Loss / Optim / Sched / AMP --------
    scaler = GradScaler(enabled=args.amp)
    optimizer = Adam(model.parameters(), lr=args.max_lr)  # base LR will be max_lr due to CLR
    scheduler = get_clr_scheduler(optimizer, args.min_lr, args.max_lr, cycle_length=args.cycle_length)
    early_stopper = EarlyStopping(patience=args.patience, delta=args.es_delta)

    # -------- Paths --------
    os.makedirs(args.ckpt_dir, exist_ok=True)
    sample_percent = int(args.sample_fraction * 100)
    name_bits = [
        f"vit3d",
        f"tgt-{args.target_field}",
        f"D{args.frames}",
        f"ps{args.frame_patch_size}x{args.image_patch_size}",
        f"dim{args.emb_dim}",
        f"depth{args.depth}",
        f"heads{args.heads}",
        f"bs{args.batch_size}",
        f"clr[{args.min_lr:.0e}-{args.max_lr:.0e}]",
        f"s{args.seed}",
        f"smp{sample_percent}"
    ]
    model_prefix = "_".join(name_bits)
    best_model_path = os.path.join(args.ckpt_dir, f"{model_prefix}_best.pt")
    final_model_path = os.path.join(args.ckpt_dir, f"{model_prefix}_final.pt")
    log_path = os.path.join(args.ckpt_dir, f"{model_prefix}_log.csv")

    # -------- Loop --------
    log_records = []
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        logger.info(f"🔁 Epoch {epoch+1}/{args.epochs} started.")
        model.train()
        epoch_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]", leave=True)
        for step, (x, y) in enumerate(loop):
            x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if args.amp:
                with autocast(dtype=torch.float16):
                    pred = model(x)
                    loss = F.mse_loss(pred, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(x)
                loss = F.mse_loss(pred, y)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item() * x.size(0)

            if step % max(1, args.log_interval) == 0:
                loop.set_postfix(loss=f"{loss.item():.5f}")

        scheduler.step()  # epoch-wise CLR
        avg_train_loss = epoch_loss / len(train_loader.dataset)
        logger.info(f"📊 Avg Train Loss: {avg_train_loss:.6f}")

        # -------- Validation --------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_step, (x_val, y_val) in enumerate(val_loader):
                x_val = x_val.to(args.device, non_blocking=True)
                y_val = y_val.to(args.device, non_blocking=True)
                if args.amp:
                    with autocast(dtype=torch.float16):
                        pred_val = model(x_val)
                        loss_val = F.mse_loss(pred_val, y_val)
                else:
                    pred_val = model(x_val)
                    loss_val = F.mse_loss(pred_val, y_val)
                val_loss += loss_val.item() * x_val.size(0)

                if val_step % max(1, args.log_interval) == 0:
                    pass  # verbose면 logger.debug로 변경 가능

        avg_val_loss = val_loss / len(val_loader.dataset)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"📉 Epoch {epoch+1:03d} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")

        log_records.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": current_lr
        })

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"✅ New best model saved at epoch {epoch+1}: {best_model_path}")

        # Early stopping
        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            logger.warning(
                f"🛑 Early stopping at epoch {epoch+1} "
                f"(best val loss tracked by ES: {early_stopper.best_loss:.6f})"
            )
            break

    # Final save & log
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"📦 Final model saved: {final_model_path}")
    pd.DataFrame(log_records).to_csv(log_path, index=False)
    logger.info(f"📝 Training log saved: {log_path}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT3D for full 3D voxel-wise regression (A-SIM).")

    # Data / split
    parser.add_argument("--yaml_path", type=str, required=True, help="Path to asim_paths.yaml")
    parser.add_argument("--target_field", type=str, choices=["rho", "tscphi"], default="rho")
    parser.add_argument("--train_val_split", type=float, default=0.8, help="Fraction of training/*.hdf5 used for train (rest for val)")
    parser.add_argument("--sample_fraction", type=float, default=1.0, help="Fraction of train files to sample (0<frac<=1)")

    # Loader
    parser.add_argument("--batch_size", type=int, default=1)           # 128³ 메모리 고려
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", type=bool, default=True)

    # Model
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--image_patch_size", type=int, default=16)
    parser.add_argument("--frame_patch_size", type=int, default=16)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp_dim", type=int, default=512)

    # Train
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--cycle_length", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--es_delta", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=10)

    # System
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt_dir", type=str, default="results/vit/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (fp16)")

    args = parser.parse_args()

    try:
        train(args)
    except Exception as e:
        import traceback
        print("🔥 Training failed due to exception:")
        traceback.print_exc()
        sys.exit(1)
