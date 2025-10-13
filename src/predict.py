"""
models/vit/predict.py

Run inference using a trained ViT3D model and save predictions to HDF5,
preserving the original input HDF5 filenames (A-SIM spec).

Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Created: 2025-07-30
"""

from __future__ import annotations

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import argparse
import yaml
import torch
import h5py
import numpy as np
from tqdm import tqdm
from glob import glob
from contextlib import nullcontext

from src.model import ViT3D
from src.logger import get_logger

logger = get_logger("predict_vit")


# ----------------------------
# Helpers
# ----------------------------
def _natkey(path: str):
    import re
    tokens = re.split(r"(\d+)", os.path.basename(path))
    return tuple(int(t) if t.isdigit() else t for t in tokens)


def _load_yaml(yaml_path: str) -> dict:
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_test_files(yaml_cfg: dict) -> list[str]:
    base = yaml_cfg["asim_datasets_hdf5"]["base_path"]
    test_rel = yaml_cfg["asim_datasets_hdf5"]["validation_set"]["path"]  # test/*.hdf5
    pattern = os.path.join(base, test_rel)
    files = sorted(glob(pattern), key=_natkey)
    if not files:
        raise FileNotFoundError(f"No test HDF5 files matched: {pattern}")
    return files


def _ensure_input_shape(x: np.ndarray) -> np.ndarray:
    """
    Accepts:
      - (2, D, H, W)
      - (1, 2, D, H, W)
      - (N, 2, D, H, W)
      - (N, 1, 2, D, H, W)  -> rare, squeeze batch dim 1
    Returns:
      - (N, 2, D, H, W) with N>=1
    """
    arr = x
    if arr.ndim == 4 and arr.shape[0] == 2:
        arr = arr[None, ...]  # (1, 2, D, H, W)
    elif arr.ndim == 5 and arr.shape[1] == 2:
        pass  # (N, 2, D, H, W)
    elif arr.ndim == 5 and arr.shape[0] == 1 and arr.shape[1] == 2:
        pass  # (1, 2, D, H, W)
    elif arr.ndim == 6 and arr.shape[1] == 1 and arr.shape[2] == 2:
        arr = np.squeeze(arr, axis=1)  # (N, 2, D, H, W)
    else:
        raise ValueError(f"Unsupported 'input' shape: {arr.shape}")
    return arr


# ----------------------------
# Inference
# ----------------------------
# ----------------------------
# Inference
# ----------------------------
from contextlib import nullcontext

def run_prediction(
    yaml_path: str,
    output_dir: str,
    model_path: str,
    device: str = "cuda",
    batch_size: int = 1,
    image_size: int = 128,
    frames: int = 128,
    image_patch_size: int = 16,
    frame_patch_size: int = 16,
    emb_dim: int = 256,
    depth: int = 6,
    heads: int = 8,
    mlp_dim: int = 512,
    amp: bool = False,
    sample_fraction: float = 1.0,   # ✔ 파일 개수만 샘플링
    sample_seed: int = 42,          # ✔ 재현성
):
    """
    sample_fraction:
        0 < f <= 1.0. If <1, randomly subsample ONLY the list of test files.
        (No within-file subsampling is performed.)
    """
    if not (0 < sample_fraction <= 1.0):
        raise ValueError(f"--sample_fraction must be in (0,1], got {sample_fraction}")

    os.makedirs(output_dir, exist_ok=True)

    # RNG for reproducible subsampling
    rng = np.random.default_rng(sample_seed)

    # 1) Resolve test set
    cfg = _load_yaml(yaml_path)
    test_files = _resolve_test_files(cfg)

    # ✔ File-level subsampling only
    if sample_fraction < 1.0:
        n_total = len(test_files)
        n_keep = max(1, int(np.ceil(sample_fraction * n_total)))
        keep_idx = np.sort(rng.choice(n_total, size=n_keep, replace=False))
        test_files = [test_files[i] for i in keep_idx]
        logger.info(f"🧪 Test files subsampled: {n_keep}/{n_total} (fraction={sample_fraction:.3f})")
    else:
        logger.info(f"🧪 Test files: {len(test_files)} found from YAML (no subsampling).")

    # 2) Build & load model
    logger.info(f"📦 Loading model from: {model_path}")
    model = ViT3D(
        image_size=image_size,
        frames=frames,
        image_patch_size=image_patch_size,
        frame_patch_size=frame_patch_size,
        dim=emb_dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        in_channels=2,    # A-SIM: ngal, vpec
        out_channels=1
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 3) Predict per file
    try:
        _ = torch.amp  # PyTorch ≥ 2.0
        autocast_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16) if amp and torch.cuda.is_available()
            else torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16) if amp and device == "cpu"
            else nullcontext()
        )
    except Exception:
        from torch.cuda.amp import autocast as legacy_autocast
        autocast_ctx = legacy_autocast(enabled=amp)

    with torch.no_grad():
        for input_path in tqdm(test_files, desc="🚀 Running test predictions"):
            filename = os.path.basename(input_path)
            output_path = os.path.join(output_dir, filename)

            # Load inputs
            with h5py.File(input_path, "r") as f:
                if "input" not in f:
                    raise KeyError(f"'input' dataset not found in {input_path}")
                x = f["input"][:]  # (1,2,D,H,W) or (2,D,H,W) or (N,2,D,H,W)
            x = _ensure_input_shape(x)  # -> (N, 2, D, H, W)

            # Sanity vs model config
            N, C, D, H, W = x.shape
            if C != 2:
                raise ValueError(f"Expected 2 input channels (ngal,vpec), got {C} in {input_path}")
            if (D, H, W) != (frames, image_size, image_size):
                raise ValueError(
                    f"Input spatial {D,H,W} does not match model ({frames},{image_size},{image_size}). "
                    f"Adjust --frames/--image_size or rebuild the model."
                )

            # Batched inference (no within-file subsampling)
            preds = []
            x_tensor = torch.from_numpy(np.ascontiguousarray(x)).float().to(device)
            for i in range(0, x_tensor.shape[0], batch_size):
                x_batch = x_tensor[i : i + batch_size]
                with autocast_ctx:
                    y_batch = model(x_batch)  # [B, 1, D, H, W]
                preds.append(y_batch.float().cpu().numpy())

            y_pred = np.concatenate(preds, axis=0)  # (N, 1, D, H, W)
            y_pred = np.squeeze(y_pred, axis=1)     # (N, D, H, W) or (D, H, W) if N=1

            # Save prediction
            with h5py.File(output_path, "w") as f_out:
                f_out.create_dataset("prediction", data=y_pred, compression="gzip")
                # Meta-info
                f_out.attrs["source_file"] = input_path
                f_out.attrs["model_path"] = model_path
                f_out.attrs["image_size"] = image_size
                f_out.attrs["frames"] = frames
                f_out.attrs["image_patch_size"] = image_patch_size
                f_out.attrs["frame_patch_size"] = frame_patch_size
                f_out.attrs["emb_dim"] = emb_dim
                f_out.attrs["depth"] = depth
                f_out.attrs["heads"] = heads
                f_out.attrs["mlp_dim"] = mlp_dim
                f_out.attrs["sample_fraction_files_only"] = float(sample_fraction)
                f_out.attrs["sample_seed"] = int(sample_seed)

            logger.info(f"✅ Saved: {output_path}")



# ----------------------------
# Main
# ----------------------------
# context manager fallback
from contextlib import nullcontext

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ViT3D inference on A-SIM test files.")
    parser.add_argument("--yaml_path", type=str, required=True, help="Path to asim_paths.yaml")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predictions")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .pt file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--amp", action="store_true", help="Enable mixed-precision inference")

    # Model settings (must match training)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--image_patch_size", type=int, default=16)
    parser.add_argument("--frame_patch_size", type=int, default=16)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp_dim", type=int, default=512)

    # Subsampling
    parser.add_argument("--sample_fraction", type=float, default=1.0,
        help="Fraction (0,1] of TEST FILES to run. Does NOT subsample within-file samples.")
    parser.add_argument("--sample_seed", type=int, default=42,
        help="Random seed for reproducible file-level subsampling.")

    args = parser.parse_args()

    run_prediction(
        yaml_path=args.yaml_path,
        output_dir=args.output_dir,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        image_size=args.image_size,
        frames=args.frames,
        image_patch_size=args.image_patch_size,
        frame_patch_size=args.frame_patch_size,
        emb_dim=args.emb_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        amp=args.amp,
    )
