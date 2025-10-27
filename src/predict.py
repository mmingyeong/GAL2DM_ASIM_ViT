# models/vit/predict.py
# -*- coding: utf-8 -*-
"""
Predict with VoxelViTUNet3D (3D voxel-wise regression) and save to HDF5.

Supports channel-ablation inference:
  --input_case {both,ch1,ch2}
  --keep_two_channels  (keep in_channels=2 and zero-pad missing channel)

Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Created: 2025-07-30
Last-Modified: 2025-10-23
"""

from __future__ import annotations
import sys, os, argparse, yaml, torch, h5py, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from tqdm import tqdm
from glob import glob
from contextlib import nullcontext

from src.model import VoxelViTUNet3D as ViT3D
from src.logger import get_logger

logger = get_logger("predict_vitunet3d")


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
    test_rel = yaml_cfg["asim_datasets_hdf5"]["validation_set"]["path"]  # e.g., test/*.hdf5
    pattern = os.path.join(base, test_rel)
    files = sorted(glob(pattern), key=_natkey)
    if not files:
        raise FileNotFoundError(f"No test HDF5 files matched: {pattern}")
    return files


def _ensure_input_shape(x: np.ndarray) -> np.ndarray:
    """
    Accepts and returns:
      - (N, C, D, H, W) with C in {1,2}
      - Also accepts (2,D,H,W) or (1,2,D,H,W) etc., normalizing to (1,C,D,H,W)
    """
    arr = x
    if arr.ndim == 4 and arr.shape[0] in (1, 2):              # (C,D,H,W)
        arr = arr[None, ...]                                   # (1,C,D,H,W)
    elif arr.ndim == 5 and arr.shape[1] in (1, 2):             # (N,C,D,H,W)
        pass
    elif arr.ndim == 6 and arr.shape[0] == 1 and arr.shape[1] in (1, 2):
        arr = np.squeeze(arr, axis=0)                          # (C,D,H,W) or (N,C,D,H,W)
        if arr.ndim == 4: arr = arr[None, ...]
    else:
        raise ValueError(f"Unsupported 'input' shape: {arr.shape}")
    return arr


def _load_checkpoint(model_path: str, device: torch.device):
    """
    Safe checkpoint load with optional weights_only.
    """
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)  # type: ignore[arg-type]
    except TypeError:
        state = torch.load(model_path, map_location=device)
    except Exception as e:
        logger.warning(f"weights_only load failed with {e}; falling back to standard torch.load")
        state = torch.load(model_path, map_location=device)

    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        elif "model" in state and isinstance(state["model"], dict):
            state = state["model"]
    return state


def _find_input_dataset(h5: h5py.File) -> str | None:
    """
    Return the dataset key to use for inputs, or None if not found.
    Tries several common keys and nested paths.
    """
    candidate_keys = [
        "input", "inputs", "X",
        "data/input", "dataset/input", "features/input",
    ]
    # exact key in root
    for k in candidate_keys:
        if k in h5:
            return k
    # nested path (group/subkey)
    for k in candidate_keys:
        if "/" in k:
            grp, dset = k.split("/", 1)
            if grp in h5 and dset in h5[grp]:
                return k
    # loose match in root (case-insensitive)
    lowered = {kk.lower(): kk for kk in h5.keys()}
    for name in ("input", "inputs", "x"):
        if name in lowered:
            return lowered[name]
    return None


def select_inputs(x: torch.Tensor, case: str, keep_two: bool) -> torch.Tensor:
    """
    x: [N,C,D,H,W] with C in {1,2} (from file)
    Returns channels according to the ablation case:
      - both:   expect/use two channels; if C==1, error
      - ch1/ch2: if keep_two -> return 2-ch tensor with zero-padded missing channel
                 else -> return 1-ch tensor of the selected channel
    """
    assert x.ndim == 5 and x.size(1) in (1, 2), f"Expected [N,1or2,D,H,W], got {tuple(x.shape)}"

    if case == "both":
        if x.size(1) != 2:
            raise ValueError(f"--input_case both requires 2 channels, but got {x.size(1)}")
        return x

    if case == "ch1":
        if x.size(1) == 2:
            if keep_two:
                ch1 = x[:, 0:1]
                z   = torch.zeros_like(ch1)
                return torch.cat([ch1, z], dim=1)
            else:
                return x[:, 0:1]
        else:  # C==1
            return x if not keep_two else torch.cat([x, torch.zeros_like(x)], dim=1)

    if case == "ch2":
        if x.size(1) == 2:
            if keep_two:
                ch2 = x[:, 1:2]
                z   = torch.zeros_like(ch2)
                return torch.cat([z, ch2], dim=1)
            else:
                return x[:, 1:2]
        else:  # C==1
            if keep_two:
                z = torch.zeros_like(x)
                return torch.cat([z, x], dim=1)
            else:
                return x
    raise ValueError(case)


# ----------------------------
# Inference
# ----------------------------
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
    image_patch_stride: int | None = None,
    frame_patch_stride: int | None = None,
    emb_dim: int = 256,
    depth: int = 3,
    heads: int = 8,
    mlp_dim: int = 512,
    amp: bool = False,
    sample_fraction: float = 1.0,
    sample_seed: int = 42,
    input_case: str = "both",
    keep_two_channels: bool = False,
    on_missing_input: str = "skip",  # NEW: "skip" | "stop"
):
    """
    Run inference on HDF5 test files with channel-ablation support.
    """
    if not (0 < sample_fraction <= 1.0):
        raise ValueError(f"--sample_fraction must be in (0,1], got {sample_fraction}")

    # case-specific subdir to avoid mixing outputs
    case_suffix = f"icase-{input_case}{'-keep2' if keep_two_channels else ''}"
    output_dir = os.path.join(output_dir, case_suffix)
    os.makedirs(output_dir, exist_ok=True)

    # RNG for reproducible file subsampling
    rng = np.random.default_rng(sample_seed)

    # 1) Resolve test set
    cfg = _load_yaml(yaml_path)
    test_files = _resolve_test_files(cfg)

    if sample_fraction < 1.0:
        n_total = len(test_files)
        n_keep = max(1, int(np.ceil(sample_fraction * n_total)))
        keep_idx = np.sort(rng.choice(n_total, size=n_keep, replace=False))
        test_files = [test_files[i] for i in keep_idx]
        logger.info(f"🧪 Test files subsampled: {n_keep}/{n_total} (fraction={sample_fraction:.3f})")
    else:
        logger.info(f"🧪 Test files: {len(test_files)} found from YAML (no subsampling).")

    # 2) Build & load model (in_channels depends on case)
    dev = torch.device(device)
    DHW = (frames, image_size, image_size)
    patch_size_3d = (frame_patch_size, image_patch_size, image_patch_size)

    if frame_patch_stride is None:
        frame_patch_stride = frame_patch_size
    if image_patch_stride is None:
        image_patch_stride = image_patch_size
    patch_stride_3d = (frame_patch_stride, image_patch_stride, image_patch_stride)
    for s, p in zip(patch_stride_3d, patch_size_3d):
        if s > p:
            raise ValueError(f"Stride {patch_stride_3d} must be <= patch {patch_size_3d} per axis.")

    # Decide in_channels same as in training
    if input_case == "both":
        in_ch = 2
    else:
        in_ch = 2 if keep_two_channels else 1

    logger.info("📦 Building model VoxelViTUNet3D ...")
    model = ViT3D(
        image_size=DHW,
        in_channels=in_ch,
        out_channels=1,
        patch_size=patch_size_3d,
        patch_stride=patch_stride_3d,
        dim=emb_dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        encoder_channels=(32, 64, 128),
        decoder_channels=(256, 128, 64),
        dropout=0.1,
    ).to(dev)
    logger.info(
        f"🧱 Model: image_size={DHW}, patch_size={patch_size_3d}, stride={patch_stride_3d}, "
        f"dim={emb_dim}, depth={depth}, heads={heads}, mlp_dim={mlp_dim}, in_channels={in_ch}, "
        f"input_case={input_case}, keep_two={keep_two_channels}"
    )

    logger.info(f"📥 Loading checkpoint: {model_path}")
    state = _load_checkpoint(model_path, dev)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Missing keys while loading: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys while loading: {unexpected}")
    model.eval()

    # AMP context
    try:
        _ = torch.amp
        if amp:
            if dev.type == "cuda":
                autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            else:
                autocast_ctx = torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
        else:
            autocast_ctx = nullcontext()
    except Exception:
        from torch.cuda.amp import autocast as legacy_autocast
        autocast_ctx = legacy_autocast(enabled=amp)

    # 3) Predict per file
    saved_files: list[str] = []
    skipped_files: list[str] = []

    torch.set_grad_enabled(False)
    with torch.no_grad():
        for input_path in tqdm(test_files, desc="🚀 Running test predictions"):
            filename = os.path.basename(input_path)
            output_path = os.path.join(output_dir, filename)

            # Load inputs (robust to missing/variant keys)
            with h5py.File(input_path, "r") as f:
                key = _find_input_dataset(f)
                if key is None:
                    msg = f"No input-like dataset found in {input_path}"
                    if on_missing_input == "stop":
                        raise KeyError(msg)
                    logger.warning(f"⚠️ {msg} — SKIP")
                    skipped_files.append(input_path)
                    continue

                try:
                    x_np = f[key][:]  # (N,C,D,H,W) or (C,D,H,W)
                except Exception as e:
                    msg = f"Failed to read dataset '{key}' in {input_path}: {e}"
                    if on_missing_input == "stop":
                        raise RuntimeError(msg)
                    logger.warning(f"⚠️ {msg} — SKIP")
                    skipped_files.append(input_path)
                    continue

            # Normalize shape
            try:
                x_np = _ensure_input_shape(x_np)  # -> (N,C,D,H,W) with C in {1,2}
            except Exception as e:
                msg = f"Invalid input shape in {input_path}: {e}"
                if on_missing_input == "stop":
                    raise
                logger.warning(f"⚠️ {msg} — SKIP")
                skipped_files.append(input_path)
                continue

            # Shape checks
            N, Cfile, D, H, W = x_np.shape
            if (D, H, W) != DHW:
                msg = f"Input spatial {(D,H,W)} does not match model {DHW} in {input_path}"
                if on_missing_input == "stop":
                    raise ValueError(msg)
                logger.warning(f"⚠️ {msg} — SKIP")
                skipped_files.append(input_path)
                continue

            # To tensor & select channels
            x_tensor = torch.from_numpy(np.ascontiguousarray(x_np)).float().to(dev)
            try:
                x_tensor = select_inputs(x_tensor, input_case, keep_two_channels)  # [N,in_ch,D,H,W]
            except Exception as e:
                msg = f"Channel selection failed for {input_path}: {e}"
                if on_missing_input == "stop":
                    raise
                logger.warning(f"⚠️ {msg} — SKIP")
                skipped_files.append(input_path)
                continue

            if x_tensor.size(1) != in_ch:
                msg = f"Post-selection channels {x_tensor.size(1)} != model.in_channels {in_ch} in {input_path}"
                if on_missing_input == "stop":
                    raise RuntimeError(msg)
                logger.warning(f"⚠️ {msg} — SKIP")
                skipped_files.append(input_path)
                continue

            # Batched inference
            preds = []
            for i in range(0, x_tensor.shape[0], batch_size):
                x_batch = x_tensor[i: i + batch_size]
                with autocast_ctx:
                    y_batch = model(x_batch)  # [B,1,D,H,W]
                preds.append(y_batch.float().cpu().numpy())

            y_pred = np.concatenate(preds, axis=0)  # (N,1,D,H,W)
            y_pred = np.squeeze(y_pred, axis=1)     # (N,D,H,W) or (D,H,W)

            # Save prediction
            with h5py.File(output_path, "w") as f_out:
                f_out.create_dataset("prediction", data=y_pred, compression="gzip")
                # Meta-info
                f_out.attrs["source_file"] = input_path
                f_out.attrs["model_path"] = model_path
                f_out.attrs["model_class"] = model.__class__.__name__
                f_out.attrs["amp"] = bool(amp)
                f_out.attrs["image_size"] = int(image_size)
                f_out.attrs["frames"] = int(frames)
                f_out.attrs["image_patch_size"] = int(image_patch_size)
                f_out.attrs["frame_patch_size"] = int(frame_patch_size)
                f_out.attrs["image_patch_stride"] = int(image_patch_stride) if image_patch_stride is not None else int(image_patch_size)
                f_out.attrs["frame_patch_stride"] = int(frame_patch_stride) if frame_patch_stride is not None else int(frame_patch_size)
                f_out.attrs["emb_dim"] = int(emb_dim)
                f_out.attrs["depth"] = int(depth)
                f_out.attrs["heads"] = int(heads)
                f_out.attrs["mlp_dim"] = int(mlp_dim)
                f_out.attrs["sample_fraction_files_only"] = float(sample_fraction)
                f_out.attrs["sample_seed"] = int(sample_seed)
                f_out.attrs["input_case"] = str(input_case)
                f_out.attrs["keep_two_channels"] = bool(keep_two_channels)

            logger.info(f"✅ Saved: {output_path}")
            saved_files.append(output_path)

    # Summary
    logger.info("====== Inference Summary ======")
    logger.info(f"Saved files : {len(saved_files)}")
    logger.info(f"Skipped     : {len(skipped_files)}")
    if skipped_files:
        logger.info("Skipped list (first 20): " + ", ".join(os.path.basename(p) for p in skipped_files[:20]))


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VoxelViTUNet3D inference on A-SIM test files.")

    # Data / Paths
    parser.add_argument("--yaml_path", type=str, required=True, help="Path to asim_paths.yaml")
    parser.add_argument("--output_dir", type=str, required=True, help="Root directory to save predictions (per-case subdir will be created)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .pt file")

    # System
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--amp", action="store_true", help="Enable mixed-precision inference")

    # Model settings (must match training)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--image_patch_size", type=int, default=16)
    parser.add_argument("--frame_patch_size", type=int, default=16)
    parser.add_argument("--image_patch_stride", type=int, default=None, help="stride for H/W (<= image_patch_size)")
    parser.add_argument("--frame_patch_stride", type=int, default=None, help="stride for D (<= frame_patch_size)")
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp_dim", type=int, default=512)

    # Subsampling
    parser.add_argument("--sample_fraction", type=float, default=1.0,
                        help="Fraction (0,1] of TEST FILES to run. Does NOT subsample within-file samples.")
    parser.add_argument("--sample_seed", type=int, default=42,
                        help="Random seed for reproducible file-level subsampling.")

    # 🔻 Channel ablation flags
    parser.add_argument("--input_case", type=str, choices=["both", "ch1", "ch2"], default="both",
                        help="Select which input channels are provided to the model.")
    parser.add_argument("--keep_two_channels", action="store_true",
                        help="If set, keep in_channels=2 and zero-pad the missing channel for single-channel cases.")

    # 🔻 Missing-input handling policy
    parser.add_argument("--on_missing_input", type=str, choices=["skip", "stop"], default="skip",
                        help="When an HDF5 file lacks the input dataset: skip it (default) or stop with error.")

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
        image_patch_stride=args.image_patch_stride,
        frame_patch_stride=args.frame_patch_stride,
        emb_dim=args.emb_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        amp=args.amp,
        sample_fraction=args.sample_fraction,
        sample_seed=args.sample_seed,
        input_case=args.input_case,
        keep_two_channels=args.keep_two_channels,
        on_missing_input=args.on_missing_input,
    )
