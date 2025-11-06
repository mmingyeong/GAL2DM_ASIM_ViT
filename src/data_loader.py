"""
src/data_loader.py  (for ViT models)

Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Created: 2025-10-13
Modified: 2025-11-03 (exclude/include list + auto-exclude paths/ids, robust normalization)

Description:
    Data loader for A-SIM HDF5 training/validation/test datasets.
    - Inputs: ngal, vpec → 2 channels (C=2)
    - Targets: output_rho (default) or output_tscphi (optional; NOT output_phi)
      * If target == 'tscphi', multiply by (0.72**-2) and subtract its mean.

Notes:
    - sample_fraction ∈ (0,1): applied equally for train/val/test.
    - validate_keys=True: drop files missing mandatory keys at init.
    - exclude_list_path / include_list_path: external list files supported.
"""

from __future__ import annotations
import os, re, yaml, h5py
from glob import glob
from typing import List, Tuple, Literal, Sequence, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from src.logger import get_logger

logger = get_logger("data_loader_vit", log_dir="logs")


# ============================================================
# Utilities
# ============================================================
def _natkey(path: str):
    tokens = re.split(r"(\d+)", os.path.basename(path))
    return tuple(int(t) if t.isdigit() else t for t in tokens)

def _norm(p: str) -> str:
    # normalize path for reliable set membership (abs + no trailing slash)
    return os.path.abspath(p).rstrip("/")

def _fname_id(p: str) -> Optional[int]:
    # extract numeric stem like ".../training/10248.hdf5" -> 10248
    base = os.path.basename(p)
    m = re.match(r"(\d+)\.hdf5$", base)
    return int(m.group(1)) if m else None

def _load_yaml(yaml_path: str) -> dict:
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _resolve_split_files(
    yaml_cfg: dict,
    split: Literal["train", "val", "test"],
    train_val_split: float = 0.8,
) -> List[str]:
    base = yaml_cfg["asim_datasets_hdf5"]["base_path"]
    train_pattern = os.path.join(base, yaml_cfg["asim_datasets_hdf5"]["training_set"]["path"])
    test_pattern  = os.path.join(base, yaml_cfg["asim_datasets_hdf5"]["validation_set"]["path"])

    train_files = sorted(glob(train_pattern), key=_natkey)
    test_files  = sorted(glob(test_pattern),  key=_natkey)
    if not train_files:
        raise FileNotFoundError(f"No training files found: {train_pattern}")

    n_total = len(train_files)
    n_split = int(n_total * train_val_split)
    if split == "train":
        selected = train_files[:n_split]
    elif split == "val":
        selected = train_files[n_split:]
    elif split == "test":
        selected = test_files
    else:
        raise ValueError(f"Invalid split '{split}'.")

    logger.info(
        f"📂 Split '{split}': {len(selected)} files "
        f"({n_split}/{n_total} train-val split, {len(test_files)} test)"
    )
    return selected


# ============================================================
# Shape normalization helpers
# ============================================================
def _squeeze_leading_ones_to_nd(arr: np.ndarray, nd: int) -> np.ndarray:
    out = arr
    while out.ndim > nd and out.shape[0] == 1:
        out = out[0]
    return out

def _ensure_input_channels(arr: np.ndarray) -> np.ndarray:
    out = _squeeze_leading_ones_to_nd(arr, nd=4)
    if out.ndim != 4 or out.shape[0] != 2:
        raise ValueError(f"'input' must be (2,D,H,W); got {arr.shape}->{out.shape}")
    return out

def _ensure_target_3d(arr: np.ndarray) -> np.ndarray:
    out = _squeeze_leading_ones_to_nd(arr, nd=3)
    if out.ndim != 3:
        raise ValueError(f"target must be 3D; got {arr.shape}->{out.shape}")
    return out


# ============================================================
# Dataset
# ============================================================
class ASIMHDF5Dataset(Dataset):
    """
    A-SIM HDF5 dataset loader for ViT models.
    Returns:
      x: FloatTensor (2, D, H, W)
      y: FloatTensor (1, D, H, W)
    """
    def __init__(self, file_paths: Sequence[str], target_field: Literal["rho", "tscphi"]="rho", dtype=torch.float32):
        self.file_paths = list(file_paths)
        self.target_field = target_field
        self.dtype = dtype
        assert target_field in ("rho", "tscphi")
        logger.info(f"🔍 ASIMHDF5Dataset init | {len(self.file_paths)} samples | target={target_field}")

    def __len__(self): return len(self.file_paths)

    def __getitem__(self, idx):
        fpath = self.file_paths[idx]
        with h5py.File(fpath, "r") as f:
            if "input" not in f:
                raise KeyError(f"'input' dataset missing in {fpath}")
            x_np = _ensure_input_channels(f["input"][:])

            if self.target_field == "rho":
                if "output_rho" not in f:
                    raise KeyError(f"'output_rho' not found in {fpath}")
                y_np = _ensure_target_3d(f["output_rho"][:])
            else:
                if "output_tscphi" not in f:
                    raise KeyError(f"'output_tscphi' not found in {fpath}")
                y_np = _ensure_target_3d(f["output_tscphi"][:])
                y_np = y_np * (0.72 ** -2)
                y_np -= y_np.mean(dtype=np.float64)

        x = torch.from_numpy(np.ascontiguousarray(x_np)).to(self.dtype)
        y = torch.from_numpy(np.ascontiguousarray(y_np)).to(self.dtype).unsqueeze(0)
        return x, y


# ============================================================
# Validation & filtering
# ============================================================
def _filter_files_by_keys(file_paths: Sequence[str], target_field: Literal["rho","tscphi"], strict=False) -> List[str]:
    req_target = "output_rho" if target_field == "rho" else "output_tscphi"
    kept, dropped = [], []
    for p in file_paths:
        try:
            with h5py.File(p, "r") as f:
                ok = ("input" in f) and (req_target in f)
            if ok:
                kept.append(p)
            else:
                dropped.append(p)
                if strict:
                    raise KeyError(f"{os.path.basename(p)} missing keys")
        except Exception as e:
            dropped.append(p)
            if strict:
                raise
            logger.warning(f"⚠️ Invalid file skipped: {p} | {e}")
    if dropped:
        logger.warning(f"⚠️ Dropped {len(dropped)} invalid file(s).")
    logger.info(f"✅ Valid files: {len(kept)}/{len(file_paths)}")
    return kept


# ============================================================
# Public API: get_dataloader
# ============================================================
def get_dataloader(
    yaml_path: str,
    split: Literal["train", "val", "test"],
    batch_size: int,
    shuffle: bool = True,
    sample_fraction: float = 1.0,
    num_workers: int = 0,
    pin_memory: bool = True,
    target_field: Literal["rho", "tscphi"] = "rho",
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = 42,
    train_val_split: float = 0.8,
    validate_keys: bool = True,
    strict: bool = False,
    exclude_list_path: Optional[str] = None,
    include_list_path: Optional[str] = None,
) -> DataLoader:
    """
    Build DataLoader with optional file include/exclude lists.
    Automatically excludes known invalid HDF5 files (by path and by numeric ID).
    """
    cfg = _load_yaml(yaml_path)
    files = _resolve_split_files(cfg, split, train_val_split=train_val_split)

    # --- Normalize all resolved file paths once
    files = [_norm(f) for f in files]

    # (0) Hardcoded auto-exclude list (known broken files)
    AUTO_EXCLUDE = {
        "/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/test/1264.hdf5",
        "/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/test/1265.hdf5",
        "/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/test/1266.hdf5",
        "/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/training/10248.hdf5",
        "/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/training/10249.hdf5",
        "/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/training/10250.hdf5",
        "/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/training/10251.hdf5",
        "/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/training/10252.hdf5",
    }
    AUTO_EXCLUDE = {_norm(p) for p in AUTO_EXCLUDE}

    # Optional: ID-based auto-exclude (robust to path layout)
    AUTO_EXCLUDE_IDS = {1264, 1265, 1266, 10248, 10249, 10250, 10251, 10252}

    before_auto = len(files)
    files = [f for f in files if (_norm(f) not in AUTO_EXCLUDE and _fname_id(f) not in AUTO_EXCLUDE_IDS)]
    removed_auto = before_auto - len(files)
    if removed_auto > 0:
        logger.warning(f"🚫 Auto-excluded {removed_auto} known broken files (A-SIM patch list, path/id).")

    # (1) include list (if provided)
    if include_list_path and os.path.exists(include_list_path):
        with open(include_list_path, "r", encoding="utf-8") as f:
            includes = {_norm(line.strip()) for line in f if line.strip()}
        before = len(files)
        files = [f for f in files if _norm(f) in includes]
        logger.info(f"✅ include_list applied ({len(files)}/{before}) from {include_list_path}")

    # (2) exclude list (if provided)
    if exclude_list_path and os.path.exists(exclude_list_path):
        with open(exclude_list_path, "r", encoding="utf-8") as f:
            excludes = {_norm(line.strip()) for line in f if line.strip()}
        before = len(files)
        files = [f for f in files if _norm(f) not in excludes]
        logger.info(f"🚫 exclude_list applied (removed {before - len(files)}) from {exclude_list_path}")

    # (3) validate HDF5 keys if requested
    if validate_keys:
        files = _filter_files_by_keys(files, target_field=target_field, strict=strict)
        if not files:
            raise RuntimeError(f"No valid HDF5 files remain for split='{split}' after validation.")

    # (4) Dataset
    dataset: Dataset = ASIMHDF5Dataset(files, target_field=target_field, dtype=dtype)

    # (5) Sample fraction
    if 0.0 < sample_fraction < 1.0:
        total_len = len(dataset)
        sample_size = max(1, int(round(sample_fraction * total_len)))
        split_offset = {"train": 0, "val": 1, "test": 2}[split]
        rng = np.random.default_rng((seed or 0) + split_offset)
        indices = np.sort(rng.choice(total_len, size=sample_size, replace=False))
        dataset = Subset(dataset, indices)
        logger.info(f"🔎 Sub-sampled {sample_size}/{total_len} ({sample_fraction*100:.1f}%)")

    logger.info(f"📦 Split='{split}' | files={len(files)} | batch={batch_size} | target='{target_field}'")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


# ============================================================
# Sanity check
# ============================================================
def sanity_check_sample(yaml_path: str, split="train", idx=0, target_field="rho"):
    cfg = _load_yaml(yaml_path)
    files = _resolve_split_files(cfg, split)
    files = _filter_files_by_keys(files, target_field, strict=False)
    if not (0 <= idx < len(files)):
        raise IndexError(f"idx out of range for split '{split}'")
    path = files[idx]
    with h5py.File(path, "r") as f:
        x = _ensure_input_channels(f["input"][:])
        if target_field == "rho":
            y = _ensure_target_3d(f["output_rho"][:])
        else:
            y = _ensure_target_3d(f["output_tscphi"][:])
            y = y*(0.72**-2); y -= y.mean(dtype=np.float64)
    logger.info(
        f"[SanityCheck] {split}[{idx}]={os.path.basename(path)} | "
        f"x.shape={x.shape}, y.shape={y.shape} | "
        f"x[min,max,mean]=({np.min(x):.3g},{np.max(x):.3g},{np.mean(x):.3g}) | "
        f"y[min,max,mean]=({np.min(y):.3g},{np.max(y):.3g},{np.mean(y):.3g})"
    )
