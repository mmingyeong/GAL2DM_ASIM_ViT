"""
src/data_loader.py

Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Created: 2025-10-13

Description:
    Data loader for A-SIM HDF5 training/validation/test datasets.
    - Inputs:  ngal (galaxy number density), vpec (peculiar velocity)  -> 2 channels
    - Targets: output_rho (default) or output_tscphi (optional; NOT output_phi)
      * If target == 'tscphi', multiply by (0.72**-2) and subtract its mean.

Notes:
    - sample_fraction ∈ (0,1): train/val/test 모든 split에 동일 적용.
    - validate_keys=True: 로더 초기화 시 파일을 선별하여 필수 키가 없으면 제외.
"""

from __future__ import annotations

import os
import re
from glob import glob
from typing import List, Tuple, Literal, Sequence, Optional

import yaml
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader

from src.logger import get_logger

logger = get_logger("data_loader", log_dir="logs")


# ----------------------------
# Utilities
# ----------------------------
def _natkey(path: str):
    """Natural sort key (numbers inside filenames sorted numerically)."""
    tokens = re.split(r"(\d+)", os.path.basename(path))
    return tuple(int(t) if t.isdigit() else t for t in tokens)


def _load_yaml(yaml_path: str) -> dict:
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _resolve_split_files(
    yaml_cfg: dict,
    split: Literal["train", "val", "test"],
    train_val_split: float = 0.8,
) -> List[str]:
    """
    Resolve file list for a given split using asim_paths.yaml.

    Rule:
        - 'training/*.hdf5' → split into train/val (default 80/20)
        - 'test/*.hdf5'     → used as test set
    """
    base = yaml_cfg["asim_datasets_hdf5"]["base_path"]
    train_pattern = os.path.join(base, yaml_cfg["asim_datasets_hdf5"]["training_set"]["path"])
    test_pattern = os.path.join(base, yaml_cfg["asim_datasets_hdf5"]["validation_set"]["path"])

    train_files = sorted(glob(train_pattern), key=_natkey)
    test_files = sorted(glob(test_pattern), key=_natkey)

    if not train_files:
        raise FileNotFoundError(f"No HDF5 training files found in {train_pattern}")

    n_train_total = len(train_files)
    n_train_split = int(n_train_total * train_val_split)

    if split == "train":
        selected = train_files[:n_train_split]
    elif split == "val":
        selected = train_files[n_train_split:]
    elif split == "test":
        selected = test_files
    else:
        raise ValueError(f"Invalid split '{split}'. Use ['train','val','test'].")

    logger.info(
        f"📂 Split '{split}': {len(selected)} files "
        f"({n_train_split}/{n_train_total} train-val split, "
        f"{len(test_files)} test files)"
    )
    return selected


# ----------------------------
# Robust shape helpers
# ----------------------------
def _squeeze_leading_ones_to_nd(arr: np.ndarray, nd: int) -> np.ndarray:
    """
    Remove *leading* singleton axes (size==1) until arr.ndim == nd
    or until the leading axis is not singleton.

    Examples:
      (1,1,128,128,128) with nd=3 -> (128,128,128)
      (1,2,128,128,128) with nd=4 -> (2,128,128,128)
    """
    out = arr
    while out.ndim > nd and out.shape[0] == 1:
        out = out[0]
    return out


def _ensure_input_channels(arr: np.ndarray) -> np.ndarray:
    """
    Make input consistently (2, D, H, W).
    Accepts e.g. (1,2,D,H,W), (2,D,H,W), (1,1,2,D,H,W) etc.
    """
    out = _squeeze_leading_ones_to_nd(arr, nd=4)
    if out.ndim != 4 or out.shape[0] != 2:
        raise ValueError(f"'input' must be (2,D,H,W) after normalization; got {arr.shape} -> {out.shape}")
    return out


def _ensure_target_3d(arr: np.ndarray) -> np.ndarray:
    """
    Make target consistently (D, H, W).
    Accepts e.g. (1,D,H,W), (D,H,W), (1,1,D,H,W) etc.
    """
    out = _squeeze_leading_ones_to_nd(arr, nd=3)
    if out.ndim != 3:
        raise ValueError(f"target must be 3D after normalization; got {arr.shape} -> {out.shape}")
    return out


# ----------------------------
# Dataset
# ----------------------------
class ASIMHDF5Dataset(Dataset):
    """
    Dataset for A-SIM HDF5 single-sample files.

    Each HDF5 file contains:
      - 'input' : shape (1, 2, 128, 128, 128) or (2, 128, 128, 128)
                  channels: [ngal, vpec]
      - 'output_rho'    : shape (1, 128, 128, 128) or (128, 128, 128); mean density already normalized to 1
      - 'output_tscphi' : shape (1, 128, 128, 128) or (128, 128, 128); must be scaled by (0.72**-2) and de-meaned

    Returns:
      x: FloatTensor (2, 128, 128, 128)
      y: FloatTensor (1, 128, 128, 128)
    """

    def __init__(
        self,
        file_paths: Sequence[str],
        target_field: Literal["rho", "tscphi"] = "rho",
        dtype: torch.dtype = torch.float32,
    ):
        self.file_paths = list(file_paths)
        self.target_field = target_field
        self.dtype = dtype

        assert self.target_field in ("rho", "tscphi"), \
            "target_field must be 'rho' or 'tscphi' (use 'tscphi', not 'phi')."

        logger.info(
            f"🔍 Initializing ASIMHDF5Dataset | samples: {len(self.file_paths)} | "
            f"target: {self.target_field}"
        )

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fpath = self.file_paths[idx]
        with h5py.File(fpath, "r") as f:
            # ---- Inputs: [ngal, vpec] -> (2,D,H,W) ----
            if "input" not in f:
                raise KeyError(f"'input' dataset not found in {fpath}")
            x_np = f["input"][:]                   # (1,2,D,H,W) or (2,D,H,W) etc.
            x_np = _ensure_input_channels(x_np)    # -> (2,D,H,W)

            # ---- Target ----
            if self.target_field == "rho":
                if "output_rho" not in f:
                    raise KeyError(f"'output_rho' not found in {fpath}")
                y_np = f["output_rho"][:]          # (1,D,H,W) or (D,H,W) etc.
                y_np = _ensure_target_3d(y_np)     # -> (D,H,W)
            else:  # 'tscphi'
                if "output_tscphi" not in f:
                    raise KeyError(
                        f"'output_tscphi' not found in {fpath}. "
                        "Do not use 'output_phi'."
                    )
                y_np = f["output_tscphi"][:]
                y_np = _ensure_target_3d(y_np)     # -> (D,H,W)
                # Scale by h^-2 (h = 0.72), then subtract mean
                y_np = y_np * (0.72 ** (-2))
                y_np = y_np - y_np.mean(dtype=np.float64)

        # Cast & shape fixes
        x = torch.from_numpy(np.ascontiguousarray(x_np)).to(self.dtype)              # (2,D,H,W)
        y = torch.from_numpy(np.ascontiguousarray(y_np)).to(self.dtype).unsqueeze(0)  # (1,D,H,W)

        return x, y


# ----------------------------
# Pre-filter invalid files
# ----------------------------
def _filter_files_by_keys(
    file_paths: Sequence[str],
    target_field: Literal["rho", "tscphi"],
    strict: bool = False,
) -> List[str]:
    """
    Keep only files that contain required datasets:
      - always requires 'input'
      - requires 'output_rho' if target='rho'
      - requires 'output_tscphi' if target='tscphi'
    If strict=True, raise on first invalid file; otherwise skip and log a warning.
    """
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
                    missing = []
                    with h5py.File(p, "r") as f:
                        if "input" not in f: missing.append("input")
                        if req_target not in f: missing.append(req_target)
                    raise KeyError(f"{os.path.basename(p)} missing keys: {missing}")
        except Exception as e:
            dropped.append(p)
            if strict:
                raise
            else:
                logger.warning(f"⚠️ Skip invalid file: {p} | reason: {e}")

    if dropped:
        logger.warning(f"⚠️ Filtered out {len(dropped)} invalid file(s) without required keys.")
    logger.info(f"✅ Valid files kept: {len(kept)} / {len(file_paths)}")
    return kept


# ----------------------------
# Public API
# ----------------------------
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
) -> DataLoader:
    """
    Construct a DataLoader for the A-SIM dataset using an asim_paths.yaml configuration.

    Parameters
    ----------
    yaml_path : str
        Path to 'asim_paths.yaml'.
    split : {'train','val','test'}
        Split to load.
    batch_size : int
        DataLoader batch size.
    shuffle : bool
        Whether to shuffle indices.
    sample_fraction : float
        Fraction of files to sample for **all** splits when 0<frac<1.0.
    num_workers : int
        Number of DataLoader workers.
    pin_memory : bool
        Pin memory for faster host->GPU transfer.
    target_field : {'rho','tscphi'}
        Supervision target. Default 'rho'.
    dtype : torch.dtype
        Returned tensor dtype (default float32).
    seed : Optional[int]
        Base RNG seed. Each split uses (seed + offset) with offset in {0,1,2}.
    train_val_split : float
        Fraction of training/*.hdf5 used as 'train' (rest used as 'val').
    validate_keys : bool
        If True, pre-scan files and drop those missing required datasets.
    strict : bool
        If True, raise error on first invalid file; else skip & log.

    Returns
    -------
    torch.utils.data.DataLoader
    """
    cfg = _load_yaml(yaml_path)
    files = _resolve_split_files(cfg, split, train_val_split=train_val_split)

    # 1) 사전 검증(선별)
    if validate_keys:
        files = _filter_files_by_keys(files, target_field=target_field, strict=strict)
        if not files:
            raise RuntimeError(f"No valid HDF5 files remain for split='{split}' after key validation.")

    # 2) Dataset 구성
    dataset: Dataset = ASIMHDF5Dataset(
        file_paths=files,
        target_field=target_field,
        dtype=dtype,
    )

    # 3) (옵션) 분할별 샘플링
    if 0.0 < sample_fraction < 1.0:
        total_len = len(dataset)
        sample_size = max(1, int(round(sample_fraction * total_len)))
        split_offset = {"train": 0, "val": 1, "test": 2}[split]
        if seed is not None:
            rng = np.random.default_rng(int(seed) + split_offset)
            indices = rng.choice(total_len, size=sample_size, replace=False)
        else:
            indices = np.random.choice(total_len, size=sample_size, replace=False)
        dataset = Subset(dataset, np.sort(indices))
        logger.info(
            f"🔎 Sub-sampled {sample_size}/{total_len} {split} files "
            f"({sample_fraction*100:.1f}%)."
        )

    # 4) 진단 로그
    logger.info(
        f"📦 Split='{split}' | batches of {batch_size} | files={len(files)} | "
        f"target='{target_field}' | dtype={dtype}"
    )

    # 5) DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


# ----------------------------
# Convenience helpers
# ----------------------------
def sanity_check_sample(
    yaml_path: str,
    split: str = "train",
    idx: int = 0,
    target_field: Literal["rho", "tscphi"] = "rho"
):
    """
    Quick I/O sanity check for a single sample (no DataLoader).
    Prints shapes and simple stats.
    """
    cfg = _load_yaml(yaml_path)
    files = _resolve_split_files(cfg, split)
    files = _filter_files_by_keys(files, target_field=target_field, strict=False)

    if not (0 <= idx < len(files)):
        raise IndexError(f"idx out of range for split '{split}': 0..{len(files)-1}")

    path = files[idx]
    with h5py.File(path, "r") as f:
        x = f["input"][:]
        x = _ensure_input_channels(x)  # -> (2,D,H,W)

        if target_field == "rho":
            y = f["output_rho"][:]
            y = _ensure_target_3d(y)   # -> (D,H,W)
        else:
            y = f["output_tscphi"][:]
            y = _ensure_target_3d(y)   # -> (D,H,W)
            y = y * (0.72 ** (-2))
            y = y - y.mean(dtype=np.float64)

    logger.info(
        f"[SanityCheck] {split}[{idx}] = {os.path.basename(path)} | "
        f"x.shape={x.shape}, y.shape={y.shape} | "
        f"x stats: min={np.min(x):.4g}, max={np.max(x):.4g}, mean={np.mean(x):.4g} | "
        f"y stats: min={np.min(y):.4g}, max={np.max(y):.4g}, mean={np.mean(y):.4g}"
    )
