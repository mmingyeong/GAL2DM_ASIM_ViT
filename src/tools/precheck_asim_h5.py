#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precheck A-SIM HDF5 files (multiprocessing + tqdm + detailed logging)

Usage:
  python precheck_asim_h5.py \
    --yaml /home/mingyeong/GAL2DM_ASIM_VNET/etc/asim_paths.yaml \
    --target_field rho \
    --outdir ./filelists \
    --num_workers 8 \
    --verbose
"""

import os
import re
import csv
import time
import argparse
import logging
from logging import handlers
from glob import glob
from datetime import datetime
import multiprocessing as mp

import yaml
import h5py
from tqdm import tqdm


# ----------------------------
# Logging setup
# ----------------------------
def setup_logger(log_dir: str, verbose: bool = False) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"precheck_{ts}.log")

    logger = logging.getLogger("precheck_asim_h5")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch_fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
    ch.setFormatter(ch_fmt)
    logger.addHandler(ch)

    fh = handlers.RotatingFileHandler(log_path, maxBytes=10_000_000, backupCount=2, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s:%(lineno)d - %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fh_fmt)
    logger.addHandler(fh)

    logger.info(f"Log file: {log_path}")
    return logger


# ----------------------------
# Utilities
# ----------------------------
def natkey(path: str):
    tokens = re.split(r"(\d+)", os.path.basename(path))
    return tuple(int(t) if t.isdigit() else t for t in tokens)


def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_patterns(cfg):
    base = cfg["asim_datasets_hdf5"]["base_path"]
    train_pat = os.path.join(base, cfg["asim_datasets_hdf5"]["training_set"]["path"])
    test_pat  = os.path.join(base, cfg["asim_datasets_hdf5"]["validation_set"]["path"])
    return train_pat, test_pat


def resolve_files(cfg, logger: logging.Logger):
    train_pat, test_pat = resolve_patterns(cfg)
    train = sorted(glob(train_pat), key=natkey)
    test  = sorted(glob(test_pat),  key=natkey)
    if not train:
        logger.warning(f"No training files found: {train_pat}")
    if not test:
        logger.warning(f"No validation/test files found: {test_pat}")
    logger.info(f"Found train={len(train)}, test={len(test)}")
    return train, test


def split_train_val(train_files, split=0.8):
    n = len(train_files)
    k = int(n * split)
    return train_files[:k], train_files[k:]


def write_list(paths, outpath):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(p + "\n")


def write_bad_reasons_csv(bad_records, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "reason"])
        w.writeheader()
        for r in bad_records:
            w.writerow(r)


# ----------------------------
# Worker function for multiprocessing
# ----------------------------
def check_file(args):
    """Check single HDF5 file for required keys."""
    path, target_field = args
    need = ["input"]
    need += ["output_rho"] if target_field == "rho" else ["output_tscphi"]

    try:
        with h5py.File(path, "r") as f:
            for k in need:
                if k not in f:
                    return (path, False, f"missing:{k}")
        return (path, True, "")
    except Exception as e:
        return (path, False, f"open_error:{type(e).__name__}:{e}")


# ----------------------------
# Parallel scanning
# ----------------------------
def scan_split_mp(name, files, target_field, outdir, logger, num_workers):
    if not files:
        logger.warning(f"[{name}] no files to scan.")
        return [], [], []

    logger.info(f"[{name}] scanning {len(files)} files with {num_workers} workers...")

    t0 = time.time()
    good, bad, bad_records = [], [], []
    reason_stats = {}

    # multiprocessing pool
    with mp.Pool(processes=num_workers) as pool:
        for path, ok, reason in tqdm(
            pool.imap_unordered(check_file, [(p, target_field) for p in files]),
            total=len(files),
            desc=f"scan:{name}",
            unit="file",
        ):
            if ok:
                good.append(path)
            else:
                bad.append(path)
                bad_records.append({"path": path, "reason": reason})
                reason_stats[reason] = reason_stats.get(reason, 0) + 1
                logger.debug(f"[{name}] BAD: {path} | {reason}")

    dt = time.time() - t0
    logger.info(f"[{name}] total={len(files)}, good={len(good)}, bad={len(bad)} | {dt:.1f}s")
    if reason_stats:
        logger.info(f"[{name}] bad reasons: " + ", ".join([f"{k}={v}" for k, v in sorted(reason_stats.items())]))

    # Save results
    write_list(good, os.path.join(outdir, f"{name}_good.txt"))
    write_list(bad,  os.path.join(outdir, f"{name}_bad.txt"))
    write_bad_reasons_csv(bad_records, os.path.join(outdir, f"bad_reasons_{name}.csv"))

    return good, bad, bad_records


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, help="Path to asim_paths.yml")
    ap.add_argument("--target_field", choices=["rho", "tscphi"], default="rho")
    ap.add_argument("--train_val_split", type=float, default=0.8)
    ap.add_argument("--outdir", default="./filelists")
    ap.add_argument("--logdir", default="./filelists")
    ap.add_argument("--num_workers", type=int, default=max(1, mp.cpu_count() // 2))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    logger = setup_logger(args.logdir, verbose=args.verbose)
    logger.info(f"Args: {vars(args)}")

    cfg = load_yaml(args.yaml)
    train_all, test = resolve_files(cfg, logger)
    train, val = split_train_val(train_all, args.train_val_split)

    g_tr, b_tr, _, = scan_split_mp("train", train, args.target_field, args.outdir, logger, args.num_workers)
    g_va, b_va, _, = scan_split_mp("val",   val,   args.target_field, args.outdir, logger, args.num_workers)
    g_te, b_te, _, = scan_split_mp("test",  test,  args.target_field, args.outdir, logger, args.num_workers)

    exclude_all = sorted(set(b_tr + b_va + b_te), key=natkey)
    write_list(exclude_all, os.path.join(args.outdir, "exclude_bad_all.txt"))

    logger.info("====== SUMMARY ======")
    logger.info(f"train: total={len(train)}, good={len(g_tr)}, bad={len(b_tr)}")
    logger.info(f"val:   total={len(val)},   good={len(g_va)}, bad={len(b_va)}")
    logger.info(f"test:  total={len(test)},  good={len(g_te)}, bad={len(b_te)}")
    logger.info(f"exclude_bad_all.txt: {len(exclude_all)} files")
    logger.info("=====================")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # for safety
    main()
