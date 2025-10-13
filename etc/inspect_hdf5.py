# /inspect_hdf5.py
import argparse
import os
import sys
import h5py
import numpy as np
from typing import Iterable

def human(n):
    # human-readable size
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"

def inspect_file(path: str, peek: bool = False, out=None):
    pr = (lambda s: print(s, file=out)) if out else print

    if not os.path.exists(path):
        pr(f"[MISS] {path} (file not found)")
        return

    try:
        stat = os.stat(path)
        pr("="*80)
        pr(f"[FILE] {path}")
        pr(f" size: {human(stat.st_size)} | mtime: {os.path.getmtime(path):.0f}")
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            pr(f" top-level keys: {keys if keys else '[]'}")

            if not keys:
                pr(" (no top-level objects)")
                return

            for k in keys:
                obj = f[k]
                if isinstance(obj, h5py.Dataset):
                    try:
                        shape = obj.shape
                        dtype = obj.dtype
                        comp = obj.compression
                        chunks = obj.chunks
                        # logical size (approx): dtype.size * prod(shape)
                        try:
                            logical = dtype.itemsize * int(np.prod(shape))
                        except Exception:
                            logical = 0
                        # physical storage on disk (HDF5 API)
                        try:
                            physical = obj.id.get_storage_size()
                        except Exception:
                            physical = 0

                        pr(f"  - [{k}] DATASET")
                        pr(f"      shape={shape}, dtype={dtype}, compression={comp}, chunks={chunks}")
                        if logical:
                            pr(f"      size: logical~{human(logical)}, physical~{human(physical)}")
                        else:
                            pr(f"      size: physical~{human(physical)}")
                        if peek:
                            # 아주 작은 코너만 살짝 보기 (0 슬라이스 가능한 축만)
                            sl = tuple(0 if s>1 else slice(None) for s in shape)
                            try:
                                sample = obj[sl]
                                # 숫자형일 때만 간단 통계
                                if np.issubdtype(sample.dtype, np.number):
                                    pr(f"      peek[{sl}]: min={np.min(sample):.4g}, max={np.max(sample):.4g}, mean={np.mean(sample):.4g}")
                                else:
                                    pr(f"      peek[{sl}]: dtype={sample.dtype}, shape={sample.shape}")
                            except Exception as e:
                                pr(f"      peek failed: {e}")

                    except Exception as e:
                        pr(f"  - [{k}] DATASET (metadata read failed: {e})")
                elif isinstance(obj, h5py.Group):
                    pr(f"  - [{k}] GROUP (contains {len(obj.keys())} items)")
                else:
                    pr(f"  - [{k}] UNKNOWN object: {type(obj)}")

    except Exception as e:
        pr(f"[ERR ] {path}: {e}")

def main(argv: Iterable[str] = None):
    ap = argparse.ArgumentParser(description="Lightweight HDF5 inspector (metadata only).")
    ap.add_argument("paths", nargs="+", help="HDF5 file paths to inspect")
    ap.add_argument("--peek", action="store_true", help="peek tiny slice from each dataset")
    ap.add_argument("--out", type=str, default=None, help="write full report to this file")
    args = ap.parse_args(argv)

    out = open(args.out, "w") if args.out else None
    try:
        for p in args.paths:
            inspect_file(p, peek=args.peek, out=out)
    finally:
        if out:
            out.close()

if __name__ == "__main__":
    sys.exit(main())
