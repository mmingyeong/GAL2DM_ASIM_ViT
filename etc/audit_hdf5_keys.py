# /audit_hdf5_keys.py
import argparse
import csv
import h5py
import os
import re
import sys
import yaml
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

def natkey(p: str):
    t = re.split(r"(\d+)", os.path.basename(p))
    return tuple(int(x) if x.isdigit() else x for x in t)

def list_paths_from_yaml(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = cfg["asim_datasets_hdf5"]["base_path"]
    train_glob = os.path.join(base, cfg["asim_datasets_hdf5"]["training_set"]["path"])
    test_glob  = os.path.join(base, cfg["asim_datasets_hdf5"]["validation_set"]["path"])  # 'test/*.hdf5'
    train_paths = sorted(glob(train_glob), key=natkey)
    test_paths  = sorted(glob(test_glob), key=natkey)
    return train_paths, test_paths

def check_file_keys(p: str):
    """Return (is_valid, missing_or_error) where:
       - is_valid: bool
       - missing_or_error: list[str] of missing keys or ["error: ..."]"""
    required_input = "input"
    alt_targets = ("output_rho", "output_tscphi")

    try:
        # 빠른 메타 접근만(데이터 로드 X)
        with h5py.File(p, "r") as f:
            has_input = (required_input in f)
            has_any_target = any(k in f for k in alt_targets)
            if has_input and has_any_target:
                return True, []
            missing = []
            if not has_input:
                missing.append(required_input)
            if not has_any_target:
                missing.append("output_rho|output_tscphi")
            return False, missing
    except Exception as e:
        return False, [f"error: {e}"]

def main():
    parser = argparse.ArgumentParser(description="Audit A-SIM HDF5 files for required keys.")
    parser.add_argument("--yaml_path", type=str,
                        default="/home/mingyeong/2510_GAL2DM_ASIM_ViT/etc/asim_paths.yaml",
                        help="Path to asim_paths.yaml")
    parser.add_argument("--out_dir", type=str, default="audit_out",
                        help="Directory to save audit results")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers (ThreadPool)")
    parser.add_argument("--include", type=str, default="train,test",
                        help="Which sets to include: comma-separated from {train,test}")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_paths, test_paths = list_paths_from_yaml(args.yaml_path)

    include_sets = {s.strip() for s in args.include.split(",")}
    paths = []
    if "train" in include_sets:
        paths += train_paths
    if "test" in include_sets:
        paths += test_paths

    print(f"[audit] Total candidate files: {len(paths)} (train={len(train_paths)}, test={len(test_paths)})")

    valid = []
    invalid = []

    # 병렬 점검 (I/O bound → ThreadPoolExecutor)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        fut2path = {ex.submit(check_file_keys, p): p for p in paths}
        for fut in as_completed(fut2path):
            p = fut2path[fut]
            ok, info = fut.result()
            if ok:
                valid.append(p)
            else:
                invalid.append((p, info))

    # === 저장: 유효/무효 리스트 ===
    valid_txt = os.path.join(args.out_dir, "audit_valid.txt")
    invalid_txt = os.path.join(args.out_dir, "audit_invalid.txt")
    invalid_csv = os.path.join(args.out_dir, "audit_invalid.csv")
    summary_txt = os.path.join(args.out_dir, "audit_summary.txt")

    with open(valid_txt, "w", encoding="utf-8") as f:
        for p in valid:
            f.write(p + "\n")

    with open(invalid_txt, "w", encoding="utf-8") as f:
        for p, miss in invalid:
            f.write(f"{p}\tmissing:{'|'.join(miss)}\n")

    with open(invalid_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "missing_keys_or_error"])
        for p, miss in invalid:
            w.writerow([p, "|".join(miss)])

    # === 요약 ===
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("A-SIM HDF5 key audit summary\n")
        f.write("----------------------------------------\n")
        f.write(f"yaml_path: {args.yaml_path}\n")
        f.write(f"include sets: {args.include}\n")
        f.write(f"workers: {args.workers}\n")
        f.write("\n")
        f.write(f"total_checked: {len(paths)}\n")
        f.write(f"valid_count : {len(valid)}\n")
        f.write(f"invalid_count: {len(invalid)}\n")
        if invalid:
            # 상위 몇 개만 예시
            f.write("\nexamples_of_invalid (up to 20):\n")
            for p, miss in invalid[:20]:
                f.write(f" - {p} | {', '.join(miss)}\n")

    print(f"[audit] Done.")
    print(f"  valid:   {len(valid)}  -> {valid_txt}")
    print(f"  invalid: {len(invalid)} -> {invalid_txt}, {invalid_csv}")
    print(f"  summary:                 {summary_txt}")

if __name__ == "__main__":
    sys.exit(main())
