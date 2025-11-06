#!/bin/bash
#SBATCH -J vitunet3d_eval_fast
#SBATCH -o logs/vitunet3d_eval_fast.%j.out
#SBATCH -e logs/vitunet3d_eval_fast.%j.err
#SBATCH -p h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmingyeong@kasi.re.kr

# ========================================
# Usage:
#   sbatch /home/mingyeong/GAL2DM_ASIM_ViT/scripts/1103_eval.sh \
#     --pred_dir /home/mingyeong/GAL2DM_ASIM_ViT/results/vit/28846 \
#     [--alex_dir /scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/predictions] \
#     [--alex_tpl '/scratch/.../test_{idx:03d}_*rho*.npy']
#
# Notes:
#   - Provide EITHER --alex_dir (directory) OR --alex_tpl (explicit template).
#   - If neither is provided, a sensible default directory is used.
# ========================================

# -------- Parse args --------
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 --pred_dir <prediction_directory> [--alex_dir <dir>] [--alex_tpl <template>]"
  exit 1
fi

PRED_DIR=""
ALEX_DIR=""
ALEX_TPL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pred_dir) PRED_DIR="$2"; shift 2;;
    --alex_dir) ALEX_DIR="$2"; shift 2;;
    --alex_tpl) ALEX_TPL="$2"; shift 2;;
    *) echo "[WARN] Unknown arg: $1"; shift;;
  esac
done

if [[ -z "${PRED_DIR}" ]]; then
  echo "[ERROR] --pred_dir is required." >&2
  exit 1
fi

# Normalize and validate
PRED_DIR="${PRED_DIR%/}"
if [[ ! -d "${PRED_DIR}" ]]; then
  echo "[ERROR] pred_dir not found: ${PRED_DIR}" >&2
  exit 1
fi

# Default Alex directory if nothing provided
if [[ -z "${ALEX_DIR}" && -z "${ALEX_TPL}" ]]; then
  ALEX_DIR="/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/predictions"
fi

# Decide what to pass to Python
if [[ -n "${ALEX_TPL}" ]]; then
  ALEX_ARG="${ALEX_TPL}"
elif [[ -n "${ALEX_DIR}" ]]; then
  # pass directory verbatim; Python will normalize to a template
  ALEX_ARG="${ALEX_DIR%/}/"
else
  echo "[ERROR] One of --alex_dir or --alex_tpl must be provided (or leave both empty to use default)." >&2
  exit 1
fi

# ========================================
# Environment setup
# ========================================
module purge
module load cuda/12.1.1
source ~/.bashrc
conda activate torch
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ========================================
# Paths
# ========================================
PROJECT_ROOT="/home/mingyeong/GAL2DM_ASIM_ViT"
EVAL_PY="${PROJECT_ROOT}/src/eval_compare.py"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"

if [[ ! -f "${EVAL_PY}" ]]; then
  echo "[ERROR] evaluator not found: ${EVAL_PY}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}" || exit 1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ========================================
# Output configuration
# ========================================
CASE_LABEL="$(basename "${PRED_DIR}")"
TS_FROM_PRED="$(basename "$(dirname "${PRED_DIR}")")"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="results/vit_eval/${RUN_TS}/${CASE_LABEL}"
mkdir -p "logs" "${OUT_ROOT}"

# Auto-discover loss CSV
CAND_DIR="${PROJECT_ROOT}/results/vit/${TS_FROM_PRED}"
if [[ -d "${CAND_DIR}" ]]; then
  LOSS_CSV_RESOLVED="$(ls -1 "${CAND_DIR}"/*log.csv 2>/dev/null | head -n1 || true)"
else
  LOSS_CSV_RESOLVED=""
fi

# ========================================
# Eval parameters (defaults)
# ========================================
SLICE_AXIS=2
SLICE_INDEX=center
MAP_COUNT=5
KS_GLOBAL_CAP=200000
JOINT_SAMPLE=50000
PDF_BINS=120
JOINT_BINS=120
VOXEL_SIZE=$(python - <<'PY'
print(205.0/250.0)
PY
)
RMAX=10.0
N_R_BINS=24

# ========================================
# Print configuration
# ========================================
echo "=== [EVAL FAST START] $(date) on $(hostname) ==="
echo "[INFO] pred_dir       : ${PRED_DIR}"
echo "[INFO] out_root       : ${OUT_ROOT}"
echo "[INFO] yaml_path      : ${YAML_PATH}"
if [[ -n "${ALEX_TPL}" ]]; then
  echo "[INFO] alex_tpl       : ${ALEX_TPL}"
else
  echo "[INFO] alex_dir       : ${ALEX_DIR}"
fi
echo "[INFO] loss_csv       : ${LOSS_CSV_RESOLVED:-<auto-not-found>}"
echo "[INFO] map_count      : ${MAP_COUNT}"
echo "[INFO] ks_global_cap  : ${KS_GLOBAL_CAP}"
which python
python - <<'PY'
import torch, os
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY
nvidia-smi || true

# ========================================
# Run evaluator
# ========================================
srun python -u "${EVAL_PY}" \
  --yaml_path "${YAML_PATH}" \
  --pred_dir "${PRED_DIR}" \
  --alex_tpl "${ALEX_ARG}" \
  --out_dir "${OUT_ROOT}" \
  --slice_axis ${SLICE_AXIS} \
  --slice_index ${SLICE_INDEX} \
  --map_count ${MAP_COUNT} \
  --ks_global_cap ${KS_GLOBAL_CAP} \
  --joint_sample ${JOINT_SAMPLE} \
  --pdf_bins ${PDF_BINS} \
  --joint_bins ${JOINT_BINS} \
  --voxel_size ${VOXEL_SIZE} \
  --rmax ${RMAX} \
  --n_r_bins ${N_R_BINS} \
  --label_pred "${CASE_LABEL}" \
  $( [ -n "${LOSS_CSV_RESOLVED}" ] && echo --loss_csv "${LOSS_CSV_RESOLVED}" ) \
  --save_latex \
  2>&1 | tee -a "logs/vitunet3d_eval_fast_${SLURM_JOB_ID}.log"

EXIT_CODE=${PIPESTATUS[0]}
echo "=== [EVAL FAST END] $(date) (exit=${EXIT_CODE}) ==="
exit ${EXIT_CODE}
