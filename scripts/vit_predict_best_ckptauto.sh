#!/bin/bash
#SBATCH -J vit_pred_auto
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_ViT/logs/vit_predict_best_ckptauto.%j.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_ViT/logs/vit_predict_best_ckptauto.%j.err
#SBATCH -p a100
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 0-03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmingyeong@kasi.re.kr
#SBATCH --chdir=/home/mingyeong/GAL2DM_ASIM_ViT

set -euo pipefail

module purge
module load cuda/12.1.1

source ~/.bashrc
conda activate torch

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_MODULE_LOADING=LAZY
ulimit -n 65535

PROJECT_ROOT="/home/mingyeong/GAL2DM_ASIM_ViT"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"

CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/results/vit/lr_1e-3/134180_1}"
SCRATCH_BASE="/scratch/mingyeong/GAL2DM_pred/vit"
HOME_BASE="/home/mingyeong/GAL2DM_pred/vit"

BATCH_SIZE="${BATCH_SIZE:-1}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-1.0}"
USE_AMP="${USE_AMP:-1}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/logs/predict/vit/${RUN_TS}"
mkdir -p "${LOG_DIR}"

cd "${PROJECT_ROOT}" || { echo "[FATAL] cd failed: ${PROJECT_ROOT}"; exit 2; }
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "=== [JOB START] $(date) on $(hostname) ==="
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "[INFO] SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-N/A}"
echo "[INFO] PWD=$(pwd)"
echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] YAML_PATH=${YAML_PATH}"
echo "[INFO] CKPT_DIR=${CKPT_DIR}"

if [ ! -d "${CKPT_DIR}" ]; then
  echo "[ERROR] CKPT_DIR not found: ${CKPT_DIR}"
  exit 1
fi

if [ -z "${MODEL_PATH:-}" ]; then
  BEST_CKPT="$(ls -t "${CKPT_DIR}"/*best*.pt 2>/dev/null | head -n 1 || true)"
  if [ -n "${BEST_CKPT}" ]; then
    MODEL_PATH="${BEST_CKPT}"
  else
    MODEL_PATH="$(ls -t "${CKPT_DIR}"/*.pt 2>/dev/null | head -n 1 || true)"
  fi
fi

if [ -z "${MODEL_PATH:-}" ]; then
  echo "[ERROR] No checkpoint (.pt) found in ${CKPT_DIR}"
  exit 1
fi

echo "[INFO] Using checkpoint: ${MODEL_PATH}"

RUN_STEM="$(basename "${CKPT_DIR}")"
LR_STEM="$(basename "$(dirname "${CKPT_DIR}")")"
if [[ "${LR_STEM}" != lr_* ]]; then
  LR_STEM="lr_unknown"
fi

# scratch 우선, 실패하면 home fallback
if mkdir -p "${SCRATCH_BASE}/${LR_STEM}/${RUN_STEM}" 2>/dev/null; then
  OUT_DIR_BASE="${SCRATCH_BASE}"
  echo "[INFO] Using scratch output base: ${OUT_DIR_BASE}"
else
  OUT_DIR_BASE="${HOME_BASE}"
  mkdir -p "${OUT_DIR_BASE}/${LR_STEM}/${RUN_STEM}"
  echo "[WARN] scratch unavailable; fallback to home output base: ${OUT_DIR_BASE}"
fi

PRED_OUT_DIR="${OUT_DIR_BASE}/${LR_STEM}/${RUN_STEM}"
mkdir -p "${PRED_OUT_DIR}"

echo "[INFO] Prediction output dir: ${PRED_OUT_DIR}"
echo "[INFO] Auto-infer mode: architecture will be inferred from checkpoint filename when args remain default"

AMP_FLAG=""
if [ "${USE_AMP}" = "1" ]; then
  AMP_FLAG="--amp"
fi

echo "=== [ENV CHECK] $(date) ==="
which python
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
nvidia-smi || echo "[WARN] nvidia-smi not available"

echo "=== [PREDICT START] $(date) ==="

srun python -u "${PROJECT_ROOT}/src/predict.py" \
  --yaml_path "${YAML_PATH}" \
  --output_dir "${PRED_OUT_DIR}" \
  --model_path "${MODEL_PATH}" \
  --device cuda \
  --batch_size "${BATCH_SIZE}" \
  --sample_fraction "${SAMPLE_FRACTION}" \
  ${AMP_FLAG} \
  2>&1 | tee -a "${LOG_DIR}/vit_predict_ckptauto_${SLURM_JOB_ID}.log"

EXIT_CODE=${PIPESTATUS[0]}

echo "=== [PREDICT END] $(date) (exit=${EXIT_CODE}) ==="
exit ${EXIT_CODE}