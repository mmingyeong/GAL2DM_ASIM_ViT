#!/bin/bash
#SBATCH -J vit_predict_best
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_ViT/logs/vit_predict_best.%j.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_ViT/logs/vit_predict_best.%j.err
#SBATCH -p a100
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 0-03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmingyeong@kasi.re.kr

set -e -o pipefail

module purge
module load cuda/12.1.1

source ~/.bashrc
conda activate torch

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_MODULE_LOADING=LAZY
ulimit -n 65535

PROJECT_ROOT="/home/mingyeong/GAL2DM_ASIM_ViT"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"

CKPT_DIR="${CKPT_DIR:-/home/mingyeong/GAL2DM_ASIM_ViT/results/vit/lr_1e-3/134180_1}"
OUT_DIR_BASE="/home/mingyeong/GAL2DM_pred/vit"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/logs/predict/vit/${RUN_TS}"
mkdir -p "${LOG_DIR}" "${OUT_DIR_BASE}"

cd "${PROJECT_ROOT}" || { echo "[FATAL] cd failed"; exit 2; }
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

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

PRED_OUT_DIR="${OUT_DIR_BASE}/${LR_STEM}/${RUN_STEM}"
mkdir -p "${PRED_OUT_DIR}"

echo "[INFO] Prediction output dir: ${PRED_OUT_DIR}"

BATCH_SIZE=1
SAMPLE_FRACTION=1.0
AMP_FLAG="--amp"

echo "=== [PREDICT START] $(date) on $(hostname) ==="
which python
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
nvidia-smi || echo "nvidia-smi not available"

srun python -u "${PROJECT_ROOT}/src/predict.py" \
  --yaml_path "${YAML_PATH}" \
  --output_dir "${PRED_OUT_DIR}" \
  --model_path "${MODEL_PATH}" \
  --device cuda \
  --batch_size ${BATCH_SIZE} \
  --sample_fraction ${SAMPLE_FRACTION} \
  --frame_patch_size 8 \
  --image_patch_size 8 \
  --frame_patch_stride 8 \
  --image_patch_stride 8 \
  ${AMP_FLAG} \
  2>&1 | tee -a "${LOG_DIR}/vit_predict_${SLURM_JOB_ID}.log"

EXIT_CODE=${PIPESTATUS[0]}
echo "=== [PREDICT END] $(date) (exit=${EXIT_CODE}) ==="
exit ${EXIT_CODE}