#!/bin/bash
#SBATCH -J vit3d_predict_ckpt
#SBATCH -o logs/vit3d_predict_ckpt.%j.out
#SBATCH -e logs/vit3d_predict_ckpt.%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 0-03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmingyeong@kasi.re.kr

# ================================
# Environment
# ================================
module purge
module load cuda/12.1.1
source ~/.bashrc
conda activate torch
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ================================
# Paths (fixed CKPT_DIR per your message)
# ================================
PROJECT_ROOT="/home/mingyeong/2510_GAL2DM_ASIM_ViT"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"
CKPT_DIR="${CKPT_DIR:-results/vit/20251014_142228/patch8}"   # <-- fixed target
OUT_DIR_BASE="results/vit_predictions"

cd "${PROJECT_ROOT}" || exit 1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/${RUN_TS}"
mkdir -p "${LOG_DIR}" "${OUT_DIR_BASE}"

# Resolve checkpoint inside CKPT_DIR
if [ -z "${MODEL_PATH:-}" ]; then
  if [ ! -d "${CKPT_DIR}" ]; then
    echo "[ERROR] CKPT_DIR not found: ${CKPT_DIR}"
    exit 1
  fi
  BEST_CKPT="$(ls -t ${CKPT_DIR}/*best*.pt 2>/dev/null | head -n 1)"
  if [ -n "${BEST_CKPT}" ]; then
    MODEL_PATH="${BEST_CKPT}"
  else
    MODEL_PATH="$(ls -t ${CKPT_DIR}/*.pt 2>/dev/null | head -n 1)"
  fi
  if [ -z "${MODEL_PATH}" ]; then
    echo "[ERROR] No .pt file in ${CKPT_DIR}"
    exit 1
  fi
fi
echo "[INFO] Using checkpoint: ${MODEL_PATH}"

# Output directory mirrors the checkpoint folder name
RUN_STEM="$(basename "${CKPT_DIR}")"
PRED_OUT_DIR="${OUT_DIR_BASE}/${RUN_STEM}"
mkdir -p "${PRED_OUT_DIR}"

# ================================
# Model/config (match training)
# ================================
IMAGE_SIZE=128
FRAMES=128
PATCH_SPATIAL=8
PATCH_DEPTH=8
EMB_DIM=256
DEPTH=3           # your debug training used 3
HEADS=8
MLP_DIM=512
BATCH_SIZE=1
AMP_FLAG="--amp"

echo "=== [PREDICT START] $(date) on $(hostname) ==="
which python
python - <<'PY'
import torch, os
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("torch.version.cuda:", getattr(torch.version, "cuda", None))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY
nvidia-smi || echo "nvidia-smi not available"

set -o pipefail
srun python -u "${PROJECT_ROOT}/src/predict.py" \
  --yaml_path "${YAML_PATH}" \
  --output_dir "${PRED_OUT_DIR}" \
  --model_path "${MODEL_PATH}" \
  --device "cuda" \
  --batch_size ${BATCH_SIZE} \
  --image_size ${IMAGE_SIZE} \
  --frames ${FRAMES} \
  --image_patch_size ${PATCH_SPATIAL} \
  --frame_patch_size ${PATCH_DEPTH} \
  --emb_dim ${EMB_DIM} \
  --depth ${DEPTH} \
  --heads ${HEADS} \
  --mlp_dim ${MLP_DIM} \
  --sample_fraction 0.01 \
  ${AMP_FLAG} \
  2>&1 | tee -a "${LOG_DIR}/vit3d_predict_${SLURM_JOB_ID}.log"

EXIT_CODE=${PIPESTATUS[0]}
echo "=== [PREDICT END] $(date) (exit=${EXIT_CODE}) ==="
exit ${EXIT_CODE}
