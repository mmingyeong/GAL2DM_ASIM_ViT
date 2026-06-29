#!/bin/bash
#SBATCH -J vitunet3d_predict_ps8
#SBATCH -o logs/vitunet3d_predict_ps8.%j.out
#SBATCH -e logs/vitunet3d_predict_ps8.%j.err
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
# Paths
# ================================
PROJECT_ROOT="/home/mingyeong/2510_GAL2DM_ASIM_ViT"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"

# 훈련 산출물 디렉토리(수정 가능): 결과 폴더 안에서 최신/베스트 체크포인트 자동 선택
CKPT_DIR="${CKPT_DIR:-results/vit/20251014_210016}"

OUT_DIR_BASE="results/vit_predictions"

cd "${PROJECT_ROOT}" || exit 1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/${RUN_TS}"
mkdir -p "${LOG_DIR}" "${OUT_DIR_BASE}"

# ================================
# Resolve checkpoint inside CKPT_DIR
# ================================
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
DEPTH=3      # training에서 3 사용
HEADS=8
mlp_dim_default=512
MLP_DIM=${MLP_DIM:-$mlp_dim_default}

# 비중첩 패치(기본): stride = patch
IMAGE_STRIDE=${IMAGE_STRIDE:-$PATCH_SPATIAL}
FRAME_STRIDE=${FRAME_STRIDE:-$PATCH_DEPTH}

# 겹침 패치 예시(50% overlap)로 실행하려면 위 두 줄 대신 다음을 사용:
# IMAGE_STRIDE=4
# FRAME_STRIDE=4

BATCH_SIZE=1
AMP_FLAG="--amp"
SAMPLE_FRACTION=1.0   # 전체 실행

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
  --image_patch_stride ${IMAGE_STRIDE} \
  --frame_patch_stride ${FRAME_STRIDE} \
  --emb_dim ${EMB_DIM} \
  --depth ${DEPTH} \
  --heads ${HEADS} \
  --mlp_dim ${MLP_DIM} \
  --sample_fraction ${SAMPLE_FRACTION} \
  ${AMP_FLAG} \
  2>&1 | tee -a "${LOG_DIR}/vitunet3d_predict_${SLURM_JOB_ID}.log"

EXIT_CODE=${PIPESTATUS[0]}
echo "=== [PREDICT END] $(date) (exit=${EXIT_CODE}) ==="
exit ${EXIT_CODE}
