#!/bin/bash
#SBATCH -J vitunet3d_predict_ps8_both
#SBATCH -o logs/vitunet3d_predict_ps8_both.%j.out
#SBATCH -e logs/vitunet3d_predict_ps8_both.%j.err
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
PRED_SCRIPT="${PROJECT_ROOT}/src/predict.py"     # predict.py 위치 (수정본 사용)

# ▶ 각 케이스의 학습 산출물 디렉토리(필요시 바꿔주세요)
CKPT_DIR_CH1="${CKPT_DIR_CH1:-results/vit/icase-ch1/20251021_163947}"
CKPT_DIR_CH2="${CKPT_DIR_CH2:-results/vit/icase-ch2/20251022_164527}"

# ▶ 출력 루트
OUT_DIR_BASE="results/vit_predictions"

cd "${PROJECT_ROOT}" || exit 1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/${RUN_TS}"
mkdir -p "${LOG_DIR}" "${OUT_DIR_BASE}"

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

# ================================
# Common model/config (match training)
# ================================
IMAGE_SIZE=128
FRAMES=128
PATCH_SPATIAL=8
PATCH_DEPTH=8
EMB_DIM=256
DEPTH=3
HEADS=8
MLP_DIM=${MLP_DIM:-512}

# Strides (non-overlap by default)
IMAGE_STRIDE=${IMAGE_STRIDE:-$PATCH_SPATIAL}
FRAME_STRIDE=${FRAME_STRIDE:-$PATCH_DEPTH}

BATCH_SIZE=1
AMP_FLAG="--amp"
SAMPLE_FRACTION=1.0

# ================================
# Helper: resolve best checkpoint in a folder
# ================================
resolve_ckpt () {
  local dir="$1"
  if [ ! -d "$dir" ]; then
    echo "[ERROR] CKPT_DIR not found: $dir" >&2
    return 1
  fi
  local best="$(ls -t "$dir"/*best*.pt 2>/dev/null | head -n 1 || true)"
  if [ -n "$best" ]; then
    echo "$best"
    return 0
  fi
  local any="$(ls -t "$dir"/*.pt 2>/dev/null | head -n 1 || true)"
  if [ -n "$any" ]; then
    echo "$any"
    return 0
  fi
  echo "[ERROR] No .pt file found in $dir" >&2
  return 1
}

# ================================
# Predict runner (one case)
# ================================
run_case () {
  local input_case="$1"      # ch1 | ch2 | both
  local ckpt_dir="$2"

  # resolve checkpoint + make output root
  local model_path
  model_path="$(resolve_ckpt "$ckpt_dir")" || return 1
  echo "[INFO] (${input_case}) Using checkpoint: ${model_path}"

  # RUN_STEM = 마지막 디렉토리명(날짜스탬프 폴더이면 그 이름)
  local run_stem
  run_stem="$(basename "$ckpt_dir")"
  local pred_out_root="${OUT_DIR_BASE}/${run_stem}"
  mkdir -p "${pred_out_root}"

  # call predict.py (predict.py가 내부에서 icase-<case> 서브폴더 생성)
  srun python -u "${PRED_SCRIPT}" \
    --yaml_path "${YAML_PATH}" \
    --output_dir "${pred_out_root}" \
    --model_path "${model_path}" \
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
    --input_case "${input_case}" \
    ${AMP_FLAG} \
    2>&1 | tee -a "${LOG_DIR}/vitunet3d_predict_${input_case}_${SLURM_JOB_ID}.log"
}

# ================================
# Run predictions for both cases
# ================================
# Case 1: ch1 (단일채널 학습 모델 사용; keep_two_channels 없음)
run_case "ch1" "${CKPT_DIR_CH1}"

# Case 2: ch2 (단일채널 학습 모델 사용; keep_two_channels 없음)
run_case "ch2" "${CKPT_DIR_CH2}"

echo "=== [PREDICT END] $(date) ==="
