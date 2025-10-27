#!/bin/bash
#SBATCH -J vit3d_train_sweep
#SBATCH -o logs/vit3d_train_sweep.%j.out
#SBATCH -e logs/vit3d_train_sweep.%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmingyeong@kasi.re.kr

# ================================
# 환경 설정
# ================================
module purge
module load cuda/12.1.1    # cu121 휠과 매칭
source ~/.bashrc
conda activate torch
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p logs results/vit

# ================================
# 경로 및 공통 하이퍼파라미터
# ================================
PROJECT_ROOT="/home/mingyeong/2510_GAL2DM_ASIM_ViT"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
BASE_CKPT_DIR="results/vit/${RUN_TS}"
BASE_LOG_DIR="logs/${RUN_TS}"
mkdir -p "${BASE_CKPT_DIR}" "${BASE_LOG_DIR}"

# 데이터/학습 설정
BATCH_SIZE=4
EPOCHS=50                 # ← 요청 반영 (50 epoch)
SAMPLE_FRACTION=1.0
TRAIN_VAL_SPLIT=0.8

# 모델 구조(공통)
IMAGE_SIZE=128
FRAMES=128
EMB_DIM=256
DEPTH=3                   # ViT block depth(모델 깊이)
HEADS=8
MLP_DIM=512

# 학습률/스케줄
MIN_LR=1e-4
MAX_LR=1e-3
CYCLE=4

# 실험할 (image_patch_size, frame_patch_size) 목록
declare -a TAGS=("patch8" "patch4" "patch2")
declare -a IPS=(8 4 2)
declare -a DPS=(8 4 2)

# ================================
# 작업 디렉토리/환경 출력
# ================================
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=== [JOB STARTED] $(date) on $(hostname) ==="
echo "PWD: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
echo "=== CUDA / PyTorch probe ==="
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

# ================================
# Sweep 실행
# ================================
for i in "${!TAGS[@]}"; do
  EXP_TAG="${TAGS[$i]}"
  PATCH_SPATIAL="${IPS[$i]}"
  PATCH_DEPTH="${DPS[$i]}"

  CKPT_DIR="${BASE_CKPT_DIR}/${EXP_TAG}"
  LOG_DIR="${BASE_LOG_DIR}/${EXP_TAG}"
  mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

  echo "=== [TRAIN] ${EXP_TAG}: image_patch_size=${PATCH_SPATIAL}, frame_patch_size=${PATCH_DEPTH} ==="
  srun python -u -m src.train \
    --yaml_path "${YAML_PATH}" \
    --target_field rho \
    --train_val_split ${TRAIN_VAL_SPLIT} \
    --sample_fraction ${SAMPLE_FRACTION} \
    --batch_size ${BATCH_SIZE} \
    --num_workers 4 \
    --pin_memory True \
    --image_size ${IMAGE_SIZE} \
    --frames ${FRAMES} \
    --image_patch_size ${PATCH_SPATIAL} \
    --frame_patch_size ${PATCH_DEPTH} \
    --emb_dim ${EMB_DIM} \
    --depth ${DEPTH} \
    --heads ${HEADS} \
    --mlp_dim ${MLP_DIM} \
    --epochs ${EPOCHS} \
    --min_lr ${MIN_LR} \
    --max_lr ${MAX_LR} \
    --cycle_length ${CYCLE} \
    --ckpt_dir "${CKPT_DIR}" \
    --seed 42 \
    --amp \
    2>&1 | tee -a "${LOG_DIR}/vit3d_train_${SLURM_JOB_ID}_${EXP_TAG}.log"

  EXIT_CODE=${PIPESTATUS[0]}
  echo "--- [${EXP_TAG}] exit=${EXIT_CODE} ---"
done

echo "=== [JOB FINISHED] $(date) ==="
