#!/bin/bash
#SBATCH -J vitunet3d_ps8_e50_single_ch1
#SBATCH -o logs/icase-ch1/vitunet3d_ps8_e50_single_ch1.%j.out
#SBATCH -e logs/icase-ch1/vitunet3d_ps8_e50_single_ch1.%j.err
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
module load cuda/12.1.1

source ~/.bashrc
conda activate torch

CASE_TAG="icase-ch1"   # ← 출력 구분 태그
mkdir -p "logs/${CASE_TAG}" "results/vit/${CASE_TAG}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ================================
# 경로 및 하이퍼파라미터
# ================================
PROJECT_ROOT="/home/mingyeong/2510_GAL2DM_ASIM_ViT"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"

# ▶ 실행 타임스탬프
RUN_TS="$(date +%Y%m%d_%H%M%S)"
CKPT_DIR="results/vit/${CASE_TAG}/${RUN_TS}"
LOG_DIR="logs/${CASE_TAG}/${RUN_TS}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

# 실험 세팅
BATCH_SIZE=4
EPOCHS=50
SAMPLE_FRACTION=1.0
TRAIN_VAL_SPLIT=0.8

# 모델 구조
IMAGE_SIZE=128
FRAMES=128
PATCH_SPATIAL=8
PATCH_DEPTH=8
EMB_DIM=256
DEPTH=3
HEADS=8
MLP_DIM=512

# 학습률/스케줄
MIN_LR=1e-4
MAX_LR=1e-3
CYCLE=8

# ---- 입력 케이스
INPUT_CASE="ch1"   # 고정: 채널1

# ================================
# 작업 디렉토리/경로 설정
# ================================
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=== [JOB STARTED] $(date) on $(hostname) ==="
echo "CASE_TAG: ${CASE_TAG}"
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
# 실행 (단일채널 모델: in_channels=1)
# ================================
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
  --input_case ${INPUT_CASE} \
  2>&1 | tee -a "${LOG_DIR}/vitunet3d_ps8_e50_single_ch1_${SLURM_JOB_ID}.console.log"

echo "=== [JOB FINISHED] $(date) ==="
