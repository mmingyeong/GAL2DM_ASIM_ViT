#!/bin/bash
#SBATCH -J vitunet3d_ps8_e50
#SBATCH -o logs/vitunet3d_ps8_e50.%j.out
#SBATCH -e logs/vitunet3d_ps8_e50.%j.err
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

mkdir -p logs results/vit

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export CUBLAS_WORKSPACE_CONFIG=:4096:8  # 재현성 엄격 모드 필요 시

# ================================
# 경로 및 하이퍼파라미터
# ================================
PROJECT_ROOT="/home/mingyeong/2510_GAL2DM_ASIM_ViT"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"

# ▶ 실행 날짜/시간(YYYYMMDD_HHMMSS)으로 체크포인트/로그 경로 생성
RUN_TS="$(date +%Y%m%d_%H%M%S)"
CKPT_DIR="results/vit/${RUN_TS}"
LOG_DIR="logs/${RUN_TS}"

mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

# 실험 세팅
BATCH_SIZE=4
EPOCHS=50                 # ← 변경: 50 epochs
SAMPLE_FRACTION=1.0       # ← 유지: 1.0
TRAIN_VAL_SPLIT=0.8

# 모델 구조
IMAGE_SIZE=128
FRAMES=128
PATCH_SPATIAL=8           # ← 변경: patch size 8 (H/W)
PATCH_DEPTH=8             # ← 변경: patch size 8 (D)
EMB_DIM=256
DEPTH=3                   # 필요 시 6으로 확장 가능
HEADS=8
MLP_DIM=512

# 학습률/스케줄
MIN_LR=1e-4
MAX_LR=1e-3
CYCLE=8                   # CLR 주기(권장 8). 필요 시 조정 가능

# ================================
# 작업 디렉토리/경로 설정
# ================================
cd "$PROJECT_ROOT"
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

# ================================
# 실행
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
  2>&1 | tee -a "${LOG_DIR}/vitunet3d_ps8_e50_${SLURM_JOB_ID}.console.log"

echo "=== [JOB FINISHED] $(date) ==="
