#!/bin/bash
#SBATCH -J vit3d_debug
#SBATCH -o logs/vit3d_debug.%j.out
#SBATCH -e logs/vit3d_debug.%j.err
#SBATCH -p gpu                      # 필요 시 gpu2080 등으로 변경
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
module load cuda/12.1.1            # cu121 휠과 매칭

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

# 디버그 세팅 → 소형 실험 세팅
BATCH_SIZE=4
EPOCHS=100        
SAMPLE_FRACTION=1.0    
TRAIN_VAL_SPLIT=0.8

# 모델 구조
IMAGE_SIZE=128
FRAMES=128
PATCH_SPATIAL=16
PATCH_DEPTH=16
EMB_DIM=256
DEPTH=3                      # 디버그용(원래 6)
HEADS=8
MLP_DIM=512

# 학습률/스케줄
MIN_LR=1e-4
MAX_LR=1e-3
CYCLE=4

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
  2>&1 | tee -a "${LOG_DIR}/vit3d_${SLURM_JOB_ID}.console.log"

echo "=== [JOB FINISHED] $(date) ==="
