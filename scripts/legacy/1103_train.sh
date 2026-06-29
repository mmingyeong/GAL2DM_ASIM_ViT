#!/bin/bash
#SBATCH -J vitunet3d_ps8_e50
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_ViT/logs/vit/%x.%j.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_ViT/logs/vit/%x.%j.err
#SBATCH -p h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmingyeong@kasi.re.kr

# -------------------------------
# 안전 설정( -u 제외 )
# -------------------------------
set -e -o pipefail

module purge
module load cuda/12.1.1

# -u 없이 bashrc 로드
source ~/.bashrc
conda activate torch

# I/O/락 안정화
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export HDF5_USE_FILE_LOCKING=FALSE
ulimit -n 65535
export CUDA_MODULE_LOADING=LAZY

# -------------------------------
# 경로/하이퍼파라미터
# -------------------------------
PROJECT_ROOT="/home/mingyeong/GAL2DM_ASIM_ViT"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"

# 상위 로그 디렉터리는 sbatch가 생성 못 하므로 '한 번만' 미리 만들어두세요:
#   mkdir -p /home/mingyeong/GAL2DM_ASIM_ViT/logs/vit

RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
CKPT_DIR="${PROJECT_ROOT}/results/vit/${RUN_ID}"
LOG_RUN_DIR="${PROJECT_ROOT}/logs/vit/${RUN_ID}"

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

# -------------------------------
# 작업 디렉토리 이동(실패 시 중단)
# -------------------------------
cd "$PROJECT_ROOT" || { echo "[FATAL] cd to PROJECT_ROOT failed"; exit 2; }
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=== [JOB STARTED] $(date) on $(hostname) ==="
echo "PWD: $(pwd)"
which python
python - <<'PY'
import torch, os
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("torch.version.cuda:", getattr(torch.version, "cuda", None))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    print("GPU Count:", torch.cuda.device_count())
    print("GPU Name[0]:", torch.cuda.get_device_name(0))
PY
nvidia-smi || echo "nvidia-smi not available"

# -------------------------------
# 프리플라이트(성공 시에만 폴더 생성)
# -------------------------------
python - <<'PY'
import sys, torch
if not torch.cuda.is_available() or torch.version.cuda is None:
    sys.stderr.write("\n[FATAL] CUDA not available (CPU-only PyTorch). Install CUDA-enabled PyTorch.\n")
    sys.exit(2)
PY

# 여기서만 생성 → 실패하면 폴더 안 생김
mkdir -p "${CKPT_DIR}" "${LOG_RUN_DIR}"
LOG_FILE="${LOG_RUN_DIR}/vitunet3d_ps8_e50_${RUN_ID}.console.log"
touch "${LOG_FILE}"

# -------------------------------
# 실행
# -------------------------------
srun python -u -m src.train \
  --yaml_path "${YAML_PATH}" \
  --target_field rho \
  --train_val_split ${TRAIN_VAL_SPLIT} \
  --sample_fraction ${SAMPLE_FRACTION} \
  --batch_size ${BATCH_SIZE} \
  --num_workers 2 \
  --pin_memory False \
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
  --validate_keys False \
  --exclude_list "${PROJECT_ROOT}/filelists/exclude_bad_all.txt" \
  2>&1 | tee -a "${LOG_FILE}"


echo "=== [JOB FINISHED] $(date) ==="
