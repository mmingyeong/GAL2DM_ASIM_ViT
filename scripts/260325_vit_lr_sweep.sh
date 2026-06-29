#!/bin/bash
#SBATCH -J vitunet3d_lr_sweep
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_ViT/logs/vit/%x.%A_%a.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_ViT/logs/vit/%x.%A_%a.err
#SBATCH -p a100
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-48:00:00
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmingyeong@kasi.re.kr

# ============================================
# Experiment: ViT-3D LR Sweep
# Date: 2026-03-25
#
# Fixed:
# - batch_size=4, epochs=200, amp=True
# - cosine warmup scheduler
# - no augmentation
#
# Sweep:
# - max_lr ∈ {3e-4, 1e-3, 3e-3}
#
# Goal:
# - Find stable & optimal learning rate
# ============================================

set -e -o pipefail

# -------------------------------
# Environment
# -------------------------------
module purge
module load cuda/12.1.1

source ~/.bashrc
conda activate torch

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export HDF5_USE_FILE_LOCKING=FALSE
ulimit -n 65535
export CUDA_MODULE_LOADING=LAZY

# -------------------------------
# Paths
# -------------------------------
PROJECT_ROOT="/home/mingyeong/GAL2DM_ASIM_ViT"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"

# -------------------------------
# LR sweep
# -------------------------------
LR_LIST=(3e-4 1e-3 3e-3)
MAX_LR=${LR_LIST[$SLURM_ARRAY_TASK_ID]}

RUN_ID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
CKPT_DIR="${PROJECT_ROOT}/results/vit/lr_${MAX_LR}/${RUN_ID}"
LOG_RUN_DIR="${PROJECT_ROOT}/logs/vit/lr_${MAX_LR}/${RUN_ID}"

# -------------------------------
# Fixed training config
# -------------------------------
BATCH_SIZE=4
EPOCHS=200
TRAIN_VAL_SPLIT=0.8

IMAGE_SIZE=128
FRAMES=128
PATCH_SPATIAL=8
PATCH_DEPTH=8

EMB_DIM=256
DEPTH=3
HEADS=8
MLP_DIM=512

# Scheduler params (train.py 기준)
WARMUP_RATIO=0.05
MIN_LR_RATIO=1e-2

# -------------------------------
# Move to project
# -------------------------------
cd "$PROJECT_ROOT" || { echo "[FATAL] cd failed"; exit 2; }
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=== [JOB STARTED] $(date) ==="
echo "MAX_LR: ${MAX_LR}"

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# -------------------------------
# Prepare dirs
# -------------------------------
mkdir -p "${CKPT_DIR}" "${LOG_RUN_DIR}"
LOG_FILE="${LOG_RUN_DIR}/train.log"

# -------------------------------
# Run training
# -------------------------------
srun python -u -m src.train \
  --yaml_path "${YAML_PATH}" \
  --target_field rho \
  --train_val_split ${TRAIN_VAL_SPLIT} \
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
  --scheduler_type cosine_warmup \
  --max_lr ${MAX_LR} \
  --warmup_ratio ${WARMUP_RATIO} \
  --min_lr_ratio ${MIN_LR_RATIO} \
  --patience 10 \
  --es_delta 0 \
  --grad_accum_steps 1 \
  --input_case both \
  --keep_two_channels \
  --ckpt_dir "${CKPT_DIR}" \
  --seed 42 \
  --amp \
  --validate_keys False \
  --exclude_list "${PROJECT_ROOT}/filelists/exclude_bad_all.txt" \
  2>&1 | tee -a "${LOG_FILE}"

echo "=== [JOB FINISHED] $(date) ==="