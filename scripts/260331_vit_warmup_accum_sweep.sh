#!/bin/bash
#SBATCH -J vit3d_wa_sweep
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_ViT/logs/vit/%x.%A_%a.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_ViT/logs/vit/%x.%A_%a.err
#SBATCH -p a100
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-48:00:00
#SBATCH --array=0-4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmingyeong@kasi.re.kr

# ============================================
# Experiment: ViT warmup_ratio / grad_accum sweep
# Date: 2026-03-31 (260331)
#
# Fixed:
# - batch_size=4, epochs=200, amp=True
# - scheduler_type=cosine_warmup
# - max_lr=1e-3
# - min_lr_ratio=1e-2
# - patience=15
#
# Sweep cases:
#   exp1: warmup_ratio=0.03, grad_accum_steps=1
#   exp2: warmup_ratio=0.10, grad_accum_steps=1
#   exp3: warmup_ratio=0.03, grad_accum_steps=2
#   exp4: warmup_ratio=0.05, grad_accum_steps=2
#   exp5: warmup_ratio=0.10, grad_accum_steps=2
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
# Fixed training config
# -------------------------------
MAX_LR=1e-3
MIN_LR_RATIO=1e-2
BATCH_SIZE=4
EPOCHS=200
TRAIN_VAL_SPLIT=0.8
PATIENCE=15
ES_DELTA=0
SEED=42

# ViT architecture
IMAGE_SIZE=128
FRAMES=128
PATCH_SPATIAL=8
PATCH_DEPTH=8
EMB_DIM=256
DEPTH=3
HEADS=8
MLP_DIM=512

# -------------------------------
# Sweep config
# -------------------------------
EXP_IDS=(
  "260331_exp1"
  "260331_exp2"
  "260331_exp3"
  "260331_exp4"
  "260331_exp5"
)

WARMUP_LIST=(
  "0.03"
  "0.10"
  "0.03"
  "0.05"
  "0.10"
)

GRAD_ACCUM_LIST=(
  "1"
  "1"
  "2"
  "2"
  "2"
)

IDX=${SLURM_ARRAY_TASK_ID}

EXP_ID=${EXP_IDS[$IDX]}
WARMUP_RATIO=${WARMUP_LIST[$IDX]}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_LIST[$IDX]}

RUN_ID="${EXP_ID}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

CKPT_DIR="${PROJECT_ROOT}/results/vit/warmup_accum/${EXP_ID}"
LOG_RUN_DIR="${PROJECT_ROOT}/logs/vit/warmup_accum/${EXP_ID}"

# -------------------------------
# Move to project
# -------------------------------
cd "$PROJECT_ROOT" || { echo "[FATAL] cd failed"; exit 2; }
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=== [JOB STARTED] $(date) ==="
echo "EXP_ID            : ${EXP_ID}"
echo "RUN_ID            : ${RUN_ID}"
echo "MAX_LR            : ${MAX_LR}"
echo "MIN_LR_RATIO      : ${MIN_LR_RATIO}"
echo "WARMUP_RATIO      : ${WARMUP_RATIO}"
echo "GRAD_ACCUM_STEPS  : ${GRAD_ACCUM_STEPS}"
echo "PATIENCE          : ${PATIENCE}"

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

# Save config snapshot
cat > "${LOG_RUN_DIR}/run_config.txt" <<EOF
EXP_ID=${EXP_ID}
RUN_ID=${RUN_ID}
MAX_LR=${MAX_LR}
MIN_LR_RATIO=${MIN_LR_RATIO}
WARMUP_RATIO=${WARMUP_RATIO}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}
BATCH_SIZE=${BATCH_SIZE}
EPOCHS=${EPOCHS}
TRAIN_VAL_SPLIT=${TRAIN_VAL_SPLIT}
PATIENCE=${PATIENCE}
ES_DELTA=${ES_DELTA}
SEED=${SEED}
EOF

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
  --patience ${PATIENCE} \
  --es_delta ${ES_DELTA} \
  --grad_accum_steps ${GRAD_ACCUM_STEPS} \
  --input_case both \
  --keep_two_channels \
  --ckpt_dir "${CKPT_DIR}" \
  --seed ${SEED} \
  --amp \
  --validate_keys False \
  --exclude_list "${PROJECT_ROOT}/filelists/exclude_bad_all.txt" \
  2>&1 | tee -a "${LOG_FILE}"

echo "=== [JOB FINISHED] $(date) ==="