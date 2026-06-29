#!/bin/bash
#SBATCH -J vit_best
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_ViT/logs/%x.%j.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_ViT/logs/%x.%j.err
#SBATCH -p h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmingyeong@kasi.re.kr

# ============================================
# Experiment: ViT final best params
# Script name : 260420_bestparams_vit.sh
# Date        : 2026-04-20
#
# Final fixed settings
# - model: ViTUNet3D
# - dtype=float32 + AMP
# - batch_size=4
# - grad_accum_steps=1
# - effective_batch=4
# - epochs=200
# - scheduler=cosine_warmup
# - max_lr=1e-3
# - warmup_ratio=0.05
# - min_lr_ratio=1e-2
# - patience=20
# - es_delta=0
# - input_case=both
# - keep_two_channels=True
# - target_field=rho
# - train_val_split=0.8
# - sample_fraction=1.0
# - use_augmentation=False
# - validate_keys=False
# - seed=42
#
# ViT geometry / final architecture
# - image_size=128
# - frames=128
# - image_patch_size=16
# - frame_patch_size=16
# - image_patch_stride=16
# - frame_patch_stride=16
# - emb_dim=320
# - depth=4
# - mlp_dim=640
# - heads=8
# - vit_dim_head=64
# - vit_encoder_channels=(32, 64, 128)
# - vit_decoder_channels=(256, 128, 64)
# - vit_dropout=0.1
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
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_MODULE_LOADING=LAZY
export HDF5_USE_FILE_LOCKING=FALSE
ulimit -n 65535

# -------------------------------
# Paths
# -------------------------------
PROJECT_ROOT="/home/mingyeong/GAL2DM_ASIM_ViT"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"

# -------------------------------
# Run / experiment naming
# -------------------------------
EXP_ID="260420_bestparams_vit"
RUN_ID="${EXP_ID}_${SLURM_JOB_ID}"

CKPT_DIR="${PROJECT_ROOT}/results/vit/bestparams/warm0.05/${EXP_ID}"
LOG_RUN_DIR="${PROJECT_ROOT}/logs/vit/bestparams/warm0.05/${EXP_ID}"

# -------------------------------
# Fixed training config
# -------------------------------
DTYPE="float32"
TARGET_FIELD="rho"

MAX_LR=1e-3
WARMUP_RATIO=0.05
MIN_LR_RATIO=1e-2

BATCH_SIZE=4
GRAD_ACCUM_STEPS=1
EPOCHS=200
TRAIN_VAL_SPLIT=0.8
SAMPLE_FRACTION=1.0

PATIENCE=20
ES_DELTA=0
SEED=42

NUM_WORKERS=4
PIN_MEMORY=True

INPUT_CASE="both"
USE_AUGMENTATION=False
VALIDATE_KEYS=False

# -------------------------------
# Fixed ViT geometry
# -------------------------------
IMAGE_SIZE=128
FRAMES=128
IMAGE_PATCH_SIZE=16
FRAME_PATCH_SIZE=16
IMAGE_PATCH_STRIDE=16
FRAME_PATCH_STRIDE=16

# -------------------------------
# Final ViT architecture
# -------------------------------
EMB_DIM=320
DEPTH=4
MLP_DIM=640
HEADS=8
DIM_HEAD=64

ENC1=32
ENC2=64
ENC3=128

DEC1=256
DEC2=128
DEC3=64

VIT_DROPOUT=0.1

EXCLUDE_LIST="${PROJECT_ROOT}/filelists/exclude_bad_all.txt"

cd "$PROJECT_ROOT" || { echo "[FATAL] cd failed"; exit 2; }

echo "=== [JOB STARTED] $(date) ==="
echo "EXP_ID              : ${EXP_ID}"
echo "RUN_ID              : ${RUN_ID}"
echo "DTYPE               : ${DTYPE}"
echo "TARGET_FIELD        : ${TARGET_FIELD}"
echo "MAX_LR              : ${MAX_LR}"
echo "WARMUP_RATIO        : ${WARMUP_RATIO}"
echo "MIN_LR_RATIO        : ${MIN_LR_RATIO}"
echo "GRAD_ACCUM_STEPS    : ${GRAD_ACCUM_STEPS}"
echo "BATCH_SIZE          : ${BATCH_SIZE}"
echo "EFFECTIVE_BATCH     : $((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "EPOCHS              : ${EPOCHS}"
echo "PATIENCE            : ${PATIENCE}"
echo "USE_AUGMENTATION    : ${USE_AUGMENTATION}"
echo "EMB_DIM             : ${EMB_DIM}"
echo "DEPTH               : ${DEPTH}"
echo "MLP_DIM             : ${MLP_DIM}"
echo "HEADS               : ${HEADS}"
echo "DIM_HEAD            : ${DIM_HEAD}"
echo "ENCODER_CHANNELS    : (${ENC1}, ${ENC2}, ${ENC3})"
echo "DECODER_CHANNELS    : (${DEC1}, ${DEC2}, ${DEC3})"
echo "VIT_DROPOUT         : ${VIT_DROPOUT}"
echo "DEVICE              : cuda"
echo "AMP                 : ON"

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(2, 2, device="cuda", dtype=torch.float32)
    print("float32 test dtype:", x.dtype)
PY

# -------------------------------
# Prepare dirs
# -------------------------------
mkdir -p "${CKPT_DIR}" "${LOG_RUN_DIR}"
LOG_FILE="${LOG_RUN_DIR}/train.log"

cat > "${LOG_RUN_DIR}/run_config.txt" <<EOF
EXP_ID=${EXP_ID}
RUN_ID=${RUN_ID}
DTYPE=${DTYPE}
TARGET_FIELD=${TARGET_FIELD}
MAX_LR=${MAX_LR}
MIN_LR_RATIO=${MIN_LR_RATIO}
WARMUP_RATIO=${WARMUP_RATIO}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}
BATCH_SIZE=${BATCH_SIZE}
EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM_STEPS))
EPOCHS=${EPOCHS}
TRAIN_VAL_SPLIT=${TRAIN_VAL_SPLIT}
SAMPLE_FRACTION=${SAMPLE_FRACTION}
PATIENCE=${PATIENCE}
ES_DELTA=${ES_DELTA}
SEED=${SEED}
NUM_WORKERS=${NUM_WORKERS}
PIN_MEMORY=${PIN_MEMORY}
INPUT_CASE=${INPUT_CASE}
KEEP_TWO_CHANNELS=True
USE_AUGMENTATION=${USE_AUGMENTATION}
VALIDATE_KEYS=${VALIDATE_KEYS}
AMP=True
SCHEDULER_TYPE=cosine_warmup
IMAGE_SIZE=${IMAGE_SIZE}
FRAMES=${FRAMES}
IMAGE_PATCH_SIZE=${IMAGE_PATCH_SIZE}
FRAME_PATCH_SIZE=${FRAME_PATCH_SIZE}
IMAGE_PATCH_STRIDE=${IMAGE_PATCH_STRIDE}
FRAME_PATCH_STRIDE=${FRAME_PATCH_STRIDE}
EMB_DIM=${EMB_DIM}
DEPTH=${DEPTH}
MLP_DIM=${MLP_DIM}
HEADS=${HEADS}
DIM_HEAD=${DIM_HEAD}
VIT_ENCODER_CHANNELS=${ENC1},${ENC2},${ENC3}
VIT_DECODER_CHANNELS=${DEC1},${DEC2},${DEC3}
VIT_DROPOUT=${VIT_DROPOUT}
EOF

# -------------------------------
# Run training
# -------------------------------
srun python -u -m src.train \
  --yaml_path "${YAML_PATH}" \
  --target_field ${TARGET_FIELD} \
  --train_val_split ${TRAIN_VAL_SPLIT} \
  --sample_fraction ${SAMPLE_FRACTION} \
  --batch_size ${BATCH_SIZE} \
  --num_workers ${NUM_WORKERS} \
  --pin_memory ${PIN_MEMORY} \
  --epochs ${EPOCHS} \
  --scheduler_type cosine_warmup \
  --max_lr ${MAX_LR} \
  --warmup_ratio ${WARMUP_RATIO} \
  --min_lr_ratio ${MIN_LR_RATIO} \
  --patience ${PATIENCE} \
  --es_delta ${ES_DELTA} \
  --grad_accum_steps ${GRAD_ACCUM_STEPS} \
  --input_case ${INPUT_CASE} \
  --keep_two_channels \
  --image_size ${IMAGE_SIZE} \
  --frames ${FRAMES} \
  --image_patch_size ${IMAGE_PATCH_SIZE} \
  --frame_patch_size ${FRAME_PATCH_SIZE} \
  --image_patch_stride ${IMAGE_PATCH_STRIDE} \
  --frame_patch_stride ${FRAME_PATCH_STRIDE} \
  --emb_dim ${EMB_DIM} \
  --depth ${DEPTH} \
  --heads ${HEADS} \
  --mlp_dim ${MLP_DIM} \
  --vit_dim_head ${DIM_HEAD} \
  --vit_encoder_channels ${ENC1} ${ENC2} ${ENC3} \
  --vit_decoder_channels ${DEC1} ${DEC2} ${DEC3} \
  --vit_dropout ${VIT_DROPOUT} \
  --ckpt_dir "${CKPT_DIR}" \
  --seed ${SEED} \
  --device cuda \
#  --dtype ${DTYPE} \
  --amp True \
  --validate_keys ${VALIDATE_KEYS} \
  --exclude_list "${EXCLUDE_LIST}" \
  2>&1 | tee -a "${LOG_FILE}"

echo "=== [JOB FINISHED] $(date) ==="