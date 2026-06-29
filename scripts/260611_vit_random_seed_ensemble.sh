#!/bin/bash
#SBATCH -J vit_seed_ens
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_ViT/logs/vit_seed_ens.%A_%a.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_ViT/logs/vit_seed_ens.%A_%a.err
#SBATCH -p h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-48:00:00
#SBATCH --array=0-9%2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmingyeong@kasi.re.kr

set -e -o pipefail

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
# Random seed configs
# -------------------------------
SEED_LIST=(1049 2037 3181 4267 5393 6421 7559 8617 9733 10891)
SEED=${SEED_LIST[$SLURM_ARRAY_TASK_ID]}

# -------------------------------
# Paths
# -------------------------------
PROJECT_ROOT="/home/mingyeong/GAL2DM_ASIM_ViT"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"
EXCLUDE_LIST="${PROJECT_ROOT}/filelists/exclude_bad_all.txt"

cd "$PROJECT_ROOT"

# -------------------------------
# Main benchmark ViT config
# VIT_BASE32_MATCH
# -------------------------------
CONFIG_NAME="VIT_BASE32_MATCH"

EMB_DIM=144
DEPTH=4
MLP_DIM=288
HEADS=4
DIM_HEAD=64

ENC1=16
ENC2=24
ENC3=48

DEC1=96
DEC2=56
DEC3=28

VIT_DROPOUT=0.1

# -------------------------------
# Run naming
# -------------------------------
EXP_ID="260611_vit_seed${SEED}"
RUN_ID="${EXP_ID}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

CKPT_DIR="${PROJECT_ROOT}/results/vit/random_seed_ensemble/${EXP_ID}"
LOG_RUN_DIR="${PROJECT_ROOT}/logs/vit/random_seed_ensemble/${EXP_ID}"

mkdir -p "${CKPT_DIR}" "${LOG_RUN_DIR}"

LOG_FILE="${LOG_RUN_DIR}/train.log"

# -------------------------------
# Fixed training config
# -------------------------------
DTYPE="float32"
TARGET_FIELD="rho"

MAX_LR=1e-3
WARMUP_RATIO=0.03
MIN_LR_RATIO=1e-2

BATCH_SIZE=4
GRAD_ACCUM_STEPS=1
EPOCHS=200

TRAIN_VAL_SPLIT=0.8
SAMPLE_FRACTION=1.0

PATIENCE=20
ES_DELTA=0

NUM_WORKERS=4
PIN_MEMORY=True

INPUT_CASE="both"
USE_AUGMENTATION=False
VALIDATE_KEYS=False

IMAGE_SIZE=128
FRAMES=128
IMAGE_PATCH_SIZE=16
FRAME_PATCH_SIZE=16

# -------------------------------
# Logging
# -------------------------------
echo "======================================================"
echo "JOB STARTED : $(date)"
echo "SLURM JOB   : ${SLURM_JOB_ID}"
echo "ARRAY ID    : ${SLURM_ARRAY_TASK_ID}"
echo "CONFIG_NAME : ${CONFIG_NAME}"
echo "EXP_ID      : ${EXP_ID}"
echo "RUN_ID      : ${RUN_ID}"
echo "SEED        : ${SEED}"
echo "CKPT_DIR    : ${CKPT_DIR}"
echo "LOG_RUN_DIR : ${LOG_RUN_DIR}"
echo "======================================================"

echo "EMB_DIM             : ${EMB_DIM}"
echo "DEPTH               : ${DEPTH}"
echo "MLP_DIM             : ${MLP_DIM}"
echo "HEADS               : ${HEADS}"
echo "DIM_HEAD            : ${DIM_HEAD}"
echo "ENCODER_CHANNELS    : (${ENC1}, ${ENC2}, ${ENC3})"
echo "DECODER_CHANNELS    : (${DEC1}, ${DEC2}, ${DEC3})"
echo "VIT_DROPOUT         : ${VIT_DROPOUT}"
echo "IMAGE_PATCH_SIZE    : ${IMAGE_PATCH_SIZE}"
echo "FRAME_PATCH_SIZE    : ${FRAME_PATCH_SIZE}"

cat > "${LOG_RUN_DIR}/run_config.txt" <<EOF
CONFIG_NAME=${CONFIG_NAME}
EXP_ID=${EXP_ID}
RUN_ID=${RUN_ID}

MODEL=ViT
RANDOM_SEED_ENSEMBLE=True
ENSEMBLE_MEMBER=${SLURM_ARRAY_TASK_ID}
SEED=${SEED}

TARGET_FIELD=${TARGET_FIELD}
DTYPE=${DTYPE}

MAX_LR=${MAX_LR}
MIN_LR_RATIO=${MIN_LR_RATIO}
WARMUP_RATIO=${WARMUP_RATIO}

BATCH_SIZE=${BATCH_SIZE}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}
EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM_STEPS))
EPOCHS=${EPOCHS}

TRAIN_VAL_SPLIT=${TRAIN_VAL_SPLIT}
SAMPLE_FRACTION=${SAMPLE_FRACTION}

PATIENCE=${PATIENCE}
ES_DELTA=${ES_DELTA}

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

EMB_DIM=${EMB_DIM}
DEPTH=${DEPTH}
MLP_DIM=${MLP_DIM}
HEADS=${HEADS}
DIM_HEAD=${DIM_HEAD}

VIT_ENCODER_CHANNELS=${ENC1},${ENC2},${ENC3}
VIT_DECODER_CHANNELS=${DEC1},${DEC2},${DEC3}
VIT_DROPOUT=${VIT_DROPOUT}
EOF

python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

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
  --amp \
  --validate_keys ${VALIDATE_KEYS} \
  --exclude_list "${EXCLUDE_LIST}" \
  2>&1 | tee -a "${LOG_FILE}"

echo "======================================================"
echo "JOB FINISHED : $(date)"
echo "======================================================"