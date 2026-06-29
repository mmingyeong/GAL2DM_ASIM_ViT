#!/bin/bash
#SBATCH -J vit_base_sweep_p16_predict
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_ViT/logs/vit_base_sweep_p16_predict.%A_%a.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_ViT/logs/vit_base_sweep_p16_predict.%A_%a.err
#SBATCH -p a100
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-12:00:00
#SBATCH --array=0-2
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

PROJECT_ROOT="/home/mingyeong/GAL2DM_ASIM_ViT"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"
EXCLUDE_LIST="${PROJECT_ROOT}/filelists/exclude_bad_all.txt"

cd "$PROJECT_ROOT"

# =========================================================
# Sweep configs: patch=(16,16,16)
# =========================================================
CONFIG_NAMES=("VIT_BASE32_MATCH" "VIT_BASE64_MATCH" "VIT_BASE128_MATCH")

DIMS=(144 256 576)
DEPTHS=(4 6 10)
MLP_DIMS=(288 512 1152)
HEADS_LIST=(4 8 9)
DIM_HEADS=(64 64 64)

ENC1_LIST=(16 24 48)
ENC2_LIST=(28 48 96)
ENC3_LIST=(48 96 176)

DEC1_LIST=(96 192 352)
DEC2_LIST=(56 96 176)
DEC3_LIST=(28 48 96)

DROPOUTS=(0.1 0.1 0.1)

IDX=${SLURM_ARRAY_TASK_ID}

CONFIG_NAME=${CONFIG_NAMES[$IDX]}
EMB_DIM=${DIMS[$IDX]}
DEPTH=${DEPTHS[$IDX]}
MLP_DIM=${MLP_DIMS[$IDX]}
HEADS=${HEADS_LIST[$IDX]}
DIM_HEAD=${DIM_HEADS[$IDX]}

ENC1=${ENC1_LIST[$IDX]}
ENC2=${ENC2_LIST[$IDX]}
ENC3=${ENC3_LIST[$IDX]}

DEC1=${DEC1_LIST[$IDX]}
DEC2=${DEC2_LIST[$IDX]}
DEC3=${DEC3_LIST[$IDX]}

VIT_DROPOUT=${DROPOUTS[$IDX]}

# Must match training EXP_ID
EXP_ID="260527_vit_base_sweep_patch16"

CKPT_DIR=$(find "${PROJECT_ROOT}/results/vit/base_sweep" \
  -maxdepth 1 -type d -name "${EXP_ID}_${CONFIG_NAME}_*" | sort | tail -n 1)

if [ -z "${CKPT_DIR}" ]; then
  echo "[ERROR] No checkpoint directory found for ${CONFIG_NAME}"
  echo "[ERROR] Expected pattern: ${PROJECT_ROOT}/results/vit/base_sweep/${EXP_ID}_${CONFIG_NAME}_*"
  exit 1
fi

CKPT_PATH=$(find "${CKPT_DIR}" \
  -type f \( -name "*best*.pt" -o -name "*best*.pth" -o -name "*.pt" -o -name "*.pth" \) | sort | head -n 1)

if [ -z "${CKPT_PATH}" ]; then
  echo "[ERROR] No checkpoint file found in ${CKPT_DIR}"
  exit 1
fi

PRED_DIR="/home/mingyeong/GAL2DM_pred/vit_base_sweep/${EXP_ID}_${CONFIG_NAME}"
LOG_RUN_DIR="${PROJECT_ROOT}/logs/vit/base_sweep_predict/${EXP_ID}_${CONFIG_NAME}"
mkdir -p "${PRED_DIR}" "${LOG_RUN_DIR}"

LOG_FILE="${LOG_RUN_DIR}/predict.log"

TARGET_FIELD="rho"
BATCH_SIZE=1
NUM_WORKERS=4
PIN_MEMORY=True
SEED=42

INPUT_CASE="both"
VALIDATE_KEYS=False

IMAGE_SIZE=128
FRAMES=128
IMAGE_PATCH_SIZE=16
FRAME_PATCH_SIZE=16

echo "=== [PREDICT STARTED] $(date) ==="
echo "CONFIG_NAME         : ${CONFIG_NAME}"
echo "CKPT_DIR            : ${CKPT_DIR}"
echo "CKPT_PATH           : ${CKPT_PATH}"
echo "PRED_DIR            : ${PRED_DIR}"
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

cat > "${LOG_RUN_DIR}/predict_config.txt" <<EOF
CONFIG_NAME=${CONFIG_NAME}
EXP_ID=${EXP_ID}
CKPT_DIR=${CKPT_DIR}
CKPT_PATH=${CKPT_PATH}
PRED_DIR=${PRED_DIR}
TARGET_FIELD=${TARGET_FIELD}
BATCH_SIZE=${BATCH_SIZE}
INPUT_CASE=${INPUT_CASE}
KEEP_TWO_CHANNELS=True
AMP=True
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

srun python -u -m src.predict \
  --yaml_path "${YAML_PATH}" \
  --output_dir "${PRED_DIR}" \
  --model_path "${CKPT_PATH}" \
  --target_field ${TARGET_FIELD} \
  --batch_size ${BATCH_SIZE} \
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
  --device cuda \
  --amp \
  --validate_keys ${VALIDATE_KEYS} \
  --exclude_list "${EXCLUDE_LIST}" \
  2>&1 | tee -a "${LOG_FILE}"

echo "=== [PREDICT FINISHED] $(date) ==="
