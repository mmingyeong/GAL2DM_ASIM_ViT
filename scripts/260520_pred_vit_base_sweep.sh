#!/bin/bash
#SBATCH -J vit_base_sweep_predict
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_ViT/logs/vit_base_sweep_predict.%A_%a.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_ViT/logs/vit_base_sweep_predict.%A_%a.err
#SBATCH -p a100
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-12:00:00
#SBATCH --array=1-2
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

CONFIG_NAMES=("VIT_BASE32_MATCH" "VIT_BASE64_MATCH" "VIT_BASE128_MATCH")

DIMS=(320 640 1152)
DEPTHS=(4 6 10)
MLP_DIMS=(640 1280 2304)
HEADS_LIST=(8 8 18)
DIM_HEADS=(64 64 64)

ENC1_LIST=(32 64 128)
ENC2_LIST=(64 128 256)
ENC3_LIST=(128 256 512)

DEC1_LIST=(256 512 1024)
DEC2_LIST=(128 256 512)
DEC3_LIST=(64 128 256)

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

EXP_ID="260520_vit_base_sweep"

CKPT_DIR=$(find "${PROJECT_ROOT}/results/vit/base_sweep" \
  -maxdepth 1 -type d -name "${EXP_ID}_${CONFIG_NAME}_*" | sort | tail -n 1)

if [ -z "${CKPT_DIR}" ]; then
  echo "[ERROR] No checkpoint directory found for ${CONFIG_NAME}"
  exit 1
fi

CKPT_PATH=$(find "${CKPT_DIR}" \
  -type f \( -name "*best*.pt" -o -name "*best*.pth" -o -name "*.pt" -o -name "*.pth" \) | sort | head -n 1)

if [ -z "${CKPT_PATH}" ]; then
  echo "[ERROR] No checkpoint file found in ${CKPT_DIR}"
  exit 1
fi

PRED_DIR="${PROJECT_ROOT}/predictions/vit/base_sweep/${EXP_ID}_${CONFIG_NAME}"
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
IMAGE_PATCH_SIZE=8
FRAME_PATCH_SIZE=8

echo "=== [PREDICT STARTED] $(date) ==="
echo "CONFIG_NAME      : ${CONFIG_NAME}"
echo "CKPT_DIR         : ${CKPT_DIR}"
echo "CKPT_PATH        : ${CKPT_PATH}"
echo "PRED_DIR         : ${PRED_DIR}"

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
  --device cuda \
  --amp \
  --validate_keys ${VALIDATE_KEYS} \
  --exclude_list "${EXCLUDE_LIST}" \
  2>&1 | tee -a "${LOG_FILE}"

echo "=== [PREDICT FINISHED] $(date) ==="