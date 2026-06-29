#!/bin/bash
#SBATCH -J pred_vitL_lr1e4
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_ViT/logs/%x.%j.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_ViT/logs/%x.%j.err
#SBATCH -p a40
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 0-24:00:00
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
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_MODULE_LOADING=LAZY
ulimit -n 65535

PROJECT_ROOT="/home/mingyeong/GAL2DM_ASIM_ViT"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"

EXP_ID="260618_vitL_patch16_lr1e4_dropout015"

# Training script output path:
# CKPT_DIR="${PROJECT_ROOT}/results/vit/retrain/${RUN_ID}"
# RUN_ID="${EXP_ID}_${SLURM_JOB_ID}"
#
# 아래 CKPT_DIR은 실제 훈련 완료 후 생성된 디렉토리명으로 맞춰서 수정해줘.
CKPT_DIR="${PROJECT_ROOT}/results/vit/retrain/${EXP_ID}_166983"

OUT_DIR="/home/mingyeong/GAL2DM_pred/vit/retrain/${EXP_ID}"
LOG_DIR="${PROJECT_ROOT}/logs/predict/vit/retrain/${EXP_ID}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

find_checkpoint() {
  local dir="$1"
  local path=""
  path="$(ls -t "${dir}"/*best*.pt 2>/dev/null | head -n 1 || true)"
  if [ -z "$path" ]; then path="$(ls -t "${dir}"/*.pt 2>/dev/null | head -n 1 || true)"; fi
  if [ -z "$path" ]; then path="$(ls -t "${dir}"/*best*.pth 2>/dev/null | head -n 1 || true)"; fi
  if [ -z "$path" ]; then path="$(ls -t "${dir}"/*.pth 2>/dev/null | head -n 1 || true)"; fi
  echo "$path"
}

MODEL_PATH="$(find_checkpoint "${CKPT_DIR}")"

if [ -z "${MODEL_PATH}" ]; then
  echo "[ERROR] No checkpoint found in ${CKPT_DIR}"
  exit 1
fi

RUN_LOG="${LOG_DIR}/predict.log"

echo "=== [PREDICTION STARTED] $(date) ==="
echo "EXP_ID      : ${EXP_ID}"
echo "CKPT_DIR    : ${CKPT_DIR}"
echo "MODEL_PATH  : ${MODEL_PATH}"
echo "OUT_DIR     : ${OUT_DIR}"
echo "LOG_DIR     : ${LOG_DIR}"

srun python -u "${PROJECT_ROOT}/src/predict.py" \
  --yaml_path "${YAML_PATH}" \
  --output_dir "${OUT_DIR}" \
  --model_path "${MODEL_PATH}" \
  --device cuda \
  --batch_size 1 \
  --sample_fraction 1.0 \
  --input_case both \
  --keep_two_channels \
  --image_size 128 \
  --frames 128 \
  --image_patch_size 16 \
  --frame_patch_size 16 \
  --emb_dim 576 \
  --depth 10 \
  --heads 9 \
  --mlp_dim 1152 \
  --vit_dim_head 64 \
  --vit_encoder_channels 48 96 176 \
  --vit_decoder_channels 352 176 96 \
  --vit_dropout 0.15 \
  --amp \
  2>&1 | tee -a "${RUN_LOG}"

echo "=== [PREDICTION FINISHED] $(date) ==="
