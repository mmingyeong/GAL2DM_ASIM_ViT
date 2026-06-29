#!/bin/bash
#SBATCH -J pred_vit_seed
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_ViT/logs/%x.%A_%a.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_ViT/logs/%x.%A_%a.err
#SBATCH -p a40
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 0-24:00:00
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
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_MODULE_LOADING=LAZY
ulimit -n 65535

PROJECT_ROOT="/home/mingyeong/GAL2DM_ASIM_ViT"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"

SEED_LIST=(1049 2037 3181 4267 5393 6421 7559 8617 9733 10891)
SEED=${SEED_LIST[$SLURM_ARRAY_TASK_ID]}

EXP_ID="260611_vit_seed${SEED}"
CKPT_DIR="${PROJECT_ROOT}/results/vit/random_seed_ensemble/${EXP_ID}"

OUT_DIR="/home/mingyeong/GAL2DM_pred/random_seed_ensemble/vit/${EXP_ID}"
LOG_DIR="${PROJECT_ROOT}/logs/predict/random_seed_ensemble/vit/${EXP_ID}"

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
  --emb_dim 144 \
  --depth 4 \
  --heads 4 \
  --mlp_dim 288 \
  --vit_dim_head 64 \
  --vit_encoder_channels 16 24 48 \
  --vit_decoder_channels 96 56 28 \
  --vit_dropout 0.1 \
  --amp \
  2>&1 | tee -a "${RUN_LOG}"