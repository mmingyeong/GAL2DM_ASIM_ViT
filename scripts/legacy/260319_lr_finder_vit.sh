#!/bin/bash
#SBATCH -J vit_lrfind_both
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_ViT/logs/lr_finder/%x.%j.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_ViT/logs/lr_finder/%x.%j.err
#SBATCH -p a40
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmingyeong@kasi.re.kr

set -e -o pipefail

module purge
module load cuda/12.1.1
source ~/.bashrc
conda activate torch

# 🔥 여기만 바뀜 (ViT repo 기준)
PROJECT_ROOT="/home/mingyeong/GAL2DM_ASIM_ViT"

CASE_TAG="icase-both"
MODEL_TAG="vit3d"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"

OUT_DIR="${PROJECT_ROOT}/results/lr_finder/${CASE_TAG}/${RUN_ID}"
LOG_DIR="${PROJECT_ROOT}/logs/lr_finder/${CASE_TAG}/${RUN_ID}"
PY_LOG_DIR="${LOG_DIR}/python_logs"

mkdir -p "${PROJECT_ROOT}/logs/lr_finder/${CASE_TAG}" "${PROJECT_ROOT}/results/lr_finder/${CASE_TAG}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${PY_LOG_DIR}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_MODULE_LOADING=LAZY
export HDF5_USE_FILE_LOCKING=FALSE
ulimit -n 65535

cd "${PROJECT_ROOT}" || { echo "[FATAL] cd failed"; exit 2; }

PLOT_PATH="${OUT_DIR}/${MODEL_TAG}_lr_finder_${RUN_ID}.png"
HISTORY_PATH="${OUT_DIR}/${MODEL_TAG}_lr_finder_${RUN_ID}.json"
CSV_PATH="${OUT_DIR}/${MODEL_TAG}_lr_finder_${RUN_ID}.csv"
SUMMARY_PATH="${OUT_DIR}/${MODEL_TAG}_lr_finder_${RUN_ID}_summary.json"
LOG_FILE="${LOG_DIR}/${MODEL_TAG}_lr_finder_${RUN_ID}.console.log"

touch "${LOG_FILE}"

echo "=== [JOB STARTED] $(date) on $(hostname) ==="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "YAML_PATH=${YAML_PATH}"
echo "RUN_ID=${RUN_ID}"
echo "OUT_DIR=${OUT_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "PY_LOG_DIR=${PY_LOG_DIR}"

which python

python - <<'PY'
import torch, os
print("Torch:", torch.__version__)
print("CUDA:", getattr(torch.version, "cuda", None))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Count:", torch.cuda.device_count())
    print("GPU Name[0]:", torch.cuda.get_device_name(0))
PY

nvidia-smi || echo "nvidia-smi not available"

python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.stderr.write("[FATAL] CUDA not available.\n")
    sys.exit(2)
PY

echo "Launching ViT LR finder..."
set +e
srun python -u -m src.lr_finder \
  --yaml_path "${YAML_PATH}" \
  --target_field rho \
  --train_val_split 0.8 \
  --sample_fraction 1.0 \
  --batch_size 4 \
  --num_workers 4 \
  --pin_memory False \
  --device cuda \
  --seed 42 \
  --input_case both \
  --keep_two_channels \
  --validate_keys False \
  --exclude_list "${PROJECT_ROOT}/filelists/exclude_bad_all.txt" \
  --image_size 128 \
  --frames 128 \
  --image_patch_size 16 \
  --frame_patch_size 16 \
  --emb_dim 256 \
  --depth 3 \
  --heads 8 \
  --mlp_dim 512 \
  --dropout 0.1 \
  --start_lr 1e-5 \
  --end_lr  3e-1 \
  --num_iter 150 \
  --step_mode exp \
  --out_dir "${OUT_DIR}" \
  --plot_path "${PLOT_PATH}" \
  --history_path "${HISTORY_PATH}" \
  --csv_path "${CSV_PATH}" \
  --summary_path "${SUMMARY_PATH}" \
  --log_dir "${PY_LOG_DIR}" \
  >> "${LOG_FILE}" 2>&1
status=$?
set -e

cat "${LOG_FILE}"

if [ ${status} -eq 0 ]; then
  echo "=== [JOB FINISHED SUCCESS] $(date) ==="
  echo "Plot saved to    : ${PLOT_PATH}"
  echo "History saved to : ${HISTORY_PATH}"
  echo "CSV saved to     : ${CSV_PATH}"
  echo "Summary saved to : ${SUMMARY_PATH}"
  echo "Console log      : ${LOG_FILE}"
  echo "Python logs dir  : ${PY_LOG_DIR}"
else
  echo "=== [JOB FAILED] $(date) | exit_code=${status} ==="
  echo "Check console log: ${LOG_FILE}"
  exit ${status}
fi