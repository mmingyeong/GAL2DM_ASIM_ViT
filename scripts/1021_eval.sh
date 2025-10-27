#!/bin/bash
#SBATCH -J vitunet3d_eval_fast
#SBATCH -o logs/vitunet3d_eval_fast.%j.out
#SBATCH -e logs/vitunet3d_eval_fast.%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmingyeong@kasi.re.kr

# ================================
# Usage (예)
#   sbatch scripts/1023_eval_fast.sh \
#     --pred_dir results/vit_predictions/20251014_210016/icase-ch1
#
# 필수 인자:
#   --pred_dir : predict.py가 생성한 <index>.hdf5들이 들어있는 디렉토리
#
# 선택 인자:
#   --alex_tpl : Alex NPY 템플릿 (default: 아래 ALEX_TPL_DEFAULT)
#   --yaml     : asim_paths.yaml 경로 (default: 프로젝트 기본값)
#   --out_root : 출력 루트 (기본: results/vit_eval/<timestamp>/<case>)
#   --alex_force {true|false}  # δ→ρ 변환 강제 여부(없으면 자동 판정)
#   --loss_csv : (선택) 학습 로그 CSV (epoch,train_loss,val_loss,lr)
#                미지정 시 자동 탐색: results/vit/<TS>/*log.csv
#   --map_count (기본 5)        # 임의 선택하여 map 저장할 인덱스 수
#   --ks_global_cap (기본 200000) # 전역 KS 샘플 상한
# ================================

# -------- Parse args --------
PRED_DIR=""
ALEX_TPL=""
YAML_PATH=""
OUT_ROOT=""
ALEX_FORCE=""
LOSS_CSV=""           # (opt)
MAP_COUNT=""          # (opt)
KS_GLOBAL_CAP=""      # (opt)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pred_dir)      PRED_DIR="$2"; shift 2;;
    --alex_tpl)      ALEX_TPL="$2"; shift 2;;
    --yaml)          YAML_PATH="$2"; shift 2;;
    --out_root)      OUT_ROOT="$2"; shift 2;;
    --alex_force)    ALEX_FORCE="$2"; shift 2;;
    --loss_csv)      LOSS_CSV="$2"; shift 2;;
    --map_count)     MAP_COUNT="$2"; shift 2;;
    --ks_global_cap) KS_GLOBAL_CAP="$2"; shift 2;;
    *) echo "[WARN] Unknown arg: $1"; shift;;
  esac
done

if [[ -z "${PRED_DIR}" ]]; then
  echo "[ERROR] --pred_dir is required." >&2
  exit 1
fi
if [[ ! -d "${PRED_DIR}" ]]; then
  echo "[ERROR] pred_dir not found: ${PRED_DIR}" >&2
  exit 1
fi

# ================================
# Environment
# ================================
module purge
module load cuda/12.1.1
source ~/.bashrc
conda activate torch
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ================================
# Paths (project defaults)
# ================================
PROJECT_ROOT="/home/mingyeong/2510_GAL2DM_ASIM_ViT"
EVAL_PY="${PROJECT_ROOT}/src/eval_fast.py"   # <<< NEW: fast evaluator
YAML_PATH_DEFAULT="${PROJECT_ROOT}/etc/asim_paths.yaml"
ALEX_TPL_DEFAULT="/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/predictions/test_{idx}_final_*.npy"

YAML_PATH="${YAML_PATH:-$YAML_PATH_DEFAULT}"
ALEX_TPL="${ALEX_TPL:-$ALEX_TPL_DEFAULT}"

cd "${PROJECT_ROOT}" || exit 1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ================================
# Label / Output dirs
# ================================
CASE_LABEL="$(basename "${PRED_DIR}")"                 # icase-ch1 / icase-ch2 등
TS_FROM_PRED="$(basename "$(dirname "${PRED_DIR}")")"  # e.g., 20251014_210016
RUN_TS="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${OUT_ROOT}" ]]; then
  OUT_ROOT="results/vit_eval/${RUN_TS}/${CASE_LABEL}"
fi
mkdir -p "logs" "${OUT_ROOT}"

# ================================
# Auto-discover loss CSV if not provided
# Looks in results/vit/<TS_FROM_PRED>/*log.csv
# ================================
LOSS_CSV_RESOLVED="${LOSS_CSV}"
if [[ -z "${LOSS_CSV_RESOLVED}" && -n "${TS_FROM_PRED}" ]]; then
  CAND_DIR="${PROJECT_ROOT}/results/vit/${TS_FROM_PRED}"
  if [[ -d "${CAND_DIR}" ]]; then
    LOSS_CSV_RESOLVED="$(ls -1 "${CAND_DIR}"/*log.csv 2>/dev/null | head -n1)"
  fi
fi
if [[ -n "${LOSS_CSV}" && ! -f "${LOSS_CSV}" ]]; then
  echo "[WARN] --loss_csv specified but not found: ${LOSS_CSV}" >&2
  LOSS_CSV_RESOLVED=""
fi

# ================================
# Eval params (FAST evaluator)
# ================================
SLICE_AXIS=${SLICE_AXIS:-2}
SLICE_INDEX=${SLICE_INDEX:-center}

# FAST 전용 파라미터
MAP_COUNT=${MAP_COUNT:-5}
KS_GLOBAL_CAP=${KS_GLOBAL_CAP:-200000}

# Aggregation 파라미터
JOINT_SAMPLE=${JOINT_SAMPLE:-50000}
PDF_BINS=${PDF_BINS:-120}
JOINT_BINS=${JOINT_BINS:-120}
VOXEL_SIZE=${VOXEL_SIZE:-$(python - <<'PY'
print(205.0/250.0)
PY
)}
RMAX=${RMAX:-10.0}
N_R_BINS=${N_R_BINS:-24}

echo "=== [EVAL FAST START] $(date) on $(hostname) ==="
echo "[INFO] pred_dir       : ${PRED_DIR}"
echo "[INFO] case label     : ${CASE_LABEL}"
echo "[INFO] timestamp      : ${TS_FROM_PRED}"
echo "[INFO] out_root       : ${OUT_ROOT}"
echo "[INFO] alex_tpl       : ${ALEX_TPL}"
echo "[INFO] yaml_path      : ${YAML_PATH}"
echo "[INFO] alex_force     : ${ALEX_FORCE:-<auto>}"
echo "[INFO] loss_csv       : ${LOSS_CSV_RESOLVED:-<auto-not-found>}"
echo "[INFO] map_count      : ${MAP_COUNT}"
echo "[INFO] ks_global_cap  : ${KS_GLOBAL_CAP}"
which python
python - <<'PY'
import torch, os
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY
nvidia-smi || true

# ================================
# Run (FAST evaluator)
# ================================
srun python -u "${EVAL_PY}" \
  --yaml_path "${YAML_PATH}" \
  --pred_dir "${PRED_DIR}" \
  --alex_tpl "${ALEX_TPL}" \
  --out_dir "${OUT_ROOT}" \
  --slice_axis ${SLICE_AXIS} \
  --slice_index ${SLICE_INDEX} \
  --map_count ${MAP_COUNT} \
  --ks_global_cap ${KS_GLOBAL_CAP} \
  --joint_sample ${JOINT_SAMPLE} \
  --pdf_bins ${PDF_BINS} \
  --joint_bins ${JOINT_BINS} \
  --voxel_size ${VOXEL_SIZE} \
  --rmax ${RMAX} \
  --n_r_bins ${N_R_BINS} \
  --label_pred "${CASE_LABEL}" \
  $( [ -n "${ALEX_FORCE}" ] && echo --alex_force "${ALEX_FORCE}" ) \
  $( [ -n "${LOSS_CSV_RESOLVED}" ] && echo --loss_csv "${LOSS_CSV_RESOLVED}" ) \
  --save_latex \
  2>&1 | tee -a "logs/vitunet3d_eval_fast_${SLURM_JOB_ID}.log"

EXIT_CODE=${PIPESTATUS[0]}
echo "=== [EVAL FAST END] $(date) (exit=${EXIT_CODE}) ==="
exit ${EXIT_CODE}
