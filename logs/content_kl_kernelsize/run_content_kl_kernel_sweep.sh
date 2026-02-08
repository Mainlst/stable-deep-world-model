#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PYTHON_BIN_DEFAULT="${PROJECT_ROOT}/.venv/bin/python3"

# =========================
# Quick sweep defaults
# =========================
TASK=${TASK:-atari_frostbite}
CONFIGS=${CONFIGS:-atari100k}
SEED=${SEED:-0}
KERNEL_SIZES=${KERNEL_SIZES:-"1 3 5"}

# Fast-run settings (override via env if needed)
STEPS=${STEPS:-120000}
EVAL_EVERY=${EVAL_EVERY:-20000}
LOG_EVERY=${LOG_EVERY:-20000}
EVAL_EPISODES=${EVAL_EPISODES:-1}
COMPILE=${COMPILE:-False}

# RTX5090 one-GPU run (set GPU_ID as needed)
GPU_ID=${GPU_ID:-0}
DEVICE=${DEVICE:-cuda:0}

RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
RUNS_DIR="${SCRIPT_DIR}/runs/${RUN_TAG}"
mkdir -p "${RUNS_DIR}"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# Prefer project-local venv Python for non-interactive robustness.
if [[ -x "${PYTHON_BIN_DEFAULT}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_DEFAULT}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Runs dir:     ${RUNS_DIR}"
echo "[INFO] Task:         ${TASK}"
echo "[INFO] Kernels:      ${KERNEL_SIZES}"
echo "[INFO] Device:       ${DEVICE} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "[INFO] Steps:        ${STEPS}"
echo "[INFO] Python:       ${PYTHON_BIN}"

for K in ${KERNEL_SIZES}; do
  LOGDIR="${RUNS_DIR}/${TASK}_vta_k${K}_seed${SEED}"
  ANALYSIS_DIR="${LOGDIR}/analysis"
  mkdir -p "${ANALYSIS_DIR}"

  echo ""
  echo "============================================================"
  echo "[RUN] kernel_size=${K}"
  echo "[RUN] logdir=${LOGDIR}"
  echo "============================================================"

  "${PYTHON_BIN}" -m src_dreamerv3.dreamer \
    --configs "${CONFIGS}" \
    --task "${TASK}" \
    --dynamics_type vta \
    --seed "${SEED}" \
    --logdir "${LOGDIR}" \
    --device "${DEVICE}" \
    --steps "${STEPS}" \
    --eval_every "${EVAL_EVERY}" \
    --log_every "${LOG_EVERY}" \
    --eval_episode_num "${EVAL_EPISODES}" \
    --compile "${COMPILE}" \
    --vta_post_boundary_kernel_size "${K}"

  echo "[EVAL] abs/context KL plot..."
  "${PYTHON_BIN}" scripts/dreamerv3/vta_abs_kl_plot.py \
    --logdirs "${LOGDIR}" \
    --tasks "${TASK}" \
    --configs "${CONFIGS}" \
    --device "${DEVICE}" \
    --out "${ANALYSIS_DIR}/vta_abs_kl_plot.png" \
    --out_ctx "${ANALYSIS_DIR}/vta_ctx_kl_plot.png" \
    --out_abs_bidirectional "${ANALYSIS_DIR}/vta_abs_kl_bidirectional.png" \
    --csv_out "${ANALYSIS_DIR}/vta_abs_ctx_kl_series.csv"

  echo "[EVAL] z_t-z_t+1 distances + boundary stats (constraints off for eval)..."
  "${PYTHON_BIN}" scripts/dreamerv3/vta_boundary_viz.py \
    --logdir "${LOGDIR}" \
    --task "${TASK}" \
    --configs "${CONFIGS}" \
    --device "${DEVICE}" \
    --output_dir "${ANALYSIS_DIR}" \
    --csv_out "${ANALYSIS_DIR}/boundary_stats.csv" \
    --dist_csv "${ANALYSIS_DIR}/zt_zt_1.csv" \
    --dist_plot_out "${ANALYSIS_DIR}/zt_zt_1_plot.png" \
    --window boundary \
    --length 32 \
    --dist_window 32 \
    --dist_stride 8 \
    --overwrite_csv \
    --vta_post_boundary_kernel_size "${K}" \
    --vta_boundary_force_scale 0 \
    --vta_max_seg_len 1000000 \
    --vta_max_seg_num 1000000

done

echo ""
echo "[SUMMARY] Aggregating kernel comparison table..."
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_kernel_results.py" \
  --runs_dir "${RUNS_DIR}" \
  --out_csv "${RUNS_DIR}/summary.csv"

echo "[DONE] ${RUNS_DIR}/summary.csv"
