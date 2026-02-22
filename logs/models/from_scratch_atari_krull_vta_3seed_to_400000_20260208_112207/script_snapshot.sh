#!/usr/bin/env bash
set -euo pipefail

# Train from scratch (VTA) and measure adjacent z_t, z_{t+1} distances
# across checkpoints for multiple seeds.
#
# Usage:
#   bash logs/models/train_krull_400k_3seed_and_zt_sweep.sh
#   bash logs/models/train_krull_400k_3seed_and_zt_sweep.sh atari_krull 400000 "0 1 2"
#
# Args:
#   $1: task (default: atari_krull)
#   $2: target env steps (default: 400000)
#   $3: seeds as space-separated string (default: "0 1 2")

TASK="${1:-atari_krull}"
TARGET_ENV_STEPS="${2:-400000}"
SEEDS_STR="${3:-0 1 2}"

ROOT="/home/haselab/projects/stable-deep-world-model"
OUT_BASE="${ROOT}/logs/models"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="from_scratch_${TASK}_vta_3seed_to_${TARGET_ENV_STEPS}_${TIMESTAMP}"
RUN_DIR="${OUT_BASE}/${RUN_NAME}"

# atari100k defaults
ACTION_REPEAT=4
CKPT_EVERY_ENV=25000
LOG_EVERY_ENV=2500
EVAL_EVERY_ENV=2500

if (( TARGET_ENV_STEPS % ACTION_REPEAT != 0 )); then
  echo "TARGET_ENV_STEPS (${TARGET_ENV_STEPS}) must be divisible by ACTION_REPEAT (${ACTION_REPEAT})." >&2
  exit 1
fi
if (( CKPT_EVERY_ENV % ACTION_REPEAT != 0 )); then
  echo "CKPT_EVERY_ENV (${CKPT_EVERY_ENV}) must be divisible by ACTION_REPEAT (${ACTION_REPEAT})." >&2
  exit 1
fi

CKPT_EVERY_INTERNAL=$((CKPT_EVERY_ENV / ACTION_REPEAT))
LOG_EVERY_INTERNAL=$((LOG_EVERY_ENV / ACTION_REPEAT))
EVAL_EVERY_INTERNAL=$((EVAL_EVERY_ENV / ACTION_REPEAT))

cd "${ROOT}"

echo "[0/3] Prepare run directory: ${RUN_DIR}"
mkdir -p "${RUN_DIR}"
printf "task=%s\ntarget_env_steps=%s\naction_repeat=%s\nseeds=%s\n" \
  "${TASK}" "${TARGET_ENV_STEPS}" "${ACTION_REPEAT}" "${SEEDS_STR}" > "${RUN_DIR}/run_config.txt"
cp -f "$0" "${RUN_DIR}/script_snapshot.sh"

IFS=' ' read -r -a SEEDS <<< "${SEEDS_STR}"
LOGDIRS=()
TASKS=()

for SEED in "${SEEDS[@]}"; do
  LOGDIR="${RUN_DIR}/${TASK}_seed${SEED}"
  LOGDIRS+=("${LOGDIR}")
  TASKS+=("${TASK}")

  if [[ -e "${LOGDIR}/latest.pt" ]]; then
    echo "Refusing to overwrite existing run: ${LOGDIR}" >&2
    exit 1
  fi

  echo "[1/3][seed=${SEED}] Train from scratch to env step ${TARGET_ENV_STEPS}"
  mkdir -p "${LOGDIR}"
  export PYTHONUNBUFFERED=1
  uv run -m src_dreamerv3.dreamer \
    --configs atari100k \
    --task "${TASK}" \
    --dynamics_type vta \
    --seed "${SEED}" \
    --logdir "${LOGDIR}" \
    --steps "${TARGET_ENV_STEPS}" \
    --resize pillow \
    --ckpt_every "${CKPT_EVERY_INTERNAL}" \
    --log_every "${LOG_EVERY_INTERNAL}" \
    --eval_every "${EVAL_EVERY_INTERNAL}" \
    2>&1 | tee "${LOGDIR}/train.log"
done

echo "[2/3] Sweep z_t -> z_{t+1} distances over checkpoints"
SWEEP_OUT="${RUN_DIR}/zt_zt_1_sweep"
mkdir -p "${SWEEP_OUT}"
uv run "${ROOT}/scripts/dreamerv3/vta_ckpt_sweep_eval.py" \
  --logdirs "${LOGDIRS[@]}" \
  --tasks "${TASKS[@]}" \
  --episodes_dir eval_eps \
  --output_dir "${SWEEP_OUT}" \
  --configs atari100k \
  --seed 0 \
  --dist_window 20 \
  --dist_stride 5 \
  --vta_boundary_force_scale 0.0 \
  --overwrite_csv

echo "[3/3] Done"
echo "Run root:      ${RUN_DIR}"
echo "Per-seed logs: ${RUN_DIR}/${TASK}_seed* (seeds: ${SEEDS_STR})"
echo "Sweep CSV:     ${SWEEP_OUT}/vta_z_summary.csv"
echo "Sweep windows: ${SWEEP_OUT}/vta_z_windows.csv"
