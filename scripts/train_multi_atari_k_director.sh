#!/bin/bash
# Train Director on multiple Atari tasks while sweeping goal update period K.

set -euo pipefail

# Configuration (override with env vars if needed)
CONFIGS="${CONFIGS:-atari100k}"
DYNAMICS_TYPE="${DYNAMICS_TYPE:-rssm}"
BASE_LOGDIR="${BASE_LOGDIR:-logdir/director_k_sweep}"
SEED="${SEED:-0}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# K sweep: 2, 4, 8, ...
K_START="${K_START:-64}"
K_MAX="${K_MAX:-64}"

# Sleep between tasks (seconds)
SLEEP_BETWEEN_TASKS="${SLEEP_BETWEEN_TASKS:-600}"

# List of Atari environments to train
TASKS=(
  "atari_bank_heist"
  "atari_frostbite"
  "atari_qbert"
)

echo "=============================================="
echo "Multi-Task Director Training (K Sweep)"
echo "=============================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Environments: ${TASKS[*]}"
echo "Dynamics: ${DYNAMICS_TYPE}"
echo "K sweep: ${K_START} -> ${K_MAX} (x2)"
echo "Sleep between tasks: ${SLEEP_BETWEEN_TASKS} sec"
echo "=============================================="

K=${K_START}
while [ "${K}" -le "${K_MAX}" ]; do
  for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    LOGDIR="${BASE_LOGDIR}/${TASK}_director_k${K}_${TIMESTAMP}"

    echo ""
    echo "=============================================="
    echo "Starting training: ${TASK} (K=${K})"
    echo "Logdir: ${LOGDIR}"
    echo "=============================================="

    uv run -m src_director.dreamer \
      --configs "${CONFIGS}" \
      --task "${TASK}" \
      --dynamics_type "${DYNAMICS_TYPE}" \
      --train_skill_duration "${K}" \
      --eval_skill_duration "${K}" \
      --logdir "${LOGDIR}" \
      --seed "${SEED}"

    echo ""
    echo "Completed: ${TASK} (K=${K})"
    echo "=============================================="

    # Sleep before next task, unless this is the last task in the current K loop
    if [ "$i" -lt $((${#TASKS[@]} - 1)) ]; then
      echo "Sleeping for ${SLEEP_BETWEEN_TASKS} seconds before next task..."
      sleep "${SLEEP_BETWEEN_TASKS}"
    fi
  done
  K=$((K * 2))
done

echo ""
echo "=============================================="
echo "All K-sweep training completed!"
echo "=============================================="