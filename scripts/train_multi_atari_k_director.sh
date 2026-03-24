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
K_START="${K_START:-2}"
K_MAX="${K_MAX:-64}"

# List of Atari environments to train
TASKS=(
  "atari_krull"
  "atari_boxing"
  "atari_breakout"
)

echo "=============================================="
echo "Multi-Task Director Training (K Sweep)"
echo "=============================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Environments: ${TASKS[*]}"
echo "Dynamics: ${DYNAMICS_TYPE}"
echo "K sweep: ${K_START} -> ${K_MAX} (x2)"
echo "=============================================="

K=${K_START}
while [ "${K}" -le "${K_MAX}" ]; do
  for TASK in "${TASKS[@]}"; do
    LOGDIR="${BASE_LOGDIR}/${TASK}_director_k${K}_${TIMESTAMP}"

    echo ""
    echo "=============================================="
    echo "Starting training: ${TASK} (K=${K})"
    echo "Logdir: ${LOGDIR}"
    echo "=============================================="

    python -m src_director.dreamer \
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
  done
  K=$((K * 2))
done

echo ""
echo "=============================================="
echo "All K-sweep training completed!"
echo "=============================================="
