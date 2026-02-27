#!/bin/bash
# Train VTA on multiple Atari environments sequentially

set -e  # Exit on error

# Configuration
CONFIGS="atari100k"
DYNAMICS_TYPE="director"
BASE_LOGDIR="logdir/director"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SEED=80
STEPS=200000  # Number of training steps per environment

# List of Atari environments to train
TASKS=(
    "atari_frostbite"
    # "atari_breakout"
    # "atari_private_eye"
)

# VTA hyperparameters: using config defaults (no overrides)
# To customize, uncomment and modify:
# MAX_SEG_LEN=50
# BOUNDARY_FORCE_SCALE=10.0
# BOUNDARY_TEMP=1.0

echo "=============================================="
echo "Multi-Environment Atari Training"
echo "=============================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Environments: ${TASKS[@]}"
echo "Dynamics: ${DYNAMICS_TYPE}"
echo "=============================================="

for TASK in "${TASKS[@]}"; do
    LOGDIR="${BASE_LOGDIR}/${TASK}_${DYNAMICS_TYPE}_${TIMESTAMP}"
    # LOGDIR="${BASE_LOGDIR}/atari_frostbite_director_20260203_204202"
    
    echo ""
    echo "=============================================="
    echo "Starting training: ${TASK}"
    echo "Logdir: ${LOGDIR}"
    echo "=============================================="
    
    python -m src_director.dreamer \
        --configs ${CONFIGS} \
        --task ${TASK} \
        --dynamics_type ${DYNAMICS_TYPE} \
        --logdir ${LOGDIR} \
        --seed ${SEED} \
        --steps ${STEPS}
    
    echo ""
    echo "Completed: ${TASK}"
    echo "=============================================="
done

echo ""
echo "=============================================="
echo "All training completed!"
echo "=============================================="
