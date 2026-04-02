#!/bin/bash
# Training script for DMC Hopper Hop using Director (RSSM)

set -e  # Exit on error

# Configuration
CONFIGS="dmc_vision"
DYNAMICS_TYPE="rssm"
TASK="dmc_hopper_hop"
BASE_LOGDIR="logdir/director"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SEED=0
STEPS=2000000

# Force EGL backend for dm_control (Fixes OpenGL AttributeError)
export MUJOCO_GL="egl"

LOGDIR="${BASE_LOGDIR}/${TASK}_director_${TIMESTAMP}"

echo "=============================================="
echo "Director Hopper Hop Training (RSSM)"
echo "=============================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Task: ${TASK}"
echo "Dynamics: ${DYNAMICS_TYPE}"
echo "Steps: ${STEPS}"
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
echo "=============================================="
echo "Training completed!"
echo "=============================================="
