#!/bin/bash
# Training script for Visual Pinpad using Director (RSSM)

set -e  # Exit on error

# Configuration
CONFIGS="visual_pinpad"
DYNAMICS_TYPE="rssm"
TASK="pinpad_four"
BASE_LOGDIR="logdir/director"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SEED=0
STEPS=3000000
GPUID="1"  # Set to empty string "" to use all available GPUs

# Force EGL backend for dm_control (Fixes OpenGL AttributeError)
export MUJOCO_GL="egl"

LOGDIR="${BASE_LOGDIR}/${TASK}_director_${TIMESTAMP}"

echo "=============================================="
echo "Director Visual Pinpad Training (RSSM)"
echo "=============================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Task: ${TASK}"
echo "Dynamics: ${DYNAMICS_TYPE}"
echo "Steps: ${STEPS}"
echo "Logdir: ${LOGDIR}"
echo "=============================================="

export CUDA_VISIBLE_DEVICES=${GPUID} 
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
