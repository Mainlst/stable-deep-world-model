#!/bin/bash

# VTA Baseline Experiments Runner
# Runs Fixed, Bernoulli, and Exponential boundary experiments SEQUENTIALLY.
# Adjusted to use python -m src_dreamerv3.dreamer consistent with train_multi_atari.sh

set -e  # Exit on error

# Task: atari_krull
TASK="atari_krull"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_LOGDIR="logs/vta_baselines/${TIMESTAMP}_${TASK}"

echo "Starting VTA Baseline Experiments for ${TASK} (SEQUENTIAL)..."
echo "Logs will be saved to: ${BASE_LOGDIR}"
echo "----------------------------------------"

# 1. Fixed-step Boundary (k=20)
echo "[1/3] Running Fixed-step Boundary (k=20)..."
LOGDIR_1="${BASE_LOGDIR}/fixed_k20"
echo "Logdir: ${LOGDIR_1}"

python -m src_dreamerv3.dreamer \
  --configs atari100k --task ${TASK} \
  --logdir "${LOGDIR_1}" \
  --dynamics_type vta \
  --ckpt_every 25000 \
  --ckpt_steps 0 \
  --vta_train_boundary_mode fixed \
  --vta_train_fixed_k 20

# 2. Bernoulli Random Boundary (p=0.05)
echo "[2/3] Running Bernoulli Random Boundary (p=0.05)..."
LOGDIR_2="${BASE_LOGDIR}/bernoulli_p0.05"
echo "Logdir: ${LOGDIR_2}"

python -m src_dreamerv3.dreamer \
  --configs atari100k --task ${TASK} \
  --logdir "${LOGDIR_2}" \
  --dynamics_type vta \
  --ckpt_every 25000 \
  --ckpt_steps 0 \
  --vta_train_boundary_mode bernoulli \
  --vta_train_bernoulli_p 0.05

# 3. Exponential Interval Boundary (lambda=0.05)
echo "[3/3] Running Exponential Interval Boundary (lambda=0.05)..."
LOGDIR_3="${BASE_LOGDIR}/exponential_lambda0.05"
echo "Logdir: ${LOGDIR_3}"

python -m src_dreamerv3.dreamer \
  --configs atari100k --task ${TASK} \
  --logdir "${LOGDIR_3}" \
  --dynamics_type vta \
  --ckpt_every 25000 \
  --ckpt_steps 0 \
  --vta_train_boundary_mode exponential \
  --vta_train_exp_lambda 0.05

echo "----------------------------------------"
echo "All experiments completed!"
