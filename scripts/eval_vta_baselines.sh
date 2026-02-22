#!/bin/bash
set -e

# Usage: ./scripts/eval_vta_baselines.sh [TIMESTAMP_DIR_NAME]
# Example: ./scripts/eval_vta_baselines.sh 20260210_203308_atari_krull

BASE_ROOT="logs/vta_baselines"

if [ -z "$1" ]; then
    # Find the latest directory if not provided
    TIMESTAMP_DIR=$(ls -t $BASE_ROOT | head -n 1)
    echo "No timestamp provided. Using latest: $TIMESTAMP_DIR"
else
    TIMESTAMP_DIR="$1"
fi

FULL_DIR="${BASE_ROOT}/${TIMESTAMP_DIR}"

if [ ! -d "$FULL_DIR" ]; then
    echo "Directory not found: $FULL_DIR"
    exit 1
fi

echo "Evaluating experiments in: $FULL_DIR"

# Define the 3 sub-experiment directories
DIR1="${FULL_DIR}/fixed_k20"
DIR2="${FULL_DIR}/bernoulli_p0.05"
DIR3="${FULL_DIR}/exponential_lambda0.05"

INPUT_DIRS=""
TASKS=""

# Check which exist and build arguments
if [ -d "$DIR1" ]; then
    INPUT_DIRS="$INPUT_DIRS $DIR1"
    TASKS="$TASKS atari_krull"
fi
if [ -d "$DIR2" ]; then
    INPUT_DIRS="$INPUT_DIRS $DIR2"
    TASKS="$TASKS atari_krull"
fi
if [ -d "$DIR3" ]; then
    INPUT_DIRS="$INPUT_DIRS $DIR3"
    TASKS="$TASKS atari_krull"
fi

if [ -z "$INPUT_DIRS" ]; then
    echo "No experiment subdirectories found in $FULL_DIR"
    exit 1
fi

OUTPUT_DIR="${FULL_DIR}/eval_results"
mkdir -p $OUTPUT_DIR

echo "Running vta_ckpt_sweep_eval.py..."
python scripts/dreamerv3/vta_ckpt_sweep_eval.py \
    --logdirs $INPUT_DIRS \
    --tasks $TASKS \
    --output_dir $OUTPUT_DIR \
    --boundary_source prior \
    --overwrite_csv

echo "Evaluation complete. Results saved to $OUTPUT_DIR"
