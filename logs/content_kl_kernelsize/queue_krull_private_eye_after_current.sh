#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_SCRIPT="${SCRIPT_DIR}/run_content_kl_kernel_sweep.sh"

# Wait condition for the currently running frostbite sweep.
# You can override this pattern if needed.
WAIT_PATTERN=${WAIT_PATTERN:-"run_content_kl_kernel_sweep.sh|src_dreamerv3\\.dreamer.*atari_frostbite"}
POLL_SEC=${POLL_SEC:-60}

# Execution settings forwarded to run_content_kl_kernel_sweep.sh
CONFIGS=${CONFIGS:-atari100k}
SEED=${SEED:-0}
KERNEL_SIZES=${KERNEL_SIZES:-"1 3 5"}
STEPS=${STEPS:-120000}
EVAL_EVERY=${EVAL_EVERY:-20000}
LOG_EVERY=${LOG_EVERY:-20000}
EVAL_EPISODES=${EVAL_EPISODES:-1}
COMPILE=${COMPILE:-False}
GPU_ID=${GPU_ID:-0}
DEVICE=${DEVICE:-cuda:0}

# Use one shared run tag so krull/private_eye are grouped together.
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_krull_private_eye}
QUEUE_LOG=${QUEUE_LOG:-"${SCRIPT_DIR}/queue_krull_private_eye_after_current.log"}

{
  trap 'rc=$?; echo "[$(date "+%F %T")] ERROR: queue script aborted (exit=${rc})"; exit ${rc}' ERR

  echo "[$(date '+%F %T')] Queue script started"
  echo "[$(date '+%F %T')] WAIT_PATTERN=${WAIT_PATTERN}"
  echo "[$(date '+%F %T')] RUN_TAG=${RUN_TAG}"

  while true; do
    mapfile -t PROCS < <(pgrep -af "${WAIT_PATTERN}" || true)

    FILTERED=()
    for p in "${PROCS[@]}"; do
      # Ignore empty lines and this queue script itself.
      if [[ -z "$p" ]]; then
        continue
      fi
      if [[ "$p" == *"${0}"* ]]; then
        continue
      fi
      FILTERED+=("$p")
    done

    if [[ ${#FILTERED[@]} -eq 0 ]]; then
      echo "[$(date '+%F %T')] No matching running process found. Start queued tasks."
      break
    fi

    echo "[$(date '+%F %T')] Waiting... matched processes:"
    for p in "${FILTERED[@]}"; do
      echo "  $p"
    done
    sleep "${POLL_SEC}"
  done

  for TASK in atari_krull atari_private_eye; do
    echo "[$(date '+%F %T')] Start queued TASK=${TASK}"
    echo "[$(date '+%F %T')] Command: TASK=${TASK} RUN_TAG=${RUN_TAG} bash ${RUN_SCRIPT}"
    TASK="${TASK}" \
    CONFIGS="${CONFIGS}" \
    SEED="${SEED}" \
    KERNEL_SIZES="${KERNEL_SIZES}" \
    STEPS="${STEPS}" \
    EVAL_EVERY="${EVAL_EVERY}" \
    LOG_EVERY="${LOG_EVERY}" \
    EVAL_EPISODES="${EVAL_EPISODES}" \
    COMPILE="${COMPILE}" \
    GPU_ID="${GPU_ID}" \
    DEVICE="${DEVICE}" \
    RUN_TAG="${RUN_TAG}" \
    bash "${RUN_SCRIPT}"

    echo "[$(date '+%F %T')] Finished TASK=${TASK}"
  done

  echo "[$(date '+%F %T')] Queue script finished"
} | tee -a "${QUEUE_LOG}"
