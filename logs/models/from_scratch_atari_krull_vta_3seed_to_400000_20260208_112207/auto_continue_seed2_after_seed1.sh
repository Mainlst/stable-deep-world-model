#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/haselab/projects/stable-deep-world-model"
RUN_DIR="${ROOT}/logs/models/from_scratch_atari_krull_vta_3seed_to_400000_20260208_112207"
SEED1_LOGDIR="${RUN_DIR}/atari_krull_seed1"
SEED2_LOGDIR="${RUN_DIR}/atari_krull_seed2"
METRICS1="${SEED1_LOGDIR}/metrics.jsonl"
TARGET_STEPS=400000
POLL_SEC=60
CHAIN_LOG="${RUN_DIR}/auto_continue_seed2_after_seed1.log"
SEED2_LOG="${SEED2_LOGDIR}/resume_seed2.nohup.log"

mkdir -p "${SEED2_LOGDIR}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${CHAIN_LOG}"
}

latest_step() {
  if [[ -s "${METRICS1}" ]]; then
    local step
    step="$(tac "${METRICS1}" | rg -m1 -o '"step"\s*:\s*[0-9]+' | rg -o '[0-9]+' | head -n 1 || true)"
    if [[ -n "${step}" ]]; then
      echo "${step}"
    else
      echo 0
    fi
  else
    echo 0
  fi
}

is_seed1_running() {
  pgrep -f "src_dreamerv3.dreamer.*--logdir ${SEED1_LOGDIR}" >/dev/null 2>&1
}

log "Watcher started. Waiting for seed1 completion (target step=${TARGET_STEPS})."

while true; do
  STEP="$(latest_step)"

  if is_seed1_running; then
    log "seed1 running. latest_step=${STEP}."
    sleep "${POLL_SEC}"
    continue
  fi

  if [[ "${STEP}" =~ ^[0-9]+$ ]] && (( STEP >= TARGET_STEPS )); then
    log "seed1 completed at step=${STEP}. Launching seed2."
    break
  fi

  log "seed1 process not found and latest_step=${STEP} (<${TARGET_STEPS}). seed2 will NOT start."
  exit 1
done

if [[ -e "${SEED2_LOGDIR}/latest.pt" ]]; then
  log "seed2 latest.pt already exists. Refusing to overwrite: ${SEED2_LOGDIR}"
  exit 1
fi

cd "${ROOT}"
export PYTHONUNBUFFERED=1

log "Starting seed2 training to ${TARGET_STEPS} steps."
/home/haselab/.local/bin/uv run -m src_dreamerv3.dreamer \
  --configs atari100k \
  --task atari_krull \
  --dynamics_type vta \
  --seed 2 \
  --logdir "${SEED2_LOGDIR}" \
  --steps "${TARGET_STEPS}" \
  --resize pillow \
  --ckpt_every 6250 \
  --log_every 625 \
  --eval_every 625 >> "${SEED2_LOG}" 2>&1

RC=$?
if (( RC == 0 )); then
  log "seed2 finished successfully."
else
  log "seed2 exited with code ${RC}."
fi
exit "${RC}"
