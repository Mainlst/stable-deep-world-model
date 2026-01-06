#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LOGROOT="${LOGROOT:-"$ROOT_DIR/logdir"}"
CONFIGS="${CONFIGS:-pinpad}"
TASK="${TASK:-}"
SEEDS="${SEEDS:-0}"
SKIP_RSSM="${SKIP_RSSM:-0}"

STEPS_SHORT="${STEPS_SHORT:-40000}"
EVAL_EVERY="${EVAL_EVERY:-10000}"
VTA_EVAL_EVERY="${VTA_EVAL_EVERY:-2000}"
PREFILL="${PREFILL:-1200}"
BATCH_SIZE="${BATCH_SIZE:-12}"
BATCH_LENGTH="${BATCH_LENGTH:-40}"
TRAIN_RATIO="${TRAIN_RATIO:-320}"
SIZE="${SIZE:-64}"
PRECISION="${PRECISION:-16}"
COMPILE="${COMPILE:-True}"
ACTION_REPEAT="${ACTION_REPEAT:-1}"
TIME_LIMIT="${TIME_LIMIT:-1000}"
PRETRAIN="${PRETRAIN:-10}"

RUN_LOG="${RUN_LOG:-"$LOGROOT/pinpad_sweep.log"}"
STATUS_FILE="${STATUS_FILE:-"$LOGROOT/pinpad_status.env"}"

if [[ -z "${TASK}" ]]; then
  echo "Set TASK to your pinpad suite task, e.g. TASK=pinpad_eight"
  exit 1
fi

mkdir -p "${LOGROOT}"
echo "Run log: ${RUN_LOG}"
echo "Status file: ${STATUS_FILE}"

total_runs=0
for _ in ${SEEDS}; do
  total_runs=$((total_runs + 4))
done
run_index=0

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

announce() {
  local msg="$1"
  echo "[$(timestamp)] ${msg}" | tee -a "${RUN_LOG}"
}

write_status() {
  local name="$1"
  local step="$2"
  local steps_target="$3"
  local percent="$4"
  local updated
  updated=$(date "+%Y-%m-%dT%H:%M:%S")
  cat > "${STATUS_FILE}" <<EOF
run_name=${name}
run_index=${run_index}
total_runs=${total_runs}
step=${step}
steps_target=${steps_target}
percent=${percent}
updated=${updated}
EOF
}

render_bar() {
  local percent="$1"
  local width=30
  local filled=$((percent * width / 100))
  local empty=$((width - filled))
  local bar=""
  for ((i=0; i<filled; i++)); do bar+="#"; done
  for ((i=0; i<empty; i++)); do bar+="-"; done
  echo "[${bar}] ${percent}%"
}

run_exp() {
  local name="$1"
  shift
  local logdir="${LOGROOT}/${name}"
  run_index=$((run_index + 1))
  announce "Start ${name} (${run_index}/${total_runs})"
  local metrics_file="${logdir}/metrics.jsonl"
  local steps_target="${STEPS_SHORT}"
  (
    set -o pipefail
    PYTHONUNBUFFERED=1 stdbuf -oL -eL python3 -m src_dreamerv3.dreamer \
      --logdir "${logdir}" \
      --configs "${CONFIGS}" \
      --task "${TASK}" \
      --steps "${STEPS_SHORT}" \
      --eval_every "${EVAL_EVERY}" \
      --log_every "${EVAL_EVERY}" \
      --envs 1 \
      --prefill "${PREFILL}" \
      --pretrain "${PRETRAIN}" \
      --batch_size "${BATCH_SIZE}" \
      --batch_length "${BATCH_LENGTH}" \
      --train_ratio "${TRAIN_RATIO}" \
      --size "${SIZE}" \
      --action_repeat "${ACTION_REPEAT}" \
      --time_limit "${TIME_LIMIT}" \
      --precision "${PRECISION}" \
      --compile "${COMPILE}" \
      "$@" 2>&1 | tee -a "${RUN_LOG}"
  ) &
  local run_pid=$!
  local percent=0
  local step=0
  while kill -0 "${run_pid}" 2>/dev/null; do
    step=0
    if [[ -f "${metrics_file}" ]]; then
      last_line=$(tail -n 1 "${metrics_file}" || true)
      if [[ "${last_line}" =~ \"step\":[[:space:]]*([0-9]+) ]]; then
        step="${BASH_REMATCH[1]}"
      fi
    fi
    # Fallback progress from saved episodes (increments by ~1000 on pinpad).
    if [[ -d "${logdir}/train_eps" ]]; then
      ep_count=$(ls -1 "${logdir}/train_eps"/*.npz 2>/dev/null | wc -l | tr -d ' ')
      if [[ "${ep_count}" =~ ^[0-9]+$ ]]; then
        ep_step=$((ep_count * 1000))
        if [[ "${ep_step}" -gt "${step}" ]]; then
          step="${ep_step}"
        fi
      fi
    fi
    if [[ "${steps_target}" -gt 0 ]]; then
      percent=$((step * 100 / steps_target))
      if [[ "${percent}" -gt 100 ]]; then
        percent=100
      fi
    else
      percent=0
    fi
    bar=$(render_bar "${percent}")
    write_status "${name}" "${step}" "${steps_target}" "${percent}"
    printf "\r%s %s (%d/%d) step=%d/%d" "${bar}" "${name}" "${run_index}" "${total_runs}" "${step}" "${steps_target}"
    sleep 5
  done
  set +e
  wait "${run_pid}"
  local exit_code=$?
  set -e
  printf "\n"
  if [[ "${exit_code}" -eq 0 ]]; then
    announce "Done ${name} (${run_index}/${total_runs})"
  else
    announce "Failed ${name} (exit=${exit_code}) (${run_index}/${total_runs})"
  fi
}

run_viz_eval() {
  local logdir="$1"
  shift
  local ckpt_path="${logdir}/latest.pt"
  if [[ ! -f "${ckpt_path}" ]]; then
    announce "Skip viz/eval (missing checkpoint): ${ckpt_path}"
    return 0
  fi
  announce "Viz ${logdir}"
  set +e
  PYTHONUNBUFFERED=1 stdbuf -oL -eL python3 scripts/dreamerv3/vta_boundary_viz.py \
    --logdir "${logdir}" \
    --configs "${CONFIGS}" \
    --task "${TASK}" \
    --length 20 \
    "$@" 2>&1 | tee -a "${RUN_LOG}"
  local viz_exit=$?
  set -e
  if [[ "${viz_exit}" -ne 0 ]]; then
    announce "Viz failed (exit=${viz_exit}): ${logdir}"
    return 0
  fi
  announce "Eval ${logdir}"
  set +e
  PYTHONUNBUFFERED=1 stdbuf -oL -eL python3 scripts/dreamerv3/vta_boundary_eval.py \
    --logdir "${logdir}" \
    --configs "${CONFIGS}" \
    --task "${TASK}" \
    --episodes 1 \
    --permutations 50 \
    "$@" 2>&1 | tee -a "${RUN_LOG}"
  local eval_exit=$?
  set -e
  if [[ "${eval_exit}" -ne 0 ]]; then
    announce "Eval failed (exit=${eval_exit}): ${logdir}"
  fi
}

for seed in ${SEEDS}; do
  if [[ "${SKIP_RSSM}" != "1" ]]; then
    run_exp "pinpad_rssm_seed${seed}" \
      --seed "${seed}" \
      --dynamics_type rssm
  else
    announce "Skip RSSM baseline (SKIP_RSSM=1)"
  fi

  run_exp "pinpad_vta_force0_t1_seed${seed}" \
    --seed "${seed}" \
    --dynamics_type vta \
    --vta_max_seg_len 30 \
    --vta_boundary_force_scale 0.0 \
    --vta_boundary_temp 1.0 \
    --vta_boundary_rate 0.0 \
    --vta_boundary_scale 0.0 \
    --compile False \
    --precision 32 \
    --eval_every "${VTA_EVAL_EVERY}" \
    --log_every "${VTA_EVAL_EVERY}"
  run_viz_eval "${LOGROOT}/pinpad_vta_force0_t1_seed${seed}" \
    --vta_max_seg_len 30 \
    --vta_boundary_force_scale 0.0 \
    --vta_boundary_temp 1.0

  run_exp "pinpad_vta_force10_t1_seed${seed}" \
    --seed "${seed}" \
    --dynamics_type vta \
    --vta_max_seg_len 30 \
    --vta_boundary_force_scale 10.0 \
    --vta_boundary_temp 1.0 \
    --vta_boundary_rate 0.0 \
    --vta_boundary_scale 0.0 \
    --compile False \
    --precision 32 \
    --eval_every "${VTA_EVAL_EVERY}" \
    --log_every "${VTA_EVAL_EVERY}"
  run_viz_eval "${LOGROOT}/pinpad_vta_force10_t1_seed${seed}" \
    --vta_max_seg_len 30 \
    --vta_boundary_force_scale 10.0 \
    --vta_boundary_temp 1.0

  run_exp "pinpad_vta_force0_t2_rate_seed${seed}" \
    --seed "${seed}" \
    --dynamics_type vta \
    --vta_max_seg_len 50 \
    --vta_boundary_force_scale 0.0 \
    --vta_boundary_temp 2.0 \
    --vta_boundary_rate 0.1 \
    --vta_boundary_scale 0.5 \
    --compile False \
    --precision 32 \
    --eval_every "${VTA_EVAL_EVERY}" \
    --log_every "${VTA_EVAL_EVERY}"
  run_viz_eval "${LOGROOT}/pinpad_vta_force0_t2_rate_seed${seed}" \
    --vta_max_seg_len 50 \
    --vta_boundary_force_scale 0.0 \
    --vta_boundary_temp 2.0 \
    --vta_boundary_rate 0.1 \
    --vta_boundary_scale 0.5
done
