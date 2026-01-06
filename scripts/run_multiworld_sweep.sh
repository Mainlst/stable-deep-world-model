#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LOGROOT="${LOGROOT:-"$ROOT_DIR/logdir"}"
CONFIGS="${CONFIGS:-defaults}"
TASK="${TASK:-multiworld_Point2D-Image-v0}"
SEEDS="${SEEDS:-0}"

STEPS_SHORT="${STEPS_SHORT:-40000}"
EVAL_EVERY="${EVAL_EVERY:-10000}"
PREFILL="${PREFILL:-1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BATCH_LENGTH="${BATCH_LENGTH:-32}"
TRAIN_RATIO="${TRAIN_RATIO:-256}"
SIZE="${SIZE:-64}"
PRECISION="${PRECISION:-16}"
COMPILE="${COMPILE:-True}"
ACTION_REPEAT="${ACTION_REPEAT:-1}"
TIME_LIMIT="${TIME_LIMIT:-1000}"

run_exp() {
  local name="$1"
  shift
  local logdir="${LOGROOT}/${name}"
  python3 -m src_dreamerv3.dreamer \
    --logdir "${logdir}" \
    --configs "${CONFIGS}" \
    --task "${TASK}" \
    --steps "${STEPS_SHORT}" \
    --eval_every "${EVAL_EVERY}" \
    --log_every "${EVAL_EVERY}" \
    --envs 1 \
    --prefill "${PREFILL}" \
    --batch_size "${BATCH_SIZE}" \
    --batch_length "${BATCH_LENGTH}" \
    --train_ratio "${TRAIN_RATIO}" \
    --size "${SIZE}" \
    --action_repeat "${ACTION_REPEAT}" \
    --time_limit "${TIME_LIMIT}" \
    --precision "${PRECISION}" \
    --compile "${COMPILE}" \
    "$@"
}

run_viz_eval() {
  local logdir="$1"
  shift
  python3 scripts/dreamerv3/vta_boundary_viz.py \
    --logdir "${logdir}" \
    --configs "${CONFIGS}" \
    --task "${TASK}" \
    --length 20 \
    "$@"
  python3 scripts/dreamerv3/vta_boundary_eval.py \
    --logdir "${logdir}" \
    --configs "${CONFIGS}" \
    --task "${TASK}" \
    --episodes 1 \
    --permutations 50 \
    "$@"
}

for seed in ${SEEDS}; do
  run_exp "multiworld_rssm_seed${seed}" \
    --seed "${seed}" \
    --dynamics_type rssm

  run_exp "multiworld_vta_force0_t1_seed${seed}" \
    --seed "${seed}" \
    --dynamics_type vta \
    --vta_max_seg_len 30 \
    --vta_boundary_force_scale 0.0 \
    --vta_boundary_temp 1.0 \
    --vta_boundary_rate 0.0 \
    --vta_boundary_scale 0.0
  run_viz_eval "${LOGROOT}/multiworld_vta_force0_t1_seed${seed}" \
    --vta_max_seg_len 30 \
    --vta_boundary_force_scale 0.0 \
    --vta_boundary_temp 1.0

  run_exp "multiworld_vta_force10_t1_seed${seed}" \
    --seed "${seed}" \
    --dynamics_type vta \
    --vta_max_seg_len 30 \
    --vta_boundary_force_scale 10.0 \
    --vta_boundary_temp 1.0 \
    --vta_boundary_rate 0.0 \
    --vta_boundary_scale 0.0
  run_viz_eval "${LOGROOT}/multiworld_vta_force10_t1_seed${seed}" \
    --vta_max_seg_len 30 \
    --vta_boundary_force_scale 10.0 \
    --vta_boundary_temp 1.0

  run_exp "multiworld_vta_force0_t2_rate_seed${seed}" \
    --seed "${seed}" \
    --dynamics_type vta \
    --vta_max_seg_len 50 \
    --vta_boundary_force_scale 0.0 \
    --vta_boundary_temp 2.0 \
    --vta_boundary_rate 0.1 \
    --vta_boundary_scale 0.5
  run_viz_eval "${LOGROOT}/multiworld_vta_force0_t2_rate_seed${seed}" \
    --vta_max_seg_len 50 \
    --vta_boundary_force_scale 0.0 \
    --vta_boundary_temp 2.0 \
    --vta_boundary_rate 0.1 \
    --vta_boundary_scale 0.5
done
