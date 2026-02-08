#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash logs/models/retrain_100k_and_zcheck.sh \
#     atari_frostbite \
#     /home/haselab/murakawa/stable-deep-world-model/logdir/atari_frostbite_vta_20260107_202443 \
#     500000
#
# Notes:
# - steps is in *environment steps* (the same "step" you see in metrics.jsonl).
# - For atari100k, action_repeat=4, so internal steps = env_steps / 4.
# - This script avoids OpenCV by training with --resize pillow.

TASK="${1:?task (e.g., atari_frostbite)}"
SRC_LOGDIR="${2:?source logdir that contains latest.pt + train_eps + eval_eps}"
TARGET_ENV_STEPS="${3:-500000}"

ROOT="/home/haselab/projects/stable-deep-world-model"
OUT_BASE="${ROOT}/logs/models"

RUN_NAME="retrain_100k_${TASK}_$(basename "${SRC_LOGDIR}")_to_${TARGET_ENV_STEPS}"
NEW_LOGDIR="${OUT_BASE}/${RUN_NAME}"

echo "[1/4] Prepare logdir: ${NEW_LOGDIR}"
mkdir -p "${NEW_LOGDIR}"

# Keep the episode dataset consistent with the original run (no huge copy).
ln -snf "${SRC_LOGDIR}/train_eps" "${NEW_LOGDIR}/train_eps"
ln -snf "${SRC_LOGDIR}/eval_eps" "${NEW_LOGDIR}/eval_eps"

# Copy the starting weights (so the original run remains untouched).
cp -f "${SRC_LOGDIR}/latest.pt" "${NEW_LOGDIR}/latest.pt"
cp -f "${SRC_LOGDIR}/latest.pt" "${NEW_LOGDIR}/init_latest.pt"

echo "[2/4] Continue training to env step ${TARGET_ENV_STEPS}"
# Show progress in real-time and keep a log.
LOGFILE="${NEW_LOGDIR}/train.log"
export PYTHONUNBUFFERED=1
# Checkpoints: every 25k env steps => every 25k/4=6250 internal steps (atari100k action_repeat=4).
uv run -m src_dreamerv3.dreamer \
  --configs atari100k \
  --task "${TASK}" \
  --logdir "${NEW_LOGDIR}" \
  --dynamics_type vta \
  --steps "${TARGET_ENV_STEPS}" \
  --resize pillow \
  --ckpt_every 6250 \
  --log_every 2500 \
  --eval_every 2500 \
  2>&1 | tee "${LOGFILE}"

echo "[3/4] Pick a fixed eval episode (longest in eval_eps) for consistent comparisons"
EP="$(
uv run - <<PY
from pathlib import Path
import re

eval_dir = Path("${NEW_LOGDIR}") / "eval_eps"
eps = list(eval_dir.glob("*.npz"))
if not eps:
  raise SystemExit(f"No .npz episodes found under {eval_dir}")

def ep_len(p: Path) -> int:
  m = re.search(r"-(\\d+)\\.npz$", p.name)
  return int(m.group(1)) if m else 0

print(max(eps, key=ep_len))
PY
)"
echo "EP=${EP}"

echo "[4/4] z(t)->z(t+1) verification for each checkpoint (constraints off)"
ZOUT="${NEW_LOGDIR}/zcheck"
mkdir -p "${ZOUT}"

shopt -s nullglob
CKPTS=( "${NEW_LOGDIR}"/init_latest.pt "${NEW_LOGDIR}"/checkpoints/step-*.pt "${NEW_LOGDIR}"/latest.pt )
for CKPT in "${CKPTS[@]}"; do
  STEM="$(basename "${CKPT%.pt}")"
  OUT_DIR="${ZOUT}/${STEM}"
  mkdir -p "${OUT_DIR}"
  uv run "${ROOT}/scripts/dreamerv3/vta_boundary_viz.py" \
    --logdir "${NEW_LOGDIR}" \
    --ckpt_path "${CKPT}" \
    --task "${TASK}" \
    --episode "${EP}" \
    --output_dir "${OUT_DIR}" \
    --dist_window 20 \
    --dist_stride 5 \
    --vta_boundary_force_scale 0 \
    --vta_max_seg_len 1000000 \
    --vta_max_seg_num 1000000
done

echo "Done."
echo "Run dir: ${NEW_LOGDIR}"
echo "zcheck:  ${ZOUT}"
