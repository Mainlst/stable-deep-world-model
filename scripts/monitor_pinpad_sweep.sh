#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATUS_FILE="${STATUS_FILE:-"$ROOT_DIR/logdir/pinpad_status.env"}"

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

while true; do
  if [[ -f "${STATUS_FILE}" ]]; then
    run_name=""
    run_index="0"
    total_runs="0"
    step="0"
    steps_target="0"
    percent="0"
    updated="unknown"
    while IFS='=' read -r key value; do
      case "${key}" in
        run_name) run_name="${value}" ;;
        run_index) run_index="${value}" ;;
        total_runs) total_runs="${value}" ;;
        step) step="${value}" ;;
        steps_target) steps_target="${value}" ;;
        percent) percent="${value}" ;;
        updated) updated="${value}" ;;
      esac
    done < "${STATUS_FILE}"
    bar=$(render_bar "${percent:-0}")
    printf "\r%s %s (%s/%s) step=%s/%s updated=%s" \
      "${bar}" "${run_name:-unknown}" "${run_index:-0}" "${total_runs:-0}" \
      "${step:-0}" "${steps_target:-0}" "${updated:-unknown}"
  else
    printf "\rStatus file not found: %s" "${STATUS_FILE}"
  fi
  sleep 2
done
