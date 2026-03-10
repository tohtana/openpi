#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/libero/track_a_common.sh
source "$SCRIPT_DIR/track_a_common.sh"

usage() {
  cat <<'USAGE'
Usage: sweep_track_a_eval.sh --matrix <path> --artifacts-root <path> --status-dir <path> [options]

Options:
  --matrix <path>          CSV file with run definitions (required)
  --artifacts-root <path>  Run artifacts root (required)
  --status-dir <path>      Row status marker directory (required)
  --max-parallel <int>     Max concurrent jobs (default: 8)
  --rows <id1,id2,...>     Optional comma-separated run_id filter
  --resume                 Rerun only failures/incomplete rows
  --dry-run                Print planned actions only
  --help                   Show this help
USAGE
}

MATRIX=""
ARTIFACTS_ROOT=""
STATUS_DIR=""
MAX_PARALLEL="8"
ROWS_FILTER=""
RESUME=0
DRY_RUN=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --matrix)
      MATRIX="${2:-}"
      shift 2
      ;;
    --artifacts-root)
      ARTIFACTS_ROOT="${2:-}"
      shift 2
      ;;
    --status-dir)
      STATUS_DIR="${2:-}"
      shift 2
      ;;
    --max-parallel)
      MAX_PARALLEL="${2:-}"
      shift 2
      ;;
    --rows)
      ROWS_FILTER="${2:-}"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      usage >&2
      fail "Unknown argument: $1" 2
      ;;
  esac
done

[[ -n "$MATRIX" && -n "$ARTIFACTS_ROOT" && -n "$STATUS_DIR" ]] || { usage >&2; fail "Missing required arguments" 2; }
[[ -f "$MATRIX" ]] || fail "Matrix file not found: $MATRIX" 2

ensure_dir "$ARTIFACTS_ROOT"
ensure_dir "$STATUS_DIR"

FAILURES_CSV="$(cd "$ARTIFACTS_ROOT/.." && pwd)/summary/failures.csv"

should_include_row() {
  local run_id="$1"
  local enabled="$2"

  if [[ "$enabled" != "1" ]]; then
    return 1
  fi

  if [[ -n "$ROWS_FILTER" ]] && ! contains_csv_token "$ROWS_FILTER" "$run_id"; then
    return 1
  fi

  if [[ "$RESUME" -eq 1 ]]; then
    local result_path="$ARTIFACTS_ROOT/$run_id/result.json"
    if [[ -f "$result_path" ]]; then
      if python - <<'PY' "$result_path"
import json
import sys
p = sys.argv[1]
with open(p, 'r', encoding='utf-8') as f:
    d = json.load(f)
ok = d.get('status') == 'pass' and int(d.get('exit_code', 1)) == 0
raise SystemExit(0 if ok else 1)
PY
      then
        return 1
      fi
    fi

    if [[ -f "$FAILURES_CSV" ]]; then
      if ! awk -F, 'NR>1 {print $1}' "$FAILURES_CSV" | grep -Fxq "$run_id" && [[ -f "$result_path" ]]; then
        return 1
      fi
    fi
  fi

  return 0
}

mapfile -t ROWS < <(python - <<'PY' "$MATRIX"
import csv
import sys
from pathlib import Path

matrix = Path(sys.argv[1])
required = ['run_id','task_suite','batch_size','n_episodes','gpu_id','seed','mujoco_gl','policy_path','enabled']
with matrix.open(newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    if reader.fieldnames != required:
        raise SystemExit(f"Unexpected matrix header: {reader.fieldnames}")
    for row in reader:
        vals = [
            row['run_id'], row['task_suite'], row['batch_size'], row['n_episodes'],
            row['gpu_id'], row['seed'], row['mujoco_gl'], row['policy_path'], row['enabled'],
        ]
        print("\t".join(vals))
PY
)

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "Dry-run matrix rows: ${#ROWS[@]}"
fi

active=0
declare -A PID_TO_RUN
FAILED_ROWS=()
attempted=0

launch_row() {
  local run_id="$1"
  local task_suite="$2"
  local batch_size="$3"
  local n_episodes="$4"
  local gpu_id="$5"
  local seed="$6"
  local mujoco_gl="$7"
  local policy_path="$8"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "Dry-run: would run $run_id on gpu=$gpu_id batch=$batch_size episodes=$n_episodes"
    return 0
  fi

  TRACK_A_STATUS_DIR="$STATUS_DIR" \
    bash "$SCRIPT_DIR/run_track_a_eval.sh" \
      --run-id "$run_id" \
      --policy-path "$policy_path" \
      --task-suite "$task_suite" \
      --batch-size "$batch_size" \
      --n-episodes "$n_episodes" \
      --gpu-id "$gpu_id" \
      --seed "$seed" \
      --mujoco-gl "$mujoco_gl" \
      --artifacts-root "$ARTIFACTS_ROOT" \
      --retry-index 0
}

wait_for_one() {
  local pid
  pid="$(jobs -pr | head -n 1 || true)"
  [[ -n "$pid" ]] || return 0
  local run_id="${PID_TO_RUN[$pid]:-unknown}"
  if wait "$pid"; then
    log "Completed: $run_id"
  else
    warn "Failed: $run_id"
    FAILED_ROWS+=("$run_id")
  fi
  unset 'PID_TO_RUN[$pid]'
  active=$((active - 1))
}

for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r run_id task_suite batch_size n_episodes gpu_id seed mujoco_gl policy_path enabled <<<"$row"

  if ! should_include_row "$run_id" "$enabled"; then
    log "Skipping row: $run_id"
    continue
  fi

  attempted=$((attempted + 1))

  if [[ "$DRY_RUN" -eq 1 ]]; then
    launch_row "$run_id" "$task_suite" "$batch_size" "$n_episodes" "$gpu_id" "$seed" "$mujoco_gl" "$policy_path"
    continue
  fi

  while [[ "$active" -ge "$MAX_PARALLEL" ]]; do
    wait_for_one
  done

  (
    launch_row "$run_id" "$task_suite" "$batch_size" "$n_episodes" "$gpu_id" "$seed" "$mujoco_gl" "$policy_path"
  ) &
  pid=$!
  PID_TO_RUN[$pid]="$run_id"
  active=$((active + 1))
done

if [[ "$DRY_RUN" -ne 1 ]]; then
  while [[ "$active" -gt 0 ]]; do
    wait_for_one
  done
fi

if [[ "${#FAILED_ROWS[@]}" -gt 0 ]]; then
  replay_rows="$(IFS=,; echo "${FAILED_ROWS[*]}")"
  printf '\nDeterministic replay command:\n'
  printf 'bash scripts/libero/sweep_track_a_eval.sh --matrix %q --artifacts-root %q --status-dir %q --rows %q --max-parallel 1 --resume\n' "$MATRIX" "$ARTIFACTS_ROOT" "$STATUS_DIR" "$replay_rows"
  exit 1
fi

log "Sweep finished; attempted rows: $attempted"
