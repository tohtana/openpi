#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_pi05_libero_finetune_example.sh [options]

Runs a bounded OpenPI LIBERO fine-tuning example for the documented `pi05_libero`
config and writes logs under `tasks/openpi-scripted-libero-finetune-example/`.

Options:
  --exp-name <name>             Training experiment name.
  --run-id <id>                 Artifact run id. Defaults to --exp-name.
  --max-frames <int>            Max frames for bounded norm-stats refresh. Default: 256
  --num-train-steps <int>       Number of training steps to run. Default: 1
  --save-interval <int>         Checkpoint save interval. Default: 1
  --log-interval <int>          Train log interval. Default: 1
  --checkpoint-base-dir <path>  Checkpoint base directory.
                                Default: /mnt/local_storage/experiments/openpi_checkpoints
  --artifacts-root <path>       Root directory for logs and manifests.
                                Default: test_openpi/tasks/openpi-scripted-libero-finetune-example/runs
  --refresh-norm-stats          Recompute norm stats even if cached stats already exist.
  --allow-overwrite             Reuse an existing run directory.
  --dry-run                     Print the resolved plan and exit.
  --help                        Show this help.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_OPENPI_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OPENPI_DIR="$TEST_OPENPI_ROOT/openpi"
CONFIG_NAME="pi05_libero"
DEFAULT_EXP_NAME="pi05_libero_scripted_$(date -u +%Y%m%dT%H%M%SZ)"
EXP_NAME="$DEFAULT_EXP_NAME"
RUN_ID=""
MAX_FRAMES="256"
NUM_TRAIN_STEPS="1"
SAVE_INTERVAL="1"
LOG_INTERVAL="1"
CHECKPOINT_BASE_DIR="/mnt/local_storage/experiments/openpi_checkpoints"
ARTIFACTS_ROOT="$TEST_OPENPI_ROOT/tasks/openpi-scripted-libero-finetune-example/runs"
REFRESH_NORM_STATS=0
ALLOW_OVERWRITE=0
DRY_RUN=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --exp-name)
      EXP_NAME="${2:-}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --max-frames)
      MAX_FRAMES="${2:-}"
      shift 2
      ;;
    --num-train-steps)
      NUM_TRAIN_STEPS="${2:-}"
      shift 2
      ;;
    --save-interval)
      SAVE_INTERVAL="${2:-}"
      shift 2
      ;;
    --log-interval)
      LOG_INTERVAL="${2:-}"
      shift 2
      ;;
    --checkpoint-base-dir)
      CHECKPOINT_BASE_DIR="${2:-}"
      shift 2
      ;;
    --artifacts-root)
      ARTIFACTS_ROOT="${2:-}"
      shift 2
      ;;
    --refresh-norm-stats)
      REFRESH_NORM_STATS=1
      shift
      ;;
    --allow-overwrite)
      ALLOW_OVERWRITE=1
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
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$EXP_NAME"
fi

if [[ ! -d "$OPENPI_DIR" ]]; then
  echo "OpenPI checkout not found at: $OPENPI_DIR" >&2
  exit 1
fi

command -v uv >/dev/null || {
  echo "uv is required but not installed" >&2
  exit 1
}

if ! [[ "$MAX_FRAMES" =~ ^[0-9]+$ && "$NUM_TRAIN_STEPS" =~ ^[0-9]+$ && "$SAVE_INTERVAL" =~ ^[0-9]+$ && "$LOG_INTERVAL" =~ ^[0-9]+$ ]]; then
  echo "Numeric arguments must be non-negative integers" >&2
  exit 2
fi

if (( MAX_FRAMES < 256 )); then
  echo "--max-frames must be at least 256 for the pi05_libero batch size" >&2
  exit 2
fi

if (( NUM_TRAIN_STEPS < 1 )); then
  echo "--num-train-steps must be at least 1" >&2
  exit 2
fi

rg -n 'name=\"pi05_libero\"' "$OPENPI_DIR/src/openpi/training/config.py" >/dev/null || {
  echo "pi05_libero config not found in $OPENPI_DIR/src/openpi/training/config.py" >&2
  exit 1
}

RUN_DIR="$ARTIFACTS_ROOT/$RUN_ID"
COMPUTE_LOG="$RUN_DIR/compute_norm_stats.log"
TRAIN_LOG="$RUN_DIR/train.log"
MANIFEST_JSON="$RUN_DIR/run_manifest.json"
RESULT_JSON="$RUN_DIR/result.json"
NORM_STATS_FILE="$OPENPI_DIR/assets/pi05_libero/physical-intelligence/libero/norm_stats.json"
CHECKPOINT_RUN_DIR="$CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME"

if [[ -e "$RUN_DIR" && "$ALLOW_OVERWRITE" -ne 1 ]]; then
  echo "Run directory already exists: $RUN_DIR (use --allow-overwrite to reuse it)" >&2
  exit 2
fi

mkdir -p "$RUN_DIR" "$CHECKPOINT_BASE_DIR"

HF_HOME_VALUE="${OPENPI_HF_HOME:-/mnt/local_storage/huggingface}"
HF_HUB_CACHE_VALUE="${OPENPI_HF_HUB_CACHE:-$HF_HOME_VALUE/hub}"
XDG_CACHE_HOME_VALUE="${OPENPI_XDG_CACHE_HOME:-/mnt/local_storage/.cache}"
OPENPI_DATA_HOME_VALUE="${OPENPI_DATA_HOME:-/mnt/local_storage/.cache/openpi}"
XLA_MEM_FRACTION_VALUE="${OPENPI_XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

export HF_HOME="$HF_HOME_VALUE"
export HF_HUB_CACHE="$HF_HUB_CACHE_VALUE"
export XDG_CACHE_HOME="$XDG_CACHE_HOME_VALUE"
export OPENPI_DATA_HOME="$OPENPI_DATA_HOME_VALUE"
export XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_MEM_FRACTION_VALUE"

python - <<'PY' "$MANIFEST_JSON" "$RUN_ID" "$TEST_OPENPI_ROOT" "$OPENPI_DIR" "$CONFIG_NAME" "$EXP_NAME" "$MAX_FRAMES" "$NUM_TRAIN_STEPS" "$SAVE_INTERVAL" "$LOG_INTERVAL" "$CHECKPOINT_BASE_DIR" "$ARTIFACTS_ROOT" "$REFRESH_NORM_STATS" "$HF_HOME" "$HF_HUB_CACHE" "$XDG_CACHE_HOME" "$OPENPI_DATA_HOME" "$XLA_PYTHON_CLIENT_MEM_FRACTION"
import json
import sys

(
    out_path,
    run_id,
    test_openpi_root,
    openpi_dir,
    config_name,
    exp_name,
    max_frames,
    num_train_steps,
    save_interval,
    log_interval,
    checkpoint_base_dir,
    artifacts_root,
    refresh_norm_stats,
    hf_home,
    hf_hub_cache,
    xdg_cache_home,
    openpi_data_home,
    xla_mem_fraction,
) = sys.argv[1:]

payload = {
    "schema_version": 1,
    "run_id": run_id,
    "test_openpi_root": test_openpi_root,
    "openpi_dir": openpi_dir,
    "config_name": config_name,
    "exp_name": exp_name,
    "max_frames": int(max_frames),
    "num_train_steps": int(num_train_steps),
    "save_interval": int(save_interval),
    "log_interval": int(log_interval),
    "checkpoint_base_dir": checkpoint_base_dir,
    "artifacts_root": artifacts_root,
    "refresh_norm_stats": refresh_norm_stats == "1",
    "env": {
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": hf_hub_cache,
        "XDG_CACHE_HOME": xdg_cache_home,
        "OPENPI_DATA_HOME": openpi_data_home,
        "XLA_PYTHON_CLIENT_MEM_FRACTION": xla_mem_fraction,
    },
}
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

print_section() {
  printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1"
}

run_and_capture() {
  local log_file="$1"
  shift
  (
    cd "$OPENPI_DIR"
    printf '$'
    printf ' %q' "$@"
    printf '\n'
    "$@"
  ) 2>&1 | tee "$log_file"
}

if (( DRY_RUN )); then
  print_section "Dry run only"
  printf 'run_dir=%s\n' "$RUN_DIR"
  printf 'norm_stats_file=%s\n' "$NORM_STATS_FILE"
  printf 'checkpoint_run_dir=%s\n' "$CHECKPOINT_RUN_DIR"
  exit 0
fi

if (( REFRESH_NORM_STATS )) || [[ ! -f "$NORM_STATS_FILE" ]]; then
  print_section "Computing bounded norm stats"
  run_and_capture "$COMPUTE_LOG" uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME" --max-frames "$MAX_FRAMES"
else
  print_section "Reusing cached norm stats"
  printf 'Using existing norm stats: %s\n' "$NORM_STATS_FILE" | tee "$COMPUTE_LOG"
fi

print_section "Launching bounded training"
run_and_capture "$TRAIN_LOG" uv run scripts/train.py "$CONFIG_NAME" \
  --exp-name "$EXP_NAME" \
  --overwrite \
  --num-train-steps "$NUM_TRAIN_STEPS" \
  --save-interval "$SAVE_INTERVAL" \
  --log-interval "$LOG_INTERVAL" \
  --no-wandb-enabled \
  --checkpoint-base-dir "$CHECKPOINT_BASE_DIR"

LATEST_CHECKPOINT_DIR="$(find "$CHECKPOINT_RUN_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"

python - <<'PY' "$RESULT_JSON" "$NORM_STATS_FILE" "$COMPUTE_LOG" "$TRAIN_LOG" "$CHECKPOINT_RUN_DIR" "$LATEST_CHECKPOINT_DIR"
import json
import os
import sys

result_path, norm_stats_file, compute_log, train_log, checkpoint_run_dir, latest_checkpoint_dir = sys.argv[1:]
payload = {
    "schema_version": 1,
    "norm_stats_file": norm_stats_file,
    "norm_stats_exists": os.path.exists(norm_stats_file),
    "compute_log": compute_log,
    "train_log": train_log,
    "checkpoint_run_dir": checkpoint_run_dir,
    "checkpoint_run_dir_exists": os.path.isdir(checkpoint_run_dir),
    "latest_checkpoint_dir": latest_checkpoint_dir or None,
}
with open(result_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

print_section "Completed"
printf 'Artifacts: %s\n' "$RUN_DIR"
printf 'Norm stats: %s\n' "$NORM_STATS_FILE"
printf 'Checkpoint run dir: %s\n' "$CHECKPOINT_RUN_DIR"
if [[ -n "$LATEST_CHECKPOINT_DIR" ]]; then
  printf 'Latest checkpoint: %s\n' "$LATEST_CHECKPOINT_DIR"
fi
