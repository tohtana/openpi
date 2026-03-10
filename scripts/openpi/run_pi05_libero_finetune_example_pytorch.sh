#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_pi05_libero_finetune_example_pytorch.sh [options]

Runs a bounded PyTorch OpenPI LIBERO fine-tuning example for the documented
`pi05_libero` config.

Options:
  --exp-name <name>               Training experiment name.
  --run-id <id>                   Artifact run id. Defaults to --exp-name.
  --batch-size <int>              Total batch size. Default: 4
  --num-train-steps <int>         Number of training steps to run. Default: 1
  --save-interval <int>           Checkpoint save interval. Default: 1
  --log-interval <int>            Train log interval. Default: 1
  --enable-profiling              Enable per-step timing breakdown in train_pytorch.py.
  --profiling-warmup-steps <int>  Skip this many initial steps before collecting timings.
                                  Default: 1
  --cuda-visible-devices <list>   CUDA_VISIBLE_DEVICES value. Default: 0
  --jax-checkpoint-dir <path>     JAX checkpoint root to convert.
                                  Default: gs://openpi-assets/checkpoints/pi05_base
  --converted-weight-dir <path>   Output directory for converted PyTorch weights.
                                  Default: /mnt/local_storage/experiments/openpi_pytorch/pi05_base_for_libero_bfloat16
  --checkpoint-base-dir <path>    Base directory for PyTorch training checkpoints.
                                  Default: /mnt/local_storage/experiments/openpi_pytorch_checkpoints
  --artifacts-root <path>         Root directory for logs and manifests.
                                  Default: test_openpi/tasks/openpi-scripted-libero-finetune-example-pytorch
  --reconvert-weights             Force reconversion even if model.safetensors already exists.
  --allow-overwrite               Reuse an existing run artifact directory.
  --dry-run                       Print the resolved plan and exit.
  --help                          Show this help.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_OPENPI_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OPENPI_DIR="$TEST_OPENPI_ROOT/openpi"
CONFIG_NAME="pi05_libero"
DEFAULT_EXP_NAME="pi05_libero_pytorch_$(date -u +%Y%m%dT%H%M%SZ)"
EXP_NAME="$DEFAULT_EXP_NAME"
RUN_ID=""
BATCH_SIZE="4"
NUM_TRAIN_STEPS="1"
SAVE_INTERVAL="1"
LOG_INTERVAL="1"
ENABLE_PROFILING=0
PROFILING_WARMUP_STEPS="1"
CUDA_VISIBLE_DEVICES_VALUE="0"
JAX_CHECKPOINT_DIR="gs://openpi-assets/checkpoints/pi05_base"
CONVERTED_WEIGHT_DIR="/mnt/local_storage/experiments/openpi_pytorch/pi05_base_for_libero_bfloat16"
CHECKPOINT_BASE_DIR="/mnt/local_storage/experiments/openpi_pytorch_checkpoints"
ARTIFACTS_ROOT="$TEST_OPENPI_ROOT/tasks/openpi-scripted-libero-finetune-example-pytorch"
RECONVERT_WEIGHTS=0
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
    --batch-size)
      BATCH_SIZE="${2:-}"
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
    --enable-profiling)
      ENABLE_PROFILING=1
      shift
      ;;
    --profiling-warmup-steps)
      PROFILING_WARMUP_STEPS="${2:-}"
      shift 2
      ;;
    --cuda-visible-devices)
      CUDA_VISIBLE_DEVICES_VALUE="${2:-}"
      shift 2
      ;;
    --jax-checkpoint-dir)
      JAX_CHECKPOINT_DIR="${2:-}"
      shift 2
      ;;
    --converted-weight-dir)
      CONVERTED_WEIGHT_DIR="${2:-}"
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
    --reconvert-weights)
      RECONVERT_WEIGHTS=1
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

if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ && "$NUM_TRAIN_STEPS" =~ ^[0-9]+$ && "$SAVE_INTERVAL" =~ ^[0-9]+$ && "$LOG_INTERVAL" =~ ^[0-9]+$ && "$PROFILING_WARMUP_STEPS" =~ ^[0-9]+$ ]]; then
  echo "Numeric arguments must be non-negative integers" >&2
  exit 2
fi

if (( BATCH_SIZE < 1 || NUM_TRAIN_STEPS < 1 )); then
  echo "--batch-size and --num-train-steps must be at least 1" >&2
  exit 2
fi

RUN_DIR="$ARTIFACTS_ROOT/$RUN_ID"
PATCH_LOG="$RUN_DIR/patch_transformers.log"
CONVERT_LOG="$RUN_DIR/convert_checkpoint.log"
TRAIN_LOG="$RUN_DIR/train_pytorch.log"
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
PYTORCH_CUDA_ALLOC_CONF_VALUE="${OPENPI_PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,expandable_segments:True}"

export HF_HOME="$HF_HOME_VALUE"
export HF_HUB_CACHE="$HF_HUB_CACHE_VALUE"
export XDG_CACHE_HOME="$XDG_CACHE_HOME_VALUE"
export OPENPI_DATA_HOME="$OPENPI_DATA_HOME_VALUE"
export PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF_VALUE"

python - <<'PY' "$MANIFEST_JSON" "$RUN_ID" "$TEST_OPENPI_ROOT" "$OPENPI_DIR" "$CONFIG_NAME" "$EXP_NAME" "$BATCH_SIZE" "$NUM_TRAIN_STEPS" "$SAVE_INTERVAL" "$LOG_INTERVAL" "$ENABLE_PROFILING" "$PROFILING_WARMUP_STEPS" "$CUDA_VISIBLE_DEVICES_VALUE" "$JAX_CHECKPOINT_DIR" "$CONVERTED_WEIGHT_DIR" "$CHECKPOINT_BASE_DIR" "$ARTIFACTS_ROOT" "$RECONVERT_WEIGHTS" "$HF_HOME" "$HF_HUB_CACHE" "$XDG_CACHE_HOME" "$OPENPI_DATA_HOME" "$PYTORCH_CUDA_ALLOC_CONF"
import json
import sys

(
    out_path,
    run_id,
    test_openpi_root,
    openpi_dir,
    config_name,
    exp_name,
    batch_size,
    num_train_steps,
    save_interval,
    log_interval,
    enable_profiling,
    profiling_warmup_steps,
    cuda_visible_devices,
    jax_checkpoint_dir,
    converted_weight_dir,
    checkpoint_base_dir,
    artifacts_root,
    reconvert_weights,
    hf_home,
    hf_hub_cache,
    xdg_cache_home,
    openpi_data_home,
    pytorch_cuda_alloc_conf,
) = sys.argv[1:]

payload = {
    "schema_version": 1,
    "run_id": run_id,
    "test_openpi_root": test_openpi_root,
    "openpi_dir": openpi_dir,
    "config_name": config_name,
    "exp_name": exp_name,
    "batch_size": int(batch_size),
    "num_train_steps": int(num_train_steps),
    "save_interval": int(save_interval),
    "log_interval": int(log_interval),
    "enable_profiling": enable_profiling == "1",
    "profiling_warmup_steps": int(profiling_warmup_steps),
    "cuda_visible_devices": cuda_visible_devices,
    "jax_checkpoint_dir": jax_checkpoint_dir,
    "converted_weight_dir": converted_weight_dir,
    "checkpoint_base_dir": checkpoint_base_dir,
    "artifacts_root": artifacts_root,
    "reconvert_weights": reconvert_weights == "1",
    "env": {
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": hf_hub_cache,
        "XDG_CACHE_HOME": xdg_cache_home,
        "OPENPI_DATA_HOME": openpi_data_home,
        "PYTORCH_CUDA_ALLOC_CONF": pytorch_cuda_alloc_conf,
    },
}
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

print_section() {
  printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1"
}

run_in_openpi() {
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

normalize_checkpoint_root() {
  local checkpoint_root="${1%/}"
  if [[ "$checkpoint_root" == */params ]]; then
    checkpoint_root="${checkpoint_root%/params}"
  fi
  printf '%s\n' "$checkpoint_root"
}

resolve_jax_checkpoint_dir() {
  local checkpoint_root
  checkpoint_root="$(normalize_checkpoint_root "$JAX_CHECKPOINT_DIR")"

  if [[ "$checkpoint_root" == *"://"* ]]; then
    printf 'Resolving JAX checkpoint root via OpenPI downloader: %s\n' "$checkpoint_root" | tee -a "$CONVERT_LOG"
    RESOLVED_JAX_CHECKPOINT_DIR="$(
      cd "$OPENPI_DIR"
      uv run python - <<'PY' "$checkpoint_root"
from openpi.shared import download
import sys

print(download.maybe_download(sys.argv[1]))
PY
    )"
    RESOLVED_JAX_CHECKPOINT_DIR="$(printf '%s\n' "$RESOLVED_JAX_CHECKPOINT_DIR" | tail -n 1)"
  elif [[ -d "$checkpoint_root" ]]; then
    RESOLVED_JAX_CHECKPOINT_DIR="$(realpath "$checkpoint_root")"
  else
    echo "JAX checkpoint root not found: $checkpoint_root" >&2
    exit 1
  fi

  printf 'Using JAX checkpoint root: %s\n' "$RESOLVED_JAX_CHECKPOINT_DIR" | tee -a "$CONVERT_LOG"

  if [[ ! -f "$RESOLVED_JAX_CHECKPOINT_DIR/params/_METADATA" ]]; then
    echo "Resolved JAX checkpoint is missing params/_METADATA: $RESOLVED_JAX_CHECKPOINT_DIR" >&2
    echo "Pass a checkpoint root that contains a params/ Orbax tree, or use the default gs://openpi-assets/checkpoints/pi05_base source." >&2
    exit 1
  fi
}

ensure_patched_transformers() {
  local site_pkg="$OPENPI_DIR/.venv/lib/python3.11/site-packages/transformers"
  local installed_file="$site_pkg/models/paligemma/modeling_paligemma.py"
  local replacement_root="$OPENPI_DIR/src/openpi/models_pytorch/transformers_replace"
  local replacement_file="$replacement_root/models/paligemma/modeling_paligemma.py"

  if [[ -f "$installed_file" ]] && cmp -s "$replacement_file" "$installed_file" && [[ "$(stat -c '%h' "$installed_file")" == "1" ]]; then
    printf 'Transformers patch already applied in local .venv\n' | tee "$PATCH_LOG"
    return
  fi

  run_in_openpi "$PATCH_LOG" uv sync --reinstall-package transformers --link-mode copy
  (
    cd "$OPENPI_DIR"
    printf '$ cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/\n' | tee -a "$PATCH_LOG"
    cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
  ) >>"$PATCH_LOG" 2>&1
}

ensure_converted_weights() {
  if [[ -f "$CONVERTED_WEIGHT_DIR/model.safetensors" && "$RECONVERT_WEIGHTS" -ne 1 ]]; then
    printf 'Using existing converted PyTorch checkpoint: %s\n' "$CONVERTED_WEIGHT_DIR" | tee "$CONVERT_LOG"
    return
  fi

  resolve_jax_checkpoint_dir
  mkdir -p "$CONVERTED_WEIGHT_DIR"
  run_in_openpi "$CONVERT_LOG" uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint-dir "$RESOLVED_JAX_CHECKPOINT_DIR" \
    --config-name "$CONFIG_NAME" \
    --output-path "$CONVERTED_WEIGHT_DIR" \
    --precision bfloat16
}

if (( DRY_RUN )); then
  print_section "Dry run only"
  printf 'run_dir=%s\n' "$RUN_DIR"
  printf 'converted_weight_dir=%s\n' "$CONVERTED_WEIGHT_DIR"
  printf 'checkpoint_run_dir=%s\n' "$CHECKPOINT_RUN_DIR"
  exit 0
fi

print_section "Preparing local PyTorch environment"
ensure_patched_transformers

print_section "Ensuring converted PyTorch weights"
ensure_converted_weights

print_section "Launching bounded PyTorch training"
TRAIN_ARGS=(
  "$CONFIG_NAME"
  --exp-name "$EXP_NAME"
  --pytorch-weight-path "$CONVERTED_WEIGHT_DIR"
  --batch-size "$BATCH_SIZE"
  --num-train-steps "$NUM_TRAIN_STEPS"
  --save-interval "$SAVE_INTERVAL"
  --log-interval "$LOG_INTERVAL"
  --checkpoint-base-dir "$CHECKPOINT_BASE_DIR"
  --no-wandb-enabled
  --overwrite
)

if (( ENABLE_PROFILING )); then
  TRAIN_ARGS+=(--enable-profiling --profiling-warmup-steps "$PROFILING_WARMUP_STEPS")
fi

(
  cd "$OPENPI_DIR"
  printf '$'
  printf ' %q' env \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
    HF_HOME="$HF_HOME" \
    HF_HUB_CACHE="$HF_HUB_CACHE" \
    XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    OPENPI_DATA_HOME="$OPENPI_DATA_HOME" \
    PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    uv run scripts/train_pytorch.py "${TRAIN_ARGS[@]}"
  printf '\n'
  env \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
    HF_HOME="$HF_HOME" \
    HF_HUB_CACHE="$HF_HUB_CACHE" \
    XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    OPENPI_DATA_HOME="$OPENPI_DATA_HOME" \
    PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    uv run scripts/train_pytorch.py "${TRAIN_ARGS[@]}"
) 2>&1 | tee "$TRAIN_LOG"

LATEST_CHECKPOINT_DIR="$(find "$CHECKPOINT_RUN_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"

python - <<'PY' "$RESULT_JSON" "$NORM_STATS_FILE" "$PATCH_LOG" "$CONVERT_LOG" "$TRAIN_LOG" "$CONVERTED_WEIGHT_DIR" "$CHECKPOINT_RUN_DIR" "$LATEST_CHECKPOINT_DIR"
import json
import os
import sys

(
    result_path,
    norm_stats_file,
    patch_log,
    convert_log,
    train_log,
    converted_weight_dir,
    checkpoint_run_dir,
    latest_checkpoint_dir,
) = sys.argv[1:]

payload = {
    "schema_version": 1,
    "norm_stats_file": norm_stats_file,
    "norm_stats_exists": os.path.exists(norm_stats_file),
    "patch_log": patch_log,
    "convert_log": convert_log,
    "train_log": train_log,
    "converted_weight_dir": converted_weight_dir,
    "converted_weight_exists": os.path.exists(os.path.join(converted_weight_dir, "model.safetensors")),
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
printf 'Converted weights: %s\n' "$CONVERTED_WEIGHT_DIR"
printf 'Checkpoint run dir: %s\n' "$CHECKPOINT_RUN_DIR"
if [[ -n "$LATEST_CHECKPOINT_DIR" ]]; then
  printf 'Latest checkpoint: %s\n' "$LATEST_CHECKPOINT_DIR"
fi
