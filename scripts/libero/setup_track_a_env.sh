#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/libero/track_a_common.sh
source "$SCRIPT_DIR/track_a_common.sh"

usage() {
  cat <<'USAGE'
Usage: setup_track_a_env.sh --artifacts-dir <path> [options]

Options:
  --env-name <name>        Environment name (default: vla_pi0)
  --python-version <ver>   Requested Python version (default: 3.10)
  --lerobot-dir <path>     LeRobot checkout path (default: /mnt/local_storage/src/lerobot)
  --hf-home <path>         HuggingFace cache directory (default: /mnt/local_storage/huggingface)
  --artifacts-dir <path>   Output directory for setup artifacts (required)
  --dry-run                Print planned actions without executing
  --help                   Show this help
USAGE
}

ENV_NAME="vla_pi0"
PYTHON_VERSION="3.10"
LEROBOT_DIR="/mnt/local_storage/src/lerobot"
HF_HOME="/mnt/local_storage/huggingface"
ARTIFACTS_DIR=""
DRY_RUN=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="${2:-}"
      shift 2
      ;;
    --python-version)
      PYTHON_VERSION="${2:-}"
      shift 2
      ;;
    --lerobot-dir)
      LEROBOT_DIR="${2:-}"
      shift 2
      ;;
    --hf-home)
      HF_HOME="${2:-}"
      shift 2
      ;;
    --artifacts-dir)
      ARTIFACTS_DIR="${2:-}"
      shift 2
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

[[ -n "$ARTIFACTS_DIR" ]] || { usage >&2; fail "--artifacts-dir is required" 2; }

ensure_dir "$ARTIFACTS_DIR"
ensure_dir "$(dirname "$LEROBOT_DIR")"
ensure_dir "$HF_HOME"

SETUP_LOG="$ARTIFACTS_DIR/setup.log"
PIP_FREEZE="$ARTIFACTS_DIR/pip_freeze.txt"
ENV_SNAPSHOT="$ARTIFACTS_DIR/env_snapshot.json"

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "Dry-run: would setup env '$ENV_NAME' (python $PYTHON_VERSION), install LeRobot in '$LEROBOT_DIR', and write artifacts under '$ARTIFACTS_DIR'"
  exit 0
fi

require_cmd git
if command -v conda >/dev/null 2>&1; then
  TOOLCHAIN="conda"
else
  TOOLCHAIN="venv"
  require_cmd python3
fi

if [[ ! -d "$LEROBOT_DIR/.git" ]]; then
  log "Cloning LeRobot into $LEROBOT_DIR"
  git clone https://github.com/huggingface/lerobot.git "$LEROBOT_DIR" >>"$SETUP_LOG" 2>&1 || fail "Failed to clone LeRobot" 4
else
  log "Using existing LeRobot repo at $LEROBOT_DIR"
fi

export HF_HOME
export PYTHONUNBUFFERED=1
export TZ=UTC
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
export LIBERO_CONFIG_PATH
PI0_TRANSFORMERS_SPEC="transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"

if [[ "$TOOLCHAIN" == "conda" ]]; then
  log "Using conda environment: $ENV_NAME"
  if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" >>"$SETUP_LOG" 2>&1 || fail "Failed to create conda env" 4
  fi

  # Remove any pip-provided cmake shim to avoid version/path conflicts.
  conda run -n "$ENV_NAME" python -m pip uninstall -y cmake >>"$SETUP_LOG" 2>&1 || true
  # egl_probe currently breaks with CMake>=4; force a conda-managed CMake<4.
  conda install -y -n "$ENV_NAME" --force-reinstall "cmake<4" >>"$SETUP_LOG" 2>&1 || fail "Failed to install pinned cmake<4" 4

  conda run -n "$ENV_NAME" python -m pip install --upgrade pip wheel >>"$SETUP_LOG" 2>&1 || fail "Failed to upgrade pip" 4
  conda run -n "$ENV_NAME" python -m pip install -e "$LEROBOT_DIR[libero]" >>"$SETUP_LOG" 2>&1 || fail "Failed to install LeRobot with [libero] extras" 4
  conda run -n "$ENV_NAME" python -m pip install --upgrade "$PI0_TRANSFORMERS_SPEC" >>"$SETUP_LOG" 2>&1 || fail "Failed to install PI0-compatible transformers branch" 4
  # Re-apply the pin in case dependency resolution upgraded cmake during install.
  conda run -n "$ENV_NAME" python -m pip uninstall -y cmake >>"$SETUP_LOG" 2>&1 || true
  conda install -y -n "$ENV_NAME" --force-reinstall "cmake<4" >>"$SETUP_LOG" 2>&1 || fail "Failed to re-pin cmake<4 after LeRobot install" 4

  CMAKE_VERSION_LINE="$(conda run -n "$ENV_NAME" cmake --version 2>>"$SETUP_LOG" | head -n1 || true)"
  [[ -n "$CMAKE_VERSION_LINE" ]] || fail "cmake is not available in env after installation" 4
  CMAKE_VERSION="$(awk '{print $3}' <<<"$CMAKE_VERSION_LINE")"
  CMAKE_MAJOR="${CMAKE_VERSION%%.*}"
  [[ "$CMAKE_MAJOR" =~ ^[0-9]+$ ]] || fail "Unable to parse cmake version: $CMAKE_VERSION_LINE" 4
  if (( CMAKE_MAJOR >= 4 )); then
    fail "Unsupported cmake version in env ($CMAKE_VERSION). Expected <4 for egl_probe builds." 4
  fi

  conda run -n "$ENV_NAME" python -m pip freeze >"$PIP_FREEZE" 2>>"$SETUP_LOG" || fail "Failed to capture pip freeze" 4
  PY_BIN="$(conda run -n "$ENV_NAME" which python | tr -d '\r')"
else
  VENV_ROOT="${VLA_ENV_ROOT:-$HOME/.venvs}"
  VENV_PATH="$VENV_ROOT/$ENV_NAME"
  ensure_dir "$VENV_ROOT"
  log "Using venv fallback: $VENV_PATH"
  if [[ ! -x "$VENV_PATH/bin/python" ]]; then
    python3 -m venv "$VENV_PATH" >>"$SETUP_LOG" 2>&1 || fail "Failed to create venv" 4
  fi
  "$VENV_PATH/bin/python" -m pip install --upgrade pip wheel >>"$SETUP_LOG" 2>&1 || fail "Failed to upgrade pip" 4
  "$VENV_PATH/bin/python" -m pip install "cmake<4" >>"$SETUP_LOG" 2>&1 || fail "Failed to install pinned cmake<4" 4
  "$VENV_PATH/bin/python" -m pip install -e "$LEROBOT_DIR[libero]" >>"$SETUP_LOG" 2>&1 || fail "Failed to install LeRobot with [libero] extras" 4
  "$VENV_PATH/bin/python" -m pip install --upgrade "$PI0_TRANSFORMERS_SPEC" >>"$SETUP_LOG" 2>&1 || fail "Failed to install PI0-compatible transformers branch" 4
  "$VENV_PATH/bin/python" -m pip freeze >"$PIP_FREEZE" 2>>"$SETUP_LOG" || fail "Failed to capture pip freeze" 4
  PY_BIN="$VENV_PATH/bin/python"
fi

# Avoid LIBERO interactive first-run prompt by writing config.yaml up front.
ensure_dir "$LIBERO_CONFIG_PATH"
LIBERO_BENCHMARK_ROOT="$("$PY_BIN" - <<'PY'
import glob
import os
import site

candidates = []
for p in site.getsitepackages():
    candidates.extend(glob.glob(os.path.join(p, "libero", "libero")))
user_site = site.getusersitepackages()
candidates.extend(glob.glob(os.path.join(user_site, "libero", "libero")))
candidates = [p for p in candidates if os.path.isdir(p)]
print(candidates[0] if candidates else "")
PY
)"
[[ -n "$LIBERO_BENCHMARK_ROOT" ]] || fail "Could not locate installed LIBERO benchmark root" 4

cat >"$LIBERO_CONFIG_PATH/config.yaml" <<EOF
benchmark_root: $LIBERO_BENCHMARK_ROOT
bddl_files: $LIBERO_BENCHMARK_ROOT/bddl_files
init_states: $LIBERO_BENCHMARK_ROOT/init_files
datasets: $LIBERO_BENCHMARK_ROOT/../datasets
assets: $LIBERO_BENCHMARK_ROOT/assets
EOF

python - <<'PY' "$ENV_SNAPSHOT" "$ENV_NAME" "$PYTHON_VERSION" "$LEROBOT_DIR" "$HF_HOME" "$TOOLCHAIN" "$PY_BIN"
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
env_name = sys.argv[2]
python_version = sys.argv[3]
lerobot_dir = sys.argv[4]
hf_home = sys.argv[5]
toolchain = sys.argv[6]
python_bin = sys.argv[7]

def cmd_output(*cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception:
        return ""

payload = {
    "schema_version": "1",
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "env_name": env_name,
    "requested_python_version": python_version,
    "toolchain": toolchain,
    "python_bin": python_bin,
    "python_version_actual": cmd_output(python_bin, "--version"),
    "lerobot_dir": lerobot_dir,
    "lerobot_commit": cmd_output("git", "-C", lerobot_dir, "rev-parse", "HEAD"),
    "hf_home": hf_home,
    "libero_config_path": os.environ.get("LIBERO_CONFIG_PATH", ""),
    "display_env": os.environ.get("DISPLAY", ""),
    "mujoco_gl": os.environ.get("MUJOCO_GL", ""),
    "tz": os.environ.get("TZ", ""),
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

log "Setup artifacts written to $ARTIFACTS_DIR"
