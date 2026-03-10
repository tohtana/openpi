#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/libero/track_a_common.sh
source "$SCRIPT_DIR/track_a_common.sh"

usage() {
  cat <<'USAGE'
Usage: run_track_a_eval.sh --run-id <id> --policy-path <path> --task-suite <suite> --batch-size <int> --n-episodes <int> --gpu-id <int> --mujoco-gl <egl|glx> --artifacts-root <path> [options]

Options:
  --run-id <id>            Run identifier (required)
  --policy-path <path>     Policy path/ID (required)
  --task-suite <suite>     LIBERO task suite (required)
  --batch-size <int>       Batch size (required)
  --n-episodes <int>       Number of episodes (required)
  --gpu-id <int>           Physical GPU id to pin (required)
  --seed <int>             Seed (default: 7)
  --mujoco-gl <egl|glx>    Mujoco backend (required)
  --artifacts-root <path>  Root directory for per-run artifacts (required)
  --timeout-secs <int>     Timeout in seconds (default: 10800)
  --retry-index <int>      Retry index (default: 0)
  --allow-overwrite        Allow overwriting an existing run directory
  --dry-run                Write manifest only; do not run eval
  --help                   Show this help
USAGE
}

RUN_ID=""
POLICY_PATH=""
TASK_SUITE=""
BATCH_SIZE=""
N_EPISODES=""
GPU_ID=""
SEED="7"
MUJOCO_GL=""
ARTIFACTS_ROOT=""
TIMEOUT_SECS="10800"
RETRY_INDEX="0"
ALLOW_OVERWRITE=0
DRY_RUN=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --policy-path)
      POLICY_PATH="${2:-}"
      shift 2
      ;;
    --task-suite)
      TASK_SUITE="${2:-}"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="${2:-}"
      shift 2
      ;;
    --n-episodes)
      N_EPISODES="${2:-}"
      shift 2
      ;;
    --gpu-id)
      GPU_ID="${2:-}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --mujoco-gl)
      MUJOCO_GL="${2:-}"
      shift 2
      ;;
    --artifacts-root)
      ARTIFACTS_ROOT="${2:-}"
      shift 2
      ;;
    --timeout-secs)
      TIMEOUT_SECS="${2:-}"
      shift 2
      ;;
    --retry-index)
      RETRY_INDEX="${2:-}"
      shift 2
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
      fail "Unknown argument: $1" 2
      ;;
  esac
done

[[ -n "$RUN_ID" && -n "$POLICY_PATH" && -n "$TASK_SUITE" && -n "$BATCH_SIZE" && -n "$N_EPISODES" && -n "$GPU_ID" && -n "$MUJOCO_GL" && -n "$ARTIFACTS_ROOT" ]] || {
  usage >&2
  fail "Missing required argument(s)" 2
}

[[ "$MUJOCO_GL" == "egl" || "$MUJOCO_GL" == "glx" ]] || fail "--mujoco-gl must be egl or glx" 2

RUN_DIR="$ARTIFACTS_ROOT/$RUN_ID"
STDOUT_LOG="$RUN_DIR/stdout.log"
STDERR_LOG="$RUN_DIR/stderr.log"
MANIFEST_JSON="$RUN_DIR/run_manifest.json"
RESULT_JSON="$RUN_DIR/result.json"

ARTIFACTS_ROOT_PARENT="$(cd "$ARTIFACTS_ROOT/.." && pwd)"
STATUS_DIR="${TRACK_A_STATUS_DIR:-$ARTIFACTS_ROOT_PARENT/state/rows}"
STATE_JSON="$STATUS_DIR/$RUN_ID.json"

if [[ -d "$RUN_DIR" && "$ALLOW_OVERWRITE" -ne 1 ]]; then
  fail "Run directory already exists: $RUN_DIR (use --allow-overwrite to override)" 2
fi

ensure_dir "$RUN_DIR"
ensure_dir "$STATUS_DIR"

START_TS="$(timestamp_utc)"
CWD_NOW="$(pwd)"
GIT_COMMIT="$(resolve_git_commit)"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
export LIBERO_CONFIG_PATH

CMD=(
  "lerobot-eval"
  "--policy.path=$POLICY_PATH"
  "--env.type=libero"
  "--env.task=$TASK_SUITE"
  "--eval.batch_size=$BATCH_SIZE"
  "--eval.n_episodes=$N_EPISODES"
  "--seed=$SEED"
)

python - <<'PY' "$MANIFEST_JSON" "$RUN_ID" "$START_TS" "$CWD_NOW" "$POLICY_PATH" "$TASK_SUITE" "$BATCH_SIZE" "$N_EPISODES" "$GPU_ID" "$SEED" "$MUJOCO_GL" "$RETRY_INDEX" "$GIT_COMMIT" "$STDOUT_LOG" "$STDERR_LOG" "$TIMEOUT_SECS" "${CMD[*]}"
import json
import os
import sys

(
    out,
    run_id,
    start_ts,
    cwd,
    policy_path,
    task_suite,
    batch_size,
    n_episodes,
    gpu_id,
    seed,
    mujoco_gl,
    retry_index,
    git_commit,
    stdout_log,
    stderr_log,
    timeout_secs,
    command,
) = sys.argv[1:]

payload = {
    "schema_version": "1",
    "run_id": run_id,
    "command": command,
    "cwd": cwd,
    "env": {
        "CUDA_VISIBLE_DEVICES": gpu_id,
        "MUJOCO_GL": mujoco_gl,
        "MUJOCO_EGL_DEVICE_ID": gpu_id,
        "HF_HOME": os.environ.get("HF_HOME", ""),
        "LIBERO_CONFIG_PATH": os.environ.get("LIBERO_CONFIG_PATH", ""),
        "PYTHONUNBUFFERED": os.environ.get("PYTHONUNBUFFERED", ""),
        "TZ": os.environ.get("TZ", ""),
    },
    "start_time_utc": start_ts,
    "policy_path": policy_path,
    "task_suite": task_suite,
    "batch_size": int(batch_size),
    "n_episodes": int(n_episodes),
    "gpu_id": int(gpu_id),
    "seed": int(seed),
    "mujoco_gl": mujoco_gl,
    "retry_index": int(retry_index),
    "timeout_secs": int(timeout_secs),
    "stdout_log": stdout_log,
    "stderr_log": stderr_log,
    "git_commit": git_commit,
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
PY

python - <<'PY' "$STATE_JSON" "$RUN_ID" "$GPU_ID" "$RETRY_INDEX" "$START_TS" "$RESULT_JSON"
import json
import sys

out, run_id, gpu_id, retry_index, started_at, result_path = sys.argv[1:]
payload = {
    "schema_version": "1",
    "run_id": run_id,
    "status": "running",
    "retry_index": int(retry_index),
    "gpu_id": int(gpu_id),
    "started_at_utc": started_at,
    "ended_at_utc": None,
    "result_path": result_path,
    "notes": "",
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
PY

if [[ "$DRY_RUN" -eq 1 ]]; then
  END_TS="$(timestamp_utc)"
  python - <<'PY' "$RESULT_JSON" "$RUN_ID" "$END_TS" "$STDOUT_LOG" "$STDERR_LOG"
import json
import sys

out, run_id, end_ts, stdout_log, stderr_log = sys.argv[1:]
payload = {
    "schema_version": "1",
    "run_id": run_id,
    "exit_code": 0,
    "status": "dry_run",
    "end_time_utc": end_ts,
    "duration_sec": 0.0,
    "successes": None,
    "episodes": None,
    "success_rate": None,
    "video_path": None,
    "stdout_log": stdout_log,
    "stderr_log": stderr_log,
    "error_message": "",
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
PY
  python - <<'PY' "$STATE_JSON" "$END_TS"
import json
import sys

path, end_ts = sys.argv[1:]
with open(path, "r", encoding="utf-8") as f:
    d = json.load(f)
d["status"] = "pass"
d["ended_at_utc"] = end_ts
d["notes"] = "dry-run"
with open(path, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=2, sort_keys=True)
    f.write("\n")
PY
  log "Dry-run complete for $RUN_ID"
  exit 0
fi

require_cmd lerobot-eval

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MUJOCO_GL
export MUJOCO_EGL_DEVICE_ID="$GPU_ID"
export PYTHONUNBUFFERED=1

if [[ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]]; then
  ensure_dir "$LIBERO_CONFIG_PATH"
  LIBERO_BENCHMARK_ROOT="$(python - <<'PY'
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
  [[ -n "$LIBERO_BENCHMARK_ROOT" ]] || fail "Could not locate installed LIBERO benchmark root" 3
  cat >"$LIBERO_CONFIG_PATH/config.yaml" <<EOF
benchmark_root: $LIBERO_BENCHMARK_ROOT
bddl_files: $LIBERO_BENCHMARK_ROOT/bddl_files
init_states: $LIBERO_BENCHMARK_ROOT/init_files
datasets: $LIBERO_BENCHMARK_ROOT/../datasets
assets: $LIBERO_BENCHMARK_ROOT/assets
EOF
fi

START_EPOCH="$(date +%s)"
EXIT_CODE=0
USE_XVFB=0

if [[ "$MUJOCO_GL" == "glx" && -z "${DISPLAY:-}" ]]; then
  require_cmd xvfb-run
  USE_XVFB=1
fi

if command -v timeout >/dev/null 2>&1; then
  if [[ "$USE_XVFB" -eq 1 ]]; then
    timeout --preserve-status "$TIMEOUT_SECS" xvfb-run -a -s "-screen 0 1280x1024x24" "${CMD[@]}" >"$STDOUT_LOG" 2>"$STDERR_LOG" || EXIT_CODE=$?
  else
    timeout --preserve-status "$TIMEOUT_SECS" "${CMD[@]}" >"$STDOUT_LOG" 2>"$STDERR_LOG" || EXIT_CODE=$?
  fi
else
  if [[ "$USE_XVFB" -eq 1 ]]; then
    xvfb-run -a -s "-screen 0 1280x1024x24" "${CMD[@]}" >"$STDOUT_LOG" 2>"$STDERR_LOG" || EXIT_CODE=$?
  else
    "${CMD[@]}" >"$STDOUT_LOG" 2>"$STDERR_LOG" || EXIT_CODE=$?
  fi
fi

END_EPOCH="$(date +%s)"
END_TS="$(timestamp_utc)"
DURATION_SEC="$((END_EPOCH - START_EPOCH))"

python - <<'PY' "$STDOUT_LOG" "$N_EPISODES" "$RESULT_JSON" "$RUN_ID" "$EXIT_CODE" "$END_TS" "$DURATION_SEC" "$STDOUT_LOG" "$STDERR_LOG"
import json
import ast
import re
import sys
from pathlib import Path

(
    stdout_path,
    n_episodes,
    result_out,
    run_id,
    exit_code,
    end_time,
    duration,
    stdout_log,
    stderr_log,
) = sys.argv[1:]

text = Path(stdout_path).read_text(encoding="utf-8", errors="replace") if Path(stdout_path).exists() else ""
successes = None
success_rate = None
episodes = None
video_path = None

m = re.search(r"# successes:\s*(\d+)\s*\(([-+]?\d+(?:\.\d+)?)%\)", text)
if m:
    successes = int(m.group(1))
    success_rate = float(m.group(2)) / 100.0

m_total_rate = re.search(r"Total success rate:\s*([-+]?\d+(?:\.\d+)?)", text)
if m_total_rate:
    val = float(m_total_rate.group(1))
    success_rate = val if val <= 1.0 else val / 100.0

m_eps = re.search(r"Total episodes:\s*(\d+)", text)
if m_eps:
    episodes = int(m_eps.group(1))
if episodes is None:
    episodes = int(n_episodes)

m_overall = re.search(r"Overall Aggregated Metrics:\s*\n(\{.*?\})", text, re.S)
if m_overall:
    try:
        overall = ast.literal_eval(m_overall.group(1))
        if success_rate is None and "pc_success" in overall:
            success_rate = float(overall["pc_success"]) / 100.0
        if episodes is None and "n_episodes" in overall:
            episodes = int(overall["n_episodes"])
        if video_path is None:
            video_paths = overall.get("video_paths")
            if isinstance(video_paths, list) and video_paths:
                video_path = str(video_paths[0])
    except Exception:
        pass

if successes is None and success_rate is not None and episodes:
    successes = int(round(success_rate * episodes))

m_video = re.search(r"(\S+\.mp4)", text)
if m_video:
    video_path = m_video.group(1)

exit_code_int = int(exit_code)
status = "pass" if exit_code_int == 0 else "fail"
result = {
    "schema_version": "1",
    "run_id": run_id,
    "exit_code": exit_code_int,
    "status": status,
    "end_time_utc": end_time,
    "duration_sec": float(duration),
    "successes": successes,
    "episodes": episodes,
    "success_rate": success_rate,
    "video_path": video_path,
    "stdout_log": stdout_log,
    "stderr_log": stderr_log,
    "error_message": "" if exit_code_int == 0 else f"lerobot-eval exited with code {exit_code_int}",
}
Path(result_out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

FINAL_STATUS="fail"
if [[ "$EXIT_CODE" -eq 0 ]]; then
  FINAL_STATUS="pass"
fi

python - <<'PY' "$STATE_JSON" "$END_TS" "$FINAL_STATUS"
import json
import sys

path, end_ts, status = sys.argv[1:]
with open(path, "r", encoding="utf-8") as f:
    d = json.load(f)
d["status"] = status
d["ended_at_utc"] = end_ts
if status == "fail":
    d["notes"] = "eval failed; inspect result.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=2, sort_keys=True)
    f.write("\n")
PY

exit "$EXIT_CODE"
