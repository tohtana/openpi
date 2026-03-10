#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/libero/track_a_common.sh
source "$SCRIPT_DIR/track_a_common.sh"

usage() {
  cat <<'USAGE'
Usage: discover_track_a_cli.sh --artifacts-dir <path> [--python-bin <path>] [--dry-run]

Options:
  --artifacts-dir <path>   Output directory for discovery artifacts (required)
  --python-bin <path>      Python executable to use (default: python)
  --dry-run                Print planned actions without executing
  --help                   Show this help
USAGE
}

ARTIFACTS_DIR=""
PYTHON_BIN="python"
DRY_RUN=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --artifacts-dir)
      ARTIFACTS_DIR="${2:-}"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="${2:-}"
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

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "Dry-run: would create $ARTIFACTS_DIR/lerobot_eval_help.txt and cli_capabilities.json"
  exit 0
fi

ensure_dir "$ARTIFACTS_DIR"
HELP_PATH="$ARTIFACTS_DIR/lerobot_eval_help.txt"
JSON_PATH="$ARTIFACTS_DIR/cli_capabilities.json"

require_cmd "$PYTHON_BIN"
if ! command -v lerobot-eval >/dev/null 2>&1; then
  fail "lerobot-eval not found in PATH" 3
fi

if ! lerobot-eval --help >"$HELP_PATH" 2>&1; then
  fail "Failed to run lerobot-eval --help" 3
fi

VERSION_TEXT=""
if lerobot-eval --version >/tmp/track_a_lerobot_eval_version.txt 2>&1; then
  VERSION_TEXT="$(cat /tmp/track_a_lerobot_eval_version.txt)"
else
  VERSION_TEXT="unknown"
fi

"$PYTHON_BIN" - <<'PY' "$HELP_PATH" "$JSON_PATH" "$VERSION_TEXT"
import json
import re
import sys
from pathlib import Path
from datetime import datetime, timezone

help_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
version_text = sys.argv[3]

help_text = help_path.read_text(encoding="utf-8", errors="replace")
flags = sorted(set(re.findall(r"--[a-zA-Z0-9][a-zA-Z0-9_.-]*", help_text)))

payload = {
    "schema_version": "1",
    "lerobot_eval_help_text_path": str(help_path),
    "supported_flags": flags,
    "version_text": version_text.strip(),
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

log "Wrote discovery artifacts: $JSON_PATH"
