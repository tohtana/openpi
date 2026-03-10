#!/usr/bin/env bash

set -euo pipefail

TRACK_A_SCHEMA_VERSION="1"

timestamp_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf '[%s] [INFO] %s\n' "$(timestamp_utc)" "$*" >&2
}

warn() {
  printf '[%s] [WARN] %s\n' "$(timestamp_utc)" "$*" >&2
}

fail() {
  local message="${1:-unspecified error}"
  local code="${2:-1}"
  printf '[%s] [ERROR] %s\n' "$(timestamp_utc)" "$message" >&2
  exit "$code"
}

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || fail "Required command not found: $cmd" 3
}

ensure_dir() {
  local dir="$1"
  mkdir -p "$dir"
}

write_json() {
  local path="$1"
  ensure_dir "$(dirname "$path")"
  local tmp
  tmp="${path}.tmp"
  cat >"$tmp"
  mv "$tmp" "$path"
}

run_logged() {
  if [[ "$#" -lt 2 ]]; then
    fail "run_logged requires at least 2 args: <logfile> <cmd...>" 2
  fi
  local logfile="$1"
  shift
  ensure_dir "$(dirname "$logfile")"
  log "Running: $*"
  "$@" 2>&1 | tee -a "$logfile"
}

resolve_git_commit() {
  if git rev-parse --verify HEAD >/dev/null 2>&1; then
    git rev-parse HEAD
  else
    echo "unknown"
  fi
}

contains_csv_token() {
  local haystack="$1"
  local needle="$2"
  local item
  IFS=',' read -r -a _items <<<"$haystack"
  for item in "${_items[@]}"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}
