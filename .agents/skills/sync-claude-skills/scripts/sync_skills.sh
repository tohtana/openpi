#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Sync Claude skills to Codex skills.

Usage:
  sync_skills.sh [--skill <name>] [--dry-run] [--source <path>] [--dest <path>]

Options:
  --skill <name>   Sync only one skill directory.
  --dry-run        Preview changes without writing.
  --source <path>  Source skills directory (default: .claude/skills).
  --dest <path>    Destination skills directory (default: .agents/skills, fallback: .agent/skills).
  -h, --help       Show help.
USAGE
}

resolve_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    printf '%s\n' "$p"
  else
    printf '%s/%s\n' "$REPO_ROOT" "$p"
  fi
}

normalize_codex_native_docs() {
  local skill_regex="$1"
  shift
  local target_dir

  for target_dir in "$@"; do
    if [[ ! -d "$target_dir" ]]; then
      continue
    fi

    while IFS= read -r -d '' md_file; do
      # Convert Claude-specific path and naming references to Codex-native forms.
      perl -i -pe '
        s{\.claude/skills}{.agents/skills}g;
        s{\.claude/rules}{.agents/references}g;
        s{\bClaude\b}{Codex}g;
        s{\bpush to update the PR\b}{ask the user to push to update the PR}g;
        s{Enables `git push` to work directly after committing}{User handles pushes; do not run `git push` in this workflow}g;
        s{\bgit push\b}{git push (user-only)}g;
      ' "$md_file"

      if [[ -n "$skill_regex" ]]; then
        # Convert slash skill invocations (/issue-fix) to Codex style ($issue-fix).
        SYNC_SKILL_RE="$skill_regex" perl -i -pe '
          my $re = $ENV{SYNC_SKILL_RE};
          if (defined $re && length $re) {
            s{(?<![\w\$])/($re)\b}{\$$1}g;
          }
        ' "$md_file"
      fi
    done < <(find "$target_dir" -type f -name '*.md' -print0)
  done
}

if ! command -v rsync >/dev/null 2>&1; then
  echo "Error: rsync is required but not installed." >&2
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
SELF_SKILL_DIR_NAME="$(basename "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")"

SKILL_NAME=""
DRY_RUN=0
SOURCE_PATH=".claude/skills"
DEST_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skill)
      if [[ $# -lt 2 ]]; then
        echo "Error: --skill requires a value." >&2
        usage
        exit 1
      fi
      SKILL_NAME="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --source)
      if [[ $# -lt 2 ]]; then
        echo "Error: --source requires a value." >&2
        usage
        exit 1
      fi
      SOURCE_PATH="$2"
      shift 2
      ;;
    --dest)
      if [[ $# -lt 2 ]]; then
        echo "Error: --dest requires a value." >&2
        usage
        exit 1
      fi
      DEST_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option '$1'." >&2
      usage
      exit 1
      ;;
  esac
done

SOURCE_DIR="$(resolve_path "$SOURCE_PATH")"

if [[ -z "$DEST_PATH" ]]; then
  if [[ -d "$REPO_ROOT/.agents/skills" ]]; then
    DEST_DIR="$REPO_ROOT/.agents/skills"
  elif [[ -d "$REPO_ROOT/.agent/skills" ]]; then
    DEST_DIR="$REPO_ROOT/.agent/skills"
  else
    DEST_DIR="$REPO_ROOT/.agents/skills"
  fi
else
  DEST_DIR="$(resolve_path "$DEST_PATH")"
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Error: source skills directory not found: $SOURCE_DIR" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"

RSYNC_OPTS=(-a --delete --checksum --itemize-changes)
if [[ "$DRY_RUN" -eq 1 ]]; then
  RSYNC_OPTS+=(--dry-run)
fi

if [[ -n "$SKILL_NAME" ]]; then
  SRC_SKILL="$SOURCE_DIR/$SKILL_NAME"
  DST_SKILL="$DEST_DIR/$SKILL_NAME"

  if [[ ! -d "$SRC_SKILL" ]]; then
    echo "Error: source skill not found: $SRC_SKILL" >&2
    exit 1
  fi

  mkdir -p "$DST_SKILL"
  echo "Sync mode: single skill"
  echo "Source: $SRC_SKILL/"
  echo "Dest:   $DST_SKILL/"
  rsync "${RSYNC_OPTS[@]}" "$SRC_SKILL/" "$DST_SKILL/"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    SKILL_REGEX="$(find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | paste -sd'|' -)"
    normalize_codex_native_docs "$SKILL_REGEX" "$DST_SKILL"
  fi
else
  # Preserve this sync skill directory so full mirror does not delete itself.
  RSYNC_OPTS+=("--filter=P ${SELF_SKILL_DIR_NAME}/")
  echo "Sync mode: full mirror"
  echo "Source: $SOURCE_DIR/"
  echo "Dest:   $DEST_DIR/"
  rsync "${RSYNC_OPTS[@]}" "$SOURCE_DIR/" "$DEST_DIR/"

  if [[ "$DRY_RUN" -eq 0 ]]; then
    SKILL_REGEX="$(find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | paste -sd'|' -)"
    if [[ -n "$SKILL_REGEX" ]]; then
      mapfile -t SOURCE_SKILLS < <(find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
      TARGET_DIRS=()
      for source_skill in "${SOURCE_SKILLS[@]}"; do
        TARGET_DIRS+=("$DEST_DIR/$source_skill")
      done
      normalize_codex_native_docs "$SKILL_REGEX" "${TARGET_DIRS[@]}"
    fi
  fi
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run complete. No files were modified."
else
  echo "Sync complete."
fi
