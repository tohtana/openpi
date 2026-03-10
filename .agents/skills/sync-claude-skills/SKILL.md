---
name: sync-claude-skills
description: Sync Codex skills from .claude/skills into the Codex skills directory (.agents/skills or .agent/skills). Use when Claude skills are updated first and Codex needs to be kept in sync.
user_invocable: true
arguments:
  - name: skill_name
    description: Optional single skill to sync (for example, issue-docs)
    required: false
  - name: dry_run
    description: Optional flag to preview changes before writing (true/false)
    required: false
---

# Sync Claude Skills To Codex

Use this skill when `.claude/skills` is the source of truth and Codex skills must mirror it.

## Workflow

1. Confirm source and destination
- Source: `.claude/skills`
- Destination: `.agents/skills` (fallback: `.agent/skills`)

2. Run preview first
- Full sync preview:
  ```bash
  .agents/skills/sync-claude-skills/scripts/sync_skills.sh --dry-run
  ```
- Single skill preview:
  ```bash
  .agents/skills/sync-claude-skills/scripts/sync_skills.sh --skill <skill_name> --dry-run
  ```

3. Apply sync
- Full sync:
  ```bash
  .agents/skills/sync-claude-skills/scripts/sync_skills.sh
  ```
- Single skill:
  ```bash
  .agents/skills/sync-claude-skills/scripts/sync_skills.sh --skill <skill_name>
  ```
- After copy, the script normalizes docs to Codex-native conventions:
  - `/skill-name` → `$skill-name` for known skills
  - `.claude/skills` → `.agents/skills`
  - `.claude/rules` → `.agents/references`
  - `Claude` → `Codex`

4. Verify
- Spot-check changed files and confirm expected parity.
- For full sync, verify no differences remain:
  ```bash
  diff -rq .claude/skills .agents/skills --exclude sync-claude-skills
  ```

## Notes
- Full sync is a mirror operation and removes stale skill files/directories from the Codex skills directory, except this `sync-claude-skills/` skill directory (self-protected).
- Single-skill sync only updates the selected skill directory.
