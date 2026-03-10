---
name: todo-impl
description: Implement a planned GitHub-tracked TODO, run verification, sync all artifacts to GitHub, and move to `todo:final-review`. GitHub mode only.
user_invocable: true
---

# TODO Impl Workflow

## When to use
- User asks to execute a TODO that is planned.
- User asks to implement work tracked by `/todo-doc` + `/todo-plan`.

## Non-negotiable rules
- GitHub-only workflow; no filesystem fallback stages.
- Do not finish this skill until **all artifacts** are mirrored to GitHub.
- `todo-impl` ends at `todo:final-review`; do not mark completed here.

## 1. Preflight (must run first)
```bash
gh --version >/dev/null
gh auth status >/dev/null
GH_REPO="$(gh repo view --json nameWithOwner --jq .nameWithOwner)"
```
If preflight fails, stop and ask user to configure GitHub CLI, then rerun.

## 2. Load metadata and verify prerequisites
Read `todo/docs/${SLUG}/github.json`:
```bash
GH_REPO="$(jq -r .repo todo/docs/${SLUG}/github.json)"
ISSUE_NUMBER="$(jq -r .issue_number todo/docs/${SLUG}/github.json)"
TODO_TYPE="$(jq -r .type todo/docs/${SLUG}/github.json)"
```

Required local docs:
- `todo/docs/${SLUG}/${SLUG}-design.md`
- `todo/docs/${SLUG}/${SLUG}-impl-plan.md`

Required stage labels:
- exactly one `todo:*`
- exactly one `type:*`
- stage is `todo:plan-ready` or `todo:in-progress` (resume)

If stage is `todo:final-review`, skip to artifact sync validation/reporting.

## 3. Move to `todo:in-progress`
If currently `todo:plan-ready`, swap:
```bash
gh issue edit "$ISSUE_NUMBER" --repo "$GH_REPO" \
  --remove-label "todo:plan-ready" --add-label "todo:in-progress"
```
Re-verify label invariant.

## 4. Implement according to plan
- Read and follow `todo/docs/${SLUG}/${SLUG}-impl-plan.md`.
- Keep implementation scoped to the plan.
- If scope changes, document rationale in `results.md`.

## 5. Run verification
- Run the tests/checks defined in the plan.
- Save a concise execution record in `todo/docs/${SLUG}/verification.md`.
- If scripts were used for reproduction/verification, save them under `todo/docs/${SLUG}/` (for example `verify_*.py`, `verify_*.sh`).

## 6. Write implementation results
Create/update:
- `todo/docs/${SLUG}/results.md`

Minimum content:
- summary of code changes
- list of executed verification commands
- pass/fail outcomes
- open risks or `None`

## 7. Sync **all artifacts** to GitHub issue (required)
By the end of `/todo-impl`, ensure the issue includes all planning/implementation artifacts, including:
- design doc
- implementation plan
- review comments summaries
- verification scripts
- verification execution notes
- results summary

Required sync actions:
1. Upsert `<!-- managed:design-doc -->` with current design doc.
2. Upsert `<!-- managed:impl-plan -->` with current impl plan.
3. Upsert `<!-- managed:impl-artifacts -->` containing:
   - artifact manifest (path + sha256)
   - inline content of text artifacts under `todo/docs/${SLUG}/` (`.md`, `.txt`, `.json`, `.py`, `.sh`)

If content exceeds comment size, split into ordered managed comments:
- `<!-- managed:impl-artifacts:1 -->`
- `<!-- managed:impl-artifacts:2 -->`
- ...

Persist all comment IDs to `github.json.comment_ids`.

## 8. Move to `todo:final-review`
If current stage is `todo:in-progress`, swap:
```bash
gh issue edit "$ISSUE_NUMBER" --repo "$GH_REPO" \
  --remove-label "todo:in-progress" --add-label "todo:final-review"
```
Verify exactly one `todo:*` label remains.

## 9. Report
Return:
- issue URL
- current stage (`todo:final-review`)
- verification summary
- explicit note that all artifacts are now synced on GitHub
- next step: `/todo-complete <slug>`
