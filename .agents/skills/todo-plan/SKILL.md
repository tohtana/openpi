---
name: todo-plan
description: Produce reviewed design and implementation plans for a GitHub-tracked TODO, then move it to `todo:plan-ready`. GitHub mode only.
user_invocable: true
---

# TODO Plan Workflow

## When to use
- User asks to plan an existing TODO created with `/todo-doc`.
- User asks for reviewed design and implementation documents before coding.

## Non-negotiable rules
- GitHub-only workflow; no filesystem fallback stages.
- Must keep stage labels consistent (`todo:*` compare-and-verify transitions).

## 1. Preflight (must run first)
```bash
gh --version >/dev/null
gh auth status >/dev/null
GH_REPO="$(gh repo view --json nameWithOwner --jq .nameWithOwner)"
```
If preflight fails, stop and ask user to configure GitHub CLI, then rerun.

## 2. Load metadata
Read `todo/docs/${SLUG}/github.json` and validate:
- `schema_version == 1`
- `repo`, `issue_number`, `slug`, `type` are present

Use repo from metadata for all operations:
```bash
GH_REPO="$(jq -r .repo todo/docs/${SLUG}/github.json)"
ISSUE_NUMBER="$(jq -r .issue_number todo/docs/${SLUG}/github.json)"
TODO_TYPE="$(jq -r .type todo/docs/${SLUG}/github.json)"
```

## 3. Validate current issue state
Fetch state and labels:
```bash
gh issue view "$ISSUE_NUMBER" --repo "$GH_REPO" --json state,labels,url
```
Requirements:
- issue open
- exactly one `todo:*` label
- exactly one `type:*` label
- stage is `todo:submitted` or `todo:in-review` (resume)

If stage is already `todo:plan-ready` or later, report and stop.

## 4. Create planning artifact directories
Ensure:
- `todo/docs/${SLUG}/`
- `todo/docs/${SLUG}/design_review/`
- `todo/docs/${SLUG}/plan_review/`

## 5. Write design doc
Create/update:
- `todo/docs/${SLUG}/${SLUG}-design.md`

Use type-aware structure:
- `feature`: motivation, proposed design, interfaces, risks
- `debug`: symptom, root cause, fix strategy, validation
- `review`: scope under review, findings, planned updates
- `docs`: docs gaps, proposed documentation changes, validation
- `infra`: operational problem, architecture/workflow changes, rollout

## 6. Move to `todo:in-review`
If current stage is `todo:submitted`, swap label:
```bash
gh issue edit "$ISSUE_NUMBER" --repo "$GH_REPO" \
  --remove-label "todo:submitted" --add-label "todo:in-review"
```
Re-read labels and verify exactly one `todo:*` (`todo:in-review`).

## 7. Review design doc
Run:
```bash
python scripts/agent/design_doc_review_cycle.py \
  "todo/docs/${SLUG}/${SLUG}-design.md" \
  --reviewer claude --cycles 2 \
  --comments-dir "todo/docs/${SLUG}/design_review" \
  --no-commit
```
Stop and report immediately if output indicates quota/rate-limit issues.

## 8. Write implementation plan
Create/update:
- `todo/docs/${SLUG}/${SLUG}-impl-plan.md`

## 9. Review implementation plan
Run:
```bash
python scripts/agent/plan_doc_review_cycle.py \
  "todo/docs/${SLUG}/${SLUG}-design.md" \
  --plan-doc "todo/docs/${SLUG}/${SLUG}-impl-plan.md" \
  --reviewer claude --cycles 2 \
  --comments-dir "todo/docs/${SLUG}/plan_review" \
  --no-commit
```
Stop and report immediately if output indicates quota/rate-limit issues.

## 10. Mirror docs to GitHub issue
Upsert managed comments:
- design: `<!-- managed:design-doc -->`
- impl plan: `<!-- managed:impl-plan -->`

Persist comment IDs to `github.json.comment_ids`.

## 11. Move to `todo:plan-ready`
Swap label:
```bash
gh issue edit "$ISSUE_NUMBER" --repo "$GH_REPO" \
  --remove-label "todo:in-review" --add-label "todo:plan-ready"
```
Re-verify exactly one `todo:*` label.

## 12. Report
Return:
- issue URL
- current stage label (`todo:plan-ready`)
- docs produced
- next step: `/todo-impl <slug>`
