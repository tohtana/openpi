---
name: todo-doc
description: Create a GitHub-tracked TODO, classify work type labels, and initialize `todo/docs/<slug>/github.json`. GitHub mode only.
user_invocable: true
---

# TODO Doc Workflow

## When to use
- User asks to add a TODO item.
- User asks to track work progress end-to-end.
- User wants status tracked on GitHub issues.

## Non-negotiable rules
- This workflow is **GitHub-only**.
- Do **not** append to `todo/TODO.md` and do **not** create filesystem-only TODO stages.
- If GitHub is not ready, stop and ask the user to configure it; do not use fallback behavior.

## 1. Preflight (must run first)
Run:
```bash
gh --version >/dev/null
gh auth status >/dev/null
GH_REPO="$(gh repo view --json nameWithOwner --jq .nameWithOwner)"
```
If any command fails:
- Stop immediately.
- Ask the user to run setup (for example `gh auth login`, and set the repo context), then rerun `/todo-doc`.

## 2. Classify work type (`type:*` label)
Assign exactly one type label.

| Label | Use for |
|------|---------|
| `type:feature` | New functionality or enhancements |
| `type:debug` | Bug fixes, regressions, incorrect behavior, failures |
| `type:review` | Review-oriented work (PR review, validation review, code/design review) |
| `type:docs` | Documentation-focused work |
| `type:infra` | Tooling, CI, environment, or infrastructure changes |

Detection order:
1. If user explicitly specifies a type, use it.
2. If request is defect/failure/regression oriented, use `debug`.
3. If request is PR/review/verification oriented, use `review`.
4. If request is docs-only, use `docs`.
5. If request is tooling/setup/automation focused, use `infra`.
6. Default to `feature`.

## 3. Derive slug
- Use a short slug with no date prefix: `short-slug`.
- Allowed pattern: `^[a-z0-9][a-z0-9-]*$`.

Collision checks:
```bash
test ! -e "todo/docs/${SLUG}/github.json"
gh issue list --repo "$GH_REPO" --state all --search "in:title [${SLUG}]" --json number,title,url
```
If a collision exists, stop and ask for a different slug.

## 4. Create tracking issue
Create issue with labels:
- `todo:submitted`
- `type:<detected-type>`

Use body template from:
- `todo/docs/_templates/issue_bodies/<type>.md`

Before issue creation, render the template with concrete content from the user request.
Do not post unresolved placeholders like `<...>`.

Create issue:
```bash
ISSUE_URL="$(gh issue create --repo "$GH_REPO" \
  --title "[${SLUG}] <short title>" \
  --body-file "todo/docs/_templates/issue_bodies/${TODO_TYPE}.md" \
  --label "todo:submitted" \
  --label "type:${TODO_TYPE}")"
ISSUE_NUMBER="$(gh issue view "$ISSUE_URL" --repo "$GH_REPO" --json number --jq .number)"
```

## 5. Persist metadata
Create `todo/docs/${SLUG}/github.json` atomically.

Required payload shape:
```json
{
  "schema_version": 1,
  "repo": "<owner/repo>",
  "issue_number": 0,
  "issue_url": "<url>",
  "slug": "<slug>",
  "type": "feature|debug|review|docs|infra",
  "pr_number": null,
  "pr_url": null,
  "comment_ids": {
    "design-doc": null,
    "impl-plan": null,
    "impl-artifacts": null,
    "final-summary": null
  }
}
```

## 6. Validate label invariants
After creation, verify issue has:
- exactly one `todo:*` label (`todo:submitted`)
- exactly one `type:*` label

If violated, write `todo/docs/${SLUG}/github-recovery.md` and stop.

## 7. Report
Return:
- `slug`
- `issue_url`
- `type`
- next step: `/todo-plan <slug>`
