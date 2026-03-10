---
name: todo-complete
description: Finalize a GitHub-tracked TODO by posting a final summary, transitioning terminal label, and closing the issue. GitHub mode only.
user_invocable: true
---

# TODO Complete Workflow

## When to use
- User explicitly asks to complete or abandon a TODO at final review.

## Non-negotiable rules
- GitHub-only workflow; no filesystem fallback stages.
- This skill requires explicit user invocation/approval.
- Never push, merge, or rewrite git history automatically.

## 1. Preflight (must run first)
```bash
gh --version >/dev/null
gh auth status >/dev/null
GH_REPO="$(gh repo view --json nameWithOwner --jq .nameWithOwner)"
```
If preflight fails, stop and ask user to configure GitHub CLI.

## 2. Load metadata and validate state
Read `todo/docs/${SLUG}/github.json`:
```bash
GH_REPO="$(jq -r .repo todo/docs/${SLUG}/github.json)"
ISSUE_NUMBER="$(jq -r .issue_number todo/docs/${SLUG}/github.json)"
```

Require:
- issue exists and is open
- exactly one `todo:*` label
- stage is `todo:final-review`

## 3. Determine terminal action
- Default terminal action: `completed`
- If user explicitly says `wontdo`, terminal action: `wontdo`

## 4. Write and sync final summary
Create/update:
- `todo/docs/${SLUG}/final_summary.md`

Include:
- implementation summary
- verification outcome
- key artifacts produced
- final disposition (`completed` or `wontdo`)

Upsert managed issue comment:
- marker: `<!-- managed:final-summary -->`
- persist ID in `github.json.comment_ids["final-summary"]`

## 5. Transition terminal label
For completed:
```bash
gh issue edit "$ISSUE_NUMBER" --repo "$GH_REPO" \
  --remove-label "todo:final-review" --add-label "todo:completed"
```

For wontdo:
```bash
gh issue edit "$ISSUE_NUMBER" --repo "$GH_REPO" \
  --remove-label "todo:final-review" --add-label "todo:wontdo"
```

Verify exactly one `todo:*` label remains.

## 6. Close issue
```bash
gh issue close "$ISSUE_NUMBER" --repo "$GH_REPO"
```

## 7. Report
Return:
- issue URL
- terminal label
- close status
- location of `final_summary.md`
