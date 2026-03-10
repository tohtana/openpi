# TODO Workflow (GitHub-Only)

This project uses a 4-skill TODO pipeline with GitHub issue labels as the source of truth.

## Flow

```text
/todo-doc -> /todo-plan -> /todo-impl -> /todo-complete
```

## Stage labels (`todo:*`)

Exactly one stage label must exist at all times.

- `todo:submitted`
- `todo:in-review`
- `todo:plan-ready`
- `todo:in-progress`
- `todo:final-review`
- `todo:completed`
- `todo:wontdo`

## Type labels (`type:*`)

Exactly one type label must exist at all times.

- `type:feature`
- `type:debug`
- `type:review`
- `type:docs`
- `type:infra`

## Required GitHub preflight

Every TODO skill must run this first:

```bash
gh --version >/dev/null
gh auth status >/dev/null
gh repo view --json nameWithOwner --jq .nameWithOwner >/dev/null
```

If preflight fails, stop and ask the user to set up GitHub CLI (`gh auth login`, repo context). No local fallback is allowed.

## Artifact contract

By the end of `/todo-impl`, all core artifacts must be mirrored to the GitHub tracking issue:

- design doc (`<slug>-design.md`)
- implementation plan (`<slug>-impl-plan.md`)
- verification scripts and notes (`verify_*.py`, `verify_*.sh`, `verification.md`, or equivalents)
- implementation results (`results.md`)
- relevant review summaries (`design_review/`, `plan_review/` summaries)

Use managed issue comments and persist comment IDs in `todo/docs/<slug>/github.json`.

## Metadata

Per-slug metadata lives at:

- `todo/docs/<slug>/github.json`

Schema template:

- `todo/docs/_templates/github.schema.v1.json`

## Legacy behavior

There is no filesystem fallback workflow for new tasks. If GitHub is unavailable, the skill must stop and request setup.
