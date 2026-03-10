# Agent Scripts

Scripts that orchestrate multi-round AI review cycles for design and implementation plan documents. Multiple AI reviewers take turns reviewing and updating a document, passing feedback between cycles.

## Scripts

### `design_doc_review_cycle.py`

Runs iterative AI review cycles on a **design document**. Each reviewer reads the document, addresses concerns, and updates it in-place. Subsequent reviewers also receive the previous reviewer's comments so they can build on (and go beyond) prior feedback.

```bash
# Review a specific document
python design_doc_review_cycle.py docs/my-design.md

# Default document (docs/autoep-design.md) when no argument given
python design_doc_review_cycle.py

# Use preset reviewers
python design_doc_review_cycle.py docs/my-design.md \
    --reviewer claude --reviewer cursor-opus

# Custom cycle count and skip git commits
python design_doc_review_cycle.py docs/my-design.md --cycles 5 --no-commit
```

### `plan_doc_review_cycle.py`

Creates and iteratively reviews an **implementation plan** derived from a design document. If the plan file does not exist yet, the first reviewer generates it from the design doc before the review loop begins. The plan is tailored for execution by AI coding agents via an orchestrator/subagent model.

```bash
# Generate and review a plan from a design doc
python plan_doc_review_cycle.py docs/my-design.md

# Explicit plan path (overrides auto-derivation)
python plan_doc_review_cycle.py docs/my-design.md --plan-doc docs/my-plan.md

# Use preset reviewers with extra context
python plan_doc_review_cycle.py docs/my-design.md \
    --reviewer claude --reviewer codex \
    --context "Focus on GPU memory constraints"
```

The plan path is auto-derived by replacing `-design` with `-impl-plan` in the filename (e.g. `docs/foo-design.md` becomes `docs/foo-impl-plan.md`).

### `todo_action_plan.py` (legacy utility)

Creates and iteratively reviews a detailed **action plan** from a TODO item. Given a TODO slug (e.g. `20260208-qwen3-vl-throughput-bench`), reads the lightweight plan from `todo/<slug>.md` and generates a comprehensive, agent-executable plan at `tasks/<slug>/plan.md`.

```bash
# Create and review a plan from a TODO slug
python todo_action_plan.py 20260208-qwen3-vl-throughput-bench \
    --reviewer claude --cycles 2

# Create plan only (no review cycles)
python todo_action_plan.py 20260208-qwen3-vl-throughput-bench \
    --reviewer claude --cycles 0

# Review an existing plan with multiple reviewers
python todo_action_plan.py 20260208-qwen3-vl-throughput-bench \
    --reviewer claude --reviewer cursor-opus
```

This helper is legacy and not part of the current primary workflow. The canonical planning entrypoint is `/todo-plan`.

### `common.py`

Shared library used by all review-cycle scripts. Provides:

- **`Reviewer`** — abstraction that invokes a CLI-based AI tool, writes prompt/output to files, emits heartbeat logs while the reviewer runs, and captures the result. Prompts are passed via stdin to avoid `ARG_MAX` limits.
- **Reviewer presets** — built-in configurations for common AI CLI tools.
- **`run_review_loop`** — the core loop that cycles reviewers through rounds of review, saves comments, and optionally commits after each iteration.
- **Comment persistence** — saves each reviewer's output to a per-document comments directory.
- **Git helpers** — stages and commits the document after each review round.
- **Common CLI arguments** — shared `argparse` options (`--cycles`, `--reviewer`, `--timeout`, `--context`, etc.).

## Reviewer Presets

All scripts support these built-in presets via the `--reviewer` flag:

| Preset | Tool | Command |
|--------|------|---------|
| `claude` | Claude CLI | `claude --dangerously-skip-permissions -p --output-format stream-json --include-partial-messages --verbose` |
| `codex` | Codex CLI | `codex --dangerously-bypass-approvals-and-sandbox exec --json` |
| `cursor-opus` | Cursor / Opus 4.6 Thinking | `cursor agent -p -f --model opus-4.6-thinking` |
| `cursor-gpt` | Cursor / GPT 5.2 Codex XHigh | `cursor agent -p -f --model gpt-5.2-codex-xhigh` |

Note: Claude's `stream-json` mode currently requires `--verbose`; the script uses it only to collect liveness events, then strips telemetry before returning reviewer output.

You can also specify fully custom reviewers:

```bash
python design_doc_review_cycle.py docs/my-doc.md \
    --reviewer-cmd "my-tool review" --reviewer-name "MyTool" \
    --reviewer-cmd "another-tool run" --reviewer-name "AnotherTool"
```

Presets and custom reviewers can be mixed. If none are specified, the script prompts interactively.

## Common Options

| Flag | Default | Description |
|------|---------|-------------|
| `--cycles N` | 3 | Number of full review rounds |
| `--reviewer PRESET` | (interactive) | Add a reviewer by preset (repeatable) |
| `--reviewer-cmd CMD` | — | Custom reviewer shell command (repeatable, paired with `--reviewer-name`) |
| `--reviewer-name NAME` | — | Display name for a custom reviewer (paired with `--reviewer-cmd`) |
| `--context TEXT` | — | Additional context included in every prompt (repeatable) |
| `--context-file FILE` | — | File whose contents are included as additional context (repeatable) |
| `--timeout SECS` | 1800 | Max seconds per reviewer invocation (0 = no limit) |
| `--stall-timeout SECS` | 0 | Kill reviewer after this much no output/CPU activity (0 = disabled) |
| `--heartbeat-secs SECS` | 30 | Emit liveness heartbeat every N seconds while reviewer runs (0 = disabled) |
| `--no-commit` | false | Skip git commits after each review |

## Output

- **Comments** are saved to `review_comments/<doc-stem>/` (design reviews), `plan_comments/<doc-stem>/` (plan reviews), or `task_comments/<doc-stem>/` (TODO action plans), with filenames like `cycle1_claude_cli.txt`.
- **Git commits** are created after each reviewer's turn (unless `--no-commit` is set), with messages like `update docs/my-design.md by Claude CLI (cycle 1)`.
- **Liveness**: each reviewer invocation prints heartbeat lines including elapsed time, idle time, output size, root/tree CPU usage, and probe state.
- **Thinking vs final output**: for `codex` and `claude` presets, structured JSON events are classified into `progress` (thinking/work-in-flight) and `final` (terminal output), and stall detection uses progress + process-tree activity.
- **Orchestrator-facing output**: only extracted final assistant text is returned to the review loop and saved to comments; raw JSON event streams stay in the temporary `stdout.log`/`stderr.log` files used for monitoring.
