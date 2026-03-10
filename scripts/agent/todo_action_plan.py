#!/usr/bin/env python3
"""Create and iteratively review an action plan from a TODO item.

Given a TODO slug (e.g. ``20260208-qwen3-vl-throughput-bench``), this script:

1. Reads the TODO description from ``todo/<slug>.md``
2. Creates a ``tasks/<slug>/`` directory
3. Generates an initial action plan at ``tasks/<slug>/plan.md`` using the
   first reviewer
4. Runs iterative review cycles to refine the plan

The action plan is designed for execution by AI coding agents via an
orchestrator/subagent model, similar to ``plan_doc_review_cycle.py``.

Four built-in reviewer presets are available:
  1) Claude CLI          (claude --dangerously-skip-permissions -p --output-format stream-json --include-partial-messages --verbose)
  2) Codex CLI           (codex --dangerously-bypass-approvals-and-sandbox exec --json)
  3) Cursor / Opus 4.6   (cursor agent -p -f --model opus-4.6-thinking)
  4) Cursor / GPT 5.2    (cursor agent -p -f --model gpt-5.2-codex-xhigh)

Automatic fallback: if a reviewer fails (non-zero exit or timeout), a
fallback reviewer is tried automatically:
  - claude  -> cursor-opus
  - codex   -> cursor-gpt
Use --no-fallback to disable this behaviour.

Usage:
    # Create and review a plan from a TODO slug
    python todo_action_plan.py 20260208-qwen3-vl-throughput-bench

    # Use preset reviewers
    python todo_action_plan.py 20260208-qwen3-vl-throughput-bench \\
        --reviewer claude --reviewer cursor-opus

    # Only generate the plan (skip review cycles)
    python todo_action_plan.py 20260208-qwen3-vl-throughput-bench --cycles 0

    # Custom cycle count and skip git commits
    python todo_action_plan.py 20260208-qwen3-vl-throughput-bench --cycles 5 --no-commit

    # Provide additional context
    python todo_action_plan.py 20260208-qwen3-vl-throughput-bench \\
        --reviewer claude --context "Focus on GPU memory constraints"

    # Explicit plan path (overrides default tasks/<slug>/plan.md)
    python todo_action_plan.py 20260208-qwen3-vl-throughput-bench \\
        --plan-doc tasks/custom/my-plan.md
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import (
    Reviewer,
    add_common_arguments,
    append_context,
    comments_dir,
    git_commit,
    postprocess_common_args,
    resolve_reviewers,
    run_review_loop,
    save_comments,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TODO_DIR = Path("todo")
TASKS_DIR = Path("tasks")
COMMENTS_BASE = Path("task_comments")


def _resolve_todo_path(slug: str) -> Path:
    """Return the TODO file path for a given slug."""
    return TODO_DIR / f"{slug}.md"


def _resolve_task_dir(slug: str) -> Path:
    """Return the task directory for a given slug."""
    return TASKS_DIR / slug


def _resolve_plan_path(slug: str) -> Path:
    """Return the default plan file path for a given slug."""
    return _resolve_task_dir(slug) / "plan.md"


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CREATION_PROMPT_TEMPLATE = """\
Read the TODO item at {todo_doc} and create a detailed action plan at \
{plan_doc}.

IMPORTANT CONTEXT: This plan will be executed by AI coding agents (e.g. \
Claude Code, Codex CLI), not human developers. The plan must be written so \
that an orchestrator agent can break it into subtasks and dispatch them to \
specialized subagents.

The TODO item contains the goal, initial plan, and any existing progress. \
Use it as the foundation but expand it into a fully actionable plan.

The action plan should include:

1. **Goal** — restate the objective clearly and concisely.

2. **Implementation phases** — ordered list of work packages with clear \
boundaries, dependencies, and estimated complexity. Each phase should be \
scoped so that a single agent session can complete it (i.e. fits within a \
reasonable context window and produces a reviewable diff).

3. **File-level change map** — for each phase, list every file to create or \
modify with a short description of the change.

4. **Interface contracts** — key function signatures, class hierarchies, and \
data structures that must be agreed upon before coding. These serve as the \
"handshake" between phases so agents working on later phases know what \
earlier phases produced.

5. **Testing strategy** — unit tests, integration tests, and manual \
validation steps for each phase. Tests should be runnable by an agent via \
shell commands (e.g. `pytest ...`, `bash scripts/...`).

6. **Risk register** — potential pitfalls, open questions, and mitigations.

7. **Acceptance criteria** — concrete, verifiable conditions for each phase \
to be considered complete. Prefer machine-checkable criteria (exit codes, \
test pass counts, grep-able output) over subjective judgments.

8. **Progress tracking** — define a progress section that can be updated \
as phases complete. Use checkboxes to track completion status.

DIFF MINIMIZATION PRINCIPLE: The plan must minimize changes to existing \
files to keep diffs small and reviews easy. When a phase requires a large \
block of new code, prefer placing it in a **new file** (with a clear \
module boundary) rather than inserting it into an existing file.

Write the plan as a Markdown file. Be specific and actionable — reference \
exact file paths, function names, and class names from the codebase. \
Include the exact commands needed to run each step.
"""

REVIEW_PROMPT_TEMPLATE = """\
Review the action plan at {plan_doc} (derived from the TODO item at \
{todo_doc}) and improve it.

This plan is designed to be executed by AI coding agents via an orchestrator \
that dispatches work to specialized subagents. Keep this execution model in \
mind during your review.

Focus on:
- **Completeness** — are any implementation steps missing?
- **Ordering & dependencies** — is the phase order optimal? Are dependency \
chains correct?
- **Specificity** — are file paths, function names, and interfaces concrete \
enough for a coding agent to start working without ambiguity?
- **Testability** — does each phase have clear, automatable acceptance \
criteria that a tester agent can verify via shell commands?
- **Risk coverage** — are edge cases, failure modes, and migration concerns \
addressed?
- **Agent-readiness** — are phases scoped to fit a single agent session? \
Are interface contracts explicit enough to serve as handshakes between \
agents working on different phases?
- **Diff minimization** — does each phase keep changes to existing files \
small and surgical?
- **Progress tracking** — is the progress section clear and easy to update?

Update {plan_doc} directly with your improvements. Keep the overall structure \
but refine, add, or reorganize content as needed.
"""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_creation_prompt(
    todo_doc: str,
    plan_doc: str,
    additional_context: str | None = None,
) -> str:
    """Build the initial plan-creation prompt."""
    prompt = CREATION_PROMPT_TEMPLATE.format(
        todo_doc=todo_doc, plan_doc=plan_doc,
    )
    return append_context(prompt, additional_context)


def _build_review_prompt(
    todo_doc: str,
    plan_doc: str,
    prev_comments_path: Path | None,
    prev_reviewer: str | None,
) -> str:
    """Build the review prompt, optionally referencing previous feedback."""
    prompt = REVIEW_PROMPT_TEMPLATE.format(
        todo_doc=todo_doc, plan_doc=plan_doc,
    )
    if prev_comments_path and prev_reviewer:
        prompt += (
            f"\nThe previous reviewer ({prev_reviewer}) left comments in "
            f"{prev_comments_path}. Please read that file for their feedback."
            "\n\nIMPORTANT: In addition to addressing the previous reviewer's "
            "feedback, actively look for issues, concerns, and improvements "
            "that the previous reviewer did NOT mention. Bring your own "
            "independent perspective — do not limit your review to only the "
            "points already raised."
        )
    return prompt


# ---------------------------------------------------------------------------
# Plan creation
# ---------------------------------------------------------------------------


def ensure_plan_exists(
    args: argparse.Namespace, reviewer: Reviewer, cdir: Path,
) -> None:
    """Generate the initial action plan if the file does not exist."""
    plan_path = Path(args.plan_doc)
    if plan_path.exists():
        print(f"\n[Plan already exists at {args.plan_doc}, skipping creation]")
        return

    # Ensure the task directory exists
    plan_path.parent.mkdir(parents=True, exist_ok=True)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Generating initial action plan with {reviewer.name}")
    print(f"{sep}\n")

    prompt = build_creation_prompt(
        args.todo_doc, args.plan_doc, args.additional_context,
    )
    try:
        output = reviewer.run(
            prompt,
            timeout=args.timeout,
            stall_timeout=args.stall_timeout,
            heartbeat_secs=args.heartbeat_secs,
        )
    except subprocess.TimeoutExpired as exc:
        limit_value = (
            exc.timeout if exc.timeout is not None else args.timeout
        )
        limit_desc = (
            f"{limit_value}s"
            if limit_value is not None
            else "the configured limit"
        )
        print(
            f"\n[TIMEOUT] {reviewer.name} exceeded {limit_desc} "
            "during plan creation. Partial output (if any) was printed above.",
            file=sys.stderr,
        )
        output = exc.output or ""

    comments_path = save_comments(
        cdir, 0, reviewer.name, output, label="creation",
    )
    print(f"\n[Saved creation output to {comments_path}]")

    if not args.no_commit and plan_path.exists():
        try:
            git_commit(
                args.plan_doc,
                f"create {args.plan_doc} from {args.todo_doc} "
                f"by {reviewer.name}",
            )
            print(f"\n[Committed: create {args.plan_doc}]")
        except subprocess.CalledProcessError as exc:
            print(f"\n[Git commit failed: {exc}]", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_todo_plan_cycle(args: argparse.Namespace) -> None:
    # Validate that the TODO document exists
    if not Path(args.todo_doc).exists():
        sys.exit(f"Error: TODO document not found: {args.todo_doc}")

    reviewers = resolve_reviewers(args)
    cdir = (
        Path(args.comments_dir)
        if args.comments_dir
        else comments_dir(COMMENTS_BASE, args.plan_doc)
    )

    print(f"\n  TODO doc:   {args.todo_doc}")
    print(f"  Plan doc:   {args.plan_doc}")
    print(f"  Task dir:   {Path(args.plan_doc).parent}")
    print(f"  Comments:   {cdir}")
    for i, r in enumerate(reviewers, 1):
        fb = f" (fallback: {r.fallback.name})" if r.fallback else ""
        rl = (
            f" (rate-limit: {r.rate_limit_fallback.name})"
            if r.rate_limit_fallback
            else ""
        )
        print(f"  Reviewer {i}: {r.name}{fb}{rl}")
    print()

    # --- Phase 1: create plan if missing ---
    ensure_plan_exists(args, reviewers[0], cdir)

    if not Path(args.plan_doc).exists():
        print(
            f"\n[ERROR] Plan file {args.plan_doc} was not created. "
            "Cannot proceed with review cycles.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Phase 2: iterative review ---
    if args.cycles <= 0:
        print("\n[Skipping review cycles (--cycles 0)]")
        return

    def build_prompt_fn(
        prev_comments_path: Path | None,
        prev_reviewer_name: str | None,
    ) -> str:
        prompt = _build_review_prompt(
            args.todo_doc,
            args.plan_doc,
            prev_comments_path,
            prev_reviewer_name,
        )
        return append_context(prompt, args.additional_context)

    run_review_loop(
        reviewers=reviewers,
        cycles=args.cycles,
        doc_path=args.plan_doc,
        build_prompt_fn=build_prompt_fn,
        cdir=cdir,
        no_commit=args.no_commit,
        timeout=args.timeout,
        stall_timeout=args.stall_timeout,
        heartbeat_secs=args.heartbeat_secs,
        cycle_label="reviewing action plan",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create and iteratively review an action plan from a TODO item",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "slug",
        help=(
            "TODO slug (e.g. 20260208-qwen3-vl-throughput-bench). "
            "Reads from todo/<slug>.md and writes to tasks/<slug>/plan.md"
        ),
    )
    p.add_argument(
        "--plan-doc",
        default=None,
        help=(
            "Path to the plan document. If omitted, defaults to "
            "tasks/<slug>/plan.md"
        ),
    )
    add_common_arguments(p)

    args = p.parse_args()
    postprocess_common_args(args)

    # Resolve paths from slug
    args.todo_doc = str(_resolve_todo_path(args.slug))
    if args.plan_doc is None:
        args.plan_doc = str(_resolve_plan_path(args.slug))

    return args


def main() -> None:
    args = parse_args()
    run_todo_plan_cycle(args)


if __name__ == "__main__":
    main()
