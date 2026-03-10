#!/usr/bin/env python3
"""Create and iteratively review an implementation plan.

Based on design_doc_review_cycle.py. One or more AI reviewers take turns reviewing
and refining an implementation plan derived from a design document.

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

If the plan file does not exist, the first reviewer generates it from the
design document before the review loop begins.

Usage:
    # Review a specific design document (plan path auto-derived)
    python plan_doc_review_cycle.py docs/autoep-deepspeedexamples-design.md

    # Default design doc (docs/autoep-design.md) when no argument given
    python plan_doc_review_cycle.py

    # Explicit plan doc path (overrides auto-derivation)
    python plan_doc_review_cycle.py docs/autoep-design.md --plan-doc docs/my-plan.md

    # Custom cycle count
    python plan_doc_review_cycle.py docs/autoep-design.md --cycles 5

    # Skip git commits (dry run)
    python plan_doc_review_cycle.py --no-commit

    # Disable automatic fallback (claude->cursor-opus, codex->cursor-gpt)
    python plan_doc_review_cycle.py docs/autoep-design.md --reviewer claude --no-fallback

    # Preset-based — any number of reviewers (1 or more)
    python plan_doc_review_cycle.py docs/autoep-design.md --reviewer claude --reviewer cursor-opus
    python plan_doc_review_cycle.py docs/autoep-design.md --reviewer claude  # single reviewer

    # Three reviewers
    python plan_doc_review_cycle.py docs/autoep-design.md \\
        --reviewer claude --reviewer cursor-opus --reviewer codex

    # Fully custom reviewer commands (repeatable, matched by position)
    python plan_doc_review_cycle.py docs/autoep-design.md \\
        --reviewer-cmd "codex --dangerously-bypass-approvals-and-sandbox exec --json" \\
        --reviewer-name "Codex" \\
        --reviewer-cmd "claude --dangerously-skip-permissions -p --output-format stream-json --include-partial-messages --verbose" \\
        --reviewer-name "Claude"

    # Mix presets and custom (presets first, then custom)
    python plan_doc_review_cycle.py docs/autoep-design.md \\
        --reviewer claude \\
        --reviewer-cmd "my-tool review" --reviewer-name "MyTool"

    # Provide additional context (inline text)
    python plan_doc_review_cycle.py docs/autoep-design.md --reviewer claude \\
        --context "Focus on GPU memory constraints"

    # Provide additional context from a file
    python plan_doc_review_cycle.py docs/autoep-design.md --reviewer claude \\
        --context-file docs/requirements.md

    # Combine inline and file context (both are concatenated)
    python plan_doc_review_cycle.py docs/autoep-design.md --reviewer claude \\
        --context "Prioritize small diffs" --context-file docs/constraints.md
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

DEFAULT_DESIGN_DOC = "docs/autoep-design.md"
DEFAULT_PLAN_DOC = "docs/autoep-impl-plan.md"
COMMENTS_BASE = Path("plan_comments")


def _derive_plan_path(design_doc: str) -> str:
    """Derive a plan doc path from a design doc path.

    Replaces '-design.md' (or '_design.md') with '-impl-plan.md' in the
    filename.  Falls back to inserting '-impl-plan' before '.md'.

    Examples:
        docs/autoep-design.md        -> docs/autoep-impl-plan.md
        docs/foo_design.md           -> docs/foo_impl-plan.md
        docs/my-proposal.md          -> docs/my-proposal-impl-plan.md
    """
    p = Path(design_doc)
    stem = p.stem  # e.g. "autoep-design"
    for sep in ("-", "_"):
        suffix = f"{sep}design"
        if stem.endswith(suffix):
            new_stem = stem[: -len(suffix)] + f"{sep}impl-plan"
            return str(p.with_name(new_stem + p.suffix))
    # Fallback: append -impl-plan before extension
    return str(p.with_name(stem + "-impl-plan" + p.suffix))


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CREATION_PROMPT_TEMPLATE = """\
Read the design document at {design_doc} and create a detailed \
implementation plan at {plan_doc}.

IMPORTANT CONTEXT: This plan will be executed by AI coding agents (e.g. \
Claude Code, Codex CLI), not human developers. The plan must be written so \
that an orchestrator agent can break it into subtasks and dispatch them to \
specialized subagents.

The plan should include:

1. **Implementation phases** — ordered list of work packages with clear \
boundaries, dependencies, and estimated complexity. Each phase should be \
scoped so that a single agent session can complete it (i.e. fits within a \
reasonable context window and produces a reviewable diff).

2. **File-level change map** — for each phase, list every file to create or \
modify with a short description of the change.

3. **Interface contracts** — key function signatures, class hierarchies, and \
data structures that must be agreed upon before coding. These serve as the \
"handshake" between phases so agents working on later phases know what \
earlier phases produced.

4. **Testing strategy** — unit tests, integration tests, and manual \
validation steps for each phase. Tests should be runnable by an agent via \
shell commands (e.g. `pytest ...`, `ds_report`, `deepspeed --num_gpus ...`).

5. **Risk register** — potential pitfalls, open questions, and mitigations.

6. **Acceptance criteria** — concrete, verifiable conditions for each phase \
to be considered complete. Prefer machine-checkable criteria (exit codes, \
test pass counts, grep-able output) over subjective judgments.

7. **Agent orchestration plan** — describe how an orchestrator agent should \
manage the implementation using subagents. Include:
   a. **Execution planner subagent** — decides the next action: write code, \
run tests, evaluate results, or revise the plan. Maintains a task queue and \
tracks phase completion status.
   b. **Coder subagent** — receives a scoped task (one phase or sub-phase) \
with the interface contracts and relevant file context, writes or modifies \
code, and returns a diff.
   c. **Reviewer subagent** — reviews each diff against the design doc and \
interface contracts. Checks for correctness, style, and consistency with \
other phases. Returns approve/request-changes with specific feedback.
   d. **Tester subagent** — runs the test commands for the current phase, \
captures output, and reports pass/fail with relevant logs.
   e. **Plan-revision subagent** — triggered when a phase fails tests or \
review. Analyzes the failure, proposes plan amendments (reorder phases, \
split a phase, add a dependency), and updates the plan doc.
   f. **Orchestration flow** — for each phase: planner dispatches to coder \
→ reviewer checks diff → if rejected, coder retries with feedback → tester \
runs acceptance criteria → if fail, plan-revision subagent adjusts → loop \
until phase passes. Then advance to the next phase.
   g. **Persistent progress record** — agent contexts can be reset at any \
time (session timeout, crash, context-window overflow). The orchestrator \
and every subagent MUST save their progress to disk so work can resume \
from the last checkpoint rather than restarting from scratch. Specify:
      - A **progress file** (e.g. `progress.json` or a YAML/Markdown \
status file in the project directory) that records: which phases are \
complete, which phase is in-progress, current sub-step within that phase, \
and any blocking issues or open questions.
      - **Update frequency** — the progress file must be updated after \
every meaningful state transition: phase started, diff produced, review \
result received, tests passed/failed, plan revised.
      - **Resume protocol** — when an agent starts (or restarts after a \
context reset), it MUST read the progress file first and continue from \
the recorded state. It must NOT re-do work that the progress file marks \
as complete.
      - **Artifact references** — the progress file should reference \
on-disk artifacts (diffs, test logs, review comments) by path so a \
resumed agent can load them instead of regenerating them.
      - **Idempotent steps** — each step should be safe to re-run in \
case the agent crashed mid-step. Prefer atomic writes (write to a temp \
file then rename) for the progress file itself.

PROGRESS PERSISTENCE PRINCIPLE: Because agent sessions are ephemeral, \
all progress MUST be recorded on disk. An agent that loses its context \
should be able to read the progress file and resume without any human \
intervention or repeated work. The plan must define the progress file \
schema, the update protocol, and the resume protocol as first-class \
concerns — not afterthoughts.

DIFF MINIMIZATION PRINCIPLE: The plan must minimize changes to existing \
files to keep diffs small and reviews easy. When a phase requires a large \
block of new code, prefer placing it in a **new file** (with a clear \
module boundary) rather than inserting it into an existing file. Existing \
files should only receive small, surgical edits — imports, registration \
calls, thin wrappers — that wire in the new module. This makes each diff \
self-contained and reviewable in isolation.

Write the plan as a Markdown file. Be specific and actionable — reference \
exact file paths, function names, and class names from the design doc. \
For agent tasks, include the exact prompts or prompt templates that the \
orchestrator should send to each subagent.
"""

REVIEW_PROMPT_TEMPLATE = """\
Review the implementation plan at {plan_doc} (derived from the design \
document at {design_doc}) and improve it.

This plan is designed to be executed by AI coding agents via an orchestrator \
that dispatches work to specialized subagents (planner, coder, reviewer, \
tester, plan-reviser). Keep this execution model in mind during your review.

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
agents working on different phases? Are the orchestration prompts clear \
and complete?
- **Failure recovery** — does the plan describe what happens when a phase \
fails review or tests? Is the plan-revision loop well-defined?
- **Progress persistence** — does the plan define a concrete progress file \
schema and resume protocol? Could an agent whose context was just reset \
pick up where the previous session left off by reading the progress file \
and on-disk artifacts? Are progress updates frequent enough (after every \
state transition) and are writes atomic/idempotent?
- **Diff minimization** — does each phase keep changes to existing files \
small and surgical? Is new logic placed in new files with clear module \
boundaries, while existing files only get minimal wiring (imports, \
registration, thin wrappers)?

Update {plan_doc} directly with your improvements. Keep the overall structure \
but refine, add, or reorganize content as needed.
"""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_creation_prompt(
    design_doc: str,
    plan_doc: str,
    additional_context: str | None = None,
) -> str:
    """Build the initial plan-creation prompt."""
    prompt = CREATION_PROMPT_TEMPLATE.format(
        design_doc=design_doc, plan_doc=plan_doc,
    )
    return append_context(prompt, additional_context)


def _build_review_prompt(
    design_doc: str,
    plan_doc: str,
    prev_comments_path: Path | None,
    prev_reviewer: str | None,
) -> str:
    """Build the review prompt, optionally referencing previous feedback."""
    prompt = REVIEW_PROMPT_TEMPLATE.format(
        design_doc=design_doc, plan_doc=plan_doc,
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
    """Generate the initial plan if the file does not exist."""
    plan_path = Path(args.plan_doc)
    if plan_path.exists():
        print(f"\n[Plan already exists at {args.plan_doc}, skipping creation]")
        return

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Generating initial plan with {reviewer.name}")
    print(f"{sep}\n")

    prompt = build_creation_prompt(
        args.design_doc, args.plan_doc, args.additional_context,
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
                f"create {args.plan_doc} from {args.design_doc} "
                f"by {reviewer.name}",
            )
            print(f"\n[Committed: create {args.plan_doc}]")
        except subprocess.CalledProcessError as exc:
            print(f"\n[Git commit failed: {exc}]", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_plan_cycle(args: argparse.Namespace) -> None:
    # Validate that the design document exists
    if not Path(args.design_doc).exists():
        sys.exit(f"Error: design document not found: {args.design_doc}")

    reviewers = resolve_reviewers(args)
    cdir = Path(args.comments_dir) if args.comments_dir else comments_dir(COMMENTS_BASE, args.design_doc)

    print(f"\n  Design doc: {args.design_doc}")
    print(f"  Plan doc:   {args.plan_doc}")
    print(f"  Comments:   {cdir}")
    for i, r in enumerate(reviewers, 1):
        fb = f" (fallback: {r.fallback.name})" if r.fallback else ""
        rl = f" (rate-limit: {r.rate_limit_fallback.name})" if r.rate_limit_fallback else ""
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
    def build_prompt_fn(
        prev_comments_path: Path | None,
        prev_reviewer_name: str | None,
    ) -> str:
        prompt = _build_review_prompt(
            args.design_doc, args.plan_doc,
            prev_comments_path, prev_reviewer_name,
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
        cycle_label="reviewing plan",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create and iteratively review an implementation plan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "design_doc",
        nargs="?",
        default=DEFAULT_DESIGN_DOC,
        help=(
            "Path to the design document "
            f"(default: {DEFAULT_DESIGN_DOC})"
        ),
    )
    p.add_argument(
        "--plan-doc",
        default=None,
        help=(
            "Path to the plan document. If omitted, derived from the "
            "design doc by replacing '-design' with '-impl-plan' "
            f"(e.g. {DEFAULT_DESIGN_DOC} -> {DEFAULT_PLAN_DOC})"
        ),
    )
    add_common_arguments(p)

    args = p.parse_args()
    postprocess_common_args(args)

    # Auto-derive plan_doc from design_doc if not explicitly given
    if args.plan_doc is None:
        args.plan_doc = _derive_plan_path(args.design_doc)

    return args


def main() -> None:
    args = parse_args()
    run_plan_cycle(args)


if __name__ == "__main__":
    main()
