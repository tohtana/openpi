#!/usr/bin/env python3
"""AI review cycle for design documents.

One or more AI reviewers take turns reviewing and updating a target
document, passing feedback between cycles.

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
    # Review a specific document
    python design_doc_review_cycle.py docs/autoep-deepspeedexamples-design.md

    # Default document (docs/autoep-design.md) when no argument given
    python design_doc_review_cycle.py

    # Preset-based — any number of reviewers (1 or more)
    python design_doc_review_cycle.py docs/my-doc.md --reviewer claude --reviewer cursor-opus
    python design_doc_review_cycle.py docs/my-doc.md --reviewer claude  # single reviewer

    # Three reviewers
    python design_doc_review_cycle.py docs/my-doc.md \\
        --reviewer claude --reviewer cursor-opus --reviewer codex

    # Custom cycle count
    python design_doc_review_cycle.py docs/my-doc.md --cycles 5

    # Skip git commits (dry run)
    python design_doc_review_cycle.py --no-commit

    # Disable automatic fallback (claude->cursor-opus, codex->cursor-gpt)
    python design_doc_review_cycle.py docs/my-doc.md --reviewer claude --no-fallback

    # Fully custom reviewer commands (repeatable, matched by position)
    python design_doc_review_cycle.py docs/my-doc.md \\
        --reviewer-cmd "codex --dangerously-bypass-approvals-and-sandbox exec --json" \\
        --reviewer-name "Codex" \\
        --reviewer-cmd "claude --dangerously-skip-permissions -p --output-format stream-json --include-partial-messages --verbose" \\
        --reviewer-name "Claude"

    # Mix presets and custom (presets first, then custom)
    python design_doc_review_cycle.py docs/my-doc.md \\
        --reviewer claude \\
        --reviewer-cmd "my-tool review" --reviewer-name "MyTool"

    # Provide additional context (inline text)
    python design_doc_review_cycle.py docs/my-doc.md --reviewer claude \\
        --context "Focus on security implications"

    # Provide additional context from a file
    python design_doc_review_cycle.py docs/my-doc.md --reviewer claude \\
        --context-file docs/requirements.md

    # Combine inline and file context (both are concatenated)
    python design_doc_review_cycle.py docs/my-doc.md --reviewer claude \\
        --context "Prioritize performance" --context-file docs/constraints.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import (
    add_common_arguments,
    append_context,
    comments_dir,
    postprocess_common_args,
    resolve_reviewers,
    run_review_loop,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DOC_PATH = "docs/autoep-design.md"
COMMENTS_BASE = Path("review_comments")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _base_prompt(doc_path: str) -> str:
    """Return the base review prompt for a given document."""
    return (
        f"Please review {doc_path} and update it to address concerns raised in "
        "the review."
    )


def _build_prompt(
    doc_path: str,
    prev_comments_path: Path | None,
    prev_reviewer: str | None,
) -> str:
    """Build the review prompt, optionally referencing the previous review file."""
    base = _base_prompt(doc_path)
    if not prev_comments_path or not prev_reviewer:
        return base
    return (
        f"{base}\n\n"
        f"The previous reviewer ({prev_reviewer}) left comments in "
        f"{prev_comments_path}. Please read that file for their feedback.\n\n"
        "IMPORTANT: In addition to addressing the previous reviewer's "
        "feedback, actively look for issues, concerns, and improvements "
        "that the previous reviewer did NOT mention. Bring your own "
        "independent perspective — do not limit your review to only the "
        "points already raised."
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_review_cycle(args: argparse.Namespace) -> None:
    doc_path: str = args.doc

    # Validate that the target document exists
    if not Path(doc_path).exists():
        sys.exit(f"Error: document not found: {doc_path}")

    reviewers = resolve_reviewers(args)
    cdir = Path(args.comments_dir) if args.comments_dir else comments_dir(COMMENTS_BASE, doc_path)

    print(f"\n  Document:   {doc_path}")
    print(f"  Comments:   {cdir}")
    for i, r in enumerate(reviewers, 1):
        fb = f" (fallback: {r.fallback.name})" if r.fallback else ""
        rl = f" (rate-limit: {r.rate_limit_fallback.name})" if r.rate_limit_fallback else ""
        print(f"  Reviewer {i}: {r.name}{fb}{rl}")
    print()

    def build_prompt_fn(
        prev_comments_path: Path | None,
        prev_reviewer_name: str | None,
    ) -> str:
        prompt = _build_prompt(doc_path, prev_comments_path, prev_reviewer_name)
        return append_context(prompt, args.additional_context)

    run_review_loop(
        reviewers=reviewers,
        cycles=args.cycles,
        doc_path=doc_path,
        build_prompt_fn=build_prompt_fn,
        cdir=cdir,
        no_commit=args.no_commit,
        timeout=args.timeout,
        stall_timeout=args.stall_timeout,
        heartbeat_secs=args.heartbeat_secs,
        cycle_label="reviewing",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI review cycle for design documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "doc",
        nargs="?",
        default=DEFAULT_DOC_PATH,
        help=f"Path to the document to review (default: {DEFAULT_DOC_PATH})",
    )
    add_common_arguments(p)

    args = p.parse_args()
    postprocess_common_args(args)
    return args


def main() -> None:
    args = parse_args()
    run_review_cycle(args)


if __name__ == "__main__":
    main()
