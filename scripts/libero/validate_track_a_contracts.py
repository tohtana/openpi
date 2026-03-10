#!/usr/bin/env python3
"""Validate Track A artifact contracts and phase handoffs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_HANDOFF_KEYS = {
    "phase",
    "status",
    "generated_at_utc",
    "git_commit",
    "inputs",
    "outputs",
    "notes",
}

REQUIRED_RUN_MANIFEST_KEYS = {
    "schema_version",
    "run_id",
    "command",
    "cwd",
    "env",
    "start_time_utc",
    "policy_path",
    "task_suite",
    "batch_size",
    "n_episodes",
    "gpu_id",
    "seed",
    "mujoco_gl",
    "retry_index",
}

REQUIRED_RESULT_KEYS = {
    "schema_version",
    "run_id",
    "exit_code",
    "status",
    "end_time_utc",
    "duration_sec",
    "successes",
    "episodes",
    "success_rate",
    "video_path",
    "stdout_log",
    "stderr_log",
    "error_message",
}

REQUIRED_PREFLIGHT_KEYS = {
    "schema_version",
    "checks",
    "gpu_count",
    "gpu_names",
    "display_env",
    "mujoco_gl_requested",
    "mujoco_gl_effective",
    "backend_attempts",
    "python_version",
    "lerobot_import_ok",
    "lerobot_eval_help_ok",
    "policy_resolve_ok",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expect_file(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"Missing required file: {path}")


def _expect_keys(path: Path, required: set[str], errors: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        errors.append(f"Missing required file: {path}")
        return None
    data = _read_json(path)
    for key in sorted(required):
        if key not in data:
            errors.append(f"Missing key '{key}' in {path}")
    return data


def _validate_phase1(root: Path, errors: list[str]) -> None:
    _expect_file(root / "configs/eval_matrix.csv", errors)
    _expect_file(root / "configs/run_schema.json", errors)
    _expect_keys(root / "artifacts/state/cli_capabilities.json", {"schema_version", "supported_flags", "version_text", "generated_at_utc"}, errors)
    handoff = _expect_keys(root / "artifacts/state/phase1_handoff.json", REQUIRED_HANDOFF_KEYS, errors)
    if handoff and handoff.get("status") != "pass":
        errors.append("phase1_handoff.json status must be 'pass'")


def _validate_phase2(root: Path, errors: list[str], warnings: list[str]) -> None:
    p1 = _expect_keys(root / "artifacts/state/phase1_handoff.json", REQUIRED_HANDOFF_KEYS, errors)
    if p1 and p1.get("status") != "pass":
        errors.append("Phase 2 requires phase1_handoff status=pass")

    preflight = _expect_keys(root / "artifacts/preflight/preflight.json", REQUIRED_PREFLIGHT_KEYS, errors)
    if preflight:
        if preflight.get("mujoco_gl_effective") not in {"egl", "glx", None}:
            errors.append("preflight.mujoco_gl_effective must be egl|glx|null")
        if preflight.get("gpu_count") != 8:
            warnings.append(f"Expected gpu_count=8 but found {preflight.get('gpu_count')}")

    handoff = _expect_keys(root / "artifacts/state/phase2_handoff.json", REQUIRED_HANDOFF_KEYS, errors)
    if handoff and handoff.get("status") not in {"pass", "blocked", "partial"}:
        errors.append("phase2_handoff status must be pass|blocked|partial")


def _validate_phase3(root: Path, errors: list[str]) -> None:
    p2 = _expect_keys(root / "artifacts/state/phase2_handoff.json", REQUIRED_HANDOFF_KEYS, errors)
    if p2 and p2.get("status") != "pass":
        errors.append("Phase 3 requires phase2_handoff status=pass")

    run_dir = root / "artifacts/runs/baseline_b1_e2"
    _expect_keys(run_dir / "run_manifest.json", REQUIRED_RUN_MANIFEST_KEYS, errors)
    _expect_keys(run_dir / "result.json", REQUIRED_RESULT_KEYS, errors)
    _expect_file(run_dir / "stdout.log", errors)
    _expect_file(run_dir / "stderr.log", errors)
    _expect_file(root / "artifacts/state/rows/baseline_b1_e2.json", errors)
    _expect_file(root / "artifacts/summary/baseline_summary.csv", errors)
    _expect_file(root / "artifacts/summary/baseline_summary.md", errors)
    _expect_keys(root / "artifacts/state/phase3_handoff.json", REQUIRED_HANDOFF_KEYS, errors)


def _validate_phase4(root: Path, errors: list[str]) -> None:
    p3 = _expect_keys(root / "artifacts/state/phase3_handoff.json", REQUIRED_HANDOFF_KEYS, errors)
    if p3 and p3.get("status") != "pass":
        errors.append("Phase 4 requires phase3_handoff status=pass")

    _expect_file(root / "artifacts/summary/all_runs.csv", errors)
    _expect_file(root / "artifacts/summary/all_runs.md", errors)
    _expect_file(root / "artifacts/summary/failures.csv", errors)
    _expect_keys(root / "artifacts/state/phase4_handoff.json", REQUIRED_HANDOFF_KEYS, errors)


def _validate_phase5(root: Path, errors: list[str]) -> None:
    for phase in ("phase1", "phase2", "phase3", "phase4", "phase5"):
        _expect_keys(root / f"artifacts/state/{phase}_handoff.json", REQUIRED_HANDOFF_KEYS, errors)

    _expect_file(root / "runbook.md", errors)
    _expect_file(root / "results.md", errors)


def validate_phase(phase: str, root: Path, strict: bool) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if phase == "phase1":
        _validate_phase1(root, errors)
    elif phase == "phase2":
        _validate_phase2(root, errors, warnings)
    elif phase == "phase3":
        _validate_phase3(root, errors)
    elif phase == "phase4":
        _validate_phase4(root, errors)
    elif phase == "phase5":
        _validate_phase5(root, errors)
    elif phase == "all":
        _validate_phase1(root, errors)
        _validate_phase2(root, errors, warnings)
        _validate_phase3(root, errors)
        _validate_phase4(root, errors)
        _validate_phase5(root, errors)
    else:
        errors.append(f"Unsupported phase: {phase}")

    if strict and warnings:
        errors.extend([f"strict-warning: {w}" for w in warnings])
    return errors, warnings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["phase1", "phase2", "phase3", "phase4", "phase5", "all"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--strict", action="store_true", help="Fail on warnings")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    errors, warnings = validate_phase(args.phase, args.root, args.strict)
    for warning in warnings:
        print(f"WARN: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
