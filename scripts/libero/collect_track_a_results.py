#!/usr/bin/env python3
"""Collect and summarize Track A run artifacts."""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any


@dataclass
class EvalRunResult:
    run_id: str
    task_suite: str
    batch_size: int
    n_episodes: int
    gpu_id: int
    success_rate: float | None
    successes: int | None
    failures: int | None
    exit_code: int
    status: str
    stdout_log: str
    stderr_log: str
    video_path: str | None


def load_run_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_success_metrics(stdout_log: Path) -> tuple[int | None, int | None, float | None]:
    if not stdout_log.exists():
        return None, None, None

    text = stdout_log.read_text(encoding="utf-8", errors="replace")

    successes: int | None = None
    episodes: int | None = None
    success_rate: float | None = None

    m = re.findall(r"# successes:\s*(\d+)\s*\(([-+]?\d+(?:\.\d+)?)%\)", text)
    if m:
        s, p = m[-1]
        successes = int(s)
        success_rate = float(p) / 100.0

    m = re.findall(r"Total episodes:\s*(\d+)", text)
    if m:
        episodes = int(m[-1])

    m = re.findall(r"Total success rate:\s*([-+]?\d+(?:\.\d+)?)", text)
    if m:
        r = float(m[-1])
        success_rate = r if r <= 1.0 else r / 100.0

    if successes is None and success_rate is not None and episodes:
        successes = int(round(success_rate * episodes))

    m = re.search(r"Overall Aggregated Metrics:\s*\n(\{.*?\})", text, re.S)
    if m:
        try:
            overall = ast.literal_eval(m.group(1))
            if success_rate is None and "pc_success" in overall:
                success_rate = float(overall["pc_success"]) / 100.0
            if episodes is None and "n_episodes" in overall:
                episodes = int(overall["n_episodes"])
            if successes is None and success_rate is not None and episodes:
                successes = int(round(success_rate * episodes))
        except Exception:
            pass

    failures: int | None = None
    if successes is not None and episodes is not None:
        failures = max(0, episodes - successes)

    return successes, failures, success_rate


def collect_results(runs_root: Path) -> list[EvalRunResult]:
    results: list[EvalRunResult] = []
    if not runs_root.exists():
        return results

    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        manifest_path = run_dir / "run_manifest.json"
        result_path = run_dir / "result.json"

        if not manifest_path.exists() and not result_path.exists():
            continue

        manifest: dict[str, Any] = {}
        result: dict[str, Any] = {}
        if manifest_path.exists():
            manifest = load_run_manifest(manifest_path)
        if result_path.exists():
            result = load_result(result_path)

        run_id = str(manifest.get("run_id") or result.get("run_id") or run_dir.name)
        task_suite = str(manifest.get("task_suite", ""))
        batch_size = int(manifest.get("batch_size", 0) or 0)
        n_episodes = int(manifest.get("n_episodes", result.get("episodes", 0)) or 0)
        gpu_id = int(manifest.get("gpu_id", 0) or 0)
        exit_code = int(result.get("exit_code", 1) if result else 1)
        status = str(result.get("status", "missing") if result else "missing")

        stdout_log = str(result.get("stdout_log") or manifest.get("stdout_log") or (run_dir / "stdout.log"))
        stderr_log = str(result.get("stderr_log") or manifest.get("stderr_log") or (run_dir / "stderr.log"))

        success_rate = result.get("success_rate") if result else None
        successes = result.get("successes") if result else None
        episodes = result.get("episodes") if result else None

        if success_rate is not None:
            success_rate = float(success_rate)
        if successes is not None:
            successes = int(successes)
        if episodes is not None:
            episodes = int(episodes)

        parsed_successes, parsed_failures, parsed_rate = parse_success_metrics(Path(stdout_log))
        if successes is None:
            successes = parsed_successes
        if success_rate is None:
            success_rate = parsed_rate

        failures: int | None = None
        if successes is not None and episodes is not None:
            failures = max(0, episodes - successes)
        if parsed_failures is not None and (
            failures is None or (episodes is not None and successes is not None and successes > episodes)
        ):
            failures = parsed_failures

        video_path = result.get("video_path") if result else None
        if video_path is not None:
            video_path = str(video_path).replace("\r", " ").replace("\n", " ").strip()

        results.append(
            EvalRunResult(
                run_id=run_id,
                task_suite=task_suite,
                batch_size=batch_size,
                n_episodes=n_episodes,
                gpu_id=gpu_id,
                success_rate=success_rate,
                successes=successes,
                failures=failures,
                exit_code=exit_code,
                status=status,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                video_path=video_path,
            )
        )
    return results


def write_csv(results: list[EvalRunResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "task_suite",
        "batch_size",
        "n_episodes",
        "gpu_id",
        "success_rate",
        "successes",
        "failures",
        "exit_code",
        "status",
        "stdout_log",
        "stderr_log",
        "video_path",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(asdict(item))


def write_markdown(results: list[EvalRunResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Track A Eval Summary",
        "",
        "| run_id | suite | batch | episodes | successes | failures | success_rate | status | exit_code |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for r in results:
        sr = "" if r.success_rate is None else f"{r.success_rate:.4f}"
        succ = "" if r.successes is None else str(r.successes)
        fail = "" if r.failures is None else str(r.failures)
        lines.append(
            f"| {r.run_id} | {r.task_suite} | {r.batch_size} | {r.n_episodes} | {succ} | {fail} | {sr} | {r.status} | {r.exit_code} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_failures(results: list[EvalRunResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_id", "status", "exit_code"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            if r.status != "pass" or r.exit_code != 0:
                writer.writerow({"run_id": r.run_id, "status": r.status, "exit_code": r.exit_code})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    parser.add_argument("--failures-csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    results = collect_results(args.runs_root)
    write_csv(results, args.summary_csv)
    write_markdown(results, args.summary_md)
    if args.failures_csv:
        write_failures(results, args.failures_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
