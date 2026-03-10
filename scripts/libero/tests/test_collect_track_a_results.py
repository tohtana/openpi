from __future__ import annotations

import csv
from pathlib import Path

from scripts.libero.collect_track_a_results import collect_results
from scripts.libero.collect_track_a_results import parse_success_metrics
from scripts.libero.collect_track_a_results import write_csv
from scripts.libero.collect_track_a_results import write_failures
from scripts.libero.collect_track_a_results import write_markdown


FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_success_metrics_extracts_values() -> None:
    stdout = FIXTURES / "runs/pass_run/stdout.log"
    successes, failures, rate = parse_success_metrics(stdout)
    assert successes == 1
    assert failures == 1
    assert rate == 0.5


def test_collect_results_and_summary_outputs(tmp_path: Path) -> None:
    runs_root = FIXTURES / "runs"
    results = collect_results(runs_root)
    assert len(results) == 2

    by_id = {r.run_id: r for r in results}
    assert by_id["pass_run"].status == "pass"
    assert by_id["pass_run"].success_rate == 0.5
    assert by_id["fail_run"].status == "fail"
    assert by_id["fail_run"].exit_code == 124

    csv_out = tmp_path / "summary.csv"
    md_out = tmp_path / "summary.md"
    failures_out = tmp_path / "failures.csv"

    write_csv(results, csv_out)
    write_markdown(results, md_out)
    write_failures(results, failures_out)

    assert csv_out.exists()
    assert md_out.exists()
    assert failures_out.exists()

    with csv_out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {r["run_id"] for r in rows} == {"pass_run", "fail_run"}

    with failures_out.open(newline="", encoding="utf-8") as f:
        failures = list(csv.DictReader(f))
    assert len(failures) == 1
    assert failures[0]["run_id"] == "fail_run"
