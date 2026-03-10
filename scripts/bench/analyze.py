#!/usr/bin/env python3
"""Analyze benchmark results and generate comparison report.

Usage: python analyze.py --results-dir <path> --num-gpus 8

Reads all JSONL files from results-dir.
For each config, computes (excluding first 5 warmup iterations):
  - mean_step_time_ms, std_step_time_ms
  - tokens_per_sec = (global_batch_size * 1024) / mean_step_time_sec
  - tflops_per_gpu (from Megatron's native logging, averaged)
  - mfu = tflops_per_gpu / 989.0
  - peak_memory_gb
Outputs:
  1. Markdown table to stdout (for inclusion in report)
  2. CSV to <results-dir>/summary.csv
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def load_jsonl(filepath: str) -> list[dict]:
    """Load JSONL file into list of dicts."""
    entries = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def analyze_config(
    entries: list[dict],
    warmup: int = 5,
    seq_length: int = 1024,
) -> dict:
    """Analyze a single benchmark config's results."""
    if len(entries) <= warmup:
        steady = entries
    else:
        steady = entries[warmup:]

    if not steady:
        return {}

    step_times = [e["step_time_ms"] for e in steady if "step_time_ms" in e]
    tflops_list = [e["tflops_per_gpu"] for e in steady if "tflops_per_gpu" in e]
    # Peak memory from ALL entries (including warmup)
    memories = [e.get("memory_gb", 0) for e in entries if e.get("memory_gb", 0) > 0]

    mean_step = sum(step_times) / len(step_times) if step_times else 0
    std_step = (
        math.sqrt(sum((t - mean_step) ** 2 for t in step_times) / len(step_times))
        if len(step_times) > 1
        else 0
    )

    gbs = steady[0].get("global_batch_size", 8)
    tokens_per_sec = gbs * seq_length / (mean_step / 1000) if mean_step > 0 else 0

    mean_tflops = sum(tflops_list) / len(tflops_list) if tflops_list else 0
    mfu = mean_tflops / 989.0 if mean_tflops > 0 else 0
    peak_mem = max(memories) if memories else 0

    return {
        "mean_step_time_ms": round(mean_step, 1),
        "std_step_time_ms": round(std_step, 1),
        "tokens_per_sec": round(tokens_per_sec, 0),
        "tflops_per_gpu": round(mean_tflops, 1),
        "mfu_percent": round(mfu * 100, 2),
        "peak_memory_gb": round(peak_mem, 1),
        "num_steady": len(steady),
    }


def parse_config_name(filename: str) -> tuple[str, str]:
    """Extract framework and config ID from filename.

    Examples:
        bridge_bridge-ep8.jsonl -> ("Bridge", "ep8")
        swift_swift-tp2-ep8.jsonl -> ("SWIFT", "tp2-ep8")
    """
    stem = Path(filename).stem  # e.g. "bridge_bridge-ep8"
    if stem.startswith("bridge_"):
        config_id = stem.replace("bridge_bridge-", "").replace("bridge_", "")
        return "Bridge", config_id
    elif stem.startswith("swift_"):
        config_id = stem.replace("swift_swift-", "").replace("swift_", "")
        return "SWIFT", config_id
    else:
        return "Unknown", stem


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--results-dir", required=True, help="Directory with JSONL result files")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs (default: 8)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations to skip (default: 5)")
    parser.add_argument("--seq-length", type=int, default=1024, help="Sequence length (default: 1024)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    jsonl_files = sorted(results_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"ERROR: No JSONL files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Analyze each config
    rows = []
    for jsonl_file in jsonl_files:
        framework, config_id = parse_config_name(jsonl_file.name)
        entries = load_jsonl(str(jsonl_file))
        stats = analyze_config(entries, warmup=args.warmup, seq_length=args.seq_length)
        if stats:
            rows.append({
                "framework": framework,
                "config": config_id,
                **stats,
            })

    if not rows:
        print("ERROR: No valid results to analyze", file=sys.stderr)
        sys.exit(1)

    # Print Markdown table
    print("\n## Benchmark Results: Qwen3-VL-30B-A3B Full Fine-Tuning Throughput")
    print(f"\nHardware: {args.num_gpus}×H100 SXM5 80GB | Seq length: {args.seq_length} | "
          f"Activation checkpointing: ON")
    print()
    header = "| Framework | Config | Step Time (ms) | Tokens/sec | TFLOPS/GPU | MFU (%) | Peak Mem (GB) | N |"
    separator = "|-----------|--------|----------------|------------|-----------|---------|---------------|---|"
    print(header)
    print(separator)
    for r in rows:
        print(
            f"| {r['framework']:<9} | {r['config']:<6} | "
            f"{r['mean_step_time_ms']:>8.1f} ± {r['std_step_time_ms']:<5.1f} | "
            f"{r['tokens_per_sec']:>10.0f} | "
            f"{r['tflops_per_gpu']:>9.1f} | "
            f"{r['mfu_percent']:>5.2f}  | "
            f"{r['peak_memory_gb']:>13.1f} | "
            f"{r['num_steady']:>1d} |"
        )
    print()

    # Write CSV
    csv_path = results_dir / "summary.csv"
    fieldnames = ["framework", "config", "mean_step_time_ms", "std_step_time_ms",
                  "tokens_per_sec", "tflops_per_gpu", "mfu_percent", "peak_memory_gb", "num_steady"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written to {csv_path}")

    # Speed comparison
    if len(rows) >= 2:
        print("\n## Relative Performance")
        # Use first row as baseline
        baseline = rows[0]
        base_tps = baseline["tokens_per_sec"]
        for r in rows:
            if base_tps > 0:
                speedup = r["tokens_per_sec"] / base_tps
                print(f"  {r['framework']} {r['config']}: {speedup:.2f}x vs {baseline['framework']} {baseline['config']}")


if __name__ == "__main__":
    main()
