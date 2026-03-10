#!/usr/bin/env python3
"""Parse Megatron-Bridge training log and output JSONL with per-iteration metrics.

Usage: python parse_bridge_log.py <logfile> [--output <jsonl_file>]

Parses Bridge training log and outputs JSONL with fields:
  {"iteration": int, "step_time_ms": float, "loss": float, "lr": float,
   "memory_gb": float, "tflops_per_gpu": float}

Extracts TFLOPS from Megatron's native throughput logging.
Skips first 5 warmup iterations for summary statistics.
Prints summary to stdout:
  mean_step_time_ms, std_step_time_ms, tokens_per_sec, tflops_per_gpu, peak_memory_gb
"""

import argparse
import json
import math
import re
import sys


def parse_log(logfile: str) -> list[dict]:
    """Parse a Megatron-Bridge training log file.

    Bridge log format (single line per iteration):
    [datetime] iteration <N>/<total> | consumed samples: ... |
    elapsed time per iteration (ms): <float> | throughput per GPU (TFLOP/s/GPU): <float> |
    learning rate: <float> | global batch size: <int> | <loss_name>: <float> | ...
    """
    entries = []

    # Regex patterns for different fields
    iter_pat = re.compile(r"iteration\s+(\d+)")
    step_time_pat = re.compile(r"elapsed time per iteration \(ms\):\s+([\d.]+)")
    tflops_pat = re.compile(r"throughput per GPU \(TFLOP/s/GPU\):\s+([\d.]+)")
    lr_pat = re.compile(r"learning rate:\s+([\d.eE+-]+)")
    loss_pat = re.compile(r"(?:lm loss|loss):\s+([\d.eE+-]+)")
    gbs_pat = re.compile(r"global batch size:\s+(\d+)")

    # Memory from "Step Time" line (separate line)
    # Step Time : 2.34s GPU utilization: 45.2MODEL_TFLOP/s/GPU
    step_time_line_pat = re.compile(r"Step Time\s*:\s+([\d.]+)s\s+GPU utilization:\s+([\d.]+)")

    # Try to capture memory from CUDA memory summary if present
    memory_pat = re.compile(r"max_memory_allocated.*?([\d.]+)\s*GiB")
    # Bridge memory format: "mem-max-allocated-gigabytes: 44.195"
    bridge_mem_pat = re.compile(r"mem-max-allocated-gigabytes:\s+([\d.]+)")

    current_entry = {}

    with open(logfile) as f:
        for line in f:
            # Check for iteration log line
            iter_match = iter_pat.search(line)
            if iter_match:
                iteration = int(iter_match.group(1))

                step_time_match = step_time_pat.search(line)
                tflops_match = tflops_pat.search(line)
                lr_match = lr_pat.search(line)
                loss_match = loss_pat.search(line)
                gbs_match = gbs_pat.search(line)

                entry = {"iteration": iteration}

                if step_time_match:
                    entry["step_time_ms"] = float(step_time_match.group(1))
                if tflops_match:
                    entry["tflops_per_gpu"] = float(tflops_match.group(1))
                if lr_match:
                    entry["lr"] = float(lr_match.group(1))
                if loss_match:
                    entry["loss"] = float(loss_match.group(1))
                if gbs_match:
                    entry["global_batch_size"] = int(gbs_match.group(1))

                # Only add if we have at least step_time
                if "step_time_ms" in entry:
                    entries.append(entry)

            # Check for memory reporting (multiple formats)
            mem_match = memory_pat.search(line)
            if mem_match and entries:
                entries[-1]["memory_gb"] = float(mem_match.group(1))
            bridge_mem_match = bridge_mem_pat.search(line)
            if bridge_mem_match and entries:
                entries[-1]["memory_gb"] = float(bridge_mem_match.group(1))

    return entries


def compute_summary(entries: list[dict], warmup: int = 5, seq_length: int = 1024) -> dict:
    """Compute summary statistics from parsed entries, skipping warmup iterations."""
    if len(entries) <= warmup:
        steady = entries
    else:
        steady = entries[warmup:]

    if not steady:
        return {}

    step_times = [e["step_time_ms"] for e in steady if "step_time_ms" in e]
    tflops = [e["tflops_per_gpu"] for e in steady if "tflops_per_gpu" in e]
    # Peak memory from ALL entries (including warmup) since Bridge only logs it once
    memories = [e.get("memory_gb", 0) for e in entries if e.get("memory_gb", 0) > 0]

    mean_step = sum(step_times) / len(step_times) if step_times else 0
    std_step = math.sqrt(sum((t - mean_step) ** 2 for t in step_times) / len(step_times)) if len(step_times) > 1 else 0

    # tokens/sec: global_batch_size * seq_length / step_time_sec
    gbs = steady[0].get("global_batch_size", 8)
    tokens_per_sec = gbs * seq_length / (mean_step / 1000) if mean_step > 0 else 0

    mean_tflops = sum(tflops) / len(tflops) if tflops else 0
    peak_mem = max(memories) if memories else 0

    return {
        "num_iterations": len(entries),
        "num_steady_state": len(steady),
        "mean_step_time_ms": round(mean_step, 1),
        "std_step_time_ms": round(std_step, 1),
        "tokens_per_sec": round(tokens_per_sec, 0),
        "tflops_per_gpu": round(mean_tflops, 1),
        "mfu_percent": round(mean_tflops / 989.0 * 100, 2) if mean_tflops > 0 else 0,
        "peak_memory_gb": round(peak_mem, 1),
        "global_batch_size": gbs,
        "seq_length": seq_length,
    }


def main():
    parser = argparse.ArgumentParser(description="Parse Megatron-Bridge training log")
    parser.add_argument("logfile", help="Path to training log file")
    parser.add_argument("--output", "-o", help="Output JSONL file (default: <logfile>.jsonl)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations to skip (default: 5)")
    parser.add_argument("--seq-length", type=int, default=1024, help="Sequence length (default: 1024)")
    args = parser.parse_args()

    entries = parse_log(args.logfile)
    if not entries:
        print(f"ERROR: No training iterations found in {args.logfile}", file=sys.stderr)
        sys.exit(1)

    # Write JSONL
    output_file = args.output or args.logfile.replace(".log", ".jsonl")
    with open(output_file, "w") as f:
        for entry in entries:
            json.dump(entry, f)
            f.write("\n")
    print(f"Wrote {len(entries)} entries to {output_file}")

    # Print summary
    summary = compute_summary(entries, warmup=args.warmup, seq_length=args.seq_length)
    print(f"\n=== Summary (skip {args.warmup} warmup) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
