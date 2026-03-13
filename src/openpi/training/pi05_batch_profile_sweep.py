from __future__ import annotations

import csv
from pathlib import Path
import re
import statistics

import matplotlib.pyplot as plt


_PROFILE_LINE_RE = re.compile(
    r"profile_ms\[(?P<timings>[^\]]+)\]\s+"
    r"profile_counts\[(?P<counts>[^\]]+)\]"
    r"(?:\s+profile_mem\[(?P<memory>[^\]]+)\])?"
)
_TIMING_ALIASES = {
    "data": "data_loading_ms",
    "h2d": "host_to_device_ms",
    "prep": "preprocess_ms",
    "vision": "vision_encoder_ms",
    "llm": "llm_ms",
    "action": "action_expert_ms",
    "backward": "backward_ms",
    "optim": "optimizer_ms",
    "other": "other_ms",
    "total": "step_total_ms",
}
_OOM_PATTERNS = (
    "cuda out of memory",
    "torch.outofmemoryerror",
    "cudnn_status_alloc_failed",
    "cuda error: out of memory",
)
_TIMING_SERIES = {
    "entire": "step_total_ms",
    "vision": "vision_encoder_ms",
    "llm": "llm_ms",
    "action": "action_expert_ms",
}


def parse_scalar_block(block: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in block.split():
        key, raw_value = item.split("=", 1)
        values[key] = float(raw_value)
    return values


def parse_profile_lines(text: str) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for match in _PROFILE_LINE_RE.finditer(text):
        record = {
            _TIMING_ALIASES.get(key, key): value
            for key, value in parse_scalar_block(match.group("timings")).items()
        }
        record.update(parse_scalar_block(match.group("counts")))
        memory_block = match.group("memory")
        if memory_block is not None:
            record.update(parse_scalar_block(memory_block))
        records.append(record)
    return records


def summarize_training_profile(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        raise ValueError("Expected at least one parsed training profile record.")

    summary = {
        f"{module}_ms": statistics.median(record[column] for record in records)
        for module, column in _TIMING_SERIES.items()
    }
    summary["entire_peak_allocated_mb"] = max(record.get("peak_allocated_mb", 0.0) for record in records)
    summary["entire_peak_reserved_mb"] = max(record.get("peak_reserved_mb", 0.0) for record in records)
    summary["profile_sample_count"] = float(len(records))
    return summary


def is_cuda_oom(text: str) -> bool:
    normalized = text.lower()
    return any(pattern in normalized for pattern in _OOM_PATTERNS)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_timing_curves(rows: list[dict[str, object]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: int(row["batch_size"]))
    batch_sizes = [int(row["batch_size"]) for row in sorted_rows]
    outputs: list[Path] = []

    combined_path = output_dir / "timing_curves.png"
    plt.figure(figsize=(10, 6))
    for module in _TIMING_SERIES:
        plt.plot(batch_sizes, [float(row[f"{module}_ms"]) for row in sorted_rows], marker="o", label=module)
    plt.xlabel("Batch Size")
    plt.ylabel("Iteration Time (ms)")
    plt.title("PI0.5 Packed Profiling Timing Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(combined_path, dpi=200)
    plt.close()
    outputs.append(combined_path)

    for module in _TIMING_SERIES:
        output_path = output_dir / f"{module}_timing_curve.png"
        plt.figure(figsize=(8, 5))
        plt.plot(batch_sizes, [float(row[f"{module}_ms"]) for row in sorted_rows], marker="o")
        plt.xlabel("Batch Size")
        plt.ylabel("Iteration Time (ms)")
        plt.title(f"{module.capitalize()} Timing Curve")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        outputs.append(output_path)

    return outputs
