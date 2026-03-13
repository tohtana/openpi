import time

import pytest
import torch

from openpi.training import profiling


def test_device_section_profiler_splits_allocated_sections_on_cpu():
    profiler = profiling.DeviceSectionProfiler(torch.device("cpu"))

    with profiler.record("vision_encoder_ms"):
        time.sleep(0.001)
    with profiler.record(allocation={"llm_ms": 3, "action_expert_ms": 1}):
        time.sleep(0.001)

    totals = profiler.finalize()

    assert totals["vision_encoder_ms"] > 0.0
    assert totals["llm_ms"] > totals["action_expert_ms"] > 0.0


def test_device_section_profiler_ignores_non_forced_sections_while_suspended():
    profiler = profiling.DeviceSectionProfiler(torch.device("cpu"))
    profiler.suspend()

    with profiler.record("llm_ms"):
        time.sleep(0.001)
    with profiler.record("backward_ms", force=True):
        time.sleep(0.001)

    totals = profiler.finalize()

    assert "llm_ms" not in totals
    assert totals["backward_ms"] > 0.0


def test_peak_memory_tracker_is_a_cpu_noop():
    tracker = profiling.PeakMemoryTracker(torch.device("cpu"))

    tracker.reset()

    assert tracker.snapshot() == {
        "peak_allocated_mb": 0.0,
        "peak_reserved_mb": 0.0,
    }


def test_peak_memory_tracker_reads_cuda_stats(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, torch.device]] = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: calls.append(("reset", device)),
    )
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: calls.append(("sync", device)),
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 3 * 1024**2)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 5 * 1024**2)

    device = torch.device("cuda:0")
    tracker = profiling.PeakMemoryTracker(device)

    tracker.reset()
    snapshot = tracker.snapshot()

    assert calls == [("reset", device), ("sync", device)]
    assert snapshot == {
        "peak_allocated_mb": 3.0,
        "peak_reserved_mb": 5.0,
    }
