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


def test_device_section_profiler_emits_torch_profiler_annotations(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, str]] = []

    class _FakeRecord:
        def __init__(self, name: str) -> None:
            self._name = name

        def __enter__(self):
            calls.append(("enter", self._name))

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", self._name))
            return False

    monkeypatch.setattr(torch.profiler, "record_function", lambda name: _FakeRecord(name))

    profiler = profiling.DeviceSectionProfiler(
        torch.device("cpu"),
        enable_timings=False,
        enable_trace=True,
    )

    with profiler.record("vision_encoder_ms"):
        pass

    assert calls == [
        ("enter", "vision_encoder_ms"),
        ("exit", "vision_encoder_ms"),
    ]


def test_device_section_profiler_does_not_emit_trace_annotations_while_suspended(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []

    class _FakeRecord:
        def __init__(self, name: str) -> None:
            self._name = name

        def __enter__(self):
            calls.append(self._name)

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(torch.profiler, "record_function", lambda name: _FakeRecord(name))

    profiler = profiling.DeviceSectionProfiler(
        torch.device("cpu"),
        enable_timings=False,
        enable_trace=True,
    )
    profiler.suspend()

    with profiler.record("llm_ms"):
        pass
    with profiler.record("backward_ms", force=True):
        pass

    assert calls == ["backward_ms"]


def test_resolve_torch_profiler_schedule_reserves_one_internal_warmup_step():
    schedule = profiling.resolve_torch_profiler_schedule(warmup_steps=50, active_steps=3)

    assert schedule.wait_steps == 49
    assert schedule.warmup_steps == 1
    assert schedule.active_steps == 3


def test_resolve_torch_profiler_schedule_allows_zero_warmup():
    schedule = profiling.resolve_torch_profiler_schedule(warmup_steps=0, active_steps=3)

    assert schedule.wait_steps == 0
    assert schedule.warmup_steps == 0
    assert schedule.active_steps == 3


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
