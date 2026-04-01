from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import pathlib
import time

import torch


_BYTES_PER_MEBIBYTE = float(1024**2)


@dataclasses.dataclass(frozen=True)
class TorchProfilerSchedule:
    wait_steps: int
    warmup_steps: int
    active_steps: int
    repeat: int = 1

    def build(self):
        return torch.profiler.schedule(
            wait=self.wait_steps,
            warmup=self.warmup_steps,
            active=self.active_steps,
            repeat=self.repeat,
        )


def resolve_torch_profiler_schedule(*, warmup_steps: int, active_steps: int) -> TorchProfilerSchedule:
    profiler_warmup_steps = 1 if warmup_steps > 0 else 0
    wait_steps = max(0, warmup_steps - profiler_warmup_steps)
    return TorchProfilerSchedule(
        wait_steps=wait_steps,
        warmup_steps=profiler_warmup_steps,
        active_steps=active_steps,
    )


@contextmanager
def record_torch_profile_section(name: str | None, *, enabled: bool) -> Iterator[None]:
    if not enabled or name is None:
        yield
        return

    with torch.profiler.record_function(name):
        yield


class TorchProfilerSession:
    """Owns the lifecycle of an optional torch.profiler session."""

    def __init__(
        self,
        profiler: torch.profiler.profile | None,
        *,
        schedule: TorchProfilerSchedule | None = None,
        trace_dir: pathlib.Path | None = None,
        worker_name: str | None = None,
    ) -> None:
        self._profiler = profiler
        self.schedule = schedule
        self.trace_dir = trace_dir
        self.worker_name = worker_name

    @property
    def enabled(self) -> bool:
        return self._profiler is not None

    def step(self) -> None:
        if self._profiler is not None:
            self._profiler.step()

    def close(self) -> None:
        if self._profiler is not None:
            self._profiler.stop()
            self._profiler = None


def create_torch_profiler_session(
    *,
    enabled: bool,
    device: torch.device,
    warmup_steps: int,
    active_steps: int,
    trace_dir: pathlib.Path,
    worker_name: str,
) -> TorchProfilerSession:
    if not enabled:
        return TorchProfilerSession(None)

    trace_dir.mkdir(parents=True, exist_ok=True)
    schedule = resolve_torch_profiler_schedule(warmup_steps=warmup_steps, active_steps=active_steps)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda" and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    profiler = torch.profiler.profile(
        activities=activities,
        schedule=schedule.build(),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir), worker_name=worker_name),
        profile_memory=False,
        record_shapes=False,
        with_stack=False,
    )
    profiler.start()
    return TorchProfilerSession(
        profiler,
        schedule=schedule,
        trace_dir=trace_dir,
        worker_name=worker_name,
    )


class WallClockTimer:
    """Accumulates wall-clock timing for host-side sections."""

    def __init__(self) -> None:
        self._totals_ms: dict[str, float] = defaultdict(float)

    @contextmanager
    def record(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._totals_ms[name] += (time.perf_counter() - start) * 1000.0

    def add_ms(self, name: str, value_ms: float) -> None:
        self._totals_ms[name] += value_ms

    def snapshot_and_reset(self) -> dict[str, float]:
        totals = dict(self._totals_ms)
        self._totals_ms.clear()
        return totals


class DeviceSectionProfiler:
    """Collects device timings with CUDA events and falls back to wall-clock on CPU."""

    def __init__(
        self,
        device: torch.device,
        *,
        enable_timings: bool = True,
        enable_trace: bool = False,
    ) -> None:
        self._device = device
        self._enable_timings = enable_timings
        self._enable_trace = enable_trace
        self._use_cuda_events = enable_timings and device.type == "cuda" and torch.cuda.is_available()
        self._recording_suspended = False
        self._totals_ms: dict[str, float] = defaultdict(float)
        self._cuda_events: list[
            tuple[str | None, torch.cuda.Event, torch.cuda.Event, dict[str, float] | None]
        ] = []

    def suspend(self) -> None:
        self._recording_suspended = True

    def resume(self) -> None:
        self._recording_suspended = False

    @contextmanager
    def record(
        self,
        name: str | None = None,
        *,
        allocation: Mapping[str, float] | None = None,
        force: bool = False,
    ) -> Iterator[None]:
        normalized_allocation = self._normalize_allocation(allocation)

        if not force and self._recording_suspended:
            yield
            return

        if name is None and normalized_allocation is None:
            raise ValueError("Either name or allocation must be provided for a timed section.")

        trace_context = record_torch_profile_section(name, enabled=self._enable_trace)

        if not self._enable_timings:
            with trace_context:
                yield
            return

        if self._use_cuda_events:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            with trace_context:
                start_event.record()
                try:
                    yield
                finally:
                    end_event.record()
                    self._cuda_events.append((name, start_event, end_event, normalized_allocation))
            return

        with trace_context:
            start = time.perf_counter()
            try:
                yield
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                self._record_elapsed(name, elapsed_ms, normalized_allocation)

    def add_ms(self, name: str, value_ms: float) -> None:
        self._totals_ms[name] += value_ms

    def finalize(self) -> dict[str, float]:
        if self._use_cuda_events:
            torch.cuda.synchronize(self._device)
            for name, start_event, end_event, allocation in self._cuda_events:
                elapsed_ms = start_event.elapsed_time(end_event)
                self._record_elapsed(name, elapsed_ms, allocation)
            self._cuda_events.clear()

        totals = dict(self._totals_ms)
        self._totals_ms.clear()
        self.resume()
        return totals

    def _record_elapsed(
        self,
        name: str | None,
        elapsed_ms: float,
        allocation: Mapping[str, float] | None,
    ) -> None:
        if allocation is not None:
            for bucket_name, share in allocation.items():
                self._totals_ms[bucket_name] += elapsed_ms * share
            return

        if name is None:
            raise ValueError("Cannot record elapsed time without a target bucket.")
        self._totals_ms[name] += elapsed_ms

    @staticmethod
    def _normalize_allocation(allocation: Mapping[str, float] | None) -> dict[str, float] | None:
        if allocation is None:
            return None

        total = sum(float(value) for value in allocation.values())
        if total <= 0:
            raise ValueError("Timing allocation must sum to a positive value.")

        return {name: float(value) / total for name, value in allocation.items()}


class PeakMemoryTracker:
    """Tracks peak CUDA memory and no-ops on CPU."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._enabled = device.type == "cuda" and torch.cuda.is_available()

    def reset(self) -> None:
        if self._enabled:
            torch.cuda.reset_peak_memory_stats(self._device)

    def snapshot(self) -> dict[str, float]:
        if not self._enabled:
            return {
                "peak_allocated_mb": 0.0,
                "peak_reserved_mb": 0.0,
            }

        torch.cuda.synchronize(self._device)
        return {
            "peak_allocated_mb": torch.cuda.max_memory_allocated(self._device) / _BYTES_PER_MEBIBYTE,
            "peak_reserved_mb": torch.cuda.max_memory_reserved(self._device) / _BYTES_PER_MEBIBYTE,
        }
