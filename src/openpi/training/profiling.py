from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import time

import torch


_BYTES_PER_MEBIBYTE = float(1024**2)


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

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._use_cuda_events = device.type == "cuda" and torch.cuda.is_available()
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

        if self._use_cuda_events:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            try:
                yield
            finally:
                end_event.record()
                self._cuda_events.append((name, start_event, end_event, normalized_allocation))
            return

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
