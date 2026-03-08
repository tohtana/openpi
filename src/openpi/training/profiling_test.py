import time

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
