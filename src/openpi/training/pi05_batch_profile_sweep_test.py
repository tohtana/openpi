from pathlib import Path

from openpi.training import pi05_batch_profile_sweep as sweep


def test_parse_profile_lines_reads_timings_counts_and_memory():
    log_text = """
    13:00:56.977 [I] step=29 loss=0.1023 lr=1.50e-07 grad_norm=1.81 time=0.9s
    profile_ms[data=7.8 h2d=1.6 prep=1.7 vision=52.0 llm=49.5 action=15.6 backward=362.7 optim=376.4 other=20.2 total=887.7]
    profile_counts[vision_ms_per_camera=26.0 prefix_token_count=527.2 vision_token_count=512.0 prompt_token_count=15.2]
    profile_mem[peak_allocated_mb=34567.0 peak_reserved_mb=40123.0]
    """.replace("\n", " ")

    records = sweep.parse_profile_lines(log_text)

    assert records == [
        {
            "action_expert_ms": 15.6,
            "backward_ms": 362.7,
            "data_loading_ms": 7.8,
            "host_to_device_ms": 1.6,
            "llm_ms": 49.5,
            "optimizer_ms": 376.4,
            "other_ms": 20.2,
            "peak_allocated_mb": 34567.0,
            "peak_reserved_mb": 40123.0,
            "prefix_token_count": 527.2,
            "preprocess_ms": 1.7,
            "prompt_token_count": 15.2,
            "step_total_ms": 887.7,
            "vision_encoder_ms": 52.0,
            "vision_ms_per_camera": 26.0,
            "vision_token_count": 512.0,
        }
    ]


def test_summarize_training_profile_uses_median_and_max_memory():
    records = [
        {
            "step_total_ms": 100.0,
            "vision_encoder_ms": 10.0,
            "llm_ms": 20.0,
            "action_expert_ms": 30.0,
            "peak_allocated_mb": 1000.0,
            "peak_reserved_mb": 1200.0,
        },
        {
            "step_total_ms": 200.0,
            "vision_encoder_ms": 40.0,
            "llm_ms": 50.0,
            "action_expert_ms": 60.0,
            "peak_allocated_mb": 1500.0,
            "peak_reserved_mb": 1600.0,
        },
        {
            "step_total_ms": 300.0,
            "vision_encoder_ms": 70.0,
            "llm_ms": 80.0,
            "action_expert_ms": 90.0,
            "peak_allocated_mb": 1400.0,
            "peak_reserved_mb": 1700.0,
        },
    ]

    summary = sweep.summarize_training_profile(records)

    assert summary == {
        "action_ms": 60.0,
        "entire_ms": 200.0,
        "entire_peak_allocated_mb": 1500.0,
        "entire_peak_reserved_mb": 1700.0,
        "llm_ms": 50.0,
        "profile_sample_count": 3.0,
        "vision_ms": 40.0,
    }


def test_is_cuda_oom_matches_expected_signatures():
    assert sweep.is_cuda_oom("RuntimeError: CUDA out of memory while allocating tensor")
    assert sweep.is_cuda_oom("torch.OutOfMemoryError: CUDA error: out of memory")
    assert not sweep.is_cuda_oom("ValueError: dataset path is missing")


def test_plot_timing_curves_writes_expected_files(tmp_path: Path):
    outputs = sweep.plot_timing_curves(
        [
            {"batch_size": 1, "entire_ms": 10.0, "vision_ms": 2.0, "llm_ms": 3.0, "action_ms": 1.0},
            {"batch_size": 2, "entire_ms": 20.0, "vision_ms": 4.0, "llm_ms": 6.0, "action_ms": 2.0},
        ],
        tmp_path,
    )

    assert [path.name for path in outputs] == [
        "timing_curves.png",
        "entire_timing_curve.png",
        "vision_timing_curve.png",
        "llm_timing_curve.png",
        "action_timing_curve.png",
    ]
    assert all(path.exists() for path in outputs)
