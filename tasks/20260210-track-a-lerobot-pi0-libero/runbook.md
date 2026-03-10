# Track A Runbook

## Environment Setup

Use this sequence whenever the env is lost or needs to be rebuilt:

```bash
# Optional hard reset of the conda env:
conda env remove -n vla_pi0 -y || true

# Rebuild env + editable LeRobot install + setup artifacts.
bash scripts/libero/setup_track_a_env.sh \
  --env-name vla_pi0 \
  --python-version 3.10 \
  --lerobot-dir /mnt/local_storage/src/lerobot \
  --hf-home /mnt/local_storage/huggingface \
  --artifacts-dir tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight

# Verify GPU visibility and keep evidence for handoff.
nvidia-smi -L | tee tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight/nvidia_smi.txt

# Sanity checks for repeatability.
conda run -n vla_pi0 python -V
conda run -n vla_pi0 python -m pip show lerobot
conda run -n vla_pi0 cmake --version
conda run -n vla_pi0 python -c "from transformers.models.siglip import check; print(check.check_whether_transformers_replace_is_installed_correctly())"
```

Expected:
- `python -V` is `3.10.x`.
- `pip show lerobot` reports editable location `/mnt/local_storage/src/lerobot`.
- `cmake --version` shows major version `<4` (required by `egl_probe` build).
- `transformers` replacement check prints `True` (required for PI0 model load).
- `nvidia_smi.txt` contains 8 lines (GPU 0..7).

If setup fails:
```bash
# Common root cause: CMake mismatch in env.
conda install -n vla_pi0 -y "cmake<4"
conda run -n vla_pi0 python -m pip uninstall -y cmake || true

# Retry setup.
bash scripts/libero/setup_track_a_env.sh \
  --env-name vla_pi0 \
  --python-version 3.10 \
  --lerobot-dir /mnt/local_storage/src/lerobot \
  --hf-home /mnt/local_storage/huggingface \
  --artifacts-dir tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight
```

## Phase 1 Validation

```bash
bash scripts/libero/discover_track_a_cli.sh --artifacts-dir tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state
bash scripts/libero/run_track_a_eval.sh --run-id dryrun --policy-path lerobot/pi0_libero_finetuned --task-suite libero_object --batch-size 1 --n-episodes 2 --gpu-id 0 --seed 7 --mujoco-gl egl --artifacts-root tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs --dry-run
python scripts/libero/validate_track_a_contracts.py --phase phase1 --root tasks/20260210-track-a-lerobot-pi0-libero
```

## Phase 2 Preflight

```bash
nvidia-smi -L | tee tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight/nvidia_smi.txt
conda run -n vla_pi0 python scripts/libero/preflight_track_a.py \
  --task-suite libero_object \
  --policy-path lerobot/pi0_libero_finetuned \
  --mujoco-gl egl \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --artifacts-dir tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight
```

## Baseline Eval (Successful)

```bash
conda run -n vla_pi0 bash scripts/libero/run_track_a_eval.sh \
  --run-id baseline_b1_e2 \
  --policy-path lerobot/pi0_libero_finetuned \
  --task-suite libero_object \
  --batch-size 1 \
  --n-episodes 2 \
  --gpu-id 0 \
  --seed 7 \
  --mujoco-gl glx \
  --artifacts-root tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs \
  --timeout-secs 1800 \
  --allow-overwrite
```

Expected result:
- `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs/baseline_b1_e2/result.json` has `status: "pass"` and `exit_code: 0`.

## Scaled Sweep (Successful)

```bash
conda run -n vla_pi0 bash scripts/libero/sweep_track_a_eval.sh \
  --matrix tasks/20260210-track-a-lerobot-pi0-libero/configs/eval_matrix.csv \
  --artifacts-root tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs \
  --status-dir tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/rows \
  --rows run_s1,run_s2,run_s3,run_s4 \
  --max-parallel 4

conda run -n vla_pi0 bash scripts/libero/sweep_track_a_eval.sh \
  --matrix tasks/20260210-track-a-lerobot-pi0-libero/configs/eval_matrix.csv \
  --artifacts-root tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs \
  --status-dir tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/rows \
  --rows run_s5,run_s6,run_s7,run_s8 \
  --max-parallel 4

conda run -n vla_pi0 python scripts/libero/collect_track_a_results.py \
  --runs-root tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs \
  --summary-csv tasks/20260210-track-a-lerobot-pi0-libero/artifacts/summary/all_runs.csv \
  --summary-md tasks/20260210-track-a-lerobot-pi0-libero/artifacts/summary/all_runs.md \
  --failures-csv tasks/20260210-track-a-lerobot-pi0-libero/artifacts/summary/failures.csv
```

Expected result:
- Row markers `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/rows/run_s*.json` all show `status: "pass"`.
- `all_runs.md` reports `run_s1`..`run_s8` as `pass`.
