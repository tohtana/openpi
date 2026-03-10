# Run Record: OpenPI LIBERO Fine-Tuning Example

## Environment Snapshot
- Date (UTC): 2026-02-20T21:43:39Z
- Workspace: `/home/ray/default/test_openpi`
- Commit: `57b6dc7b41f06b16bb716d548e6b08c657161258`
- Tooling: `uv 0.9.17`
- Key env vars used:
  - `HF_HOME=/mnt/local_storage/huggingface`
  - `HF_HUB_CACHE=/mnt/local_storage/huggingface/hub`
  - `XDG_CACHE_HOME=/mnt/local_storage/.cache`
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (training)

## Commands Run
1. Preflight checks (config/docs/prereqs):
```bash
rg --line-number "pi05_libero|compute_norm_stats|scripts/train.py|libero" openpi/README.md openpi/src/openpi/training/config.py
cd openpi && uv run python -V
```

2. Full norm-stats attempt (documented command; hit HF rate-limits):
```bash
cd openpi
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```
Log: `tasks/20260219-openpi-libero-finetune-example/compute_norm_stats.log`

3. Retry attempts for mitigation / cache reuse:
```bash
cd openpi
HF_HUB_DISABLE_XET=1 uv run scripts/compute_norm_stats.py --config-name pi05_libero
```
Log: `tasks/20260219-openpi-libero-finetune-example/compute_norm_stats_retry_no_xet.log`

```bash
cd openpi
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```
Log: `tasks/20260219-openpi-libero-finetune-example/compute_norm_stats_retry_after_cooldown.log`

4. Completed bounded norm-stats run (deviation to complete task under quota constraints):
```bash
cd openpi
uv run scripts/compute_norm_stats.py --config-name pi05_libero --max-frames 8192
```
Log: `tasks/20260219-openpi-libero-finetune-example/compute_norm_stats_bounded.log`

5. Completed bounded training run:
```bash
cd openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero \
  --exp-name todo_impl_bounded_20260220 \
  --overwrite \
  --num-train-steps 2 \
  --save-interval 1 \
  --log-interval 1 \
  --no-wandb-enabled \
  --checkpoint-base-dir /mnt/local_storage/experiments/openpi_checkpoints
```
Log: `tasks/20260219-openpi-libero-finetune-example/train_bounded.log`

## Artifacts
- Norm stats file:
  - `openpi/assets/pi05_libero/physical-intelligence/libero/norm_stats.json`
- Checkpoint directory:
  - `/mnt/local_storage/experiments/openpi_checkpoints/pi05_libero/todo_impl_bounded_20260220/1`
  - Size observed: `41G`
- Training step metrics observed in log:
  - `Step 0: grad_norm=0.4948, loss=0.0879, param_norm=1802.3865`
  - `Step 1: grad_norm=0.4343, loss=0.0826, param_norm=1802.3865`

## Outcome
- Preflight validation: completed.
- Norm stats: completed with bounded run; full documented command was attempted but rate-limited.
- Training: completed bounded startup/step/checkpoint validation for `pi05_libero`.
- Reproducibility artifacts and logs are saved under `tasks/20260219-openpi-libero-finetune-example/`.

## Blockers and Mitigations
- Blocker: HuggingFace Hub `429 Too Many Requests` while downloading `physical-intelligence/libero` during full `compute_norm_stats`.
- Mitigation:
  - Reused cached dataset downloads.
  - Retried after cooldown.
  - Completed this TODO with bounded `--max-frames 8192` norm-stats execution, then bounded training.

