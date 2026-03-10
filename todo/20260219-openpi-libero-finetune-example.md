# Run OpenPI LIBERO Fine-Tuning Example

## Goal
- [x] Execute one documented openpi LIBERO fine-tuning example (`compute_norm_stats` + `train`) and capture reproducible commands and artifact locations.

## Plan
- [x] Step 1: Validate training prerequisites against openpi docs/config.
  - [x] Substep: Confirm chosen training config exists in `openpi/src/openpi/training/config.py` (target: `pi05_libero`).
  - [x] Substep: Confirm required command sequence in `openpi/README.md` (`scripts/compute_norm_stats.py` then `scripts/train.py`).
- [x] Step 2: Run normalization stats computation.
  - [x] Substep: Execute `uv run scripts/compute_norm_stats.py --config-name pi05_libero` (attempted; bounded fallback used due HF 429 limits).
  - [x] Substep: Verify norm-stats artifacts are produced and readable.
- [x] Step 3: Run a bounded fine-tuning example.
  - [x] Substep: Launch training with explicit experiment name and overwrite flag.
  - [x] Substep: For this task, use a bounded/short run configuration (or resume-safe limit) to validate pipeline execution without full convergence runtime.
  - [x] Substep: Verify checkpoint directory creation and log outputs.
- [x] Step 4: Document outcomes.
  - [x] Substep: Save exact commands, config used, and produced artifact paths under `tasks/20260219-openpi-libero-finetune-example/`.
  - [x] Substep: Record any blockers (dataset access, memory, dependency mismatch) and mitigation steps.

## Progress
- Created: Completed preflight checks, bounded norm-stats + bounded training runs, and added `tasks/20260219-openpi-libero-finetune-example/run_record.md` with commands/artifacts.
- Issue: Full `compute_norm_stats` command encountered HuggingFace Hub 429 rate limits; completed using bounded `--max-frames 8192` fallback.
- Next action: None (completed)

## Verification
- [x] `uv run scripts/compute_norm_stats.py --config-name pi05_libero` was attempted; bounded run completed successfully as a documented workaround.
- [x] Training command starts and writes checkpoints under configured checkpoint path.
- [x] A concise run record exists under `tasks/20260219-openpi-libero-finetune-example/` with commands, config, and artifact paths.
- [x] Any deviations from docs are explicitly captured with reason and workaround.
