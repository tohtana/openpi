# Action Plan: OpenPI LIBERO Fine-Tuning Example

## Objective
Execute one reproducible OpenPI LIBERO fine-tuning example (`compute_norm_stats` then `train`) using `pi05_libero`, capture exact commands, and document artifact locations and blockers.

## Scope
- In scope: preflight checks, normalization stats run, bounded training run, artifact verification, run record.
- Out of scope: full convergence training, hyperparameter sweeps, multi-dataset benchmarking.

## Deliverables
- `tasks/20260219-openpi-libero-finetune-example/run_record.md` with exact commands, timestamps, artifact paths, and observed outcomes.
- Verified norm-stats artifacts and at least one checkpoint/log directory path.
- Updated TODO tracking remains unchecked until implementation completes.

## Assumptions
- Workspace root: `/home/ray/default/test_openpi`.
- Required runtime/tooling is available (`uv`, CUDA/PyTorch stack, dataset access).
- Large artifacts and caches are written under `/mnt/local_storage/`.

## Phase 1: Preflight Validation

### Tasks
- Confirm config `pi05_libero` exists in `openpi/src/openpi/training/config.py`.
- Confirm command sequence in `openpi/README.md` (`scripts/compute_norm_stats.py`, then `scripts/train.py`).
- Validate runtime prerequisites:
  - `uv --version`
  - dataset path/access for LIBERO
  - writable checkpoint/output target under `/mnt/local_storage/`
- Define bounded training parameters before execution (short run cap, fixed experiment name).

### Exit Criteria
- Commands, config name, and output locations are resolved and documented.
- Any missing dependency/path is recorded with a mitigation.

## Phase 2: Normalization Stats Run

### Tasks
- Execute from `openpi/`:
  - `uv run scripts/compute_norm_stats.py --config-name pi05_libero`
- Capture stdout/stderr to a log file in `tasks/20260219-openpi-libero-finetune-example/`.
- Verify produced norm-stats artifact paths and basic readability.

### Exit Criteria
- Norm-stats command exits `0`.
- Artifact path(s) are recorded and accessible.

## Phase 3: Bounded Fine-Tuning Run

### Tasks
- Launch bounded training with explicit experiment name and overwrite-safe behavior.
- Ensure outputs/checkpoints are directed to `/mnt/local_storage/`.
- Capture logs and monitor for startup health (data loading, model init, first steps).
- Stop at predetermined short-run bound and preserve outputs.

### Exit Criteria
- Training process starts successfully and writes logs/checkpoints.
- At least one checkpoint/log artifact path is confirmed.

## Phase 4: Documentation and Verification

### Tasks
- Create `tasks/20260219-openpi-libero-finetune-example/run_record.md` containing:
  - Environment snapshot (date, commit, key env vars/paths)
  - Exact commands run
  - Artifact paths
  - Duration and outcome
  - Blockers + mitigation used
- Cross-check results against TODO verification checklist.

### Exit Criteria
- `run_record.md` is complete and reproducible.
- Remaining blockers (if any) are clearly actionable.

## Risks and Mitigations
- Dataset access failure: verify path/permissions first; fail fast in Phase 1.
- OOM or long runtime: enforce bounded run params and smaller batch/step limits.
- Artifact sprawl in home/project: force output/cache paths to `/mnt/local_storage/`.

## Verification Checklist
- `uv run scripts/compute_norm_stats.py --config-name pi05_libero` was attempted; bounded retry with `--max-frames 8192` completed successfully due HF rate-limits.
- Training starts and writes checkpoints/logs to configured output path.
- `tasks/20260219-openpi-libero-finetune-example/run_record.md` captures commands and artifacts.
- Any docs deviations are explicitly documented with rationale.

## Progress
- Created: Executed preflight checks, bounded norm-stats run (`--max-frames 8192`), bounded `pi05_libero` training run (2 steps), and captured artifacts/logs in `tasks/20260219-openpi-libero-finetune-example/`.
- Issue: Full `compute_norm_stats` command hit HuggingFace Hub 429 rate limits mid-download; bounded retry used to complete this TODO run.
- Next action: None (completed)
