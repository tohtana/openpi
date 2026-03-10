# Track A LeRobot pi0 LIBERO Eval

## Goal
- [ ] Validate Track A from `initial_plan.md` by running `lerobot-eval` with a pretrained pi0 flow policy on LIBERO in headless mode on an 8xH100 machine and capturing reproducible results.

## Plan
- [x] Step 1: Prepare environment and dependencies for LeRobot LIBERO evaluation.
  - [x] Substep: Create a Python 3.10 env (`vla_pi0`) and verify all 8 H100 GPUs are visible (`nvidia-smi`).
  - [x] Substep: Clone `lerobot` and install with `pip install -e ".[libero]"`.
  - [x] Substep: Configure offscreen rendering (headless, no monitor/X display) for the host before running eval.
- [x] Step 2: Run a baseline evaluation with a small episode count.
  - [x] Substep: Execute a headless baseline `lerobot-eval --policy.path=lerobot/pi0_libero_finetuned --env.type=libero --env.task=libero_object --eval.batch_size=1 --eval.n_episodes=2`.
- [x] Step 3: Scale evaluation and collect artifacts.
  - [x] Substep: Increase episode count and batch size based on available GPU/CPU resources, targeting efficient use of 8xH100.
  - [x] Substep: Save final commands, suite name, episode count, success rate, and log/video paths in a downloadable directory.

## Progress
- Created: Re-ran Step 1 environment provisioning on 2026-02-19 (rebuilt `vla_pi0`, reinstalled editable `lerobot[libero]`, regenerated preflight setup artifacts, and re-verified 8xH100 visibility via `nvidia-smi`).
- Updated: Hardened `scripts/libero/setup_track_a_env.sh` to enforce `cmake<4` before and after `lerobot[libero]` install (with final version guard) and added repeatable rebuild + recovery instructions to `tasks/20260210-track-a-lerobot-pi0-libero/runbook.md`.
- Updated: Confirmed access to gated model `google/paligemma-3b-pt-224` and removed the prior auth blocker; patched wrappers for non-interactive LIBERO init (`~/.libero/config.yaml`), headless `glx` via auto-`xvfb-run`, and PI0-compatible transformers pin (`fix/lerobot_openpi`) during env setup.
- Updated: Added an explicit eval dataflow explainer (sim observations -> policy -> actions -> metrics/videos) and concrete `libero_object` task-spec list to `tasks/20260210-track-a-lerobot-pi0-libero/results.md`.
- Updated: Completed baseline rerun on 2026-02-20 with `--mujoco-gl glx` via `xvfb-run`; `baseline_b1_e2` now exits `0` and reports `pc_success=65.0` over 20 aggregated episodes (13 successes, 7 failures) in `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs/baseline_b1_e2/stdout.log`.
- Updated: Completed scaled sweep rows `run_s1`..`run_s8` (all pass, exit code `0`) and regenerated aggregate summaries in `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/summary/`.
- Updated: Final contract validation passed after refreshing handoff markers: `conda run -n vla_pi0 python scripts/libero/validate_track_a_contracts.py --phase all --root tasks/20260210-track-a-lerobot-pi0-libero`.
- Next action: None (track complete).

## Verification
- [x] `lerobot-eval` completes without runtime errors on `libero_object`.
- [x] Eval runs without an attached monitor and without requiring an active GUI session.
- [x] Results include suite name, episode count, success rate, and artifact locations.
- [x] Video artifacts are written to a known path for later download from this environment.
- [x] A concise runbook of exact commands used for the successful run is documented.
