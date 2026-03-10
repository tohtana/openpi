# Track A Results

## Final Status

- Status: **Complete**
- Current phase: **Phase 5 complete**
- Blockers addressed:
  - Gated model access for `google/paligemma-3b-pt-224` approved and verified.
  - Non-interactive LIBERO config bootstrap added to setup/run scripts.
  - Headless `glx` fallback via auto-`xvfb-run` added to eval runner.
  - PI0-compatible `transformers` branch (`fix/lerobot_openpi`) pinned in env setup.

## Environment / Preflight

- Task suite: `libero_object`
- Policy path: `lerobot/pi0_libero_finetuned`
- GPUs detected: `8x NVIDIA H100 80GB HBM3`
- Selected backend: `glx` (with `xvfb-run` when `DISPLAY` is unset)
- Preflight artifact: `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight/preflight.json`

## Baseline Run (Completed)

- Run id: `baseline_b1_e2`
- Episodes requested: `2`
- Exit code: `0` (completed on 2026-02-20)
- Aggregated episodes (suite-wide): `20` (10 tasks x 2 episodes)
- Successes: `13`
- Failures: `7`
- Success rate: `0.65` (`65.0%`)
- Result artifact: `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs/baseline_b1_e2/result.json`
- Stdout log: `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs/baseline_b1_e2/stdout.log`
- Video root: `outputs/eval/2026-02-20/09-44-25_libero_pi0/videos/`

## Scaled Sweep (Completed)

- Matrix rows: `run_s1`..`run_s8`
- Batch size / episodes per row: `2 / 16`
- Run status: `8/8 pass`, all with `exit_code=0`
- Mean success rate across scaled rows: `0.7625` (`76.25%`)
- Total successes/failures across scaled rows: `97 / 31`
- Per-run artifacts: `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs/run_s*/`
- Aggregate summary:
  - `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/summary/all_runs.csv`
  - `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/summary/all_runs.md`
  - `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/summary/failures.csv`

## Eval Dataflow (What The Model Receives)

The evaluation command runs a closed-loop simulation rollout, not random inputs.

1. LIBERO simulator resets a task from suite `libero_object`.
2. Simulator renders camera observations (for this setup: agent view + wrist camera).
3. Simulator provides robot state features (joint/gripper/end-effector state).
4. Task context from the suite/spec is used for the episode.
5. Policy `lerobot/pi0_libero_finetuned` receives these observations and predicts actions.
6. Predicted actions are applied in simulation and the environment steps forward.
7. Success metrics, logs, and optional video artifacts are written to the run directory.

Task suite used by this run (`libero_object`) contains 10 pick-and-place tasks:
- `pick_up_the_alphabet_soup_and_place_it_in_the_basket`
- `pick_up_the_cream_cheese_and_place_it_in_the_basket`
- `pick_up_the_salad_dressing_and_place_it_in_the_basket`
- `pick_up_the_bbq_sauce_and_place_it_in_the_basket`
- `pick_up_the_ketchup_and_place_it_in_the_basket`
- `pick_up_the_tomato_sauce_and_place_it_in_the_basket`
- `pick_up_the_butter_and_place_it_in_the_basket`
- `pick_up_the_milk_and_place_it_in_the_basket`
- `pick_up_the_chocolate_pudding_and_place_it_in_the_basket`
- `pick_up_the_orange_juice_and_place_it_in_the_basket`

LIBERO path roles in this workflow:
- `bddl_files`: task/goal specifications.
- `init_files`: initial states for episode starts.
- `assets`: simulation resources (object/scene/robot model assets).
- `datasets`: offline data location (not required for this pure eval rollout).

## Current Artifacts

- Baseline summary CSV: `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/summary/baseline_summary.csv`
- All-runs summary CSV: `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/summary/all_runs.csv`
- Failures CSV: `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/summary/failures.csv`
- Phase handoffs:
  - `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/phase1_handoff.json`
  - `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/phase2_handoff.json`
  - `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/phase3_handoff.json`
  - `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/phase4_handoff.json`
  - `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/phase5_handoff.json`

## Required Next Step

None (track complete).
