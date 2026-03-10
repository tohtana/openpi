# Action Plan: Track A LeRobot pi0 LIBERO Eval

## Goal
Run a reproducible, headless `lerobot-eval` workflow for `lerobot/pi0_libero_finetuned` on `libero_object` in this 8xH100 environment, then publish machine-checkable artifacts (commands, logs, videos, and success-rate summary) under `tasks/20260210-track-a-lerobot-pi0-libero/`.

This plan is optimized for orchestrated AI subagents: every phase has explicit handoff files, deterministic gate checks, and minimal overlap in writable paths to avoid merge/race conflicts.

## Implementation Phases
### Phase 1: Automation Scaffold and Contracts
- Scope: add small, reviewable automation entrypoints and schemas before any expensive GPU runs.
- Complexity: Medium.
- Dependencies: none.
- Session boundary: one agent session creates scripts, tests, config templates, and no GPU execution.
- Handoff in: `todo/20260210-track-a-lerobot-pi0-libero.md`, this plan.
- Handoff out:
  - Executable wrappers in `scripts/libero/` with `--help` and `--dry-run`.
  - CLI capability snapshot at `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/cli_capabilities.json` (resolved from `lerobot-eval --help` and `--version`).
  - Schema + matrix templates in `tasks/20260210-track-a-lerobot-pi0-libero/configs/`.
  - Parser/unit tests that pass without GPU.
  - `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/phase1_handoff.json`.
- Work package:
1. Create `scripts/libero/track_a_common.sh` for shared strict-mode helpers and consistent logging.
2. Create `scripts/libero/discover_track_a_cli.sh` to capture available `lerobot-eval` flags/options and write `cli_capabilities.json`.
3. Create setup/preflight/run/sweep/collect/validate scripts with explicit CLI contracts and non-zero exit behavior.
4. Add schema and matrix templates, plus parser fixtures/tests.
5. Add a phase handoff marker with generated-at timestamp, git commit, and produced file list.

### Phase 2: Environment Provisioning and Headless Preflight
- Scope: create/activate `vla_pi0`, install LeRobot + LIBERO extras, and verify offscreen rendering.
- Complexity: Medium.
- Dependencies: Phase 1 scripts and schema contracts.
- Session boundary: one agent session runs setup + preflight and produces only preflight artifacts.
- Handoff in:
  - `artifacts/state/phase1_handoff.json` with `status="pass"`.
  - `artifacts/state/cli_capabilities.json`.
  - `tasks/20260210-track-a-lerobot-pi0-libero/configs/eval_matrix.csv`.
  - `tasks/20260210-track-a-lerobot-pi0-libero/configs/run_schema.json`.
- Handoff out:
  - `artifacts/preflight/preflight.json`, `nvidia_smi.txt`, `pip_freeze.txt`, and setup logs.
  - Updated `eval_matrix.csv` only if resource constraints require smaller defaults.
  - `artifacts/state/phase2_handoff.json` with pass/fail checks and selected backend.
- Work package:
1. Run setup script with deterministic cache/env vars (`HF_HOME`, `PYTHONUNBUFFERED`, `TZ=UTC`).
2. Capture GPU inventory, CUDA driver/runtime, Python package snapshot, and whether `DISPLAY` is unset.
3. Run preflight with `MUJOCO_GL=egl`; automatically retry once with `MUJOCO_GL=glx` if EGL fails.
4. Record final backend (`egl|glx`) and blocker reason if both backends fail.
5. If preflight is blocked, write `phase2_handoff.json` with `status="blocked"` and stop downstream phases.

### Phase 3: Baseline Evaluation (Small Episode Count)
- Scope: run 2-episode baseline (`batch_size=1`) and validate end-to-end command -> artifacts -> parsed summary.
- Complexity: Medium.
- Dependencies: Phase 2 preflight status must be `pass`.
- Session boundary: one agent session executes exactly one baseline run and parses results.
- Handoff in: Phase 2 preflight artifacts and chosen rendering backend (`mujoco_gl_effective`).
- Handoff out:
  - Baseline run directory with manifest/log/result.
  - Baseline summary CSV/Markdown rows.
  - Per-row state marker: `artifacts/state/rows/baseline_b1_e2.json`.
  - `artifacts/state/phase3_handoff.json` containing run_id and exit status.
- Work package:
1. Resolve backend dynamically from preflight report (do not hard-code `egl`).
2. Execute wrapper for `run_id=baseline_b1_e2` with pinned seed/backend and bounded timeout.
3. Persist command, env snapshot, stdout/stderr, and resolved video path(s).
4. Parse metrics into normalized structured outputs; preserve parser warnings when fields are missing.
5. Write row state marker for orchestrator visibility.

### Phase 4: Scaled Evaluation Across 8xH100
- Scope: run bounded evaluation matrix and aggregate all runs with deterministic rerun support.
- Complexity: Large.
- Dependencies: Phase 3 baseline must complete with parseable output.
- Session boundary: one agent session per matrix shard (recommended shard size: <=4 rows) plus one aggregation session.
- Handoff in:
  - `artifacts/state/phase3_handoff.json` with baseline metadata.
  - Finalized matrix and selected backend.
  - Existing row state markers in `artifacts/state/rows/` for resume decisions.
- Handoff out:
  - Per-run artifacts for all enabled rows.
  - Row state markers for every attempted row.
  - Aggregate summaries (`all_runs.csv`, `all_runs.md`, `failures.csv`).
  - `artifacts/state/phase4_handoff.json` with completed/failed row counts.
- Work package:
1. Dispatch matrix rows in shards with explicit GPU pinning (`CUDA_VISIBLE_DEVICES` + row `gpu_id`).
2. Skip `enabled=0`; do not mutate successful run directories on rerun.
3. Retry failed rows once with conservative batch size override recorded in manifest.
4. Ensure only the aggregation session writes shared summary files to avoid cross-agent write races.
5. Aggregate metrics and unresolved failures into machine-readable outputs.

### Phase 5: Final Runbook and TODO Closure Artifacts
- Scope: publish concise reproducibility docs and completion evidence for `$todo-impl`.
- Complexity: Small.
- Dependencies: Phases 3-4 artifacts and summaries.
- Session boundary: one agent session writes docs and updates tracking files only.
- Handoff in: all prior phase handoff markers + summary outputs.
- Handoff out:
  - `runbook.md` and `results.md` with exact successful commands and caveats.
  - Updated progress sections in `todo/20260210-track-a-lerobot-pi0-libero.md` and this plan.
  - `artifacts/state/phase5_handoff.json` and final completion note.
- Work package:
1. Document exact commands that produced accepted baseline/scaled results.
2. Publish final result table (episodes, successes, success rate, backend, artifact path).
3. Record unresolved caveats, failed rows (if any), and deterministic rerun command.
4. Run final contract validation before marking TODO complete.
5. Update TODO tracking fields only after all acceptance criteria are met.

## File-Level Change Map
### Phase 1
- Create `scripts/libero/track_a_common.sh`: shared shell helpers (`require_cmd`, `ensure_dir`, `run_logged`, `write_json`, `fail`).
- Create `scripts/libero/discover_track_a_cli.sh`: captures `lerobot-eval` capabilities used by downstream wrappers.
- Create `scripts/libero/setup_track_a_env.sh`: idempotent setup for `vla_pi0`, LeRobot install, and environment snapshots.
- Create `scripts/libero/preflight_track_a.py`: preflight checks and JSON report emitter.
- Create `scripts/libero/run_track_a_eval.sh`: single-run wrapper for `lerobot-eval` with run manifest output.
- Create `scripts/libero/sweep_track_a_eval.sh`: matrix launcher with shard support, GPU assignment, skip/resume, retry-once policy.
- Create `scripts/libero/collect_track_a_results.py`: parse run artifacts/logs and emit CSV/Markdown summaries.
- Create `scripts/libero/validate_track_a_contracts.py`: validates handoff markers and artifact schema keys per phase.
- Create `scripts/libero/tests/test_collect_track_a_results.py`: parser/unit tests from synthetic logs.
- Create `scripts/libero/tests/test_validate_track_a_contracts.py`: contract-validator unit tests with fixtures.
- Create `scripts/libero/tests/fixtures/`: synthetic stdout/manifests/results/handoff fixtures for parser and validator tests.
- Create `tasks/20260210-track-a-lerobot-pi0-libero/configs/eval_matrix.csv`: baseline + scaled run definitions.
- Create `tasks/20260210-track-a-lerobot-pi0-libero/configs/run_schema.json`: schema for per-run manifest/result JSON.
- Create `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/cli_capabilities.json`: discovered `lerobot-eval` CLI features.
- Create `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/phase1_handoff.json`: phase completion marker.

### Phase 2
- Create `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight/preflight.json`: machine-readable check outcomes.
- Create `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight/nvidia_smi.txt`: GPU inventory evidence.
- Create `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight/pip_freeze.txt`: dependency snapshot.
- Create `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight/setup.log`: setup execution log.
- Modify `tasks/20260210-track-a-lerobot-pi0-libero/configs/eval_matrix.csv`: optional conservative tuning after preflight only.
- Create `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/phase2_handoff.json`: phase completion marker.

### Phase 3
- Create `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs/baseline_b1_e2/run_manifest.json`: command/env/parameters for baseline.
- Create `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs/baseline_b1_e2/stdout.log`: raw eval stdout.
- Create `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs/baseline_b1_e2/stderr.log`: raw eval stderr.
- Create `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs/baseline_b1_e2/result.json`: parsed metrics + exit code.
- Create `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/rows/baseline_b1_e2.json`: row status marker for orchestrator resume.
- Create `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/summary/baseline_summary.csv`: baseline rollup row.
- Create `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/summary/baseline_summary.md`: baseline human-readable summary.
- Create `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/phase3_handoff.json`: phase completion marker.

### Phase 4
- Create `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs/<run_id>/...`: per-run manifest/log/result for every scaled row.
- Create `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/rows/<run_id>.json`: row status marker (`pending|running|pass|fail`).
- Create `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/summary/all_runs.csv`: aggregated structured results.
- Create `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/summary/all_runs.md`: human-readable summary table.
- Create `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/summary/failures.csv`: failed/incomplete rows for replay.
- Create `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/phase4_handoff.json`: phase completion marker.

### Phase 5
- Create `tasks/20260210-track-a-lerobot-pi0-libero/runbook.md`: exact setup/eval commands that succeeded.
- Create `tasks/20260210-track-a-lerobot-pi0-libero/results.md`: final success-rate report with artifact paths.
- Modify `todo/20260210-track-a-lerobot-pi0-libero.md`: update TODO progress fields.
- Modify `todo/TODO.md`: mark main checkbox complete only when all acceptance criteria are met.
- Modify `tasks/20260210-track-a-lerobot-pi0-libero/plan.md`: check phase boxes and append progress lines.
- Create `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/phase5_handoff.json`: final completion marker.

## Interface Contracts
### Existing Codebase Contracts to Respect
- `openpi/examples/libero/main.py` defines `Args` and `eval_libero(args: Args)`; keep task-suite names aligned (`libero_object`, `libero_spatial`, `libero_goal`, `libero_10`, `libero_90`).
- `openpi/examples/libero/main.py` uses `OffScreenRenderEnv`; preflight must validate the same offscreen assumptions and backend fallback (`egl` -> `glx`).
- `openpi/examples/libero/compose.yml` documents `MUJOCO_GL` and `MUJOCO_EGL_DEVICE_ID`; wrappers must expose equivalent env knobs instead of hard-coding.
- Keep existing repository behavior unchanged: no edits to `openpi/examples/libero/*` unless a blocker is documented in the phase handoff marker.

### New Script Contracts (Phase Handshake)
1. `scripts/libero/discover_track_a_cli.sh`
- CLI:
  - `--artifacts-dir <path>` required
  - `--python-bin <path>` default `python`
  - `--dry-run`
- Exit behavior:
  - `0` discovery complete
  - `2` bad CLI args
  - `3` `lerobot-eval` not found/invokable
- Outputs:
  - `cli_capabilities.json` with `schema_version`, `lerobot_eval_help_text_path`, `supported_flags`, `version_text`, `generated_at_utc`.
  - `lerobot_eval_help.txt` raw capture for debugging drift.

2. `scripts/libero/setup_track_a_env.sh`
- CLI:
  - `--env-name <name>` default `vla_pi0`
  - `--python-version <ver>` default `3.10`
  - `--lerobot-dir <path>` default `/mnt/local_storage/src/lerobot`
  - `--hf-home <path>` default `/mnt/local_storage/huggingface`
  - `--artifacts-dir <path>` required
  - `--dry-run`
- Exit behavior:
  - `0` success
  - `2` bad CLI args
  - `3` missing dependency/tool
  - `4` installation failure
- Outputs: `setup.log`, `pip_freeze.txt`, `env_snapshot.json`.

3. `scripts/libero/preflight_track_a.py`
- Dataclass:
```python
@dataclass
class PreflightConfig:
    task_suite: str
    policy_path: str
    mujoco_gl: str
    gpu_ids: list[int]
    artifacts_dir: Path
```
- Functions:
  - `def run_preflight(cfg: PreflightConfig) -> dict[str, Any]: ...`
  - `def write_preflight_report(report: dict[str, Any], out_path: Path) -> None: ...`
- CLI contract:
  - Emits `preflight.json` with `schema_version`, backend attempts, and check records.
  - Returns non-zero if required checks fail.

4. `scripts/libero/run_track_a_eval.sh`
- CLI:
  - `--run-id <id>` required
  - `--policy-path <hf_or_local_path>` required
  - `--task-suite <suite>` required
  - `--batch-size <int>` required
  - `--n-episodes <int>` required
  - `--gpu-id <int>` required
  - `--seed <int>` default `7`
  - `--mujoco-gl <egl|glx>` required (must come from preflight decision)
  - `--artifacts-root <path>` required
  - `--timeout-secs <int>` default `10800`
  - `--retry-index <int>` default `0`
  - `--allow-overwrite`
  - `--dry-run`
- Behavior:
  - Fails if run directory exists without `--allow-overwrite`.
  - Always writes `run_manifest.json` before launch and `result.json` after completion/failure.
  - Sets `CUDA_VISIBLE_DEVICES` to `gpu_id` for child process.
  - Writes row marker at `artifacts/state/rows/<run_id>.json`.

5. `scripts/libero/sweep_track_a_eval.sh`
- Input CSV columns (exact): `run_id,task_suite,batch_size,n_episodes,gpu_id,seed,mujoco_gl,policy_path,enabled`.
- CLI:
  - `--matrix <path>` required
  - `--artifacts-root <path>` required
  - `--status-dir <path>` required
  - `--max-parallel <int>` default `8`
  - `--rows <comma-separated-run-ids>` optional shard filter
  - `--resume`
  - `--dry-run`
- Behavior:
  - Skips `enabled=0`.
  - Continue-on-error and record failures.
  - `--resume` mode reruns only rows in `failures.csv` or rows without successful `result.json`.
  - Emits deterministic replay command at end.
  - Does not write aggregate summary files (aggregation is a separate step/session).

6. `scripts/libero/collect_track_a_results.py`
- Dataclass:
```python
@dataclass
class EvalRunResult:
    run_id: str
    task_suite: str
    batch_size: int
    n_episodes: int
    gpu_id: int
    success_rate: float | None
    successes: int | None
    failures: int | None
    exit_code: int
    status: str
    stdout_log: str
    stderr_log: str
    video_path: str | None
```
- Functions:
  - `def load_run_manifest(path: Path) -> dict[str, Any]: ...`
  - `def load_result(path: Path) -> dict[str, Any]: ...`
  - `def parse_success_metrics(stdout_log: Path) -> tuple[int | None, int | None, float | None]: ...`
  - `def collect_results(runs_root: Path) -> list[EvalRunResult]: ...`
  - `def write_csv(results: list[EvalRunResult], out_path: Path) -> None: ...`
  - `def write_markdown(results: list[EvalRunResult], out_path: Path) -> None: ...`

7. `scripts/libero/validate_track_a_contracts.py`
- CLI:
  - `--phase <phase1|phase2|phase3|phase4|phase5|all>` required
  - `--root <tasks/20260210-track-a-lerobot-pi0-libero>` required
  - `--strict` (fail on warnings)
- Behavior:
  - Validates required file existence, required JSON keys, and status transitions.
  - Exits non-zero when current phase or dependency markers are invalid.

### Artifact Data Contracts
- `run_manifest.json` required keys:
  - `schema_version`, `run_id`, `command`, `cwd`, `env`, `start_time_utc`, `policy_path`, `task_suite`, `batch_size`, `n_episodes`, `gpu_id`, `seed`, `mujoco_gl`, `retry_index`.
- `result.json` required keys:
  - `schema_version`, `run_id`, `exit_code`, `status`, `end_time_utc`, `duration_sec`, `successes`, `episodes`, `success_rate`, `video_path`, `stdout_log`, `stderr_log`, `error_message`.
- `preflight.json` required keys:
  - `schema_version`, `checks`, `gpu_count`, `gpu_names`, `display_env`, `mujoco_gl_requested`, `mujoco_gl_effective`, `backend_attempts`, `python_version`, `lerobot_import_ok`, `lerobot_eval_help_ok`, `policy_resolve_ok`.
- `artifacts/state/rows/<run_id>.json` required keys:
  - `schema_version`, `run_id`, `status`, `retry_index`, `gpu_id`, `started_at_utc`, `ended_at_utc`, `result_path`, `notes`.
- `phase*_handoff.json` required keys:
  - `phase`, `status`, `generated_at_utc`, `git_commit`, `inputs`, `outputs`, `notes`.
  - `status` allowed values: `pass`, `blocked`, `partial`.

## Testing Strategy
Run commands from repo root with `set -euo pipefail`.

### Phase 1 Tests
1. Shell syntax:
```bash
bash -n scripts/libero/track_a_common.sh
bash -n scripts/libero/discover_track_a_cli.sh
bash -n scripts/libero/setup_track_a_env.sh
bash -n scripts/libero/run_track_a_eval.sh
bash -n scripts/libero/sweep_track_a_eval.sh
```
2. Python static sanity:
```bash
python -m py_compile scripts/libero/preflight_track_a.py scripts/libero/collect_track_a_results.py scripts/libero/validate_track_a_contracts.py
```
3. Unit tests:
```bash
pytest -q scripts/libero/tests/test_collect_track_a_results.py scripts/libero/tests/test_validate_track_a_contracts.py
```
4. CLI capability discovery:
```bash
bash scripts/libero/discover_track_a_cli.sh --artifacts-dir tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state
python - <<'PY'
import json
from pathlib import Path
d = json.loads(Path('tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/cli_capabilities.json').read_text())
for k in ['schema_version','supported_flags','version_text','generated_at_utc']:
    assert k in d, k
PY
```
5. Dry-run integration:
```bash
bash scripts/libero/run_track_a_eval.sh --run-id dryrun --policy-path lerobot/pi0_libero_finetuned --task-suite libero_object --batch-size 1 --n-episodes 2 --gpu-id 0 --seed 7 --mujoco-gl egl --artifacts-root tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs --dry-run
test -f tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs/dryrun/run_manifest.json
```
6. Schema/header checks:
```bash
python - <<'PY'
import csv, json
from pathlib import Path
root = Path('tasks/20260210-track-a-lerobot-pi0-libero/configs')
cols = next(csv.reader((root / 'eval_matrix.csv').open()))
expected = ['run_id','task_suite','batch_size','n_episodes','gpu_id','seed','mujoco_gl','policy_path','enabled']
assert cols == expected, (cols, expected)
schema = json.loads((root / 'run_schema.json').read_text())
assert 'run_manifest' in schema and 'result' in schema
PY
```
7. Phase gate validation:
```bash
python scripts/libero/validate_track_a_contracts.py --phase phase1 --root tasks/20260210-track-a-lerobot-pi0-libero
```

### Phase 2 Tests
1. Environment setup:
```bash
bash scripts/libero/setup_track_a_env.sh --env-name vla_pi0 --python-version 3.10 --lerobot-dir /mnt/local_storage/src/lerobot --hf-home /mnt/local_storage/huggingface --artifacts-dir tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight
```
2. GPU visibility:
```bash
nvidia-smi -L | tee tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight/nvidia_smi.txt
test "$(wc -l < tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight/nvidia_smi.txt)" -eq 8
```
3. Preflight with backend fallback:
```bash
python scripts/libero/preflight_track_a.py --task-suite libero_object --policy-path lerobot/pi0_libero_finetuned --mujoco-gl egl --gpu-ids 0,1,2,3,4,5,6,7 --artifacts-dir tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight
```
4. Preflight JSON contract validation:
```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight/preflight.json')
d = json.loads(p.read_text())
for k in ['schema_version','checks','gpu_count','display_env','mujoco_gl_effective','backend_attempts','lerobot_import_ok','lerobot_eval_help_ok','policy_resolve_ok']:
    assert k in d, k
assert d['gpu_count'] == 8
assert d['lerobot_import_ok'] is True
assert d['lerobot_eval_help_ok'] is True
assert d['policy_resolve_ok'] is True
assert d['mujoco_gl_effective'] in ('egl','glx')
PY
```
5. Phase gate validation:
```bash
python scripts/libero/validate_track_a_contracts.py --phase phase2 --root tasks/20260210-track-a-lerobot-pi0-libero
```

### Phase 3 Tests
1. Resolve backend from preflight:
```bash
BACKEND="$(python - <<'PY'
import json
from pathlib import Path
d = json.loads(Path('tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight/preflight.json').read_text())
print(d['mujoco_gl_effective'])
PY
)"
test "$BACKEND" = "egl" -o "$BACKEND" = "glx"
```
2. Baseline execution:
```bash
bash scripts/libero/run_track_a_eval.sh --run-id baseline_b1_e2 --policy-path lerobot/pi0_libero_finetuned --task-suite libero_object --batch-size 1 --n-episodes 2 --gpu-id 0 --seed 7 --mujoco-gl "$BACKEND" --artifacts-root tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs
```
3. Baseline parsing:
```bash
python scripts/libero/collect_track_a_results.py --runs-root tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs --summary-csv tasks/20260210-track-a-lerobot-pi0-libero/artifacts/summary/baseline_summary.csv --summary-md tasks/20260210-track-a-lerobot-pi0-libero/artifacts/summary/baseline_summary.md
```
4. Baseline artifact contract validation:
```bash
python - <<'PY'
import json
from pathlib import Path
root = Path('tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs/baseline_b1_e2')
manifest = json.loads((root / 'run_manifest.json').read_text())
result = json.loads((root / 'result.json').read_text())
for k in ['run_id','command','task_suite','batch_size','n_episodes','gpu_id','mujoco_gl','retry_index']:
    assert k in manifest, k
for k in ['run_id','exit_code','status','episodes','success_rate','duration_sec','stdout_log','stderr_log']:
    assert k in result, k
assert result['run_id'] == 'baseline_b1_e2'
PY
```
5. Phase gate validation:
```bash
python scripts/libero/validate_track_a_contracts.py --phase phase3 --root tasks/20260210-track-a-lerobot-pi0-libero
```

### Phase 4 Tests
1. Sweep execution (single-session full matrix):
```bash
bash scripts/libero/sweep_track_a_eval.sh --matrix tasks/20260210-track-a-lerobot-pi0-libero/configs/eval_matrix.csv --artifacts-root tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs --status-dir tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/rows --max-parallel 8 --resume
```
2. Sweep execution (optional shard mode for multiple agents):
```bash
bash scripts/libero/sweep_track_a_eval.sh --matrix tasks/20260210-track-a-lerobot-pi0-libero/configs/eval_matrix.csv --artifacts-root tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs --status-dir tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/rows --rows run_s1,run_s2,run_s3 --max-parallel 3 --resume
```
3. Aggregation (single writer session only):
```bash
python scripts/libero/collect_track_a_results.py --runs-root tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs --summary-csv tasks/20260210-track-a-lerobot-pi0-libero/artifacts/summary/all_runs.csv --summary-md tasks/20260210-track-a-lerobot-pi0-libero/artifacts/summary/all_runs.md --failures-csv tasks/20260210-track-a-lerobot-pi0-libero/artifacts/summary/failures.csv
```
4. Sweep output validation:
```bash
test -f tasks/20260210-track-a-lerobot-pi0-libero/artifacts/summary/all_runs.csv
test -f tasks/20260210-track-a-lerobot-pi0-libero/artifacts/summary/failures.csv
python - <<'PY'
import csv
from pathlib import Path
rows = list(csv.DictReader(Path('tasks/20260210-track-a-lerobot-pi0-libero/artifacts/summary/all_runs.csv').open()))
matrix = list(csv.DictReader(Path('tasks/20260210-track-a-lerobot-pi0-libero/configs/eval_matrix.csv').open()))
enabled = {r['run_id'] for r in matrix if r['enabled'] == '1'}
recorded = {r['run_id'] for r in rows}
assert enabled.issubset(recorded), (enabled - recorded)
assert any(r['run_id'] != 'baseline_b1_e2' for r in rows), 'no scaled rows recorded'
PY
```
5. Phase gate validation:
```bash
python scripts/libero/validate_track_a_contracts.py --phase phase4 --root tasks/20260210-track-a-lerobot-pi0-libero
```

### Phase 5 Tests
1. Runbook completeness checks:
```bash
rg -n "lerobot-eval|libero_object|success rate|artifacts|MUJOCO_GL|rerun" tasks/20260210-track-a-lerobot-pi0-libero/runbook.md tasks/20260210-track-a-lerobot-pi0-libero/results.md
```
2. TODO linkage checks:
```bash
rg -n "tasks/20260210-track-a-lerobot-pi0-libero/plan.md|Phase [1-5]|Next action" todo/20260210-track-a-lerobot-pi0-libero.md tasks/20260210-track-a-lerobot-pi0-libero/plan.md
```
3. Handoff marker completeness:
```bash
python - <<'PY'
import json
from pathlib import Path
root = Path('tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state')
for phase in ['phase1_handoff.json','phase2_handoff.json','phase3_handoff.json','phase4_handoff.json','phase5_handoff.json']:
    p = root / phase
    d = json.loads(p.read_text())
    for k in ['phase','status','generated_at_utc','git_commit','inputs','outputs','notes']:
        assert k in d, (phase, k)
    assert d['status'] in ('pass','blocked','partial')
PY
```
4. Final contract validation:
```bash
python scripts/libero/validate_track_a_contracts.py --phase all --root tasks/20260210-track-a-lerobot-pi0-libero --strict
```

## Risk Register
1. `lerobot-eval` CLI/metric output format may differ by version.
- Mitigation: Phase 1 writes `cli_capabilities.json` from actual local help/version; wrappers rely on discovered flags.
2. EGL offscreen rendering may fail on this host/driver stack.
- Mitigation: explicit fallback order (`egl` then `glx`) and backend recorded in preflight + each run manifest.
3. 8-GPU visibility can regress due to scheduler/container constraints.
- Mitigation: hard gate at Phase 2 (`gpu_count == 8`) and stop with blocker note if unmet.
4. HF/model downloads can be slow, interrupted, auth-gated, or quota-limited.
- Mitigation: cache in `/mnt/local_storage/huggingface`; preflight checks policy resolution; setup/preflight are idempotent and resume-safe.
5. Long sweeps can be interrupted mid-run.
- Mitigation: per-run immutable directories, row state markers, explicit `--resume`, and deterministic replay from `failures.csv`.
6. Artifact growth can exceed repository storage.
- Mitigation: keep heavy runtime outputs under `tasks/.../artifacts/` and document optional pruning in `runbook.md`.
7. Cross-agent write races in orchestrated execution (especially summaries/failure lists).
- Mitigation: row sessions write only per-run outputs + row marker files; one aggregation session owns shared summary writes.
8. OOM or hung eval process during scaled runs.
- Mitigation: wrapper-level timeout, single retry with reduced batch size, and explicit failure reason in `result.json`.
9. Migration concern: future switch from `lerobot/pi0_libero_finetuned` to a new checkpoint can silently change behavior.
- Mitigation: persist exact policy identifier in each manifest and in final `results.md`.

## Acceptance Criteria
### Phase 1
- All new shell wrappers pass `bash -n`.
- `pytest -q scripts/libero/tests/test_collect_track_a_results.py scripts/libero/tests/test_validate_track_a_contracts.py` exits `0`.
- `artifacts/state/cli_capabilities.json` exists and contains required keys.
- `eval_matrix.csv` header matches required columns exactly.
- `run_schema.json` includes both `run_manifest` and `result` sections.
- `artifacts/state/phase1_handoff.json` exists with required keys and `status="pass"`.

### Phase 2
- `test "$(wc -l < tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight/nvidia_smi.txt)" -eq 8` exits `0`.
- `artifacts/preflight/preflight.json` exists and validates required keys.
- `lerobot_import_ok=true`, `lerobot_eval_help_ok=true`, and `policy_resolve_ok=true` in preflight report.
- Effective backend (`mujoco_gl_effective`) is recorded (`egl` or `glx`).
- `artifacts/state/phase2_handoff.json` exists with `status="pass"` or `status="blocked"` and explicit notes.

### Phase 3
- Baseline run produces `run_manifest.json`, `stdout.log`, `stderr.log`, `result.json`, and row marker JSON.
- Baseline command uses backend from `preflight.json` (not a hard-coded value).
- `result.json` has non-null `episodes` and contains `status` + `exit_code`.
- Baseline summary files are generated (`baseline_summary.csv` and `baseline_summary.md`).
- `artifacts/state/phase3_handoff.json` exists with baseline run metadata.

### Phase 4
- Sweep writes per-run artifacts and per-row marker files for all enabled matrix rows.
- Aggregation writes `all_runs.csv`, `all_runs.md`, and `failures.csv`.
- Every enabled row appears in `all_runs.csv` (success or failure), with no silent omissions.
- Resume mode does not rerun rows that already have successful `result.json`.
- `artifacts/state/phase4_handoff.json` exists with completed/failed counts.

### Phase 5
- `runbook.md` includes exact setup/eval commands and rerun command for failures.
- `results.md` includes suite name, policy path, backend, episodes, success rate, and artifact locations.
- Progress fields updated in both `todo/20260210-track-a-lerobot-pi0-libero.md` and this plan.
- `todo/TODO.md` checkbox is marked complete only when all above criteria are met.
- `artifacts/state/phase5_handoff.json` exists with final status.
- `python scripts/libero/validate_track_a_contracts.py --phase all --root tasks/20260210-track-a-lerobot-pi0-libero --strict` exits `0`.

## Progress Tracking
- [x] Phase 1 complete: automation scaffold/contracts landed and `phase1_handoff.json` written.
- [x] Phase 2 complete: environment + headless preflight passed (or blocked with explicit reason) and `phase2_handoff.json` written.
- [x] Phase 3 complete: baseline eval parsed and `phase3_handoff.json` written.
- [x] Phase 4 complete: scaled sweep aggregated and `phase4_handoff.json` written.
- [x] Phase 5 complete: runbook/results/TODO updates finished and `phase5_handoff.json` written.

Update rule for each phase completion:
1. Check the corresponding box in this section.
2. Append one line to `## Progress` using this template:
   - `YYYY-MM-DDTHH:MM:SSZ | phase=<phaseN> | owner=<agent_id> | status=<pass|blocked|partial> | marker=<path> | next=<next_action>`
3. Keep prior progress lines (append-only; do not rewrite history).
4. Record marker path for orchestrator pickup.

## Progress
- Created: Detailed action plan at `tasks/20260210-track-a-lerobot-pi0-libero/plan.md`.
- Updated: Added explicit phase handoff contracts, schema/test gates, resume semantics, and progress update rules for orchestrated subagents.
- Updated: Added CLI capability discovery, row-level state handshakes, single-writer aggregation rules, dynamic backend gating, and a dedicated contract-validator script.
- 2026-02-10T22:36:16Z | phase=phase1 | owner=codex | status=pass | marker=tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/phase1_handoff.json | next=Execute Phase 2 setup/preflight.
- 2026-02-10T22:37:03Z | phase=phase2 | owner=codex | status=pass | marker=tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/phase2_handoff.json | next=Run baseline Phase 3 eval.
- 2026-02-10T22:42:37Z | phase=phase3 | owner=codex | status=blocked | marker=tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/phase3_handoff.json | next=Grant gated HF model access for google/paligemma-3b-pt-224 and rerun baseline.
- 2026-02-10T22:43:20Z | phase=phase4 | owner=codex | status=blocked | marker=tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/phase4_handoff.json | next=Unblock Phase 3, then execute sweep and aggregation.
- 2026-02-10T22:43:20Z | phase=phase5 | owner=codex | status=partial | marker=tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/phase5_handoff.json | next=After access is granted, rerun baseline/sweep and finalize TODO closure.
- 2026-02-19T17:52:13Z | phase=phase2 | owner=codex | status=pass | marker=tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight/env_snapshot.json | next=Rerun baseline eval once gated HF model access is granted.
- 2026-02-19T17:57:21Z | phase=phase2 | owner=codex | status=pass | marker=tasks/20260210-track-a-lerobot-pi0-libero/runbook.md | next=Use documented rebuild flow for future env resets before baseline reruns.
- 2026-02-19T18:00:00Z | phase=phase2 | owner=codex | status=pass | marker=scripts/libero/setup_track_a_env.sh | next=Use hardened setup script for repeat env rebuilds; it auto re-pins cmake<4.
- 2026-02-19T18:52:57Z | phase=phase3 | owner=codex | status=partial | marker=tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs/baseline_b1_e2/stderr.log | next=Replace interactive LIBERO init with non-interactive config setup in wrappers.
- 2026-02-19T18:56:05Z | phase=phase3 | owner=codex | status=partial | marker=tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs/baseline_b1_e2/stdout.log | next=Use glx+xvfb fallback and PI0-compatible transformers branch, then rerun baseline to completion.
- 2026-02-19T19:11:00Z | phase=phase2 | owner=codex | status=pass | marker=scripts/libero/setup_track_a_env.sh | next=Rebuild envs with pinned PI0 transformers + LIBERO config bootstrap to avoid recurring setup failures.
- 2026-02-19T19:20:00Z | phase=phase5 | owner=codex | status=partial | marker=tasks/20260210-track-a-lerobot-pi0-libero/results.md | next=Keep baseline/sweep pending, but preserve task-spec and eval dataflow documentation for handoff.
- 2026-02-20T17:56:15Z | phase=phase3 | owner=codex | status=pass | marker=tasks/20260210-track-a-lerobot-pi0-libero/artifacts/runs/baseline_b1_e2/result.json | next=Execute Phase 4 sweep (`run_s1`..`run_s8`) and aggregate final summaries.
- 2026-02-20T21:02:27Z | phase=phase4 | owner=codex | status=pass | marker=tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/phase4_handoff.json | next=Finalize Phase 5 docs/trackers and run full contract validation.
- 2026-02-20T21:02:27Z | phase=phase5 | owner=codex | status=pass | marker=tasks/20260210-track-a-lerobot-pi0-libero/artifacts/state/phase5_handoff.json | next=None (completed).
