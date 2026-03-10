# Track A Environment Setup (Repeatable)

This is the canonical, repeatable setup for TODO `20260210-track-a-lerobot-pi0-libero`.

Use this whenever `vla_pi0` is missing, broken, or you want a clean rebuild.

## Scope

- Creates/repairs conda env `vla_pi0` with Python 3.10.
- Clones/uses LeRobot at `/mnt/local_storage/src/lerobot`.
- Installs `lerobot[libero]` editable.
- Installs PI0-compatible `transformers` from `huggingface/transformers@fix/lerobot_openpi`.
- Avoids known `egl_probe` build issues by enforcing `cmake<4`.
- Writes non-interactive LIBERO config at `${LIBERO_CONFIG_PATH:-~/.libero}/config.yaml`.
- Writes setup artifacts under `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight/`.

## One-Time/Per-Rebuild Commands

Run from repo root:

```bash
# Optional: hard reset env for a fully clean rebuild.
conda env remove -n vla_pi0 -y || true

# Rebuild env and install dependencies.
bash scripts/libero/setup_track_a_env.sh \
  --env-name vla_pi0 \
  --python-version 3.10 \
  --lerobot-dir /mnt/local_storage/src/lerobot \
  --hf-home /mnt/local_storage/huggingface \
  --artifacts-dir tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight
```

## Verification Commands

```bash
# GPU visibility evidence (expect 8 lines: GPU 0..7).
nvidia-smi -L | tee tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight/nvidia_smi.txt

# Env/tooling checks.
conda run -n vla_pi0 python -V
conda run -n vla_pi0 python -m pip show lerobot
conda run -n vla_pi0 cmake --version
conda run -n vla_pi0 python -c "from transformers.models.siglip import check; print(check.check_whether_transformers_replace_is_installed_correctly())"
```

Expected:

- `python -V` shows `3.10.x`.
- `pip show lerobot` contains `Editable project location: /mnt/local_storage/src/lerobot`.
- `cmake --version` shows major version `<4`.
- `transformers` check prints `True`.
- `nvidia_smi.txt` has 8 GPU lines.

## Artifacts Produced

- `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight/setup.log`
- `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight/pip_freeze.txt`
- `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight/env_snapshot.json`
- `tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight/nvidia_smi.txt`

## Troubleshooting

If setup fails during `lerobot[libero]` install:

```bash
conda run -n vla_pi0 python -m pip uninstall -y cmake || true
conda install -n vla_pi0 -y --force-reinstall "cmake<4"

bash scripts/libero/setup_track_a_env.sh \
  --env-name vla_pi0 \
  --python-version 3.10 \
  --lerobot-dir /mnt/local_storage/src/lerobot \
  --hf-home /mnt/local_storage/huggingface \
  --artifacts-dir tasks/20260210-track-a-lerobot-pi0-libero/artifacts/preflight
```

If baseline eval fails with PI0 transformer compatibility:

```bash
conda run -n vla_pi0 python -m pip install --upgrade \
  "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"
```

If baseline eval fails with headless rendering:
- `MUJOCO_GL=egl` may fail on this host (`Cannot initialize a EGL device display`).
- Use `--mujoco-gl glx` with `scripts/libero/run_track_a_eval.sh`; it auto-wraps with `xvfb-run` when `DISPLAY` is unset.
