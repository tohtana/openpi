# Scripted OpenPI LIBERO Fine-Tune Example

This directory contains thin wrappers around the documented OpenPI `pi05_libero`
fine-tuning flow.

JAX entrypoint:

```bash
scripts/openpi/run_pi05_libero_finetune_example.sh --exp-name my_run
```

Default behavior:

- reuses `openpi/assets/pi05_libero/physical-intelligence/libero/norm_stats.json` if it already exists
- runs a bounded training example with `--num-train-steps 1`
- writes logs and manifests under `tasks/openpi-scripted-libero-finetune-example/runs/<run-id>/`
- writes checkpoints under `/mnt/local_storage/experiments/openpi_checkpoints/pi05_libero/<exp-name>/`
- sets `OPENPI_DATA_HOME=/mnt/local_storage/.cache/openpi` by default so OpenPI-downloaded weights and assets stay on local storage

To force a fresh bounded norm-stats pass before training:

```bash
scripts/openpi/run_pi05_libero_finetune_example.sh \
  --exp-name my_run \
  --refresh-norm-stats \
  --max-frames 256
```

The script sets cache-related environment variables to `/mnt/local_storage` by
default so repeated runs reuse local artifacts.

PyTorch entrypoint:

```bash
scripts/openpi/run_pi05_libero_finetune_example_pytorch.sh --exp-name my_run
```

Default PyTorch behavior:

- ensures the local `.venv` has OpenPI's patched `transformers` files without reusing hardlinked cache files
- resolves `gs://openpi-assets/checkpoints/pi05_base` through OpenPI's downloader and converts it to a PyTorch checkpoint if needed
- runs a bounded `scripts/train_pytorch.py pi05_libero` example with `--batch-size 4 --num-train-steps 1`
- writes logs and manifests under `tasks/openpi-scripted-libero-finetune-example-pytorch/<run-id>/`
- writes checkpoints under `/mnt/local_storage/experiments/openpi_pytorch_checkpoints/pi05_libero/<exp-name>/`
- sets `OPENPI_DATA_HOME=/mnt/local_storage/.cache/openpi` by default so OpenPI-downloaded weights and assets stay on local storage

To capture a per-step timing breakdown during the PyTorch run:

```bash
scripts/openpi/run_pi05_libero_finetune_example_pytorch.sh \
  --exp-name my_profile_run \
  --num-train-steps 4 \
  --log-interval 1 \
  --enable-profiling \
  --profiling-warmup-steps 1
```

The trainer logs averaged `data`, `host->device`, `preprocess`, `vision`, `llm`,
`action`, `backward`, `optimizer`, `other`, and `total` timings in milliseconds.
