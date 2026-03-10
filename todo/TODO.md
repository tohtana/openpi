# TODO

- [x] Try Track A from `initial_plan.md`: run LeRobot + pi0 (flow) on LIBERO in headless mode on this 8xH100 environment and capture reproducible eval commands and results ([plan](todo/20260210-track-a-lerobot-pi0-libero.md))
  - [x] Create the `vla_pi0` environment and install LeRobot with `.[libero]`.
  - [x] Run a small then scaled headless `lerobot-eval` with `lerobot/pi0_libero_finetuned`, using 8xH100 capacity where practical.
  - [x] Record success rate plus log/video artifact paths for later file download.
- [x] Run one documented openpi LIBERO fine-tuning example end-to-end and capture reproducible commands/artifacts ([plan](todo/20260219-openpi-libero-finetune-example.md))
  - [x] Run norm-stats computation for a chosen LIBERO training config.
  - [x] Launch a bounded fine-tuning run and verify checkpoints/logging output.
  - [x] Document exact command line, config, and artifact paths in `tasks/`.
