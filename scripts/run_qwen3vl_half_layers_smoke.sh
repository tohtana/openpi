#!/usr/bin/env bash
set -euo pipefail

source /home/ray/anaconda3/etc/profile.d/conda.sh
conda activate moe

export HF_HOME=/mnt/local_storage/huggingface
export PYTHONPATH="/home/ray/default/moe_bench/Megatron-Bridge/src:/home/ray/default/moe_bench/Megatron-LM"
export PYTORCH_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc-per-node=8 --master_port=29509 \
  /home/ray/default/moe_bench/Megatron-Bridge/examples/recipes/qwen_vl/finetune_qwen_vl.py \
  --recipe qwen3_vl_3b_active_30b_moe_finetune_config \
  --dataset-type mock \
  train.train_iters=50 \
  train.global_batch_size=8 \
  train.micro_batch_size=1 \
  train.eval_interval=0 \
  checkpoint.save_interval=0 \
  checkpoint.save=/mnt/local_storage/experiments/qwen3vl30b_half_layers_smoke \
  model.num_layers=24 \
  model.seq_length=256 \
  dataset.sequence_length=256 \
  model.expert_model_parallel_size=8 \
  model.tensor_model_parallel_size=1 \
  model.pipeline_model_parallel_size=1 \
  model.moe_permute_fusion=false \
  logger.log_interval=1
