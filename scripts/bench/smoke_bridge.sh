#!/usr/bin/env bash
# Smoke test for Megatron-Bridge with reduced Qwen3-VL layers.
# Usage: bash scripts/bench/smoke_bridge.sh <llm_layers> <vit_layers> "<deepstack_indices>"
# Example: bash scripts/bench/smoke_bridge.sh 24 14 "[4,8,12]"
# Runs 3 iterations at seq_len=1024 with activation checkpointing enabled.
# Prints peak memory and success/failure.
# Exit code 0 = fits, 1 = OOM or other failure.
set -euo pipefail

LLM_LAYERS="${1:?Usage: $0 <llm_layers> <vit_layers> <deepstack_indices>}"
VIT_LAYERS="${2:?Usage: $0 <llm_layers> <vit_layers> <deepstack_indices>}"
DEEPSTACK="${3:?Usage: $0 <llm_layers> <vit_layers> <deepstack_indices>}"

echo "=== Bridge Smoke Test: LLM=${LLM_LAYERS}, ViT=${VIT_LAYERS}, Deepstack=${DEEPSTACK} ==="

source /home/ray/anaconda3/etc/profile.d/conda.sh
conda activate moe

export HF_HOME=/mnt/local_storage/huggingface
export PYTHONPATH="/home/ray/default/moe_bench/Megatron-Bridge/src:/home/ray/default/moe_bench/Megatron-LM"
export PYTORCH_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc-per-node=8 --master_port=29540 \
  /home/ray/default/moe_bench/Megatron-Bridge/examples/recipes/qwen_vl/finetune_qwen_vl.py \
  --recipe qwen3_vl_3b_active_30b_moe_finetune_config \
  --dataset-type mock \
  train.train_iters=3 \
  train.global_batch_size=8 \
  train.micro_batch_size=1 \
  train.eval_interval=0 \
  checkpoint.save_interval=0 \
  checkpoint.save=/mnt/local_storage/experiments/smoke_bridge_${LLM_LAYERS}_${VIT_LAYERS} \
  model.num_layers=${LLM_LAYERS} \
  model.seq_length=1024 \
  dataset.sequence_length=1024 \
  model.expert_model_parallel_size=8 \
  model.tensor_model_parallel_size=1 \
  model.pipeline_model_parallel_size=1 \
  model.moe_permute_fusion=false \
  model.freeze_language_model=false \
  model.freeze_vision_model=false \
  model.freeze_vision_projection=false \
  +model.deepstack_visual_indexes="${DEEPSTACK}" \
  +model.vision_config.depth=${VIT_LAYERS} \
  model.recompute_granularity=selective \
  logger.log_interval=1 \
  logger.log_throughput=true \
  2>&1 | tee /tmp/smoke_bridge_${LLM_LAYERS}_${VIT_LAYERS}.log

echo "=== Bridge Smoke Test PASSED: LLM=${LLM_LAYERS}, ViT=${VIT_LAYERS} ==="
