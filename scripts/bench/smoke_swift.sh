#!/usr/bin/env bash
# Smoke test for SWIFT (Megatron backend) with reduced Qwen3-VL layers.
# Usage: bash scripts/bench/smoke_swift.sh <llm_layers> <vit_layers>
# Example: bash scripts/bench/smoke_swift.sh 24 14
# Runs 3 iterations at seq_len=1024 with activation checkpointing enabled.
# Prints peak memory and success/failure.
# Exit code 0 = fits, 1 = OOM or other failure.
set -euo pipefail

LLM_LAYERS="${1:?Usage: $0 <llm_layers> <vit_layers>}"
VIT_LAYERS="${2:?Usage: $0 <llm_layers> <vit_layers>}"

echo "=== SWIFT Smoke Test: LLM=${LLM_LAYERS}, ViT=${VIT_LAYERS} ==="

source /home/ray/anaconda3/etc/profile.d/conda.sh
conda activate moe

export HF_HOME=/mnt/local_storage/huggingface
export PYTORCH_ALLOC_CONF='expandable_segments:True'

# Create patched config for reduced layers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHED_DIR="/mnt/local_storage/qwen3vl_reduced_${LLM_LAYERS}_${VIT_LAYERS}"

python "${SCRIPT_DIR}/patch_swift_config.py" \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --num-llm-layers "${LLM_LAYERS}" \
  --vit-depth "${VIT_LAYERS}" \
  --output-dir "${PATCHED_DIR}"

NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29545 \
megatron sft \
    --model "${PATCHED_DIR}" \
    --load_safetensors true \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#1000' \
    --freeze_vit false \
    --freeze_llm false \
    --freeze_aligner false \
    --num_layers "${LLM_LAYERS}" \
    --tensor_model_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --moe_permute_fusion false \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --finetune true \
    --lr 1e-5 \
    --min_lr 1e-6 \
    --save /mnt/local_storage/experiments/smoke_swift_${LLM_LAYERS}_${VIT_LAYERS} \
    --eval_interval 10000 \
    --save_interval 10000 \
    --max_length 1024 \
    --log_interval 1 \
    --log_throughput true \
    --attention_backend flash \
    --train_iters 3 \
    --recompute_granularity selective \
    2>&1 | tee /tmp/smoke_swift_${LLM_LAYERS}_${VIT_LAYERS}.log

echo "=== SWIFT Smoke Test PASSED: LLM=${LLM_LAYERS}, ViT=${VIT_LAYERS} ==="
