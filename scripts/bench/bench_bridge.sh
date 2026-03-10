#!/usr/bin/env bash
# Megatron-Bridge benchmark script for Qwen3-VL-30B-A3B throughput testing.
#
# Usage: bash scripts/bench/bench_bridge.sh --config <config_id> --llm-layers <N> --vit-layers <N> \
#            --deepstack "[i,j,k]" --iters <N> --output-dir <path>
# Seq length is fixed at 1024. Activation checkpointing is always on.
# Writes raw log to <output-dir>/bridge_<config_id>.log
set -euo pipefail

# Parse arguments
CONFIG=""
LLM_LAYERS=""
VIT_LAYERS=""
DEEPSTACK=""
ITERS=50
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --config) CONFIG="$2"; shift 2;;
    --llm-layers) LLM_LAYERS="$2"; shift 2;;
    --vit-layers) VIT_LAYERS="$2"; shift 2;;
    --deepstack) DEEPSTACK="$2"; shift 2;;
    --iters) ITERS="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    *) echo "Unknown argument: $1"; exit 1;;
  esac
done

if [[ -z "$CONFIG" || -z "$LLM_LAYERS" || -z "$VIT_LAYERS" || -z "$DEEPSTACK" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 --config <id> --llm-layers <N> --vit-layers <N> --deepstack \"[i,j,k]\" --output-dir <path>"
  exit 1
fi

# Determine parallelism from config ID
case "$CONFIG" in
  bridge-ep8)
    EP=8; TP=1
    ;;
  bridge-ep4-tp2)
    EP=4; TP=2
    ;;
  *)
    echo "Unknown config: $CONFIG. Valid: bridge-ep8, bridge-ep4-tp2"
    exit 1
    ;;
esac

echo "=== Bridge Benchmark: config=${CONFIG}, LLM=${LLM_LAYERS}, ViT=${VIT_LAYERS}, EP=${EP}, TP=${TP}, iters=${ITERS} ==="

source /home/ray/anaconda3/etc/profile.d/conda.sh
conda activate moe

export HF_HOME=/mnt/local_storage/huggingface
export PYTHONPATH="/home/ray/default/moe_bench/Megatron-Bridge/src:/home/ray/default/moe_bench/Megatron-LM"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT_DIR}"
LOGFILE="${OUTPUT_DIR}/bridge_${CONFIG}.log"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc-per-node=8 --master_port=29540 \
  /home/ray/default/moe_bench/Megatron-Bridge/examples/recipes/qwen_vl/finetune_qwen_vl.py \
  --recipe qwen3_vl_3b_active_30b_moe_finetune_config \
  --dataset-type mock \
  train.train_iters=${ITERS} \
  train.global_batch_size=8 \
  train.micro_batch_size=1 \
  train.eval_interval=0 \
  checkpoint.save_interval=0 \
  checkpoint.save=/mnt/local_storage/experiments/bench_bridge_${CONFIG} \
  model.num_layers=${LLM_LAYERS} \
  model.seq_length=1024 \
  dataset.sequence_length=1024 \
  model.expert_model_parallel_size=${EP} \
  model.tensor_model_parallel_size=${TP} \
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
  2>&1 | tee "${LOGFILE}"

echo "=== Bridge Benchmark DONE: ${CONFIG} ==="
echo "Log: ${LOGFILE}"
