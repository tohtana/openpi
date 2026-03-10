#!/usr/bin/env bash
# SWIFT (Megatron backend) benchmark script for Qwen3-VL-30B-A3B throughput testing.
#
# Usage: bash scripts/bench/bench_swift.sh --config <config_id> --llm-layers <N> --vit-layers <N> \
#            --iters <N> --output-dir <path>
# Seq length is fixed at 1024. Activation checkpointing is always on.
# Calls patch_swift_config.py internally to create reduced config.
# Writes raw log to <output-dir>/swift_<config_id>.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
CONFIG=""
LLM_LAYERS=""
VIT_LAYERS=""
ITERS=50
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --config) CONFIG="$2"; shift 2;;
    --llm-layers) LLM_LAYERS="$2"; shift 2;;
    --vit-layers) VIT_LAYERS="$2"; shift 2;;
    --iters) ITERS="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    *) echo "Unknown argument: $1"; exit 1;;
  esac
done

if [[ -z "$CONFIG" || -z "$LLM_LAYERS" || -z "$VIT_LAYERS" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 --config <id> --llm-layers <N> --vit-layers <N> --output-dir <path>"
  exit 1
fi

# Determine parallelism and extra args from config ID
EXTRA_ARGS=""
case "$CONFIG" in
  swift-ep8)
    EP=8; TP=1; SEQ_PAR=""
    ;;
  swift-tp2-ep8)
    EP=8; TP=2; SEQ_PAR="--sequence_parallel true"
    EXTRA_ARGS="--moe_grouped_gemm true --moe_shared_expert_overlap true"
    ;;
  swift-ep8-optimized)
    EP=8; TP=1; SEQ_PAR=""
    EXTRA_ARGS="--moe_grouped_gemm true --cross_entropy_loss_fusion true --packing true"
    ;;
  *)
    echo "Unknown config: $CONFIG. Valid: swift-ep8, swift-tp2-ep8, swift-ep8-optimized"
    exit 1
    ;;
esac

echo "=== SWIFT Benchmark: config=${CONFIG}, LLM=${LLM_LAYERS}, ViT=${VIT_LAYERS}, EP=${EP}, TP=${TP}, iters=${ITERS} ==="

source /home/ray/anaconda3/etc/profile.d/conda.sh
conda activate moe

export HF_HOME=/mnt/local_storage/huggingface
export PYTORCH_ALLOC_CONF='expandable_segments:True'

# Create patched config for reduced layers
PATCHED_DIR="/mnt/local_storage/qwen3vl_reduced_${LLM_LAYERS}_${VIT_LAYERS}"
python "${SCRIPT_DIR}/patch_swift_config.py" \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --num-llm-layers "${LLM_LAYERS}" \
  --vit-depth "${VIT_LAYERS}" \
  --output-dir "${PATCHED_DIR}"

mkdir -p "${OUTPUT_DIR}"
LOGFILE="${OUTPUT_DIR}/swift_${CONFIG}.log"

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
    --tensor_model_parallel_size "${TP}" \
    --expert_model_parallel_size "${EP}" \
    --moe_permute_fusion false \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --finetune true \
    --lr 1e-5 \
    --min_lr 1e-6 \
    --save /mnt/local_storage/experiments/bench_swift_${CONFIG} \
    --eval_interval 10000 \
    --save_interval 10000 \
    --max_length 1024 \
    --log_interval 1 \
    --log_throughput true \
    --attention_backend flash \
    --train_iters "${ITERS}" \
    --recompute_granularity selective \
    ${SEQ_PAR} \
    ${EXTRA_ARGS} \
    2>&1 | tee "${LOGFILE}"

echo "=== SWIFT Benchmark DONE: ${CONFIG} ==="
echo "Log: ${LOGFILE}"
