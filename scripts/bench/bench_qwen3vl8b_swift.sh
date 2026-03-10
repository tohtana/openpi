#!/usr/bin/env bash
# SWIFT (Megatron backend) benchmark for Qwen3-VL-8B (dense) throughput testing.
#
# Usage: bash scripts/bench/bench_qwen3vl8b_swift.sh --config <config_id> \
#            --seq-length <N> --iters <N> --output-dir <path>
# Writes raw log to <output-dir>/swift_<config_id>.log
set -euo pipefail

# Parse arguments
CONFIG=""
SEQ_LENGTH=1024
ITERS=50
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --config) CONFIG="$2"; shift 2;;
    --seq-length) SEQ_LENGTH="$2"; shift 2;;
    --iters) ITERS="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    *) echo "Unknown argument: $1"; exit 1;;
  esac
done

if [[ -z "$CONFIG" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 --config <id> --output-dir <path> [--seq-length N] [--iters N]"
  exit 1
fi

# Determine parallelism and extra args from config ID
EXTRA_ARGS=""
case "$CONFIG" in
  swift-dp8)
    TP=1; SEQ_PAR=""
    ;;
  swift-dp4-tp2)
    TP=2; SEQ_PAR="--sequence_parallel true"
    ;;
  swift-dp8-optimized)
    TP=1; SEQ_PAR=""
    EXTRA_ARGS="--cross_entropy_loss_fusion true --packing true"
    ;;
  swift-dp8-recompute)
    TP=1; SEQ_PAR=""
    EXTRA_ARGS="--recompute_granularity selective"
    ;;
  *)
    echo "Unknown config: $CONFIG"
    echo "Valid: swift-dp8, swift-dp4-tp2, swift-dp8-optimized, swift-dp8-recompute"
    exit 1
    ;;
esac

echo "=== SWIFT 8B Benchmark: config=${CONFIG}, TP=${TP}, seq=${SEQ_LENGTH}, iters=${ITERS} ==="

source /home/ray/anaconda3/etc/profile.d/conda.sh
conda activate moe

export HF_HOME=/mnt/local_storage/huggingface
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

mkdir -p "${OUTPUT_DIR}"
LOGFILE="${OUTPUT_DIR}/swift_${CONFIG}.log"

# No patch_swift_config.py needed — full model loaded directly from HF
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29545 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
megatron sft \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --load_safetensors true \
    --save_safetensors true \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#1000' \
    --freeze_vit false \
    --freeze_llm false \
    --freeze_aligner false \
    --tensor_model_parallel_size "${TP}" \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --finetune true \
    --lr 1e-5 \
    --min_lr 1e-6 \
    --save /mnt/local_storage/experiments/bench_qwen3vl8b_swift_${CONFIG} \
    --eval_interval 10000 \
    --save_interval 10000 \
    --max_length "${SEQ_LENGTH}" \
    --log_interval 1 \
    --log_throughput true \
    --attention_backend flash \
    --train_iters "${ITERS}" \
    ${SEQ_PAR} \
    ${EXTRA_ARGS} \
    2>&1 | tee "${LOGFILE}"

echo "=== SWIFT 8B Benchmark DONE: ${CONFIG} ==="
echo "Log: ${LOGFILE}"
