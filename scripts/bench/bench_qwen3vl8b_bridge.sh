#!/usr/bin/env bash
# Megatron-Bridge benchmark for Qwen3-VL-8B (dense) throughput testing.
#
# Usage: bash scripts/bench/bench_qwen3vl8b_bridge.sh --config <config_id> \
#            --seq-length <N> --iters <N> --output-dir <path>
# Writes raw log to <output-dir>/bridge_<config_id>.log
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
  bridge-dp8)
    TP=1; EXTRA_ARGS=""
    ;;
  bridge-dp4-tp2)
    TP=2; EXTRA_ARGS=""
    ;;
  bridge-dp2-tp4)
    TP=4; EXTRA_ARGS=""
    ;;
  bridge-dp8-recompute)
    TP=1; EXTRA_ARGS="model.recompute_granularity=selective"
    ;;
  *)
    echo "Unknown config: $CONFIG"
    echo "Valid: bridge-dp8, bridge-dp4-tp2, bridge-dp2-tp4, bridge-dp8-recompute"
    exit 1
    ;;
esac

echo "=== Bridge 8B Benchmark: config=${CONFIG}, TP=${TP}, seq=${SEQ_LENGTH}, iters=${ITERS} ==="

source /home/ray/anaconda3/etc/profile.d/conda.sh
conda activate moe

export HF_HOME=/mnt/local_storage/huggingface
export PYTHONPATH="/home/ray/default/moe_bench/Megatron-Bridge/src:/home/ray/default/moe_bench/Megatron-LM"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT_DIR}"
LOGFILE="${OUTPUT_DIR}/bridge_${CONFIG}.log"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc-per-node=8 --master_port=29540 \
  /home/ray/default/moe_bench/Megatron-Bridge/examples/recipes/qwen_vl/finetune_qwen_vl.py \
  --recipe qwen3_vl_8b_finetune_config \
  --dataset-type mock \
  train.train_iters=${ITERS} \
  train.global_batch_size=8 \
  train.micro_batch_size=1 \
  train.eval_interval=0 \
  checkpoint.save_interval=0 \
  checkpoint.save=/mnt/local_storage/experiments/bench_qwen3vl8b_bridge_${CONFIG} \
  model.seq_length=${SEQ_LENGTH} \
  dataset.sequence_length=${SEQ_LENGTH} \
  model.tensor_model_parallel_size=${TP} \
  model.pipeline_model_parallel_size=1 \
  model.freeze_language_model=false \
  model.freeze_vision_model=false \
  model.freeze_vision_projection=false \
  logger.log_interval=1 \
  logger.log_throughput=true \
  ${EXTRA_ARGS} \
  2>&1 | tee "${LOGFILE}"

echo "=== Bridge 8B Benchmark DONE: ${CONFIG} ==="
echo "Log: ${LOGFILE}"
