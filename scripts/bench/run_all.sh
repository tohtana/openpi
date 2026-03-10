#!/usr/bin/env bash
# Master benchmark script: runs all Bridge and SWIFT configs sequentially.
#
# Usage: bash scripts/bench/run_all.sh --llm-layers <N> --vit-layers <N> \
#            --deepstack "[i,j,k]" --output-dir <path> [--iters <N>]
# Each run: 50 iterations by default.
# Saves all logs and parsed JSONL to <output-dir>/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
LLM_LAYERS=""
VIT_LAYERS=""
DEEPSTACK=""
ITERS=50
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --llm-layers) LLM_LAYERS="$2"; shift 2;;
    --vit-layers) VIT_LAYERS="$2"; shift 2;;
    --deepstack) DEEPSTACK="$2"; shift 2;;
    --iters) ITERS="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    *) echo "Unknown argument: $1"; exit 1;;
  esac
done

if [[ -z "$LLM_LAYERS" || -z "$VIT_LAYERS" || -z "$DEEPSTACK" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 --llm-layers <N> --vit-layers <N> --deepstack \"[i,j,k]\" --output-dir <path>"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo "Qwen3-VL-30B-A3B Throughput Benchmark"
echo "LLM layers: ${LLM_LAYERS}, ViT layers: ${VIT_LAYERS}"
echo "Deepstack: ${DEEPSTACK}"
echo "Iterations per run: ${ITERS}"
echo "Output: ${OUTPUT_DIR}"
echo "========================================"

# Track results
RESULTS_LOG="${OUTPUT_DIR}/run_all.log"
echo "Benchmark started at $(date)" | tee "${RESULTS_LOG}"

# --- Bridge benchmarks ---
BRIDGE_CONFIGS=("bridge-ep8" "bridge-ep4-tp2")

for cfg in "${BRIDGE_CONFIGS[@]}"; do
  echo ""
  echo ">>> Running Bridge config: ${cfg}" | tee -a "${RESULTS_LOG}"
  echo "    Started at $(date)" | tee -a "${RESULTS_LOG}"

  if bash "${SCRIPT_DIR}/bench_bridge.sh" \
    --config "${cfg}" \
    --llm-layers "${LLM_LAYERS}" \
    --vit-layers "${VIT_LAYERS}" \
    --deepstack "${DEEPSTACK}" \
    --iters "${ITERS}" \
    --output-dir "${OUTPUT_DIR}"; then
    echo "    PASSED at $(date)" | tee -a "${RESULTS_LOG}"

    # Parse log
    python "${SCRIPT_DIR}/parse_bridge_log.py" \
      "${OUTPUT_DIR}/bridge_${cfg}.log" \
      --output "${OUTPUT_DIR}/bridge_${cfg}.jsonl" \
      | tee -a "${RESULTS_LOG}"
  else
    echo "    FAILED at $(date)" | tee -a "${RESULTS_LOG}"
  fi
done

# --- SWIFT benchmarks ---
SWIFT_CONFIGS=("swift-ep8" "swift-tp2-ep8" "swift-ep8-optimized")

for cfg in "${SWIFT_CONFIGS[@]}"; do
  echo ""
  echo ">>> Running SWIFT config: ${cfg}" | tee -a "${RESULTS_LOG}"
  echo "    Started at $(date)" | tee -a "${RESULTS_LOG}"

  if bash "${SCRIPT_DIR}/bench_swift.sh" \
    --config "${cfg}" \
    --llm-layers "${LLM_LAYERS}" \
    --vit-layers "${VIT_LAYERS}" \
    --iters "${ITERS}" \
    --output-dir "${OUTPUT_DIR}"; then
    echo "    PASSED at $(date)" | tee -a "${RESULTS_LOG}"

    # Parse log
    python "${SCRIPT_DIR}/parse_swift_log.py" \
      "${OUTPUT_DIR}/swift_${cfg}.log" \
      --output "${OUTPUT_DIR}/swift_${cfg}.jsonl" \
      | tee -a "${RESULTS_LOG}"
  else
    echo "    FAILED at $(date)" | tee -a "${RESULTS_LOG}"
  fi
done

echo ""
echo "========================================"
echo "All benchmarks completed at $(date)" | tee -a "${RESULTS_LOG}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Run analysis: python ${SCRIPT_DIR}/analyze.py --results-dir ${OUTPUT_DIR} --num-gpus 8"
echo "========================================"
