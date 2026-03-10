#!/usr/bin/env bash
# Master benchmark script for Qwen3-VL-8B dense model.
#
# Usage: bash scripts/bench/run_all_8b.sh --output-dir <path> [--iters <N>] [--seq-length <N>]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
ITERS=50
SEQ_LENGTH=1024
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --iters) ITERS="$2"; shift 2;;
    --seq-length) SEQ_LENGTH="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    *) echo "Unknown argument: $1"; exit 1;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 --output-dir <path> [--iters N] [--seq-length N]"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo "Qwen3-VL-8B Dense Throughput Benchmark"
echo "Seq length: ${SEQ_LENGTH}, Iterations: ${ITERS}"
echo "Output: ${OUTPUT_DIR}"
echo "========================================"

RESULTS_LOG="${OUTPUT_DIR}/run_all.log"
echo "Benchmark started at $(date)" | tee "${RESULTS_LOG}"

# --- Bridge benchmarks ---
BRIDGE_CONFIGS=("bridge-dp8" "bridge-dp4-tp2" "bridge-dp2-tp4" "bridge-dp8-recompute")

for cfg in "${BRIDGE_CONFIGS[@]}"; do
  echo ""
  echo ">>> Running Bridge config: ${cfg}" | tee -a "${RESULTS_LOG}"
  echo "    Started at $(date)" | tee -a "${RESULTS_LOG}"

  if bash "${SCRIPT_DIR}/bench_qwen3vl8b_bridge.sh" \
    --config "${cfg}" \
    --seq-length "${SEQ_LENGTH}" \
    --iters "${ITERS}" \
    --output-dir "${OUTPUT_DIR}"; then
    echo "    PASSED at $(date)" | tee -a "${RESULTS_LOG}"

    python "${SCRIPT_DIR}/parse_bridge_log.py" \
      "${OUTPUT_DIR}/bridge_${cfg}.log" \
      --output "${OUTPUT_DIR}/bridge_${cfg}.jsonl" \
      --seq-length "${SEQ_LENGTH}" \
      | tee -a "${RESULTS_LOG}"
  else
    echo "    FAILED at $(date)" | tee -a "${RESULTS_LOG}"
  fi
done

# --- SWIFT benchmarks ---
SWIFT_CONFIGS=("swift-dp8" "swift-dp4-tp2" "swift-dp8-optimized" "swift-dp8-recompute")

for cfg in "${SWIFT_CONFIGS[@]}"; do
  echo ""
  echo ">>> Running SWIFT config: ${cfg}" | tee -a "${RESULTS_LOG}"
  echo "    Started at $(date)" | tee -a "${RESULTS_LOG}"

  if bash "${SCRIPT_DIR}/bench_qwen3vl8b_swift.sh" \
    --config "${cfg}" \
    --seq-length "${SEQ_LENGTH}" \
    --iters "${ITERS}" \
    --output-dir "${OUTPUT_DIR}"; then
    echo "    PASSED at $(date)" | tee -a "${RESULTS_LOG}"

    python "${SCRIPT_DIR}/parse_swift_log.py" \
      "${OUTPUT_DIR}/swift_${cfg}.log" \
      --output "${OUTPUT_DIR}/swift_${cfg}.jsonl" \
      --seq-length "${SEQ_LENGTH}" \
      | tee -a "${RESULTS_LOG}"
  else
    echo "    FAILED at $(date)" | tee -a "${RESULTS_LOG}"
  fi
done

echo ""
echo "========================================"
echo "All benchmarks completed at $(date)" | tee -a "${RESULTS_LOG}"
echo "Output: ${OUTPUT_DIR}"
echo "Analyze: python ${SCRIPT_DIR}/analyze.py --results-dir ${OUTPUT_DIR} --num-gpus 8 --seq-length ${SEQ_LENGTH}"
echo "========================================"
