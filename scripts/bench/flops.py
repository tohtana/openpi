#!/usr/bin/env python3
"""Thin wrapper around Megatron's built-in FLOPS calculator.

Provides:
  1. extract_tflops_from_log(logfile) -> list of per-iteration TFLOPS values
  2. compute_total_param_flops(...) -> total FLOPS counting ALL expert params
  3. compute_mfu(tflops_per_gpu, peak_tflops=989.0) -> MFU fraction
"""

import re
import sys
from pathlib import Path


def extract_tflops_from_log(logfile: str) -> list[dict]:
    """Parse TFLOPS values from a Megatron training log.

    Handles both Megatron-Bridge and SWIFT (Megatron backend) log formats:
    - Bridge: "throughput per GPU (TFLOP/s/GPU): 45.2"
    - SWIFT:  "throughput per GPU (TFLOP/s/GPU): 45.2"
    Both use the same Megatron log format.

    Returns list of dicts with keys: iteration, tflops_per_gpu
    """
    results = []
    # Pattern: "iteration <N>/<total>" or "iteration <N>"
    iter_pattern = re.compile(r"iteration\s+(\d+)")
    # Pattern: "throughput per GPU (TFLOP/s/GPU): <float>"
    tflops_pattern = re.compile(r"throughput per GPU \(TFLOP/s/GPU\):\s+([\d.]+)")

    with open(logfile) as f:
        for line in f:
            iter_match = iter_pattern.search(line)
            tflops_match = tflops_pattern.search(line)
            if iter_match and tflops_match:
                results.append({
                    "iteration": int(iter_match.group(1)),
                    "tflops_per_gpu": float(tflops_match.group(1)),
                })
    return results


def compute_total_param_flops(
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    seq_length: int,
    batch_size: int,
    num_experts: int = 128,
    topk: int = 8,
    ffn_hidden_size: int = 768,
    gated_linear: bool = True,
    kv_channels: int | None = None,
    num_kv_heads: int | None = None,
) -> dict:
    """Compute FLOPS for a forward+backward pass, both active and total.

    Active FLOPS: only count topk routed experts (matches Megatron's
    num_floating_point_operations() which uses moe_router_topk).
    Total FLOPS: count ALL expert parameters (hypothetical, not used by Megatron).

    Uses Megatron's standard formula:
      expansion_factor = 12 (3 for fwd+bwd × 2 for GEMMs × 2 for FLOP per MAC)
      For gated linear: multiply MLP by 1.5
    """
    expansion = 12
    gated_mult = 1.5 if gated_linear else 1.0

    if kv_channels is None:
        kv_channels = hidden_size // num_attention_heads
    if num_kv_heads is None:
        num_kv_heads = num_attention_heads

    # Attention FLOPS per layer (GQA-aware)
    # Q, K, V projections + output projection + attention scores
    attn_flops = expansion * batch_size * seq_length * hidden_size * (
        hidden_size  # Q projection
        + hidden_size * (num_kv_heads / num_attention_heads)  # K projection (GQA)
        + hidden_size * (num_kv_heads / num_attention_heads)  # V projection (GQA)
        + hidden_size  # Output projection
    )
    # Attention score computation (softmax(QK^T)V)
    attn_score_flops = expansion * batch_size * num_attention_heads * seq_length * (
        seq_length * kv_channels  # QK^T
        + seq_length * kv_channels  # score × V
    )

    # MLP FLOPS per layer - active (topk experts)
    mlp_active_flops = expansion * batch_size * seq_length * (
        hidden_size * ffn_hidden_size * topk * gated_mult * 2  # up + down projections
    )

    # MLP FLOPS per layer - total (all experts)
    mlp_total_flops = expansion * batch_size * seq_length * (
        hidden_size * ffn_hidden_size * num_experts * gated_mult * 2
    )

    # Per-layer totals
    active_per_layer = attn_flops + attn_score_flops + mlp_active_flops
    total_per_layer = attn_flops + attn_score_flops + mlp_total_flops

    # Embedding + LM head (approximate)
    vocab_size = 151936  # Qwen3-VL
    embed_flops = expansion * batch_size * seq_length * hidden_size * vocab_size

    active_total = active_per_layer * num_layers + embed_flops
    total_total = total_per_layer * num_layers + embed_flops

    return {
        "active_flops": active_total,
        "total_flops": total_total,
        "active_tflops_per_step": active_total / 1e12,
        "total_tflops_per_step": total_total / 1e12,
    }


def compute_mfu(tflops_per_gpu: float, peak_tflops: float = 989.0) -> float:
    """Compute Model FLOPS Utilization.

    Args:
        tflops_per_gpu: Measured TFLOPS per GPU (from Megatron logs).
        peak_tflops: Theoretical peak TFLOPS for the GPU.
            H100 SXM5 bf16: 989 TFLOPS (Tensor Core).

    Returns:
        MFU as a fraction in [0, 1].
    """
    if peak_tflops <= 0:
        return 0.0
    return max(0.0, min(1.0, tflops_per_gpu / peak_tflops))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python flops.py <logfile>")
        print("  Extracts TFLOPS from Megatron training log.")
        sys.exit(1)

    logfile = sys.argv[1]
    entries = extract_tflops_from_log(logfile)
    if not entries:
        print(f"No TFLOPS entries found in {logfile}")
        sys.exit(1)

    print(f"Found {len(entries)} entries")
    for e in entries:
        mfu = compute_mfu(e["tflops_per_gpu"])
        print(f"  iter {e['iteration']:>4d}: {e['tflops_per_gpu']:.1f} TFLOP/s/GPU  MFU={mfu:.1%}")

    # Summary (skip first 5 warmup)
    if len(entries) > 5:
        steady = [e["tflops_per_gpu"] for e in entries[5:]]
        avg = sum(steady) / len(steady)
        print(f"\nSteady-state avg (skip 5 warmup): {avg:.1f} TFLOP/s/GPU  MFU={compute_mfu(avg):.1%}")
