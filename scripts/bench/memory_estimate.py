#!/usr/bin/env python3
"""Analytical memory estimator for Qwen3-VL-30B-A3B reduced-layer configurations.

Estimates per-GPU memory usage with:
- AdamW bf16 training (params bf16 + grads bf16 + 2x optimizer states fp32 = 20 bytes/param)
- Expert Parallelism (EP=8): MoE experts sharded across 8 GPUs
- Activation checkpointing always enabled (recompute_granularity=full)

Output format (tab-separated):
  CONFIG  LLM_LAYERS  VIT_LAYERS  DEEPSTACK_INDICES  EST_MEM_GB  FITS_80GB
"""

import sys


def estimate_memory(
    llm_layers: int,
    vit_layers: int,
    deepstack_indices: list[int],
    ep_size: int = 8,
    tp_size: int = 1,
    seq_length: int = 1024,
    micro_batch_size: int = 1,
    hidden_size: int = 2048,
    num_attention_heads: int = 32,
    num_kv_heads: int = 4,
    ffn_hidden_size: int = 768,  # per-expert FFN hidden
    num_experts: int = 128,
    vit_hidden_size: int = 1152,
    vit_ffn_hidden: int = 4304,
    vit_num_heads: int = 16,
    vocab_size: int = 151936,
    gpu_memory_gb: float = 80.0,
) -> dict:
    """Estimate per-GPU memory for a given configuration.

    Returns dict with component breakdowns and total.
    """
    bytes_per_param = 20  # AdamW bf16: 2(param) + 2(grad) + 4+4(adam states) + 8(master) = 20

    # --- LLM layers ---
    # Attention: Q, K, V projections + output projection
    # Q: hidden_size * hidden_size
    # K: hidden_size * (kv_channels * num_kv_heads) = hidden_size * (hidden_size * num_kv_heads / num_attention_heads)
    # V: same as K
    # O: hidden_size * hidden_size
    kv_channels = hidden_size // num_attention_heads
    q_params = hidden_size * hidden_size
    k_params = hidden_size * (kv_channels * num_kv_heads)
    v_params = hidden_size * (kv_channels * num_kv_heads)
    o_params = hidden_size * hidden_size
    attn_params = q_params + k_params + v_params + o_params
    # Biases, layer norms (small)
    attn_misc = hidden_size * 4  # 2x LN (2 * hidden_size), biases etc.

    # MoE MLP per expert: gate_proj + up_proj + down_proj (SwiGLU)
    # gate: hidden_size * ffn_hidden_size
    # up:   hidden_size * ffn_hidden_size
    # down: ffn_hidden_size * hidden_size
    expert_params = 3 * hidden_size * ffn_hidden_size
    total_expert_params = expert_params * num_experts

    # Router: hidden_size * num_experts
    router_params = hidden_size * num_experts

    # Per LLM layer total params
    llm_layer_params = attn_params + attn_misc + total_expert_params + router_params

    # With EP sharding: experts are split across ep_size GPUs
    # Attention is replicated (or TP-sharded if TP>1)
    expert_params_per_gpu = total_expert_params / ep_size
    attn_params_per_gpu = (attn_params + attn_misc) / tp_size
    router_params_per_gpu = router_params  # replicated
    llm_layer_per_gpu = expert_params_per_gpu + attn_params_per_gpu + router_params_per_gpu

    # --- ViT layers ---
    # Self-attention: Q, K, V, O projections
    vit_attn_params = 4 * vit_hidden_size * vit_hidden_size
    # MLP: 2 projections (hidden -> ffn, ffn -> hidden)
    vit_mlp_params = 2 * vit_hidden_size * vit_ffn_hidden
    # LN, biases
    vit_misc = vit_hidden_size * 4
    vit_layer_params = vit_attn_params + vit_mlp_params + vit_misc
    # ViT is replicated (no EP/TP for ViT typically)
    vit_layer_per_gpu = vit_layer_params

    # --- Non-layer components ---
    # Embedding: vocab_size * hidden_size (shared with lm_head typically)
    embed_params = vocab_size * hidden_size
    # Final LN: hidden_size
    final_ln_params = hidden_size * 2
    llm_nonlayer = embed_params + final_ln_params
    llm_nonlayer_per_gpu = llm_nonlayer / tp_size

    # ViT non-layer: patch_embed, merger, deepstack_merger_list
    # patch_embed: in_channels * temporal_patch_size * patch_size^2 * vit_hidden_size
    patch_embed_params = 3 * 2 * 14 * 14 * vit_hidden_size  # ~1.1M
    # merger: merges spatial features
    merger_params = vit_hidden_size * 4 * hidden_size * 2  # ~18M (approx)
    # deepstack mergers: one per deepstack index
    deepstack_merger_params = len(deepstack_indices) * vit_hidden_size * 4 * hidden_size
    vit_nonlayer = patch_embed_params + merger_params + deepstack_merger_params
    vit_nonlayer_per_gpu = vit_nonlayer  # replicated

    # --- Total params per GPU ---
    total_params_per_gpu = (
        llm_layers * llm_layer_per_gpu
        + vit_layers * vit_layer_per_gpu
        + llm_nonlayer_per_gpu
        + vit_nonlayer_per_gpu
    )

    # Memory in GB
    param_memory_gb = total_params_per_gpu * bytes_per_param / (1024**3)

    # Activation memory estimate with checkpointing
    # With full recompute (recompute_num_layers=1), only store:
    # - Input activations at each recompute boundary
    # - Current layer's activations during forward
    # Rough estimate: ~2 * micro_batch_size * seq_length * hidden_size * 2 bytes per layer boundary
    # Plus working memory for current computation
    act_per_boundary = 2 * micro_batch_size * seq_length * hidden_size * 2  # bytes
    # With full recompute every 1 layer, we have num_layers boundaries
    act_memory = act_per_boundary * (llm_layers + vit_layers)
    # Working memory for current layer (attention scores, MLP intermediates)
    # Attention: micro_batch * num_heads * seq_len * seq_len * 2 bytes
    attn_working = micro_batch_size * num_attention_heads * seq_length * seq_length * 2
    # MLP working: micro_batch * seq_len * ffn_hidden * topk * 2 bytes (per expert)
    mlp_working = micro_batch_size * seq_length * ffn_hidden_size * 8 * 2  # topk=8
    working_memory = max(attn_working, mlp_working)

    act_memory_gb = (act_memory + working_memory) / (1024**3)

    # CUDA overhead + fragmentation (typically 2-5 GB)
    cuda_overhead_gb = 3.0

    total_memory_gb = param_memory_gb + act_memory_gb + cuda_overhead_gb

    return {
        "llm_layers": llm_layers,
        "vit_layers": vit_layers,
        "deepstack_indices": deepstack_indices,
        "llm_layer_params_per_gpu": llm_layer_per_gpu,
        "vit_layer_params_per_gpu": vit_layer_per_gpu,
        "total_params_per_gpu": total_params_per_gpu,
        "param_memory_gb": param_memory_gb,
        "act_memory_gb": act_memory_gb,
        "cuda_overhead_gb": cuda_overhead_gb,
        "total_memory_gb": total_memory_gb,
        "fits_80gb": total_memory_gb <= gpu_memory_gb,
    }


def main():
    configs = [
        {"name": "A", "llm": 24, "vit": 14, "deepstack": [4, 8, 12]},
        {"name": "B", "llm": 16, "vit": 9, "deepstack": [2, 5, 8]},
        {"name": "C", "llm": 12, "vit": 7, "deepstack": [2, 4, 6]},
    ]

    print("Qwen3-VL-30B-A3B Memory Estimation (EP=8, AdamW bf16, activation checkpointing ON)")
    print("=" * 100)
    print(f"{'CONFIG':<8} {'LLM_LAYERS':<12} {'VIT_LAYERS':<12} {'DEEPSTACK':<18} "
          f"{'PARAMS_GB':<12} {'ACT_GB':<10} {'TOTAL_GB':<12} {'FITS_80GB':<10}")
    print("-" * 100)

    for cfg in configs:
        result = estimate_memory(
            llm_layers=cfg["llm"],
            vit_layers=cfg["vit"],
            deepstack_indices=cfg["deepstack"],
        )
        deepstack_str = str(cfg["deepstack"])
        params_gb = result["param_memory_gb"]
        act_gb = result["act_memory_gb"]
        total_gb = result["total_memory_gb"]
        fits = "YES" if result["fits_80gb"] else "NO"

        print(f"{cfg['name']:<8} {cfg['llm']:<12} {cfg['vit']:<12} {deepstack_str:<18} "
              f"{params_gb:<12.1f} {act_gb:<10.1f} {total_gb:<12.1f} {fits:<10}")

    print("-" * 100)
    print("\nNotes:")
    print("  - Memory = params×20B (AdamW bf16) + activations (with recompute) + 3GB CUDA overhead")
    print("  - EP=8 shards MoE experts across 8 GPUs; attention is replicated")
    print("  - ViT is replicated across all GPUs (no EP/TP for ViT)")
    print("  - Activation checkpointing: recompute_granularity=full, recompute_num_layers=1")
    print("  - Seq length: 1024, micro_batch_size: 1")


if __name__ == "__main__":
    main()
