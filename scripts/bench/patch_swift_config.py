#!/usr/bin/env python3
"""Create a patched HF config.json with reduced ViT depth + LLM layers for SWIFT.

Creates a minimal HF-compatible model directory at <output-dir> with:
  - config.json (patched num_hidden_layers and vision_config.depth)
  - tokenizer files (symlinked from original)
Prints the output directory path to stdout.

Usage:
  python patch_swift_config.py --model Qwen/Qwen3-VL-30B-A3B-Instruct \
      --num-llm-layers 24 --vit-depth 14 --output-dir /mnt/local_storage/qwen3vl_reduced
"""

import argparse
import json
import os
import sys
from pathlib import Path


def find_snapshot_dir(model_name: str) -> Path:
    """Find the HF model snapshot directory."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    # Convert model name to cache directory format
    cache_name = f"models--{model_name.replace('/', '--')}"
    cache_dir = Path(hf_home) / "hub" / cache_name

    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Model cache not found at {cache_dir}. "
            f"Download it first: huggingface-cli download {model_name}"
        )

    # Find the latest snapshot
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"No snapshots found at {snapshots_dir}")

    snapshots = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not snapshots:
        raise FileNotFoundError(f"No snapshot directories in {snapshots_dir}")

    return snapshots[0]


def patch_config(
    model_name: str,
    num_llm_layers: int,
    vit_depth: int,
    output_dir: str,
) -> str:
    """Create patched config directory.

    Returns the output directory path.
    """
    snapshot_dir = find_snapshot_dir(model_name)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Read and patch config.json
    config_path = snapshot_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Patch LLM layers
    if "num_hidden_layers" in config:
        config["num_hidden_layers"] = num_llm_layers
    if "text_config" in config and "num_hidden_layers" in config["text_config"]:
        config["text_config"]["num_hidden_layers"] = num_llm_layers

    # Patch ViT depth
    if "vision_config" in config:
        config["vision_config"]["depth"] = vit_depth
        # Also update deepstack_layer_list if present (for SWIFT's auto deepstack)
        if "fullstack_layer_list" in config["vision_config"]:
            # Evenly space deepstack layers within the reduced ViT
            original = config["vision_config"]["fullstack_layer_list"]
            # Keep same number of deepstack points, but scale to new depth
            num_points = len(original)
            step = max(1, vit_depth // (num_points + 1))
            config["vision_config"]["fullstack_layer_list"] = [
                step * (i + 1) for i in range(num_points)
            ]

    # Write patched config
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Symlink ALL files from original snapshot (tokenizer, weights, etc.)
    # except config.json which we've already written with patches.
    # SWIFT needs model weights for HF->Megatron conversion.
    for item in snapshot_dir.iterdir():
        if item.name == "config.json":
            continue  # skip - we wrote our own patched version
        dst = out / item.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(item)

    return str(out)


def main():
    parser = argparse.ArgumentParser(description="Patch HF config for reduced-layer SWIFT benchmark")
    parser.add_argument("--model", required=True, help="HF model name (e.g. Qwen/Qwen3-VL-30B-A3B-Instruct)")
    parser.add_argument("--num-llm-layers", type=int, required=True, help="Number of LLM layers")
    parser.add_argument("--vit-depth", type=int, required=True, help="ViT depth (number of vision layers)")
    parser.add_argument("--output-dir", required=True, help="Output directory for patched config")
    args = parser.parse_args()

    result = patch_config(args.model, args.num_llm_layers, args.vit_depth, args.output_dir)
    print(result)


if __name__ == "__main__":
    main()
