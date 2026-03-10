import inspect
import os
from typing import Any, Dict, Tuple

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.safe_config_loader import safe_load_config_with_retry
from megatron.core import parallel_state, tensor_parallel


def _get_image_size_and_patch(config) -> Tuple[int, int]:
    image_size = getattr(config, "image_size", 448)
    if isinstance(image_size, (tuple, list)):
        image_size = image_size[0]
    patch_size = getattr(config, "patch_size", 14)
    if isinstance(patch_size, (tuple, list)):
        patch_size = patch_size[0]
    return int(image_size), int(patch_size)


def _prepare_vit_inputs(vision_model, device: torch.device) -> Dict[str, Any]:
    signature = inspect.signature(vision_model.forward)
    params = signature.parameters

    img_size, patch_size = _get_image_size_and_patch(vision_model.config)
    batch = 1

    kwargs: Dict[str, Any] = {}
    if "pixel_values" in params:
        pixel_values = torch.randn(batch, 3, img_size, img_size, device=device)
        kwargs["pixel_values"] = pixel_values
    elif "hidden_states" in params:
        grid_h = img_size // patch_size
        grid_w = img_size // patch_size
        grid_t = 1
        seq_len = grid_t * grid_h * grid_w
        in_channels = int(getattr(vision_model.config, "in_channels", 3))
        temporal_patch = int(getattr(vision_model.config, "temporal_patch_size", 2))
        patch_volume = in_channels * temporal_patch * patch_size * patch_size
        hidden_states = torch.randn(seq_len, patch_volume, device=device)
        kwargs["hidden_states"] = hidden_states
    else:
        raise RuntimeError(
            "Unexpected vision model signature: missing pixel_values or hidden_states."
        )

    if "grid_thw" in params:
        grid_h = img_size // patch_size
        grid_w = img_size // patch_size
        grid_thw = torch.tensor([[1, grid_h, grid_w]], device=device, dtype=torch.int64)
        kwargs["grid_thw"] = grid_thw

    return kwargs


def _broadcast_inputs(kwargs: Dict[str, Any], src: int = 0) -> None:
    for value in kwargs.values():
        if torch.is_tensor(value):
            torch.distributed.broadcast(value, src=src)


def _hash_tensor(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    return torch.tensor(
        [
            value.mean().item(),
            value.std(unbiased=False).item(),
            value.abs().max().item(),
        ],
        device=value.device,
    )


def _extract_output_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
    if isinstance(output, dict):
        for item in output.values():
            if torch.is_tensor(item):
                return item
    raise RuntimeError("Vision model output did not contain a tensor.")


def main() -> None:
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    world_size = torch.distributed.get_world_size()
    if world_size != 4:
        raise RuntimeError(f"Expected world_size=4, got {world_size}.")

    if torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"Need at least {world_size} GPUs, found {torch.cuda.device_count()}."
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
    )
    tensor_parallel.model_parallel_cuda_manual_seed(1234)

    hf_config = safe_load_config_with_retry("Qwen/Qwen3-VL-8B-Instruct")
    bridge = AutoBridge.from_hf_config(hf_config)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    model_cfg.tensor_model_parallel_size = 4
    model_cfg.finalize()

    torch.manual_seed(1234)
    model = model_cfg.provide(pre_process=True, post_process=True)
    vision_model = model.vision_model
    vision_model.eval()

    first_param = next(vision_model.parameters())
    param_hash = _hash_tensor(first_param.detach())

    kwargs = _prepare_vit_inputs(vision_model, device)
    if torch.distributed.get_rank() == 0:
        _broadcast_inputs(kwargs, src=0)
    else:
        _broadcast_inputs(kwargs, src=0)

    with torch.no_grad():
        output = vision_model(**kwargs)

    output_tensor = _extract_output_tensor(output)
    input_tensor = next(v for v in kwargs.values() if torch.is_tensor(v))

    input_hash = _hash_tensor(input_tensor)
    output_hash = _hash_tensor(output_tensor)

    gathered_param = [torch.zeros_like(param_hash) for _ in range(world_size)]
    gathered_input = [torch.zeros_like(input_hash) for _ in range(world_size)]
    gathered_output = [torch.zeros_like(output_hash) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_param, param_hash)
    torch.distributed.all_gather(gathered_input, input_hash)
    torch.distributed.all_gather(gathered_output, output_hash)

    if torch.distributed.get_rank() == 0:
        print("ViT first-param hash per rank (mean, std, max):")
        for idx, tensor in enumerate(gathered_param):
            print(f"  rank {idx}: {tensor.tolist()}")
        print("ViT input hash per rank (mean, std, max):")
        for idx, tensor in enumerate(gathered_input):
            print(f"  rank {idx}: {tensor.tolist()}")
        print("ViT output hash per rank (mean, std, max):")
        for idx, tensor in enumerate(gathered_output):
            print(f"  rank {idx}: {tensor.tolist()}")

    torch.distributed.barrier()
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
