from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
from pathlib import Path
import time
import traceback

import jax
import torch

import openpi.training.config as _config
import openpi.training.data_loader as _data
from openpi.training import pi05_batch_profile_sweep as _sweep
import openpi.training.profiling as _profiling
import train_pytorch as _train_pytorch


def configure_default_cache_env() -> None:
    hf_home = os.environ.setdefault("HF_HOME", "/mnt/local_storage/huggingface")
    os.environ.setdefault("HF_HUB_CACHE", f"{hf_home}/hub")
    os.environ.setdefault("XDG_CACHE_HOME", "/mnt/local_storage/.cache")
    os.environ.setdefault("OPENPI_DATA_HOME", "/mnt/local_storage/.cache/openpi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PI0.5 packed training submodules on one real batch.")
    parser.add_argument("--config-name", default="pi05_libero")
    parser.add_argument("--exp-name", default="pi05_libero_module_benchmark")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--pytorch-weight-path", required=True)
    parser.add_argument("--vision-encoder-image-mode", default="packed", choices=("iterative", "packed"))
    parser.add_argument("--precision", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--module-step-kind", default="forward_only", choices=("forward_only", "train_like"))
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["vision", "llm", "action", "entire"],
        choices=("vision", "llm", "action", "entire"),
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> _config.TrainConfig:
    config = _config.get_config(args.config_name)
    model = config.model
    if hasattr(model, "vision_encoder_image_mode"):
        model = dataclasses.replace(model, vision_encoder_image_mode=args.vision_encoder_image_mode)
    return dataclasses.replace(
        config,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pytorch_weight_path=args.pytorch_weight_path,
        pytorch_training_precision=args.precision,
        wandb_enabled=False,
        model=model,
    )


def reset_runtime_state(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def clear_gradients(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.detach_()
            parameter.grad = None


def measure_region(device: torch.device, fn):
    reset_runtime_state(device)
    tracker = _profiling.PeakMemoryTracker(device)
    tracker.reset()
    started_at = time.perf_counter()
    outputs = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    metrics = tracker.snapshot()
    return outputs, elapsed_ms, metrics


def run_vision(model, images, img_masks):
    if model.config.vision_encoder_image_mode == "packed":
        return model._embed_images_packed(images, img_masks)
    return model._embed_images_iterative(images, img_masks)


def run_llm(model, image_embs, image_pad_masks, image_att_masks, lang_tokens, lang_masks):
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix_from_image_embeddings(
        image_embs,
        image_pad_masks,
        image_att_masks,
        lang_tokens,
        lang_masks,
    )
    prefix_out, _ = model.forward_transformer(
        prefix_embs=prefix_embs,
        prefix_pad_masks=prefix_pad_masks,
        prefix_att_masks=prefix_att_masks,
    )
    return prefix_out


def run_action(model, state, x_t, timestep):
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(state, x_t, timestep)
    _, suffix_out = model.forward_transformer(
        suffix_embs=suffix_embs,
        suffix_pad_masks=suffix_pad_masks,
        suffix_att_masks=suffix_att_masks,
        adarms_cond=adarms_cond,
    )
    suffix_out = suffix_out[:, -model.config.action_horizon :].to(dtype=torch.float32)
    return model.action_out_proj(suffix_out)


def build_optimizer(config: _config.TrainConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    return build_optimizer_for_parameters(config, list(model.parameters()))


def build_optimizer_for_parameters(
    config: _config.TrainConfig,
    parameters: list[torch.nn.Parameter],
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        parameters,
        lr=config.lr_schedule.peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )


def run_entire(model, optimizer, observation, actions, clip_gradient_norm: float):
    losses = model(observation, actions)
    if isinstance(losses, list | tuple):
        losses = torch.stack(losses)
    elif not isinstance(losses, torch.Tensor):
        losses = torch.tensor(losses, device=actions.device, dtype=torch.float32)
    loss = losses.mean()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_gradient_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    clear_gradients(model)
    return {
        "loss": float(loss.item()),
        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
    }


def get_module_parameters(model: torch.nn.Module, mode: str) -> list[torch.nn.Parameter]:
    prefix_map = {
        "vision": (
            "paligemma_with_expert.paligemma.model.vision_tower",
            "paligemma_with_expert.paligemma.model.multi_modal_projector",
        ),
        "llm": ("paligemma_with_expert.paligemma.model.language_model",),
        "action": (
            "paligemma_with_expert.gemma_expert.model",
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ),
    }
    prefixes = prefix_map[mode]
    parameters = [param for name, param in model.named_parameters() if any(name.startswith(prefix) for prefix in prefixes)]
    if not parameters:
        raise ValueError(f"No parameters matched mode={mode!r}")
    return parameters


def set_trainable_parameters(model: torch.nn.Module, trainable_parameters: list[torch.nn.Parameter]) -> list[tuple[torch.nn.Parameter, bool]]:
    trainable_ids = {id(param) for param in trainable_parameters}
    original_states: list[tuple[torch.nn.Parameter, bool]] = []
    for param in model.parameters():
        original_states.append((param, param.requires_grad))
        param.requires_grad_(id(param) in trainable_ids)
    return original_states


def restore_trainable_parameters(original_states: list[tuple[torch.nn.Parameter, bool]]) -> None:
    for param, requires_grad in original_states:
        param.requires_grad_(requires_grad)


def dummy_loss_from_outputs(outputs) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        if not outputs.requires_grad:
            raise TypeError("Tensor output does not require grad.")
        return outputs.float().square().mean()
    if isinstance(outputs, list | tuple):
        losses = []
        for output in outputs:
            try:
                losses.append(dummy_loss_from_outputs(output))
            except TypeError:
                continue
        if not losses:
            raise TypeError("Expected at least one differentiable tensor output for dummy loss construction.")
        return torch.stack(losses).sum()
    raise TypeError(f"Unsupported output type for dummy loss: {type(outputs)!r}")


def run_module_train_like(forward_fn, optimizer: torch.optim.Optimizer) -> dict[str, object]:
    outputs = forward_fn()
    loss = dummy_loss_from_outputs(outputs)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "dummy_loss": float(loss.item()),
    }


def main() -> int:
    args = parse_args()
    configure_default_cache_env()
    _train_pytorch.init_logging()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = build_config(args)
    device = torch.device(args.device)
    _train_pytorch.set_seed(config.seed, 0)

    payload: dict[str, object] = {
        "schema_version": 1,
        "config_name": config.name,
        "batch_size": config.batch_size,
        "device": str(device),
        "vision_encoder_image_mode": args.vision_encoder_image_mode,
        "pytorch_weight_path": args.pytorch_weight_path,
        "module_step_kind": args.module_step_kind,
        "modes": list(args.modes),
        "status": "ok",
        "results": {},
    }

    try:
        data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        observation, actions = next(iter(data_loader))
        observation = jax.tree.map(lambda x: x.to(device), observation)
        actions = actions.to(torch.float32).to(device)

        model, _, _ = _train_pytorch.build_pytorch_model(config, device)
        _train_pytorch.load_pytorch_weights_if_needed(model, config)
        model.train()

        images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=True)
        noise = model.sample_noise(actions.shape, actions.device)
        timestep = model.sample_time(actions.shape[0], actions.device)
        time_expanded = timestep[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions

        image_embs, image_pad_masks, image_att_masks = run_vision(model, images, img_masks)
        image_embs = [emb.detach() for emb in image_embs]
        mode_order = [mode for mode in args.modes if mode != "entire"]
        if "entire" in args.modes:
            mode_order.append("entire")

        for mode in mode_order:
            clear_gradients(model)
            module_optimizer = None
            restore_states = None
            if mode == "vision":
                forward_fn = lambda: run_vision(model, images, img_masks)
            elif mode == "llm":
                forward_fn = lambda: run_llm(
                    model,
                    image_embs,
                    image_pad_masks,
                    image_att_masks,
                    lang_tokens,
                    lang_masks,
                )
            elif mode == "action":
                forward_fn = lambda: run_action(model, state, x_t, timestep)
            else:
                entire_optimizer = build_optimizer(config, model)
                run = lambda: run_entire(
                    model,
                    entire_optimizer,
                    observation,
                    actions,
                    config.optimizer.clip_gradient_norm,
                )

            if mode != "entire":
                if args.module_step_kind == "train_like":
                    module_parameters = get_module_parameters(model, mode)
                    restore_states = set_trainable_parameters(model, module_parameters)
                    module_optimizer = build_optimizer_for_parameters(config, module_parameters)
                    run = lambda forward_fn=forward_fn, module_optimizer=module_optimizer: run_module_train_like(
                        forward_fn,
                        module_optimizer,
                    )
                else:
                    run = forward_fn

            try:
                outputs, elapsed_ms, peak_memory = measure_region(device, run)
                result = {
                    "status": "ok",
                    "elapsed_ms": elapsed_ms,
                    **peak_memory,
                }
                if isinstance(outputs, dict):
                    result.update(outputs)
                payload["results"][mode] = result
                del outputs
            finally:
                clear_gradients(model)
                if restore_states is not None:
                    restore_trainable_parameters(restore_states)
                del module_optimizer

    except Exception as exc:  # noqa: BLE001
        payload["status"] = "oom" if _sweep.is_cuda_oom("".join(traceback.format_exception(exc))) else "error"
        payload["error"] = "".join(traceback.format_exception(exc))
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 88 if payload["status"] == "oom" else 1

    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
