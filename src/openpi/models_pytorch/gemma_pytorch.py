from contextlib import nullcontext
from typing import Literal

import pytest
import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward_shared_attention(
        self,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        inputs_embeds: list[torch.FloatTensor | None],
        adarms_cond: list[torch.Tensor | None] | None = None,
        profiler=None,
    ) -> list[torch.FloatTensor | None]:
        def record_section(name: str | None = None, *, allocation: dict[str, float] | None = None):
            if profiler is None:
                return nullcontext()
            return profiler.record(name, allocation=allocation)

        if adarms_cond is None:
            adarms_cond = [None, None]

        branch_specs = []
        if inputs_embeds[0] is not None:
            branch_specs.append((0, self.paligemma.language_model, "llm_ms"))
        if inputs_embeds[1] is not None:
            branch_specs.append((1, self.gemma_expert.model, "action_expert_ms"))
        if not branch_specs:
            raise ValueError("at least one branch input is required")

        active_inputs = [inputs_embeds[slot_idx] for slot_idx, _, _ in branch_specs]
        active_conds = [adarms_cond[slot_idx] for slot_idx, _, _ in branch_specs]
        num_layers = len(branch_specs[0][1].layers)

        use_gradient_checkpointing = self.training and (
            any(
                hasattr(model, "gradient_checkpointing") and model.gradient_checkpointing
                for _, model, _ in branch_specs
            )
            or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing)
        )

        def compute_layer_complete(layer_idx, layer_inputs, attention_mask, position_ids, layer_conds):
            query_states = []
            key_states = []
            value_states = []
            gates = []

            for branch_idx, (_, model, bucket_name) in enumerate(branch_specs):
                hidden_states = layer_inputs[branch_idx]
                with record_section(bucket_name):
                    layer = model.layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=layer_conds[branch_idx])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)

            query_states = torch.cat(query_states, dim=2)
            key_states = torch.cat(key_states, dim=2)
            value_states = torch.cat(value_states, dim=2)

            dummy_tensor = torch.zeros(
                query_states.shape[0],
                query_states.shape[2],
                query_states.shape[-1],
                device=query_states.device,
                dtype=query_states.dtype,
            )
            cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
            query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=1
            )

            attention_allocation = {
                bucket_name: float(layer_inputs[branch_idx].shape[1])
                for branch_idx, (_, _, bucket_name) in enumerate(branch_specs)
            }
            scaling = branch_specs[0][1].layers[layer_idx].self_attn.scaling
            with record_section("shared_attention_ms", allocation=attention_allocation):
                att_output, _ = modeling_gemma.eager_attention_forward(
                    branch_specs[0][1].layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )

            batch_size = query_states.shape[0]
            head_dim = branch_specs[0][1].layers[layer_idx].self_attn.head_dim
            att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

            outputs_embeds = []
            start_pos = 0
            for branch_idx, (_, model, bucket_name) in enumerate(branch_specs):
                hidden_states = layer_inputs[branch_idx]
                with record_section(bucket_name):
                    layer = model.layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    model_att_output = att_output
                    if model_att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        model_att_output = model_att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(model_att_output[:, start_pos:end_pos])

                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[branch_idx])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=layer_conds[branch_idx])
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

            return outputs_embeds

        for layer_idx in range(num_layers):
            if use_gradient_checkpointing:
                active_inputs = torch.utils.checkpoint.checkpoint(
                    compute_layer_complete,
                    layer_idx,
                    active_inputs,
                    attention_mask,
                    position_ids,
                    active_conds,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                active_inputs = compute_layer_complete(layer_idx, active_inputs, attention_mask, position_ids, active_conds)

        def compute_final_norms(layer_inputs, layer_conds):
            outputs_embeds = []
            for branch_idx, (_, model, _) in enumerate(branch_specs):
                out_emb, _ = model.norm(layer_inputs[branch_idx], cond=layer_conds[branch_idx])
                outputs_embeds.append(out_emb)
            return outputs_embeds

        if use_gradient_checkpointing:
            active_outputs = torch.utils.checkpoint.checkpoint(
                compute_final_norms,
                active_inputs,
                active_conds,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            active_outputs = compute_final_norms(active_inputs, active_conds)

        outputs: list[torch.FloatTensor | None] = [None, None]
        for branch_idx, (slot_idx, _, _) in enumerate(branch_specs):
            outputs[slot_idx] = active_outputs[branch_idx]
        return outputs

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
        profiler=None,
    ):
        def record_section(name: str | None = None, *, allocation: dict[str, float] | None = None):
            if profiler is None:
                return nullcontext()
            return profiler.record(name, allocation=allocation)

        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            with record_section("llm_ms"):
                prefix_output = self.paligemma.language_model.forward(
                    inputs_embeds=inputs_embeds[0],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
                )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            with record_section("action_expert_ms"):
                suffix_output = self.gemma_expert.model.forward(
                    inputs_embeds=inputs_embeds[1],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
                )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            outputs_embeds = self.forward_shared_attention(
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                adarms_cond=adarms_cond,
                profiler=profiler,
            )
            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values
