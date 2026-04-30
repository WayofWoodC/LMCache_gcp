# SPDX-License-Identifier: Apache-2.0
# Standard
from abc import ABC, abstractmethod
import inspect

# Third Party
from torch import nn
import torch

# First Party
from lmcache.v1.compute.attention.utils import infer_attn_backend_from_vllm
from lmcache.v1.compute.positional_encoding import get_fused_rope

# TODO(Jiayi): A few things need to be tested/supported:
# TP, PP, Multimodal


class LMCBaseModel(nn.Module, ABC):
    def __init__(
        self,
        vllm_model,
        blender,
        enable_sparse: bool = False,
    ):
        super().__init__()
        self.vllm_model = vllm_model

        self.num_layers = len(vllm_model.model.layers)

        self.vllm_attn_layers = []
        self.lmc_attn_layers = []
        for i in range(self.num_layers):
            vllm_attn = vllm_model.model.layers[i].self_attn.attn
            self.vllm_attn_layers.append(vllm_attn)

            self.lmc_attn_layers.append(
                infer_attn_backend_from_vllm(vllm_attn, enable_sparse)
            )

        # NOTE(Jiayi): better not to pass the blender in init
        # if we want to make this LMCModel more general.
        self.blender = blender

        # remove hard code
        rotary_emb = vllm_model.model.layers[0].self_attn.rotary_emb
        head_dim = rotary_emb.head_size
        max_position_embeddings = rotary_emb.max_position_embeddings
        rope_scaling = None
        base = rotary_emb.base
        is_neox_style = rotary_emb.is_neox_style
        dtype = rotary_emb.dtype
        self.fused_rotary_emb = get_fused_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=base,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )
        self._embed_input_ids_fn = self._resolve_embed_input_ids_fn()

    def _resolve_embed_input_ids_fn(self):
        """Resolve a compatible embedding entrypoint across vLLM versions."""
        model = self.vllm_model

        # Legacy path used by older integrations.
        get_input_embeddings = getattr(model, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            try:
                sig = inspect.signature(get_input_embeddings)
                # For bound methods:
                # - old style: get_input_embeddings(input_ids)
                # - HF style: get_input_embeddings()
                takes_input_ids = len(sig.parameters) >= 1
            except (TypeError, ValueError):
                takes_input_ids = False

            if takes_input_ids:
                def _via_get_input_embeddings(input_ids: torch.Tensor) -> torch.Tensor:
                    return get_input_embeddings(input_ids)

                return _via_get_input_embeddings

            # HF-style API: get_input_embeddings() -> embedding module.
            emb_layer = get_input_embeddings()
            if callable(emb_layer):

                def _via_hf_style(input_ids: torch.Tensor) -> torch.Tensor:
                    return emb_layer(input_ids)

                return _via_hf_style

        # vLLM 0.11+ commonly exposes embed_input_ids at top-level model.
        embed_input_ids = getattr(model, "embed_input_ids", None)
        if callable(embed_input_ids):

            def _via_embed_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
                return embed_input_ids(input_ids)

            return _via_embed_input_ids

        inner = getattr(model, "model", None)
        if inner is not None:
            inner_embed_input_ids = getattr(inner, "embed_input_ids", None)
            if callable(inner_embed_input_ids):

                def _via_inner_embed_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
                    return inner_embed_input_ids(input_ids)

                return _via_inner_embed_input_ids

            embed_tokens = getattr(inner, "embed_tokens", None)
            if callable(embed_tokens):

                def _via_embed_tokens(input_ids: torch.Tensor) -> torch.Tensor:
                    return embed_tokens(input_ids)

                return _via_embed_tokens

        raise AttributeError(
            f"Cannot resolve embedding function for vLLM model type {type(model).__name__}"
        )

    @abstractmethod
    def _process_qkv(self, q, k, v, layer):
        """Process QKV tensors. Model-specific implementation."""
        pass

    @torch.compile
    def compute_layer(
        self,
        input_ids: torch.Tensor,
    ):
        input_ids = input_ids.cuda()
        hidden_states = self._embed_input_ids_fn(input_ids)
        residual = None

        attn_output = None

        # TODO(Jiayi): Need to build `attn_metadata` more elegantly.
        attn_metadata = self.lmc_attn_layers[0].init_attn_metadata(
            input_ids=input_ids,
        )

        for idx, layer in enumerate(
            self.vllm_model.model.layers[
                self.vllm_model.model.start_layer : self.vllm_model.model.end_layer
            ]
        ):
            # TODO(Jiayi) The last layer doesn't have to be computed
            # hidden_states, residual = layer(positions, hidden_states, residual)

            # Self Attention
            if residual is None:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            else:
                hidden_states, residual = layer.input_layernorm(hidden_states, residual)
            # hidden_states = self.self_attn(positions=positions,
            #                            hidden_states=hidden_states)

            qkv, _ = layer.self_attn.qkv_proj(hidden_states)
            q, k, v = qkv.split(
                [
                    layer.self_attn.q_size,
                    layer.self_attn.kv_size,
                    layer.self_attn.kv_size,
                ],
                dim=-1,
            )

            # Model-specific QKV processing
            q, k, v = self._process_qkv(q, k, v, layer)

            q, k, v, residual, attn_output, attn_metadata = self.blender.process_qkv(
                q, k, v, residual, idx, attn_output, attn_metadata
            )

            num_heads = self.vllm_attn_layers[idx].num_heads
            num_kv_heads = self.vllm_attn_layers[idx].num_kv_heads
            head_size = self.vllm_attn_layers[idx].head_size

            q = q.view(-1, num_heads, head_size)
            k = k.view(-1, num_kv_heads, head_size)
            v = v.view(-1, num_kv_heads, head_size)
            attn_output = attn_output.view(-1, num_heads, head_size)

            attn_output = self.lmc_attn_layers[idx].forward_contiguous(
                q, k, v, attn_output, attn_metadata
            )

            attn_output = attn_output.view(-1, num_heads * head_size)
            k = k.view(-1, num_kv_heads * head_size)
            v = v.view(-1, num_kv_heads * head_size)

            hidden_states, _ = layer.self_attn.o_proj(attn_output)

            # Fully Connected
            hidden_states, residual = layer.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states = layer.mlp(hidden_states)

            yield
