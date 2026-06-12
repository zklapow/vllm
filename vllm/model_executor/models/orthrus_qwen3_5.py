# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Orthrus-Qwen3.6: Qwen3.5-family hybrid backbone plus per-layer
diffusion-head (``*_diff``) projections, served as a text-only CausalLM.

The Orthrus checkpoint is the frozen AR base under stock HF Qwen3.5 names plus
merged diffusion-head tensors carrying a ``_diff`` suffix (LoRA already folded
offline by ``export_orthrus_hf.py`` in the prototype repo):

* full-attention layers: ``self_attn.{q,k,v,o}_proj_diff.weight`` and
  ``self_attn.{q,k}_norm_diff.weight``
* GatedDeltaNet layers: ``linear_attn.{in_proj_qkv,in_proj_z,out_proj,
  in_proj_a,in_proj_b}_diff.weight`` and ``linear_attn.conv1d_diff.weight``

The AR/verify path is untouched stock Qwen3.5: same modules, same weights,
same kernels. The diff modules feed ``diffusion_forward()`` — the parallel
mask-token drafting pass used by the Orthrus speculative-decoding proposer.

NOTE: the diff modules are plain (non-tensor-parallel) layers; M1 targets
TP=1, bs=1 greedy. See docs/orthrus/INTEGRATION_PLAN.md.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3_5RMSNorm,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .interfaces import IsHybrid, SupportsMRoPE
from .qwen3_5 import Qwen3_5ForCausalLMBase
from .utils import PPMissingLayer

logger = init_logger(__name__)


class OrthrusQwen3_5ForCausalLM(Qwen3_5ForCausalLMBase, IsHybrid, SupportsMRoPE):
    """Qwen3.5/3.6 hybrid CausalLM with Orthrus diffusion-head modules."""

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list,
    ) -> tuple[torch.Tensor, int]:
        """Text-only M-RoPE: T/H/W positions are all the token index.

        The checkpoint config carries ``mrope_section`` (interleaved), which
        with identical T/H/W rows reduces exactly to standard partial RoPE.
        """
        assert not mm_features, "Orthrus serves text-only requests"
        n = len(input_tokens)
        llm_positions = (
            torch.arange(n, dtype=torch.long).unsqueeze(0).expand(3, -1)
        )
        return llm_positions, 0

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = vllm_config.model_config.hf_text_config
        # Orthrus metadata block written by the export script
        # (block_size, mask_token_id, ...).
        self.orthrus_config: dict = getattr(config, "orthrus", None) or {}
        self._attach_diff_modules(config)

    def _attach_diff_modules(self, config) -> None:
        """Attach the diffusion-head projections onto the stock layers.

        Module attribute names are chosen so that the resulting parameter
        names (``model.layers.<i>.self_attn.q_proj_diff.weight``, ...) match
        the exported checkpoint exactly.
        """
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", hidden_size // num_heads)
        attn_bias = getattr(config, "attention_bias", False)

        key_dim = config.linear_key_head_dim * config.linear_num_key_heads
        value_dim = config.linear_value_head_dim * config.linear_num_value_heads
        conv_dim = 2 * key_dim + value_dim
        conv_kernel = config.linear_conv_kernel_dim
        num_v_heads = config.linear_num_value_heads

        num_attached = 0
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if layer.layer_type == "full_attention":
                attn = layer.self_attn
                # q_proj_diff packs per-head [query, gate] like the AR q_proj.
                attn.q_proj_diff = nn.Linear(
                    hidden_size, num_heads * head_dim * 2, bias=attn_bias
                )
                attn.k_proj_diff = nn.Linear(
                    hidden_size, num_kv_heads * head_dim, bias=attn_bias
                )
                attn.v_proj_diff = nn.Linear(
                    hidden_size, num_kv_heads * head_dim, bias=attn_bias
                )
                attn.o_proj_diff = nn.Linear(
                    num_heads * head_dim, hidden_size, bias=attn_bias
                )
                attn.q_norm_diff = Qwen3_5RMSNorm(head_dim, eps=config.rms_norm_eps)
                attn.k_norm_diff = Qwen3_5RMSNorm(head_dim, eps=config.rms_norm_eps)
            elif layer.layer_type == "linear_attention":
                dn = layer.linear_attn
                # Merged-LoRA projections (residual + scaling*(A@B).T folded
                # offline) + diffusion gates + diffusion conv.
                dn.in_proj_qkv_diff = nn.Linear(hidden_size, conv_dim, bias=False)
                dn.in_proj_z_diff = nn.Linear(hidden_size, value_dim, bias=False)
                dn.out_proj_diff = nn.Linear(value_dim, hidden_size, bias=False)
                dn.in_proj_a_diff = nn.Linear(hidden_size, num_v_heads, bias=False)
                dn.in_proj_b_diff = nn.Linear(hidden_size, num_v_heads, bias=False)
                dn.conv1d_diff = nn.Conv1d(
                    in_channels=conv_dim,
                    out_channels=conv_dim,
                    bias=False,
                    kernel_size=conv_kernel,
                    groups=conv_dim,
                    padding=conv_kernel - 1,
                )
            num_attached += 1
        logger.info("Orthrus: attached diff modules to %d layers", num_attached)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load ``*_diff`` tensors directly; delegate the rest to stock loading.

        The stock Qwen3_5 stacked-params mapping must not see the diff names:
        e.g. ``in_proj_qkv_diff`` contains the shard name ``in_proj_qkv`` and
        would be silently renamed to a nonexistent ``in_proj_qkvz_diff``.
        """
        params_dict = dict(self.named_parameters())
        diff_loaded: set[str] = set()

        def base_weights():
            for name, loaded_weight in weights:
                if "_diff" not in name:
                    yield name, loaded_weight
                    continue
                if name not in params_dict:
                    raise KeyError(
                        f"Orthrus diff tensor {name} has no matching parameter"
                    )
                param = params_dict[name]
                if param.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Orthrus diff tensor {name}: checkpoint shape "
                        f"{tuple(loaded_weight.shape)} != param shape "
                        f"{tuple(param.shape)}"
                    )
                default_weight_loader(param, loaded_weight)
                diff_loaded.add(name)

        loaded = super().load_weights(base_weights())
        logger.info("Orthrus: loaded %d diff tensors", len(diff_loaded))
        return loaded | diff_loaded

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()
