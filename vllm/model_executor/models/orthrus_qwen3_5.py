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

import os
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3_5RMSNorm,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .interfaces import IsHybrid, SupportsMRoPE
from .qwen3_5 import Qwen3_5ForCausalLMBase
from .utils import PPMissingLayer

logger = init_logger(__name__)

# M3 drafter-speed knobs (drafter-only: the target model verifies every
# draft, so these can only move acceptance, never correctness).
#   ORTHRUS_FP8_DRAFT=diff  quantize the *_diff projections to fp8 in place
#   ORTHRUS_FP8_DRAFT=full  also fp8-copy the shared MLP + lm_head weights
#                           for the diffusion pass (AR/verify stays bf16)
#   ORTHRUS_FWD_ONLY_SCAN=1 skip the backward DN scan (halves scan cost;
#                           the head was trained bidirectional, expect an
#                           acceptance drop)
_FWD_ONLY_SCAN = os.environ.get("ORTHRUS_FWD_ONLY_SCAN", "0") == "1"

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max


def _gemma_rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Inline GemmaRMSNorm (x * (1+w), fp32 mul then cast).

    The module's forward routes through an opaque IR op that stays eager at
    the torch.compile boundary (161 separate reduce+elementwise kernels per
    drafter replay); pure-torch math here lets inductor fuse them into the
    surrounding graph."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * (1.0 + weight.float())).to(x.dtype)


def _quant_fp8_rowwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel e4m3 quantization of a (N, K) weight.

    Returns the fp8 weight (N, K) and a (1, N) fp32 scale laid out for
    ``torch._scaled_mm``'s rowwise scale_b."""
    amax = w.detach().abs().amax(dim=1, keepdim=True).float().clamp_(min=1e-12)
    scale = amax / _FP8_MAX
    wq = (w.float() / scale).clamp_(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
    return wq, scale.t().contiguous()


def _fp8_linear(
    x: torch.Tensor, wq: torch.Tensor, w_scale: torch.Tensor
) -> torch.Tensor:
    """W8A8 e4m3 GEMM: dynamic per-token activation scales (M, 1) x
    per-channel weight scales (1, N) via cuBLASLt rowwise scaled_mm."""
    amax = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-6)
    x_scale = amax / _FP8_MAX
    xq = (x.float() / x_scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
    return torch._scaled_mm(
        xq,
        wq.t(),
        scale_a=x_scale,
        scale_b=w_scale,
        out_dtype=x.dtype,
        use_fast_accum=True,
    )


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
        self._fuse_diff_projections()
        self._quantize_draft_weights()
        return loaded | diff_loaded

    def _fuse_diff_projections(self) -> None:
        """Concatenate per-layer diff projection weights into single GEMMs.

        The eager diffusion pass is launch-bound (64 layers x many small
        matmuls); fusing the input projections cuts the launch count without
        changing the math (one bf16 GEMM split afterwards). The original
        nn.Linear modules are dropped to keep memory neutral.
        """
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if layer.layer_type == "full_attention":
                attn = layer.self_attn
                assert attn.q_proj_diff.bias is None
                # Plain tensors (not Parameters): created post-load, so they
                # must stay invisible to the loader's coverage tracking.
                attn.diff_qkv_weight = torch.cat(
                    [
                        attn.q_proj_diff.weight,
                        attn.k_proj_diff.weight,
                        attn.v_proj_diff.weight,
                    ],
                    dim=0,
                )
                del attn.q_proj_diff, attn.k_proj_diff, attn.v_proj_diff
            elif layer.layer_type == "linear_attention":
                dn = layer.linear_attn
                dn.diff_in_weight = torch.cat(
                    [
                        dn.in_proj_qkv_diff.weight,
                        dn.in_proj_z_diff.weight,
                        dn.in_proj_b_diff.weight,
                        dn.in_proj_a_diff.weight,
                    ],
                    dim=0,
                )
                del (
                    dn.in_proj_qkv_diff,
                    dn.in_proj_z_diff,
                    dn.in_proj_b_diff,
                    dn.in_proj_a_diff,
                )
        torch.cuda.empty_cache()

    def _quantize_draft_weights(self) -> None:
        """fp8-quantize the drafter-path GEMM weights (ORTHRUS_FP8_DRAFT).

        ``diff``: the *_diff projections are drafter-only, so they are
        quantized in place (halves their footprint). ``full``: additionally
        make fp8 *copies* of the weights the diffusion pass shares with the
        AR/verify path (per-layer MLP, lm_head) — the AR path must stay
        bf16-exact, so the originals are untouched. Runs before vLLM's KV
        memory profiling, so the extra footprint is accounted for
        automatically."""
        mode = os.environ.get("ORTHRUS_FP8_DRAFT", "0").lower()
        if mode in ("0", "", "off"):
            return
        full = mode in ("1", "full", "all")
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if layer.layer_type == "full_attention":
                attn = layer.self_attn
                attn.diff_qkv_weight_q, attn.diff_qkv_weight_s = (
                    _quant_fp8_rowwise(attn.diff_qkv_weight)
                )
                del attn.diff_qkv_weight
                attn.o_proj_diff_q, attn.o_proj_diff_s = _quant_fp8_rowwise(
                    attn.o_proj_diff.weight
                )
                del attn.o_proj_diff
            elif layer.layer_type == "linear_attention":
                dn = layer.linear_attn
                dn.diff_in_weight_q, dn.diff_in_weight_s = _quant_fp8_rowwise(
                    dn.diff_in_weight
                )
                del dn.diff_in_weight
                dn.out_proj_diff_q, dn.out_proj_diff_s = _quant_fp8_rowwise(
                    dn.out_proj_diff.weight
                )
                del dn.out_proj_diff
            if full:
                mlp = layer.mlp
                mlp.draft_gate_up_q, mlp.draft_gate_up_s = _quant_fp8_rowwise(
                    mlp.gate_up_proj.weight
                )
                mlp.draft_down_q, mlp.draft_down_s = _quant_fp8_rowwise(
                    mlp.down_proj.weight
                )
        if full:
            # Slice off vocab padding so an fp8 argmax can never return a
            # padding token id (padded rows quantize to logit 0, which could
            # win when every real logit is negative).
            vocab = getattr(
                self.lm_head, "org_vocab_size", self.lm_head.weight.shape[0]
            )
            self.draft_lm_head_q, self.draft_lm_head_s = _quant_fp8_rowwise(
                self.lm_head.weight[:vocab]
            )
        torch.cuda.empty_cache()
        logger.info(
            "Orthrus: fp8 draft weights ready (mode=%s, shared MLP/lm_head "
            "copies=%s)",
            mode,
            full,
        )

    # ------------------------------------------------------------------
    # Orthrus diffusion forward (M1: dense-fallback attention, bs=1, eager)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def diffusion_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_len: "int | torch.Tensor",
        attn_block_tables: dict[int, torch.Tensor],
        attn_block_size: int,
        gdn_state_indices: dict[int, int],
        gdn_conv_offset: "int | torch.Tensor" = 0,
    ) -> torch.Tensor:
        """One parallel diffusion pass over a K-token block.

        Args:
            input_ids: (K,) anchor token + K-1 mask tokens.
            positions: (K,) absolute positions ``[cur, cur+K)``.
            seq_len: committed prefix length ``cur`` (tokens whose KV/DN state
                is already in the caches). May be a 0-dim device tensor: then
                the pass runs in *static* (CUDA-graph capturable) mode — every
                page in ``attn_block_tables`` is gathered (fixed shape) and an
                additive bias built from ``seq_len`` masks the invalid slots.
            attn_block_tables: layer_idx -> page ids of this request in that
                layer's KV-cache group (hybrid models split layers across
                several groups, so the page ids are per-layer).
            attn_block_size: tokens per KV page.
            gdn_state_indices: layer_idx -> (conv_state_idx, ssm_state_idx)
                ints, or (1,) device tensors in static mode. Under spec decode
                the conv window lives in slot 0 (sliding, select via
                gdn_conv_offset) while the recurrent state of the committed
                prefix is the accepted per-draft-step slot.
            gdn_conv_offset: first column of the committed conv window inside
                the conv state slot (= num_accepted - 1; 0 without spec).
                0-dim device tensor in static mode.

        Returns:
            (K, vocab) logits. Position i predicts the token at ``cur+i+1``
            (same shifted convention as the AR head), so ``logits[:-1].argmax``
            are the K-1 drafted tokens following the anchor.

        Mirrors the prototype's ``is_diffusion_pass=True`` forward: full-attn
        layers read the committed prefix from the paged KV non-causally and a
        dense bidirectional K-block of diff K/V; GDN layers run the dual scan
        seeded by the request's recurrent state. Reads caches, writes nothing.
        """
        static_mode = isinstance(seq_len, torch.Tensor)
        if static_mode:
            # Fixed-shape prefix: gather every page, mask validity with one
            # additive bias shared by all attention layers.
            any_pages = next(iter(attn_block_tables.values()))
            prefix_len = any_pages.numel() * attn_block_size
            kv_pos = torch.arange(prefix_len, device=input_ids.device)
            num_tokens = input_ids.shape[0]
            bias = torch.zeros(
                prefix_len + num_tokens,
                dtype=self.model.embed_tokens.weight.dtype,
                device=input_ids.device,
            )
            bias[:prefix_len] = torch.where(
                kv_pos < seq_len, 0.0, float("-inf")
            ).to(bias.dtype)
            attn_bias = bias[None, None, None, :]
        else:
            prefix_len = seq_len
            attn_bias = None

        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            residual = hidden
            ln = layer.input_layernorm
            h = _gemma_rms(hidden, ln.weight, ln.variance_epsilon)
            if layer.layer_type == "linear_attention":
                idx = gdn_state_indices[layer.layer_idx]
                conv_idx, ssm_idx = idx if isinstance(idx, tuple) else (idx, idx)
                h = self._gdn_diffusion(
                    layer.linear_attn, h, conv_idx, ssm_idx, gdn_conv_offset
                )
            else:
                h = self._attn_diffusion(
                    layer.self_attn,
                    h,
                    positions,
                    prefix_len,
                    attn_block_tables[layer.layer_idx],
                    attn_block_size,
                    attn_bias,
                )
            hidden = residual + h
            residual = hidden
            ln = layer.post_attention_layernorm
            h = _gemma_rms(hidden, ln.weight, ln.variance_epsilon)
            mlp = layer.mlp
            if getattr(mlp, "draft_gate_up_q", None) is not None:
                gu = _fp8_linear(h, mlp.draft_gate_up_q, mlp.draft_gate_up_s)
                half = gu.shape[-1] // 2
                h = F.silu(gu[..., :half]) * gu[..., half:]
                h = _fp8_linear(h, mlp.draft_down_q, mlp.draft_down_s)
            else:
                h = mlp(h)
            hidden = residual + h
        hidden = _gemma_rms(
            hidden, self.model.norm.weight, self.model.norm.variance_epsilon
        )
        if getattr(self, "draft_lm_head_q", None) is not None:
            return _fp8_linear(hidden, self.draft_lm_head_q, self.draft_lm_head_s)
        return self.compute_logits(hidden)

    def _attn_diffusion(
        self,
        attn,
        h: torch.Tensor,
        positions: torch.Tensor,
        seq_len: int,
        block_table: torch.Tensor,
        block_size: int,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Diffusion attention: non-causal over [prefix KV from cache ‖ diff
        block KV], dense fallback (single softmax — no LSE merge needed).

        ``seq_len`` here is the number of prefix tokens gathered from the
        cache; in static mode that is all pages of ``block_table`` and
        ``attn_bias`` (additive, broadcast over heads/queries) carries the
        true-prefix validity instead."""
        num_tokens = h.shape[0]
        num_heads = attn.num_heads
        num_kv_heads = attn.num_kv_heads
        head_dim = attn.head_dim

        if getattr(attn, "diff_qkv_weight_q", None) is not None:
            qkv = _fp8_linear(h, attn.diff_qkv_weight_q, attn.diff_qkv_weight_s)
        else:
            qkv = F.linear(h, attn.diff_qkv_weight)
        qg, k, v = qkv.split(
            [
                num_heads * head_dim * 2,
                num_kv_heads * head_dim,
                num_kv_heads * head_dim,
            ],
            dim=-1,
        )
        q, gate = qg.view(num_tokens, num_heads, 2 * head_dim).chunk(2, dim=-1)
        gate = gate.reshape(num_tokens, num_heads * head_dim)
        q = _gemma_rms(
            q.contiguous(),
            attn.q_norm_diff.weight,
            attn.q_norm_diff.variance_epsilon,
        )
        k = _gemma_rms(
            k.reshape(num_tokens, num_kv_heads, head_dim),
            attn.k_norm_diff.weight,
            attn.k_norm_diff.variance_epsilon,
        )
        v = v.view(num_tokens, num_kv_heads, head_dim)
        q, k = attn.rotary_emb(
            positions, q.reshape(num_tokens, -1), k.reshape(num_tokens, -1)
        )
        q = q.view(num_tokens, num_heads, head_dim)
        k = k.view(num_tokens, num_kv_heads, head_dim)

        k_pref, v_pref = self._gather_prefix_kv(
            attn.attn, block_table, block_size, seq_len, num_kv_heads, head_dim
        )
        k_all = torch.cat([k_pref.to(k.dtype), k], dim=0)
        v_all = torch.cat([v_pref.to(v.dtype), v], dim=0)

        out = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k_all.transpose(0, 1).unsqueeze(0),
            v_all.transpose(0, 1).unsqueeze(0),
            attn_mask=attn_bias,
            is_causal=False,
            enable_gqa=True,
        )
        out = out.squeeze(0).transpose(0, 1).reshape(num_tokens, num_heads * head_dim)
        out = out * torch.sigmoid(gate)
        if getattr(attn, "o_proj_diff_q", None) is not None:
            return _fp8_linear(out, attn.o_proj_diff_q, attn.o_proj_diff_s)
        return attn.o_proj_diff(out)

    @staticmethod
    def _gather_prefix_kv(
        attn_impl,
        block_table: torch.Tensor,
        block_size: int,
        seq_len: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read the committed prefix K/V from the layer's paged KV cache.

        Normalizes the backend layout to (seq_len, num_kv_heads, head_dim).
        """
        kv = attn_impl.kv_cache
        if isinstance(kv, (list, tuple)):
            kv = kv[0]
        assert kv.numel() > 0, "KV cache not bound"
        if kv.ndim != 5:
            raise NotImplementedError(f"unexpected KV cache rank {kv.ndim}")
        if kv.shape[0] == 2 and kv.shape[1] != 2:
            k_cache, v_cache = kv[0], kv[1]
        elif kv.shape[1] == 2:
            k_cache, v_cache = kv[:, 0], kv[:, 1]
        else:
            raise NotImplementedError(f"unexpected KV cache shape {kv.shape}")
        # Normalize per-page layout to (block_size, num_kv_heads, head_dim).
        # NHD: (nb, bs, h, d); HND: (nb, h, bs, d). Disambiguate via dims
        # (block_size != num_kv_heads for any sane config).
        if k_cache.shape[1] == num_kv_heads and k_cache.shape[2] == block_size:
            k_cache = k_cache.transpose(1, 2)
            v_cache = v_cache.transpose(1, 2)
        elif not (
            k_cache.shape[1] == block_size and k_cache.shape[2] == num_kv_heads
        ):
            raise NotImplementedError(f"unexpected page layout {k_cache.shape}")
        num_blocks = (seq_len + block_size - 1) // block_size
        pages = block_table[:num_blocks].to(torch.long)
        k = k_cache.index_select(0, pages).reshape(-1, num_kv_heads, head_dim)
        v = v_cache.index_select(0, pages).reshape(-1, num_kv_heads, head_dim)
        return k[:seq_len], v[:seq_len]

    def _gdn_diffusion(
        self,
        dn,
        h: torch.Tensor,
        conv_state_idx: "int | torch.Tensor",
        ssm_state_idx: "int | torch.Tensor",
        conv_offset: "int | torch.Tensor" = 0,
    ) -> torch.Tensor:
        """Diffusion GDN: diff conv (seeded by AR conv state) + dual-scan
        bidirectional delta rule seeded by the AR recurrent state.

        Static (graph-capturable) mode: ``conv_state_idx``/``ssm_state_idx``
        are (1,) device tensors and ``conv_offset`` a 0-dim device tensor."""
        num_tokens = h.shape[0]
        if getattr(dn, "diff_in_weight_q", None) is not None:
            proj = _fp8_linear(h, dn.diff_in_weight_q, dn.diff_in_weight_s)
        else:
            proj = F.linear(h, dn.diff_in_weight)
        mixed, z_flat, b_raw, a_raw = proj.split(
            [dn.conv_dim, dn.value_dim, dn.num_v_heads, dn.num_v_heads], dim=-1
        )

        conv_pool = dn.kv_cache[0]
        ssm_pool = dn.kv_cache[1]
        if not is_conv_state_dim_first():
            conv_pool = conv_pool.transpose(-1, -2)
        # (conv_dim, kernel-1) most-recent-last window of pre-conv columns of
        # the committed prefix. Under spec decode the slot is wider
        # (kernel-1+num_spec) and the committed window starts at conv_offset.
        width = dn.conv_kernel_size - 1
        if isinstance(conv_state_idx, torch.Tensor):
            cols = (
                torch.arange(width, device=h.device, dtype=torch.long)
                + conv_offset
            )
            conv_state = (
                conv_pool.index_select(0, conv_state_idx)
                .squeeze(0)
                .index_select(-1, cols)
            )
        else:
            conv_state = conv_pool[
                conv_state_idx, :, conv_offset : conv_offset + width
            ]

        window = torch.cat([conv_state.to(mixed.dtype), mixed.t()], dim=-1)
        conv_out = F.conv1d(
            window.unsqueeze(0), dn.conv1d_diff.weight, groups=dn.conv_dim
        ).squeeze(0)  # (conv_dim, K)
        x = F.silu(conv_out).t()  # (K, conv_dim)

        q, k, v = torch.split(x, [dn.key_dim, dn.key_dim, dn.value_dim], dim=-1)
        q = q.reshape(1, num_tokens, dn.num_k_heads, dn.head_k_dim)
        k = k.reshape(1, num_tokens, dn.num_k_heads, dn.head_k_dim)
        v = v.reshape(1, num_tokens, dn.num_v_heads, dn.head_v_dim)
        n_rep = dn.num_v_heads // dn.num_k_heads
        if n_rep > 1:
            q = q.repeat_interleave(n_rep, dim=2)
            k = k.repeat_interleave(n_rep, dim=2)

        beta = b_raw.sigmoid().view(1, num_tokens, dn.num_v_heads)
        g = (
            -dn.A_log.float().exp()
            * F.softplus(a_raw.float() + dn.dt_bias.float())
        ).view(1, num_tokens, dn.num_v_heads)

        # Dual scan in ONE kernel call: row 0 forward (seeded by the AR
        # recurrent state), row 1 the flipped block from a zero state.
        if isinstance(ssm_state_idx, torch.Tensor):
            init_fwd = ssm_pool.index_select(0, ssm_state_idx)
        else:
            init_fwd = ssm_pool[ssm_state_idx].unsqueeze(0)
        if _FWD_ONLY_SCAN:
            out2, _ = chunk_gated_delta_rule(
                q,
                k,
                v,
                g=g,
                beta=beta,
                initial_state=init_fwd,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
            core = out2[0]  # (K, H, head_v_dim)
        else:
            q2 = torch.cat([q, torch.flip(q, dims=[1])], dim=0)
            k2 = torch.cat([k, torch.flip(k, dims=[1])], dim=0)
            v2 = torch.cat([v, torch.flip(v, dims=[1])], dim=0)
            g2 = torch.cat([g, torch.flip(g, dims=[1])], dim=0)
            b2 = torch.cat([beta, torch.flip(beta, dims=[1])], dim=0)
            init2 = torch.cat([init_fwd, torch.zeros_like(init_fwd)], dim=0)

            out2, _ = chunk_gated_delta_rule(
                q2,
                k2,
                v2,
                g=g2,
                beta=b2,
                initial_state=init2,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
            core = out2[0] + torch.flip(out2[1], dims=[0])  # (K, H, head_v_dim)

        z = z_flat.reshape(-1, dn.head_v_dim)
        core = core.reshape(-1, dn.head_v_dim)
        # Inline the gated RMS norm (same math as RMSNormGated.forward_static)
        # so it fuses into the compiled drafter graph instead of running as
        # standalone eager kernels at the FLA graph break.
        nw = dn.norm
        cf = core.float()
        cf = cf * torch.rsqrt(cf.pow(2).mean(dim=-1, keepdim=True) + nw.eps)
        cf = cf * nw.weight.float()
        zf = z.float()
        gate = torch.sigmoid(zf) if nw.activation == "sigmoid" else F.silu(zf)
        core = (cf * gate).to(core.dtype)
        core = core.reshape(num_tokens, dn.value_dim)
        if getattr(dn, "out_proj_diff_q", None) is not None:
            return _fp8_linear(core, dn.out_proj_diff_q, dn.out_proj_diff_s)
        return dn.out_proj_diff(core)

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
