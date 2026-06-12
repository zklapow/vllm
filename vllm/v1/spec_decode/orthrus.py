# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Orthrus proposer: parallel block-diffusion drafting with the target model.

The draft "model" is the target model's own diffusion head
(OrthrusQwen3_5ForCausalLM.diffusion_forward): one parallel forward over
[anchor, mask x K-1] at positions [cur, cur+K) reading the target's paged KV
and GDN states. K = num_speculative_tokens + 1.

M1 scope: greedy, eager, CPU-list proposer path (runs after bookkeeping like
ngram). Each request is drafted with a separate bs=1 diffusion forward —
correct for any batch size, optimized later (M3 batches the block dim).

State-slot conventions (see gdn_attn.py + fused_sigmoid_gating.py +
causal_conv1d.py): with ``a = len(sampled)`` accepted tokens in the step that
just ran, the GDN recurrent state after the committed prefix is slot
``block_ids[a-1]`` and the conv window starts at column ``a-1`` of conv slot
``block_ids[0]``.
"""

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class OrthrusProposer:
    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner):
        assert vllm_config.speculative_config is not None
        self.vllm_config = vllm_config
        self.device = device
        self.runner = runner
        self.num_spec_tokens = vllm_config.speculative_config.num_speculative_tokens
        self.K = self.num_spec_tokens + 1

        orthrus_cfg = (
            getattr(vllm_config.model_config.hf_text_config, "orthrus", None) or {}
        )
        if "mask_token_id" not in orthrus_cfg:
            raise ValueError(
                "method='orthrus' requires an Orthrus checkpoint (config.json "
                "with an 'orthrus' block: mask_token_id, block_size, ...)"
            )
        self.mask_token_id = int(orthrus_cfg["mask_token_id"])
        trained_k = orthrus_cfg.get("block_size")
        if trained_k is not None and self.K > trained_k:
            logger.warning(
                "Orthrus: K=%d (num_speculative_tokens+1) exceeds the trained "
                "block size %d; acceptance will degrade.",
                self.K,
                trained_k,
            )

        # Resolved lazily (KV caches don't exist at construction time).
        self._layer_maps = None

        # Acceptance stats: each propose() call follows a verify step;
        # len(sampled)-1 of the previous cycle's drafts were accepted.
        self.stat_cycles = 0
        self.stat_accepted = 0

    def load_model(self, *args, **kwargs) -> None:
        # The "draft model" is the target model itself.
        pass

    def _resolve_layer_maps(self):
        """Map decoder layer index -> KV-cache group index, and pre-compute
        the kernel-page expansion for the attention groups."""
        from vllm.model_executor.models.utils import extract_layer_index
        from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

        runner = self.runner
        attn_map: dict[int, int] = {}
        gdn_map: dict[int, int] = {}
        attn_block_size = None
        for gi, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                li = extract_layer_index(layer_name)
                if isinstance(spec, MambaSpec):
                    gdn_map[li] = gi
                elif isinstance(spec, AttentionSpec):
                    attn_map[li] = gi
                    attn_block_size = spec.block_size
        assert attn_map and gdn_map and attn_block_size is not None
        kernel_bs = {
            gi: runner._kernel_block_sizes[gi] for gi in set(attn_map.values())
        }
        self._layer_maps = (attn_map, gdn_map, attn_block_size, kernel_bs)
        return self._layer_maps

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        slot_mappings=None,
    ) -> list[list[int]]:
        """Draft ``num_spec_tokens`` tokens per request.

        Runs after bookkeeping: ``input_batch.num_tokens_no_spec`` already
        includes this step's sampled tokens; the last sampled token (the
        anchor) has no KV/DN state yet.
        """
        runner = self.runner
        model = runner.model
        maps = self._layer_maps or self._resolve_layer_maps()
        attn_map, gdn_map, attn_block_size, kernel_bs = maps

        drafts: list[list[int]] = []
        for i, sampled in enumerate(sampled_token_ids):
            if not sampled:
                drafts.append([])
                continue
            req_id = runner.input_batch.req_ids[i]
            req_state = runner.requests[req_id]
            total = int(runner.input_batch.num_tokens_no_spec[i])
            cur = total - 1
            anchor = sampled[-1]
            num_accepted = len(sampled)
            # Count acceptance for every post-draft verify step (any step
            # where drafts could have been scheduled, i.e. not the prefill).
            if total - len(sampled) > len(req_state.prompt_token_ids or []):
                self.stat_cycles += 1
                self.stat_accepted += num_accepted - 1

            if cur + self.K > runner.max_model_len:
                drafts.append([])
                continue

            block_ids = req_state.block_ids
            input_ids = torch.full(
                (self.K,), self.mask_token_id, dtype=torch.long, device=self.device
            )
            input_ids[0] = anchor
            positions = torch.arange(
                cur, cur + self.K, dtype=torch.long, device=self.device
            )
            attn_bts = {}
            for li, gi in attn_map.items():
                ratio = attn_block_size // kernel_bs[gi]
                pages = [
                    b * ratio + j for b in block_ids[gi] for j in range(ratio)
                ]
                attn_bts[li] = torch.tensor(
                    pages, dtype=torch.long, device=self.device
                )
            gdn_idxs = {}
            for li, gi in gdn_map.items():
                row = block_ids[gi]
                # Conv state: sliding window in slot row[0], committed window
                # starts at column num_accepted-1. Recurrent state: the verify
                # pass wrote per-draft-step states into row[j]; slot
                # row[num_accepted-1] is the committed one. (After a
                # prefill/non-spec step num_accepted == 1 -> row[0].)
                gdn_idxs[li] = (row[0], row[min(num_accepted, len(row)) - 1])

            logits = model.diffusion_forward(
                input_ids=input_ids,
                positions=positions,
                seq_len=cur,
                attn_block_tables=attn_bts,
                attn_block_size=next(iter(kernel_bs.values())),
                gdn_state_indices=gdn_idxs,
                gdn_conv_offset=num_accepted - 1,
            )
            draft = logits[: self.num_spec_tokens].argmax(dim=-1).tolist()
            drafts.append(draft)
        return drafts
