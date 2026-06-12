# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Orthrus proposer: parallel block-diffusion drafting with the target model.

The draft "model" is the target model's own diffusion head
(OrthrusQwen3_5ForCausalLM.diffusion_forward): one parallel forward over
[anchor, mask x K-1] at positions [cur, cur+K) reading the target's paged KV
and GDN states. K = num_speculative_tokens + 1.

CPU-list proposer path (runs after bookkeeping like ngram). bs=1 requests are
drafted through a manually captured CUDA graph of ``diffusion_forward`` in
static mode (fixed K, fixed max-prefix page buffer, validity carried by an
additive bias built from a device scalar — same pattern as the prototype's
static_infer.py). Anything else (bs>1, prefix overflow) falls back to the
eager per-request path, which stays bit-equivalent in output structure.

State-slot conventions (see gdn_attn.py + fused_sigmoid_gating.py +
causal_conv1d.py): with ``a = len(sampled)`` accepted tokens in the step that
just ran, the GDN recurrent state after the committed prefix is slot
``block_ids[a-1]`` and the conv window starts at column ``a-1`` of conv slot
``block_ids[0]``.
"""

import os

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class _DiffusionGraph:
    """One captured CUDA graph of the static-mode diffusion forward.

    All inputs live in fixed device buffers; per-cycle host work is a handful
    of small async H2D copies followed by ``graph.replay()`` and one D2H sync
    to read the K-1 drafted token ids.
    """

    def __init__(
        self,
        model,
        K: int,
        num_spec_tokens: int,
        attn_map: dict[int, int],
        gdn_map: dict[int, int],
        attn_block_size: int,
        kernel_bs: dict[int, int],
        max_pages: int,
        device: torch.device,
    ):
        self.model = model
        self.K = K
        self.num_spec_tokens = num_spec_tokens
        self.max_pages = max_pages
        self.kernel_block_size = next(iter(kernel_bs.values()))
        self.attn_block_size = attn_block_size
        self.kernel_bs = kernel_bs

        attn_groups = set(attn_map.values())
        assert len(attn_groups) == 1, "expected a single attention KV group"
        self.attn_group = next(iter(attn_groups))
        self.gdn_groups = sorted(set(gdn_map.values()))

        dev = device
        self.input_ids = torch.zeros(K, dtype=torch.long, device=dev)
        self.positions = torch.zeros(K, dtype=torch.long, device=dev)
        self.seq_len_t = torch.zeros((), dtype=torch.long, device=dev)
        self.pages = torch.zeros(max_pages, dtype=torch.long, device=dev)
        self.conv_idx = {
            gi: torch.zeros(1, dtype=torch.long, device=dev)
            for gi in self.gdn_groups
        }
        self.ssm_idx = {
            gi: torch.zeros(1, dtype=torch.long, device=dev)
            for gi in self.gdn_groups
        }
        self.conv_off = torch.zeros((), dtype=torch.long, device=dev)
        self.draft_out = torch.zeros(
            num_spec_tokens, dtype=torch.long, device=dev
        )

        # Pinned host staging buffers (one per device buffer: each is the
        # source of an async H2D copy and must not be reused mid-flight).
        self.host_tokens = torch.zeros(K, dtype=torch.long, pin_memory=True)
        self.host_pos = torch.zeros(K, dtype=torch.long, pin_memory=True)
        self.host_pages = torch.zeros(
            max_pages, dtype=torch.long, pin_memory=True
        )
        self._ratio = attn_block_size // kernel_bs[self.attn_group]

        self.attn_bts = {li: self.pages for li in attn_map}
        self.gdn_idxs = {
            li: (self.conv_idx[gi], self.ssm_idx[gi])
            for li, gi in gdn_map.items()
        }
        self.graph: torch.cuda.CUDAGraph | None = None

    def _forward(self) -> None:
        logits = self.model.diffusion_forward(
            input_ids=self.input_ids,
            positions=self.positions,
            seq_len=self.seq_len_t,
            attn_block_tables=self.attn_bts,
            attn_block_size=self.kernel_block_size,
            gdn_state_indices=self.gdn_idxs,
            gdn_conv_offset=self.conv_off,
        )
        torch.argmax(
            logits[: self.num_spec_tokens], dim=-1, out=self.draft_out
        )

    def capture(self) -> None:
        """Warm up (FLA autotune, optional torch.compile) and capture.
        Buffers must already hold a valid request state so warmup touches
        real cache slots.

        torch.compile fuses the long elementwise/cat/reduce tail of the eager
        diffusion pass (~3000 tiny kernels) before graph capture; graph breaks
        at the FLA Triton kernels are harmless because the captured CUDA graph
        eliminates the CPU dispatch between subgraphs anyway."""
        fn = self._forward
        if os.environ.get("ORTHRUS_COMPILE_DRAFT", "1") != "0":
            try:
                compiled = torch.compile(self._forward, dynamic=False)
                compiled()  # compile + smoke outside the capture stream
                torch.cuda.synchronize()
                fn = compiled
            except Exception:
                logger.exception(
                    "Orthrus: torch.compile of diffusion forward failed; "
                    "capturing the eager version"
                )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        self.graph = graph

    def fill(
        self,
        anchor: int,
        mask_token_id: int,
        cur: int,
        block_ids,
        num_accepted: int,
    ) -> bool:
        """Stage one request into the static buffers.

        Returns False if the request does not fit (caller falls back to the
        eager path)."""
        needed = -(-cur // self.kernel_block_size)
        if needed > self.max_pages:
            return False

        self.host_tokens.fill_(mask_token_id)
        self.host_tokens[0] = anchor
        self.input_ids.copy_(self.host_tokens, non_blocking=True)
        torch.arange(cur, cur + self.K, out=self.host_pos)
        self.positions.copy_(self.host_pos, non_blocking=True)
        self.seq_len_t.fill_(cur)

        blocks = torch.tensor(
            block_ids[self.attn_group], dtype=torch.long
        )
        expanded = (
            blocks.unsqueeze(1) * self._ratio
            + torch.arange(self._ratio, dtype=torch.long)
        ).flatten()
        n = min(expanded.numel(), self.max_pages)
        self.host_pages[:n] = expanded[:n]
        self.host_pages[n:] = 0  # null block; masked by the validity bias
        self.pages.copy_(self.host_pages, non_blocking=True)

        for gi in self.gdn_groups:
            row = block_ids[gi]
            self.conv_idx[gi].fill_(row[0])
            self.ssm_idx[gi].fill_(row[min(num_accepted, len(row)) - 1])
        self.conv_off.fill_(num_accepted - 1)
        return True

    def run(self) -> list[int]:
        assert self.graph is not None
        self.graph.replay()
        return self.draft_out.tolist()

    def profile_once(self) -> None:
        """Dump a per-kernel table of one replay (ORTHRUS_PROFILE=1)."""
        torch.cuda.synchronize()
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            self.graph.replay()
            torch.cuda.synchronize()
        print(
            prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=25
            ),
            flush=True,
        )


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

        # CUDA-graph drafting (default on; ORTHRUS_DRAFT_CUDAGRAPH=0 to
        # force the eager per-request path). Also disabled when the engine
        # itself is eager.
        self.use_graph = (
            os.environ.get("ORTHRUS_DRAFT_CUDAGRAPH", "1") != "0"
            and not vllm_config.model_config.enforce_eager
        )
        # One graph per prefix bucket (power-of-two page counts): the dense
        # prefix gather + masked SDPA cost scales with the captured prefix
        # length, so short sequences shouldn't pay for max_model_len.
        self._graphs: dict[int, _DiffusionGraph] = {}

        # Resolved lazily (KV caches don't exist at construction time).
        self._layer_maps = None

        # Acceptance stats: each propose() call follows a verify step;
        # len(sampled)-1 of the previous cycle's drafts were accepted.
        self.stat_cycles = 0
        self.stat_accepted = 0
        # Wall-clock spent inside propose() (each call ends with a device
        # sync at draft_out.tolist(), so this is honest GPU+CPU time).
        self.stat_propose_s = 0.0

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

    def _get_graph(self, cur: int) -> "_DiffusionGraph":
        attn_map, gdn_map, attn_block_size, kernel_bs = self._layer_maps
        kernel_block = next(iter(kernel_bs.values()))
        limit = -(-self.runner.max_model_len // kernel_block)
        needed = max(-(-cur // kernel_block), 1)
        bucket = 64
        while bucket < needed:
            bucket *= 2
        bucket = min(bucket, limit)
        if bucket not in self._graphs:
            self._graphs[bucket] = _DiffusionGraph(
                self.runner.model,
                self.K,
                self.num_spec_tokens,
                attn_map,
                gdn_map,
                attn_block_size,
                kernel_bs,
                bucket,
                self.device,
            )
        return self._graphs[bucket]

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
        import time

        t0 = time.perf_counter()
        runner = self.runner
        maps = self._layer_maps or self._resolve_layer_maps()

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
            if self.use_graph:
                graph = self._get_graph(cur)
                if graph.fill(
                    anchor, self.mask_token_id, cur, block_ids, num_accepted
                ):
                    if graph.graph is None:
                        tc = time.perf_counter()
                        graph.capture()
                        self.stat_capture_s = time.perf_counter() - tc
                        self.stat_propose_s -= self.stat_capture_s
                        logger.info(
                            "Orthrus: captured diffusion CUDA graph "
                            "(K=%d, max_pages=%d) in %.1fs",
                            self.K,
                            graph.max_pages,
                            self.stat_capture_s,
                        )
                    if (
                        os.environ.get("ORTHRUS_PROFILE") == "1"
                        and self.stat_cycles == 40
                    ):
                        graph.profile_once()
                    drafts.append(graph.run())
                    continue
            drafts.append(
                self._propose_one_eager(maps, block_ids, anchor, cur, num_accepted)
            )
        self.stat_propose_s += time.perf_counter() - t0
        return drafts

    def _propose_one_eager(
        self, maps, block_ids, anchor: int, cur: int, num_accepted: int
    ) -> list[int]:
        attn_map, gdn_map, attn_block_size, kernel_bs = maps
        input_ids = torch.full(
            (self.K,), self.mask_token_id, dtype=torch.long, device=self.device
        )
        input_ids[0] = anchor
        positions = torch.arange(
            cur, cur + self.K, dtype=torch.long, device=self.device
        )
        # One expanded page tensor per KV-cache group, shared by every
        # attention layer in that group (they have identical block ids).
        group_pages: dict[int, torch.Tensor] = {}
        for gi in set(attn_map.values()):
            ratio = attn_block_size // kernel_bs[gi]
            blocks = torch.tensor(block_ids[gi], dtype=torch.long)
            pages = (blocks.unsqueeze(1) * ratio
                     + torch.arange(ratio, dtype=torch.long)).flatten()
            group_pages[gi] = pages.to(self.device, non_blocking=True)
        attn_bts = {li: group_pages[gi] for li, gi in attn_map.items()}
        gdn_idxs = {}
        for li, gi in gdn_map.items():
            row = block_ids[gi]
            # Conv state: sliding window in slot row[0], committed window
            # starts at column num_accepted-1. Recurrent state: the verify
            # pass wrote per-draft-step states into row[j]; slot
            # row[num_accepted-1] is the committed one. (After a
            # prefill/non-spec step num_accepted == 1 -> row[0].)
            gdn_idxs[li] = (row[0], row[min(num_accepted, len(row)) - 1])

        logits = self.runner.model.diffusion_forward(
            input_ids=input_ids,
            positions=positions,
            seq_len=cur,
            attn_block_tables=attn_bts,
            attn_block_size=next(iter(kernel_bs.values())),
            gdn_state_indices=gdn_idxs,
            gdn_conv_offset=num_accepted - 1,
        )
        return logits[: self.num_spec_tokens].argmax(dim=-1).tolist()
