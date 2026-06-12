# Orthrus M1 harness scripts

Working scripts for the Orthrus→vLLM integration (docs/orthrus/INTEGRATION_PLAN.md).
All run on the B200 box (GPU 6) against the run17 27B export at
`~/checkpoints/orthrus/run17_27b/export_hf`.

Environments on the box:
- `~/vllm-dev-env` — editable install of this fork at `~/src/vllm-orthrus`
  (`VLLM_USE_PRECOMPILED=1`). Run with `PATH=$HOME/vllm-dev-env/bin:$PATH`
  (flashinfer JIT needs `ninja` from the venv bin).
- `~/src/qwen_orthrus/.venv` — the prototype venv (HF transformers fork),
  used only for the parity oracle.

## Increment 1 — model loads & serves vanilla
```
CUDA_VISIBLE_DEVICES=6 PATH=$HOME/vllm-dev-env/bin:$PATH \
  ~/vllm-dev-env/bin/python tools/orthrus/smoke_generate.py \
    --model ~/checkpoints/orthrus/run17_27b/export_hf \
    --baseline ~/orthrus_m1/baseline_mtp_greedy.json
```
Baseline file was banked from the stock vLLM 0.22.1 MTP server (greedy ==
lossless AR) before it was torn down.

## Increment 2 — diffusion-forward parity vs prototype
Phase A (vLLM side; in-process engine, steps requests manually, calls
`model.diffusion_forward()` at commit points):
```
VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=6 \
  PATH=$HOME/vllm-dev-env/bin:$PATH \
  ~/vllm-dev-env/bin/python tools/orthrus/parity_vllm.py \
    --model ~/checkpoints/orthrus/run17_27b/export_hf \
    --out ~/orthrus_m1/parity_vllm.pt
```
Phase B (prototype oracle; loads the same export bit-exactly, replays the
same committed prefixes + diffusion blocks):
```
cd ~/src/qwen_orthrus && CUDA_VISIBLE_DEVICES=6 .venv/bin/python \
  ~/src/vllm-orthrus/tools/orthrus/parity_proto.py \
    --export ~/checkpoints/orthrus/run17_27b/export_hf \
    --in ~/orthrus_m1/parity_vllm.pt --out ~/orthrus_m1/parity_proto.pt
```
Gate (margin-aware argmax agreement, pattern from prototype
validate_packed.py):
```
python tools/orthrus/parity_compare.py \
  --vllm ~/orthrus_m1/parity_vllm.pt --proto ~/orthrus_m1/parity_proto.pt
```

## Increment 3/4 — speculative decoding end to end
```
VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=6 \
  PATH=$HOME/vllm-dev-env/bin:$PATH \
  ~/vllm-dev-env/bin/python tools/orthrus/bench_spec.py \
    --model ~/checkpoints/orthrus/run17_27b/export_hf --k 16 \
    --reference ~/orthrus_m1/smoke_orthrus.json
```
Serving (M2: no --enforce-eager; engine cudagraphs + captured drafter):
`vllm serve <export_hf> --speculative-config
'{"method": "orthrus", "num_speculative_tokens": 15}' --max-num-seqs 1`.
First request per prefix bucket pays a one-time drafter capture (1-8s).

M1 results (2026-06-12, B200 GPU 6, bs=1 greedy, 128 tok, demo prompts,
all eager):
- AR baseline (this fork, eager): 25.8 tok/s
- Orthrus K=8:  36.3 tok/s (1.41x), 3.23 accepted drafts/cycle, lossless 3/3
- Orthrus K=16: 38.1 tok/s (1.48x), 3.42 accepted drafts/cycle, lossless 3/3
- Orthrus K=32: 38.6 tok/s (1.50x), 3.57 accepted drafts/cycle, lossless 3/3
- `vllm serve` (multiproc engine, K=16): 37.7 tok/s over HTTP, lossless 3/3;
  vLLM SpecDecoding metrics report mean acceptance length 4.60
- (banked cudagraph numbers from the 0.22.1 server for context: AR 89.7,
  MTP k=4 ~236 tok/s — closing that gap is M2/M3 work: cudagraph capture,
  batched diffusion, paged Option-A attention)

## M2 — cudagraph speed (2026-06-12)

Increments (bs=1 greedy, 256 tok, demo prompts, vs the M2 AR-cudagraph
baseline of 85.0 tok/s measured on this fork; banked MTP bar 236 tok/s):

| increment | overall | code prompt | accept | propose ms/cycle |
|---|---|---|---|---|
| M1 (all eager, K=32)                      | 38.6  | —     | 3.57 | — |
| 1. engine cudagraphs, eager drafter (K=16)| 50.4  | 61.2  | 3.51 | ~75 |
| 3. drafter CUDA graph, max-len prefix     | 86.0  | 100.3 | 3.54 | 38.2 |
| 3b. bucketed prefix graphs (64-page)      | 102.6 | 116.4 | 3.74 | 30.5 |
| 3c. torch.compile + capture (K=16)        | 133.2 | 160.2 | 3.60 | 19.6 |
| 3c at K=32                                | 127.9 | 151.0 | 3.64 | 19.3 |
| 3c + ORTHRUS_COMPILE_MODE=max-autotune-no-cudagraphs | 136.4 | 163.2 | 3.62 | 19.0 |

- K sweep at speed (increment 3): K=16 86.0 / K=32 84.7 / K=48 79.9 tok/s;
  acceptance only 3.54 -> 3.64, so larger K does not pay for run17 weights on
  these prompts. K=16 is the operating point.
- Remaining gap to the MTP bar (236 tok/s code prompt), decomposed: cycle is
  ~33ms = ~14.4ms verify+engine (parity with MTP's whole 15.2ms cycle) +
  ~19ms drafter. Drafter replay is 18.5ms CUDA: 9.4ms weight GEMMs (floor —
  the diffusion pass is a full-depth 64-layer forward; MTP's drafter is one
  layer, ~1ms), ~2ms FLA dual-scan, ~1ms fp32 SDPA, rest fused elementwise.
  At 19ms drafter the bar needs tpf ~7.9; our tpf is 4.6 (already above
  MTP's accept-len 3.59 — Orthrus loses purely on drafter depth). The lever
  is acceptance: with the K=48-trained Run 18/19 checkpoints at accept ~7-8
  (prototype saw ~11 on code), the same engine clears 236. Option-A paged
  diffusion attention was not built: the dense gather is already a single
  fused triton kernel at ~0.6ms/replay and is not on the critical path.
- Drafter graphs: `OrthrusProposer` captures `diffusion_forward` in *static
  mode* (fixed K, power-of-two page-bucket prefix, gather-everything +
  additive validity bias from a 0-dim `seq_len` tensor; GDN state slots
  selected via device-tensor `index_select`). One graph per prefix bucket,
  captured lazily; `ORTHRUS_DRAFT_CUDAGRAPH=0` forces eager,
  `ORTHRUS_COMPILE_DRAFT=0` skips the torch.compile-before-capture step,
  `ORTHRUS_PROFILE=1` dumps a kernel table of one replay at cycle 40.
- Replay decomposition (uncompiled, 25.4ms): ~9.1ms weight GEMMs (floor —
  same weights any K<=48 forward must read), ~8-10ms elementwise/cat/copy
  zoo across ~3000 tiny kernels (what torch.compile fuses), ~2ms FLA chunk
  kernels, ~1.2ms fp32 SDPA GEMMs.
- LOSSLESSNESS: byte-identity vs the AR run is *tie-fragile*. Drafts change
  acceptance boundaries -> the same committed token can be computed by a
  different verify-chunk alignment -> bf16 logits move by 1 ulp -> argmax
  flips when the top-2 logits are 1 ulp apart (0.125 at logit scale ~20).
  `probe_margin.py` confirmed the only observed divergences are such
  1-ulp ties (gap exactly 0.125 = 1 bf16 ulp; both candidates legitimate
  greedy choices). Increment 3c is byte-identical 3/3; intermediate
  numerics variants flipped 1-2 prompts at single tie points. This is
  inherent to any spec method on bf16 (incl. stock MTP), not a rollback or
  state bug.

## M3 — drafter cost (2026-06-12)

Goal: drafter 19ms -> ~12ms without losing acceptance. Drafter-only changes
cannot break correctness (the target verifies everything), so they are gated
on acceptance, not output equality.

| increment | overall | code prompt | accept | propose ms/cycle |
|---|---|---|---|---|
| M2 best (re-measured, K=16)              | 121-136 | 163-176 | 3.57 | 19.0-21.4 |
| 1. inline norms in the compile region    | 151.7 | 176.2 | 3.48 | 14.4 |
| 2. + ORTHRUS_FP8_DRAFT=full              | 160.1 | 187.2 | 3.55 | 13.3 |

- Increment 1: GemmaRMSNorm / RMSNormGated modules route through opaque IR
  ops that stayed eager at the torch.compile boundary — 161 reduce kernels +
  2x161 elementwise per replay (~2.4ms). Inlining the norm math as pure
  torch in `diffusion_forward` lets inductor fuse them. Lossless check:
  2/3 byte-identical, 1 prompt flips at a 1-ulp tie (expected tie-fragility,
  same class as M2's intermediate variants).
- Increment 2 (`ORTHRUS_FP8_DRAFT`): W8A8 e4m3 for the drafter-path GEMMs.
  `diff` quantizes the *_diff projections in place (drafter-only weights,
  frees ~7GB); `full` additionally makes fp8 *copies* of the shared
  per-layer MLP + lm_head weights (~18GB, accounted before KV profiling;
  AR/verify path stays bf16-exact). Dynamic per-token activation scales +
  per-channel weight scales via rowwise `torch._scaled_mm` (sm100 OK).
  Acceptance 3.55 vs 3.57 bf16 — no real drop; lossless 3/3 byte-identical.
  First capture pays ~2min of max-autotune over the fp8 GEMM shapes.

## Notes / gotchas discovered
- Internal request ids are randomized; `LLMEngine.add_request` returns the
  internal id (keys `runner.requests`), `RequestOutput.request_id` is the
  external one.
- KV-cache groups on the 27B hybrid: 3 mamba groups (16 GDN layers each) +
  1 attention group; per-request block ids are per-group.
- Attention logical block size is forced to 400 tokens (mamba page parity)
  but the cache tensor uses 16-token kernel pages — expand block ids by
  `400/16` (see `runner._kernel_block_sizes`).
- GDN pools: conv `(slots, kernel-1[+num_spec], conv_dim)` (transpose to
  dim-first), ssm `(slots, H, head_v_dim, head_k_dim)` fp32.
- Spec-slot conventions for the future proposer: after a verify step the
  conv slot holds `[history..., input_1..input_{num_spec+1}]` and the window
  for the committed prefix starts at column `num_accepted-1`; the recurrent
  state after the committed prefix is `ssm[spec_state_indices[req,
  num_accepted-1]]`.
