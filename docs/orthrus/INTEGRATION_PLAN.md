# Orthrus Production Integration Plan: vLLM (recommended) vs SGLang

**Date**: 2026-06-11
**Sources studied**: vLLM @ `6f573f4` (2026-06-12), SGLang @ `2e74ff1` (2026-06-11),
`latent-data/orthrus-serve` (HF-transformers reference server), this repo
(`model.py`, `static_infer.py`, `bench_tpf.py`, ONBOARDING.md).

**Goal**: serve Orthrus-Qwen3.6-27B (48 GatedDeltaNet + 16 full-attention layers,
frozen AR backbone + 2.2B-param diffusion head) in a production inference stack
with continuous batching, at ≥ the 1.885x bs=1 speedup our static-cache CUDA-graph
harness already demonstrates.

---

## 1. Recommendation: vLLM

Both frameworks already run Qwen3.5-family GDN hybrids **and** speculative decoding
on them (built-in MTP). The decisive differences:

| Criterion | vLLM | SGLang |
|---|---|---|
| Qwen3.5 hybrid model | `model_executor/models/qwen3_5.py` (`QwenGatedDeltaNetAttention`) | `srt/models/qwen3_5.py` + `hybrid_linear_attn_backend.py` |
| MTP spec decode on the hybrid | `qwen3_5_mtp.py`, works | `qwen3_5_mtp.py`, works |
| **DN state rollback on partial acceptance** | **Method-agnostic, in the attention backend**: `v1/attention/backends/gdn_attn.py` builds `spec_state_indices_tensor [batch, num_spec+1]` + `num_accepted_tokens`; the fused GDN kernels (`fused_sigmoid_gating_delta_rule_update`, `causal_conv1d_update`) write per-draft-step conv+SSM states into separate slots and *select* the accepted-step slot as the next initial state. Any spec method gets this for free. | Per-step `intermediate_ssm` / `intermediate_conv_window` caches + fused scatter (`update_mamba_state_after_mtp_verify`), but the commit logic is **called from inside each spec worker** (`eagle_worker_v2._mamba_verify_update`, duplicated in `dflash_worker_v2`, `ngram_worker`). A new method must replicate it. |
| Parallel block-diffusion drafting precedent | **DFlash** (`v1/spec_decode/dflash.py` + `models/qwen3_dflash.py`): mask-token queries, one parallel pass, `parallel_drafting=True`, draft attention runs **non-causal** via `attention_config.use_non_causal`. Explicitly supports Qwen3.5 targets. | DFlash worker exists too (`dflash_worker_v2.py`), also supports GDN targets. |
| Read-only access to target KV for the proposer | `Attention(kv_sharing_target_layer_name=...)` (`model_executor/layers/attention/attention.py:203`) — a layer reads another layer's paged KV without writing. SGLang's closest analog is the `frozen_kv_mtp` pool-view swap. | `frozen_kv_mtp_utils.frozen_kv_target_view` (backend pool swap) — workable but a worker-level hack. |
| GDN kernels available for the dual scan | FLA vendored: `model_executor/layers/fla/ops/chunk.py` `chunk_gated_delta_rule(initial_state=…, cu_seqlens=…)` — varlen with per-sequence initial state, exactly what batched dual-scan needs. | FLA vendored under `layers/attention/fla/`. |
| Plug-in surface | `SpecDecodeBaseProposer` subclass; new `method="orthrus"` (the `custom_class` hook only receives CPU token lists — ngram-shaped, unusable). Out-of-tree model via `ModelRegistry.register_model` + platform plugin. | `SpeculativeAlgorithm.register` plugin decorator (clean), but the worker it registers must reimplement mamba bookkeeping. |
| Codebase risk | Single active architecture (V1). | v1/v2 spec workers coexist and churn fast. |

**Why vLLM wins**: the make-or-break requirement — per-request DN state
snapshot/rollback under partial acceptance, batched — is already implemented
*below* the spec-method layer, so the Orthrus verify side works essentially for
free. The proposer side has a direct template (DFlash: mask tokens, parallel
drafting, non-causal attention). SGLang is a viable fallback; its mamba radix
cache (`mamba_track_interval`, prefix-cache state tracking) is the one feature
vLLM lacks that we'd eventually want, but it isn't on the critical path.

**Naming note**: neither framework knows "Qwen3.6" as a distinct arch; our 27B
loads through the `Qwen3_5*` modeling classes (same as this repo's `model.py`).
Verify the checkpoint's `architectures` string resolves to
`Qwen3_5ForCausalLM`/`Qwen3_5ForConditionalGeneration` in M0.

### Prior art: latent-data/orthrus-serve

They served Orthrus-Qwen3-8B (full attention only, no DN) as a **bespoke FastAPI
wrapper around HF `model.generate`** — bs=1, requests serialized behind an asyncio
semaphore, TPF counted with a forward pre-hook. No vLLM/SGLang work to reuse.
What *is* useful: their quantization study (the diffusion drafter survives fp8 and
NVFP4 with TPF ≥ 4 across all schemes) — de-risks serving Orthrus quantized in M4 —
and their OpenAI tool-call surface matches what our `serve_orthrus.py` already does.

---

## 2. Architecture mapping (Orthrus → vLLM)

The structural insight: in vLLM terms, **Orthrus is "DFlash where the draft model
is the target model itself"** — same depth, shared MLP/embed/lm_head weights,
shared KV cache and DN states, plus 2.2B params of per-layer `_diff` projections.
That rules out vLLM's separate-draft-model loading path and dictates a
target-model extension:

| Orthrus component (this repo) | vLLM interface |
|---|---|
| `OrthrusForCausalLM` (model.py) | New `OrthrusQwen3_5ForCausalLM` extending `Qwen3_5ForCausalLMBase` (`models/qwen3_5.py`), registered out-of-tree via `ModelRegistry.register_model`. Adds per-layer diff modules and an explicit `diffusion_forward()` entry point. AR/verify path is untouched stock Qwen3_5. |
| `OrthrusAttention` diff projections (`q/k/v/o_proj_diff`, `q/k_norm_diff`) | Extra modules on the attention layers. The diffusion attention is a *second* `Attention` instance per full-attn layer with `kv_sharing_target_layer_name` → the AR layer (reads target paged KV, writes nothing). |
| `OrthrusGatedDeltaNet` LoRA + gates + `conv1d_diff` | Extra modules on `QwenGatedDeltaNetAttention` layers. **LoRA is merged offline** (see weights below) → plain `_diff` linear projections, mirroring `merge_lora_for_inference()` (+0.10x measured; 144 fewer matmul pairs). |
| `bidirectional_delta.py` dual scan | Port to vendored FLA: forward scan = `chunk_gated_delta_rule(initial_state=accepted_state_slot, cu_seqlens=fixed-K varlen)`; backward scan = flip each K-block, zero initial state, same kernel; additive fusion. Read-only on the GDN state pool. |
| Diffusion proposer loop (`bench_tpf.py` / `static_infer.py` cycle) | `OrthrusProposer(SpecDecodeBaseProposer)` with `parallel_drafting=True`, modeled line-for-line on `DFlashProposer`. `load_model()` binds the target model instead of loading a draft. `propose()` builds K−1 mask tokens + anchor per request at positions `[cur, cur+K)` and calls `target_model.diffusion_forward()` under `set_forward_context`. |
| Longest-prefix acceptance / verify | Stock vLLM: scheduler + rejection sampler greedy path already implements longest-prefix-match verification with a bonus token. `num_speculative_tokens = K − 1`; the anchor is `next_token_ids`. Lossless vs greedy AR holds by construction. |
| DN snapshot + masked replay (`replay_state_masked`) | **Deleted on the happy path** — replaced by vLLM's per-step state slots + `num_accepted_tokens` selection. Retained as the *memory fallback* for large K (see hard problem #1). |
| `StaticHybridCache` / CUDA-graph capture | Replaced by vLLM paged KV + `MambaSpec` pool + vLLM's spec-decode cudagraph machinery. |
| Trainable-head checkpoint loading (`diff_params.pt` over frozen base) | Offline export script → one HF safetensors checkpoint: original frozen base weights + merged diff tensors (`*.q_proj_diff.weight`, `*.in_proj_qkv_diff.weight`, …). Careful: PiSSA puts the principal components in LoRA and a *residual* in `lora_*.base`; merged diff weight = `residual + scaling·(A@B)ᵀ`, while the AR path keeps the **original** base weight. `load_weights()` maps the extra names; everything else is stock. |

### Bidirectional block attention — how it's expressed

The Orthrus diffusion mask is: K block queries see the committed prefix `[0, cur)`
**plus the whole K block** (bidirectional). Two implementation options, no custom
kernel required for either:

- **Option A (default)** — two-part attention with LSE merge: (1) non-causal
  varlen attention of the K queries against the target's paged KV (read via
  kv-sharing; every query sees the whole prefix — causal=False *is* the right
  mask here), (2) dense K×K bidirectional attention over the block's own
  `k/v_proj_diff` outputs, (3) combine with vLLM's `merge_attn_states` (the
  cascade-attention LSE merge). All standard kernels.
- **Option B (optimization)** — temporarily write the block's diff K/V into the
  request's own pages at `[cur, cur+K)` and run one non-causal read over pages
  `[0, cur+K)`; the verify pass overwrites those exact slots with AR K/V in the
  same cycle. Zero extra memory, one kernel — but interacts with prefix caching
  and write-ordering; only attempt after A is correct.

---

## 3. Hard problems, ranked

### P1 — Mamba spec-slot memory at Orthrus block sizes (NEW problem, biggest risk)
vLLM sizes the GDN pool as `(num_spec + 1)` state slots per request
(`MambaSpec.num_speculative_blocks`, `v1/kv_cache_interface.py:614`). MTP uses
num_spec = 1–3; Orthrus wants K−1 = 7–47. Per request: 48 DN layers ×
(num_v_heads·d_k·d_v fp32 + conv state) × (K) slots — order ~100 MB/slot-set per
layer-stack on the 27B, i.e. **multiple GB per request at K=48**.
**Approach**: (a) start at K=8–16 where slot memory is tolerable; (b) for large K,
swap selection-rollback for our **masked DN replay trick** (`replay_state_masked`
in model.py: beta=0, g=0 ⇒ state identity; replay measured at 6.2 ms across all
48 layers under CUDA graphs) — 1 snapshot slot + one cheap DN-only pass instead of
K slots. This is a contained change: the GDN backend already isolates the spec
path, and the replay needs only the verify-pass mixer inputs (we capture them via
a preallocated buffer, as `dn_input_buf` does today). Decide per-K by measurement
in M3.

### P2 — Diffusion attention correctness/perf (prefix + bidirectional block)
Option A's LSE merge must match the prototype's single-softmax over
`[K_AR ‖ K_diff]` exactly (it does mathematically; bf16 numerics need the parity
harness). Perf: two attention calls + merge per layer per cycle. Fallback:
FlexAttention backend with a custom block mask (vLLM has one;
flex on sm_100 needed `FORCE_USE_FLEX_ATTENTION` in our prototype — known sharp
edge). **Approach**: build the M1 logit-parity harness (prototype `model.py` as
oracle, same checkpoint, same prompts) before any spec-decode wiring.

### P3 — Full-depth proposer integration + CUDA graphs
The diffusion pass is a full 64-layer forward at K tokens/request with its own
attention metadata — unlike EAGLE's shallow draft. Cost is fine (that's the
Orthrus design: ~2 forwards per cycle for ~5 committed tokens), but it must be
captured in vLLM's draft cudagraph machinery (DFlash already captures parallel-
drafting graphs; reuse its buffer-stability patterns — our `static_infer.py`
experience maps directly). Risk: vendored-FLA Triton kernels inside capture —
proven workable in our prototype (manual capture records Triton fine).

### P4 — Weight export & loading
Mostly mechanical (export script + `load_weights` name mapping), but two traps we
already hit in the prototype: (a) PiSSA residual-vs-original base weights — the AR
path must get the *original* weights, the diff path the merged ones; (b) config
fidelity (`text_config` vs reconstructed config halved the cache layout once —
commit c7f3a30). Unit-test the export against `load_model()` + checkpoint in this
repo.

### P5 — Batched speculation dynamics
Per-request variable acceptance is native to vLLM. Remaining Orthrus-specific
bits: every request drafts exactly K−1 tokens every cycle (uniform — simpler than
EAGLE trees); mask-token id must be pinned in the served config; EOS-inside-block
and max-len trimming are handled by the scheduler. Sampling: lossless guarantee is
greedy-only — M2–M3 restrict to greedy/temp=0; standard rejection sampling for
temperature>0 is a research follow-up (the head was distilled on argmax agreement).

---

## 4. Phased milestones

| Phase | Deliverable | Exit criteria | Effort |
|---|---|---|---|
| **M0 — baseline & export** | Vanilla Qwen3.6-27B serving in stock vLLM on the B200 box; AR + built-in MTP throughput baselines; checkpoint export script (frozen base + merged diff → one safetensors) | vLLM serves the base model; MTP baseline numbers recorded (the 1.5–2x bar Orthrus must beat); export round-trips against prototype `load_model` | ~1 wk |
| **M1 — Orthrus model in vLLM** | `OrthrusQwen3_5ForCausalLM` plugin: diff modules, weight loading, eager `diffusion_forward()` | Logit parity vs prototype `model.py` (AR pass bit-exact; diffusion pass within bf16 tol) on 16 fixture prompts | 1–2 wk |
| **M2 — spec decode, bs=1 greedy** | `OrthrusProposer` (method="orthrus"), Option-A diffusion attention, dual-scan on vendored FLA, verify via stock GDN spec path | Lossless vs greedy AR; accept ≥ 4 avg (parity with `bench_tpf.py`); wall-clock ≥ 1.3x vs vLLM AR at bs=1, K=8–16 | 2–4 wk |
| **M3 — batched** | Varlen batched diffusion + capture; K-vs-slot-memory decision (selection-rollback vs masked replay); throughput sweeps | Correct + faster than AR at bs ∈ {1,4,16,32}; beats MTP baseline at equal batch; no per-request cross-talk (lossless holds per request) | 2–3 wk |
| **M4 — serving parity** | Prefix caching interaction, metrics/logging (accept-rate per request), OpenAI surface A/B (port `serve_orthrus.py` semantics), optional fp8 | Production checklist: stop sequences, tool calls, long prompts, graceful K fallback to AR; fp8 spot-check (orthrus-serve evidence says drafter survives) | 1–2 wk |

Total: roughly 7–12 weeks of focused work, with M2 the highest-variance phase.

---

## 5. Reuse vs rewrite from this prototype

**Reused (logic or directly ported)**
- `merge_lora_for_inference()` → the offline export script (same math, offline).
- `bidirectional_delta.py` dual-scan formulation → re-targeted at vLLM's vendored
  `chunk_gated_delta_rule` (supports `initial_state` + `cu_seqlens`; our kernel
  use is unchanged, batching becomes varlen instead of bs=1).
- **Masked DN replay trick** (`replay_state_masked`) → the large-K memory fallback
  for P1; the beta=0/g=0 state-identity insight is the only known alternative to
  K-per-step state slots.
- `dn_input_buf` capture pattern (graph-safe, hook-free mixer-input capture) → if
  the replay fallback is used.
- `bench_tpf.py` / `sweep_acceptance.py` / `eval_acceptance.py` → acceptance
  parity oracles for M2 (same prompts, accept-length distributions must match).
- `serve_orthrus.py` → request-level mode=ar A/B semantics and per-request orthrus
  stats for M4; tool-call validation prompts.
- `static_infer.py` → the *experience* (capture-safe buffer discipline, padded
  last block, Triton-in-graph viability), not the code.

**Rewritten / discarded**
- `StaticHybridCache`, manual 5-graph engine, the python orchestration loop in
  `orthrus_generate_graphed` → vLLM scheduler, paged KV, MambaSpec pool, drafter
  cudagraphs.
- DN snapshot ring (`snap_buf`/`_foreach_copy_`) → vLLM per-step slot selection
  (happy path).
- `DynamicCache` crop / `ar_seq_len` plumbing → paged KV makes it moot.
- Prototype attention (explicit matmul softmax, additive bias masks) → vLLM
  attention backends + Option-A merge.

---

## 6. Open questions to resolve early

1. Exact GDN state bytes/layer on the 27B (drives the P1 K-vs-memory curve) — read
   from the served config in M0.
2. Does the Qwen3.6 checkpoint's `architectures` field resolve in vLLM's registry
   as-is, or do we ship a config patch with the plugin?
3. `merge_attn_states` numerics vs single-softmax at bf16 on sm_100 (M1 harness).
4. Whether vLLM's drafter cudagraph capture tolerates the kv-sharing read path
   (DFlash materializes its own context KV instead — we'd be the first kv-sharing
   proposer).
