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
