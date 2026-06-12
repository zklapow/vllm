# Orthrus integration (work in progress)

Orthrus is a speculative-decoding scheme for **Qwen3.6-27B**, a hybrid model with
48 GatedDeltaNet (linear-attention) layers and 16 full-attention layers. A frozen
AR backbone is paired with a 2.2B-parameter diffusion head that drafts K-token
blocks in parallel; the target model verifies them with longest-prefix acceptance
(lossless vs greedy AR).

**Status**: plan/groundwork only. There are no vLLM source changes on this branch
yet — milestone M0 (baselines, checkpoint export) used stock pip vLLM 0.22.1.
The proposer work (M1: `OrthrusQwen3_5ForCausalLM` + `OrthrusProposer`) starts
from this branch.

**Results so far**: 1.885x bs=1 speedup measured in our standalone static-cache
CUDA-graph harness on a B200. **Target**: beat vLLM's native MTP on the same
model (~236 tok/s at bs=1 on B200).

See [INTEGRATION_PLAN.md](INTEGRATION_PLAN.md) for the full architecture mapping,
hard problems, and milestones (M0–M4).

**Pointers**
- Prototype repo (training + standalone inference harness):
  `~/src/scratch/qwen_orthrus` on the dev machine.
- Exported HF checkpoint (frozen base + merged diff tensors):
  `~/checkpoints/orthrus/run17_27b/export_hf` on the GPU box.
