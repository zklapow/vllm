# SPDX-License-Identifier: Apache-2.0
"""Probe the logit margin at a spec-vs-AR divergence point.

Loads the model AR-only (no spec), replays the common prefix of the two
outputs, and reports the top-2 logprob gap at the first divergent token.
A near-zero gap means the divergence is a bf16 argmax tie flipped by
chunk-shape numerics (verify processes 16-token chunks, AR 1-token steps),
not a speculative-decoding correctness bug.

  VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=6 \
    python tools/orthrus/probe_margin.py --model <export_hf> \
      --ref ~/orthrus_m2/ar_cudagraph.json --spec ~/orthrus_m2/spec16_v2.json
"""

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()

    with open(os.path.expanduser(args.ref)) as f:
        ref = json.load(f)
    with open(os.path.expanduser(args.spec)) as f:
        spec = json.load(f)

    cases = []
    for r, s in zip(ref, spec):
        assert r["prompt"] == s["prompt"]
        m = min(len(r["text"]), len(s["text"]))
        if r["text"][:m] == s["text"][:m]:
            continue
        i = next(j for j in range(m) if r["text"][j] != s["text"][j])
        cases.append((r["prompt"], r["text"][:i], r["text"][i : i + 30],
                      s["text"][i : i + 30]))
    if not cases:
        print("no divergences found")
        return

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=os.path.expanduser(args.model),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.85,
        max_num_seqs=1,
    )
    for prompt, common, ref_next, spec_next in cases:
        outs = llm.generate(
            [prompt + common],
            SamplingParams(temperature=0.0, max_tokens=1, logprobs=8),
        )
        lp = outs[0].outputs[0].logprobs[0]
        ranked = sorted(lp.values(), key=lambda x: -x.logprob)
        print("=" * 70)
        print("common prefix tail:", repr(common[-60:]))
        print("ref  continues:", repr(ref_next))
        print("spec continues:", repr(spec_next))
        for e in ranked[:4]:
            print(f"  cand {e.rank}: {e.decoded_token!r} logprob={e.logprob:.6f}")
        if len(ranked) >= 2:
            gap = ranked[0].logprob - ranked[1].logprob
            print(f"  top-2 logprob gap: {gap:.6e}"
                  f"  ({'TIE-CLASS' if gap < 5e-3 else 'REAL GAP'})")


if __name__ == "__main__":
    main()
