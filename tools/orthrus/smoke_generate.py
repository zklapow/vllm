# SPDX-License-Identifier: Apache-2.0
"""M1 increment-1 smoke test: load the Orthrus export_hf checkpoint through the
OrthrusQwen3_5ForCausalLM vLLM model and produce greedy completions.

Compare against a baseline JSON banked from a stock vLLM server (greedy MTP
output == greedy AR output, lossless): the raw-completion texts must match.

Usage (on the GPU box):
  CUDA_VISIBLE_DEVICES=6 ~/vllm-dev-env/bin/python tools/orthrus/smoke_generate.py \
      --model ~/checkpoints/orthrus/run17_27b/export_hf \
      --baseline ~/orthrus_m1/baseline_mtp_greedy.json \
      --out ~/orthrus_m1/smoke_orthrus.json
"""

import argparse
import json
import os

PROMPTS = [
    "Explain in two short paragraphs why speculative decoding speeds up LLM "
    "inference without changing the model's output.",
    "Write a Python function that parses an Apache access log line into a "
    "dict with fields ip, timestamp, method, path, status, and bytes. Use a "
    "compiled regex, include type hints and a short docstring.",
    "Implement an LRU cache class in Python with O(1) get and put using a "
    "dict and a doubly-linked list. Include type hints.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=16384)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=os.path.expanduser(args.model),
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.85,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    outputs = llm.generate(PROMPTS, sp)
    results = [
        {"prompt": o.prompt, "text": o.outputs[0].text} for o in outputs
    ]

    if args.out:
        with open(os.path.expanduser(args.out), "w") as f:
            json.dump(results, f, indent=2)

    for r in results:
        print("=" * 70)
        print("PROMPT:", r["prompt"][:80])
        print(r["text"])

    if args.baseline:
        with open(os.path.expanduser(args.baseline)) as f:
            baseline = json.load(f)["completions"]
        n_match = 0
        for r, b in zip(results, baseline):
            assert r["prompt"] == b["prompt"], "prompt order mismatch"
            ok = r["text"][: len(b["text"])] == b["text"][: len(r["text"])]
            n_match += ok
            if not ok:
                print("MISMATCH for prompt:", r["prompt"][:60])
                for i, (a, c) in enumerate(zip(r["text"], b["text"])):
                    if a != c:
                        print(f"  first divergence at char {i}:")
                        print("  orthrus:", repr(r["text"][max(0, i - 40) : i + 40]))
                        print("  baseline:", repr(b["text"][max(0, i - 40) : i + 40]))
                        break
        print(f"BASELINE MATCH: {n_match}/{len(baseline)}")


if __name__ == "__main__":
    main()
