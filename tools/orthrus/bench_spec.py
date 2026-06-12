# SPDX-License-Identifier: Apache-2.0
"""M1 increment-4 bench: bs=1 greedy Orthrus speculative decoding in vLLM.

Measures tok/s + acceptance on the demo prompts and (optionally) checks
losslessness against a reference output JSON (produced by smoke_generate.py
on the same checkpoint without spec decode).

  VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=6 \
    PATH=$HOME/vllm-dev-env/bin:$PATH \
    ~/vllm-dev-env/bin/python tools/orthrus/bench_spec.py \
      --model ~/checkpoints/orthrus/run17_27b/export_hf --k 16 \
      --reference ~/orthrus_m1/smoke_orthrus.json
"""

import argparse
import json
import os
import time

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
    ap.add_argument("--k", type=int, default=16,
                    help="Orthrus block size K (num_speculative_tokens = K-1)")
    ap.add_argument("--no-spec", action="store_true",
                    help="AR baseline (no speculative config)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--reference", default=None,
                    help="JSON from smoke_generate.py for lossless check")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    kwargs = {}
    if not args.no_spec:
        kwargs["speculative_config"] = {
            "method": "orthrus",
            "num_speculative_tokens": args.k - 1,
        }
    llm = LLM(
        model=os.path.expanduser(args.model),
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.85,
        max_num_seqs=1,
        async_scheduling=False,
        **kwargs,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # Warmup
    llm.generate([PROMPTS[0]], SamplingParams(temperature=0.0, max_tokens=16))

    results = []
    total_tokens = 0
    total_time = 0.0
    for prompt in PROMPTS:
        start = time.perf_counter()
        outs = llm.generate([prompt], sp)
        elapsed = time.perf_counter() - start
        n = len(outs[0].outputs[0].token_ids)
        total_tokens += n
        total_time += elapsed
        results.append({"prompt": prompt, "text": outs[0].outputs[0].text,
                        "tokens": n, "seconds": elapsed,
                        "tok_s": n / elapsed})
        print(f"{n} tokens in {elapsed:.2f}s = {n / elapsed:.1f} tok/s")

    print(f"\nOVERALL: {total_tokens} tokens in {total_time:.2f}s "
          f"= {total_tokens / total_time:.1f} tok/s "
          f"({'AR baseline' if args.no_spec else f'orthrus K={args.k}'})")

    if not args.no_spec:
        # Acceptance stats live on the drafter (in-process engine required).
        try:
            ec = llm.llm_engine.engine_core.engine_core
            worker = ec.model_executor.driver_worker
            worker = getattr(worker, "worker", worker)
            drafter = worker.model_runner.drafter
            cycles, accepted = drafter.stat_cycles, drafter.stat_accepted
            if cycles:
                print(f"acceptance: {accepted} drafts accepted over {cycles} "
                      f"cycles = {accepted / cycles:.2f} avg accepted/cycle "
                      f"(tpf ~ {1 + accepted / cycles:.2f})")
        except AttributeError as e:
            print(f"(no drafter stats: {e})")

    if args.reference:
        with open(os.path.expanduser(args.reference)) as f:
            ref = json.load(f)
        n_match = 0
        for r, b in zip(results, ref):
            assert r["prompt"] == b["prompt"]
            m = min(len(r["text"]), len(b["text"]))
            ok = r["text"][:m] == b["text"][:m]
            n_match += ok
            if not ok:
                for i in range(m):
                    if r["text"][i] != b["text"][i]:
                        print(f"DIVERGENCE at char {i}:")
                        print("  spec:", repr(r["text"][max(0, i - 40): i + 40]))
                        print("  ref: ", repr(b["text"][max(0, i - 40): i + 40]))
                        break
        print(f"LOSSLESS CHECK: {n_match}/{len(ref)} prompts match reference")

    if args.out:
        with open(os.path.expanduser(args.out), "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
