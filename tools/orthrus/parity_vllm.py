# SPDX-License-Identifier: Apache-2.0
"""M1 increment-2 parity harness, vLLM side (phase A).

Drives the in-process v1 engine step by step so requests stay RUNNING with
live paged KV + GDN states, then calls ``model.diffusion_forward()`` at chosen
commit points and saves the logits plus everything phase B (the prototype
oracle, parity_proto.py in this directory run under the prototype venv) needs
to reproduce the exact same diffusion inputs.

Run on the GPU box:
  VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=6 \
    PATH=$HOME/vllm-dev-env/bin:$PATH \
    ~/vllm-dev-env/bin/python tools/orthrus/parity_vllm.py \
      --model ~/checkpoints/orthrus/run17_27b/export_hf \
      --out ~/orthrus_m1/parity_vllm.pt
"""

import argparse
import os

import torch

PROMPTS = [
    "Explain in two short paragraphs why speculative decoding speeds up LLM "
    "inference without changing the model's output.",
    "Write a Python function that parses an Apache access log line into a "
    "dict with fields ip, timestamp, method, path, status, and bytes. Use a "
    "compiled regex, include type hints and a short docstring.",
    "Implement an LRU cache class in Python with O(1) get and put using a "
    "dict and a doubly-linked list. Include type hints.",
    "Give three debugging tips for a CUDA out-of-memory error.",
    "Solve: if 3x + 7 = 31, what is x? Show the steps.",
    "Write a SQL query to count users by signup month.",
]


def get_runner(llm):
    ec = llm.llm_engine.engine_core.engine_core  # InprocClient -> EngineCore
    worker = ec.model_executor.driver_worker
    # UniProcExecutor wraps the worker; unwrap if needed.
    worker = getattr(worker, "worker", worker)
    return worker.model_runner


def map_layers_to_groups(runner):
    """Map each decoder layer index to its KV-cache group.

    Hybrid models split layers across several groups (e.g. 48 GDN layers in 3
    mamba groups + 16 attention layers in 1 group), so block ids are
    per-layer. Returns (attn_layer_to_group, gdn_layer_to_group,
    attn_block_size).
    """
    from vllm.model_executor.models.utils import extract_layer_index
    from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

    attn_map: dict[int, int] = {}
    gdn_map: dict[int, int] = {}
    attn_block_size = None
    for gi, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        for layer_name in group.layer_names:
            layer_idx = extract_layer_index(layer_name)
            if isinstance(spec, MambaSpec):
                gdn_map[layer_idx] = gi
            elif isinstance(spec, AttentionSpec):
                attn_map[layer_idx] = gi
                attn_block_size = spec.block_size
    assert attn_map and gdn_map and attn_block_size
    return attn_map, gdn_map, attn_block_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block-size", type=int, default=None,
                    help="diffusion K; default = config orthrus.block_size")
    ap.add_argument("--commit-points", type=int, nargs="+", default=[1, 9, 33],
                    help="run a diffusion pass after this many sampled tokens")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--probe", action="store_true",
                    help="print runtime structures and exit after one prompt")
    args = ap.parse_args()

    assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", (
        "run with VLLM_ENABLE_V1_MULTIPROCESSING=0 (in-process engine)"
    )

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=os.path.expanduser(args.model),
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.85,
        max_num_seqs=1,
        # Keep step() strictly synchronous: one scheduled batch per step so
        # runner state (num_computed_tokens, caches) matches the sampled
        # tokens we see after each step.
        async_scheduling=False,
    )
    runner = get_runner(llm)
    model = runner.model
    orthrus_cfg = model.orthrus_config
    K = args.block_size or orthrus_cfg["block_size"]
    mask_token_id = orthrus_cfg["mask_token_id"]
    print(f"orthrus config: {orthrus_cfg}; harness K={K}")

    attn_map, gdn_map, attn_block_size = map_layers_to_groups(runner)
    print(f"attn layers -> groups: {sorted(set(attn_map.values()))} "
          f"(block_size={attn_block_size}); "
          f"gdn layers -> groups: {sorted(set(gdn_map.values()))}")
    # The logical KV block size (e.g. 400, picked to match the mamba page
    # size) is split into smaller kernel pages (e.g. 16) in the actual cache
    # tensor; expand logical block ids to kernel page ids accordingly.
    kernel_block_sizes = runner._kernel_block_sizes
    attn_kernel_bs = {gi: kernel_block_sizes[gi] for gi in set(attn_map.values())}
    print(f"kernel block sizes per group: {kernel_block_sizes}")

    engine = llm.llm_engine
    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=max(args.commit_points) + 1,
                        ignore_eos=True)

    results = []
    for pi, prompt in enumerate(PROMPTS):
        ext_id = f"parity-{pi}"
        # add_request returns the randomized INTERNAL id (used by the model
        # runner); RequestOutputs carry the EXTERNAL id.
        int_id = engine.add_request(ext_id, prompt, sp)
        sampled: list[int] = []
        records = []
        steps = 0
        while True:
            outs = engine.step()
            steps += 1
            assert steps < 200, "engine.step() loop did not terminate"
            done = False
            for out in outs:
                if out.request_id not in (ext_id, int_id):
                    continue
                new = list(out.outputs[0].token_ids)
                sampled = new
                done = out.finished
            t = len(sampled)
            if t in args.commit_points and not done:
                if int_id not in runner.requests:
                    print("runner.requests keys:", list(runner.requests))
                req_state = runner.requests[int_id]
                prompt_ids = list(req_state.prompt_token_ids)
                block_ids = req_state.block_ids
                if args.probe:
                    print("block_ids:", block_ids)
                    print("num_computed_tokens:", req_state.num_computed_tokens)
                    for ln, layer in [
                        ("attn3", model.model.layers[3].self_attn.attn),
                        ("gdn0", model.model.layers[0].linear_attn),
                    ]:
                        kv = layer.kv_cache
                        if isinstance(kv, (list, tuple)):
                            print(ln, "kv_cache entries:", [tuple(x.shape) for x in kv])
                        else:
                            print(ln, "kv_cache:", tuple(kv.shape))
                # The committed prefix in the caches: prompt + t-1 outputs
                # (the t-th sampled token is the anchor — not forwarded yet).
                # Note req_state.num_computed_tokens lags by the tokens of
                # the just-executed step; the caches themselves are current.
                cur = len(prompt_ids) + t - 1
                anchor = sampled[-1]
                device = next(model.parameters()).device
                input_ids = torch.full((K,), mask_token_id, dtype=torch.long,
                                       device=device)
                input_ids[0] = anchor
                positions = torch.arange(cur, cur + K, dtype=torch.long,
                                         device=device)
                attn_bts = {}
                for li, gi in attn_map.items():
                    ratio = attn_block_size // attn_kernel_bs[gi]
                    pages = [
                        b * ratio + j for b in block_ids[gi] for j in range(ratio)
                    ]
                    attn_bts[li] = torch.tensor(pages, dtype=torch.long,
                                                device=device)
                gdn_idxs = {li: block_ids[gi][0] for li, gi in gdn_map.items()}
                logits = model.diffusion_forward(
                    input_ids=input_ids,
                    positions=positions,
                    seq_len=cur,
                    attn_block_tables=attn_bts,
                    attn_block_size=next(iter(attn_kernel_bs.values())),
                    gdn_state_indices=gdn_idxs,
                )
                records.append(
                    dict(
                        t=t,
                        cur=cur,
                        anchor=anchor,
                        committed=prompt_ids + sampled[:-1],
                        logits=logits.to(torch.float32).cpu(),
                    )
                )
                print(f"prompt {pi} t={t} cur={cur}: diffusion logits "
                      f"{tuple(logits.shape)}; draft head: "
                      f"{logits[:8].argmax(dim=-1).tolist()}")
                if args.probe:
                    return
            if done or t > max(args.commit_points):
                break
        # drain/abort
        engine.abort_request([int_id])
        results.append(
            dict(prompt=prompt, prompt_ids=prompt_ids, sampled=sampled,
                 records=records)
        )
        text = tokenizer.decode(sampled)
        print(f"prompt {pi}: sampled {len(sampled)} tokens: {text[:80]!r}")

    torch.save(
        dict(K=K, mask_token_id=mask_token_id, results=results),
        os.path.expanduser(args.out),
    )
    print(f"saved {sum(len(r['records']) for r in results)} records to {args.out}")


if __name__ == "__main__":
    main()
