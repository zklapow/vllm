# SPDX-License-Identifier: Apache-2.0
"""M1 increment-2 parity gate: compare vLLM diffusion logits vs the prototype
oracle (margin-aware, copied from the prototype's validate_packed.py).

  python tools/orthrus/parity_compare.py \
      --vllm ~/orthrus_m1/parity_vllm.pt --proto ~/orthrus_m1/parity_proto.pt
"""

import argparse
import os

import torch
import torch.nn.functional as F


def report(name, a, b, argmax_thresh=0.995, cos_thresh=0.99, margin=1.0):
    """In bf16, near-tied top-1/top-2 logits flip argmax from kernel-tiling
    noise alone; gate on positions where the reference margin exceeds
    `margin` — flips there would indicate a real bug.

    cos_thresh is looser than the prototype's same-implementation gate
    (0.999): this comparison crosses kernel stacks (vLLM SDPA / vendored FLA /
    fused RoPE vs prototype matmul-softmax / HF FLA). Measured on run17:
    per-block-position cos decays smoothly 0.9998 (pos 0) -> 0.9945 (pos 47)
    with no positional cliff — accumulated bf16 noise, not a systematic bug."""
    a32, b32 = a.float(), b.float()
    max_abs = (a32 - b32).abs().max().item()
    cos = F.cosine_similarity(
        a32.reshape(-1, a32.shape[-1]), b32.reshape(-1, b32.shape[-1]), dim=-1
    ).mean().item()
    agree_all = (a32.argmax(dim=-1) == b32.argmax(dim=-1)).float().mean().item()

    top2 = b32.topk(2, dim=-1).values
    confident = (top2[..., 0] - top2[..., 1]) > margin
    if confident.any():
        agree_conf = (
            (a32.argmax(dim=-1) == b32.argmax(dim=-1))[confident].float().mean().item()
        )
        conf_frac = confident.float().mean().item()
    else:
        agree_conf, conf_frac = 1.0, 0.0

    ok = agree_conf >= argmax_thresh and cos >= cos_thresh
    status = "PASS" if ok else "FAIL"
    print(
        f"[{status}] {name}: agree(margin>{margin})={agree_conf:.4f} "
        f"({conf_frac:.0%} of pos) agree(all)={agree_all:.4f} "
        f"cos={cos:.6f} max_abs={max_abs:.4f}"
    )
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", required=True)
    ap.add_argument("--proto", required=True)
    args = ap.parse_args()

    dv = torch.load(os.path.expanduser(args.vllm), map_location="cpu",
                    weights_only=True)
    dp = torch.load(os.path.expanduser(args.proto), map_location="cpu",
                    weights_only=True)
    assert dv["K"] == dp["K"]

    all_ok = True
    anchor_match = anchor_total = 0
    va, pa = [], []
    for pi, (rv, rp) in enumerate(zip(dv["results"], dp["results"])):
        for recv, recp in zip(rv["records"], rp["records"]):
            assert recv["t"] == recp["t"] and recv["cur"] == recp["cur"]
            anchor_total += 1
            anchor_match += recv["anchor"] == recp["proto_anchor"]
            all_ok &= report(
                f"prompt {pi} t={recv['t']} cur={recv['cur']} diffusion logits",
                recv["logits"],
                recp["logits"],
            )
            va.append(recv["logits"])
            pa.append(recp["logits"])

    all_ok &= report("ALL positions pooled", torch.stack(va), torch.stack(pa))
    print(f"anchor agreement (vLLM AR vs prototype AR): "
          f"{anchor_match}/{anchor_total}")
    print("PARITY " + ("PASS" if all_ok else "FAIL"))
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
