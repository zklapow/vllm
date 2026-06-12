# SPDX-License-Identifier: Apache-2.0
"""M1 increment-2 parity harness, prototype side (phase B, the oracle).

Replays the commit points recorded by parity_vllm.py through the prototype
implementation (~/src/qwen_orthrus model.py): teacher-forced prefill of the
exact committed prefix, then one diffusion forward with the same anchor, K and
mask token. Saves logits for parity_compare.py.

Loads weights from the EXPORTED merged checkpoint (export_hf) rather than the
training pipeline: the export was verified bit-exact against the live
load_model + diff_params.pt + merge_lora_for_inference model, and loading it
directly skips the expensive PiSSA SVD re-init.

Run on the GPU box under the PROTOTYPE venv (needs its transformers fork):
  cd ~/src/qwen_orthrus && CUDA_VISIBLE_DEVICES=6 .venv/bin/python \
      ~/src/vllm-orthrus/tools/orthrus/parity_proto.py \
      --export ~/checkpoints/orthrus/run17_27b/export_hf \
      --in ~/orthrus_m1/parity_vllm.pt --out ~/orthrus_m1/parity_proto.pt
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.expanduser("~/src/qwen_orthrus"))

# exported name -> prototype post-merge module name
LORA_RENAMES = {
    "in_proj_qkv_diff": "lora_in_proj_qkv",
    "in_proj_z_diff": "lora_in_proj_z",
    "out_proj_diff": "lora_out_proj",
}


def proto_name(key: str) -> str:
    for old, new in LORA_RENAMES.items():
        if f".{old}." in key:
            return key.replace(f".{old}.", f".{new}.")
    return key


def load_exported_model(export_dir: str):
    from safetensors.torch import safe_open
    from transformers import AutoConfig

    from model import OrthrusForCausalLM, merge_lora_for_inference

    config = AutoConfig.from_pretrained(export_dir)
    config = config.text_config if hasattr(config, "text_config") else config

    with torch.device("cpu"):
        model = OrthrusForCausalLM(config, lora_rank=8)  # rank irrelevant: merged away
    model = model.bfloat16().eval()
    merged = merge_lora_for_inference(model)
    print(f"swapped {merged} LoRALinear modules for plain Linears", flush=True)

    with open(os.path.join(export_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)
    own = model.state_dict()
    seen = set()
    by_file: dict[str, list[str]] = {}
    for k, fname in index["weight_map"].items():
        by_file.setdefault(fname, []).append(k)
    for fname, keys in by_file.items():
        with safe_open(os.path.join(export_dir, fname), framework="pt",
                       device="cpu") as fh:
            for k in keys:
                pk = proto_name(k)
                assert pk in own, f"exported tensor {k} -> {pk} not in model"
                t = fh.get_tensor(k)
                assert own[pk].shape == t.shape, (pk, own[pk].shape, t.shape)
                own[pk].copy_(t)
                seen.add(pk)
    missing = [k for k in own if k not in seen]
    assert not missing, f"params not covered by export: {missing[:10]}"
    model.load_state_dict(own, strict=True)
    print(f"loaded {len(seen)} tensors from export", flush=True)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = torch.load(os.path.expanduser(args.inp), map_location="cpu",
                      weights_only=True)
    K = data["K"]
    mask_token_id = data["mask_token_id"]

    model = load_exported_model(os.path.expanduser(args.export))
    model = model.cuda()

    out_results = []
    with torch.inference_mode():
        for pi, res in enumerate(data["results"]):
            recs = []
            for rec in res["records"]:
                committed = torch.tensor([rec["committed"]], device="cuda")
                cur = rec["cur"]
                assert committed.shape[1] == cur, (committed.shape, cur)
                prefill = model(input_ids=committed, use_cache=True,
                                is_diffusion_pass=False)
                # AR cross-check: the prototype's next-token prediction at the
                # commit point should (modulo bf16 near-ties) equal the anchor
                # vLLM sampled.
                proto_anchor = prefill.logits[0, -1].argmax().item()

                diff_ids = torch.full((1, K), mask_token_id, dtype=torch.long,
                                      device="cuda")
                diff_ids[0, 0] = rec["anchor"]
                positions = torch.arange(cur, cur + K, device="cuda").unsqueeze(0)
                diff_out = model(
                    input_ids=diff_ids,
                    position_ids=positions,
                    past_key_values=prefill.past_key_values,
                    use_cache=False,
                    is_diffusion_pass=True,
                    ar_seq_len=cur,
                )
                recs.append(
                    dict(
                        t=rec["t"],
                        cur=cur,
                        anchor=rec["anchor"],
                        proto_anchor=proto_anchor,
                        ar_last_logits=prefill.logits[0, -1].float().cpu(),
                        logits=diff_out.logits[0].float().cpu(),
                    )
                )
                print(f"prompt {pi} t={rec['t']} cur={cur}: "
                      f"anchor={rec['anchor']} proto_anchor={proto_anchor} "
                      f"draft head: {diff_out.logits[0, :8].argmax(dim=-1).tolist()}",
                      flush=True)
            out_results.append(dict(records=recs))

    torch.save(dict(K=K, results=out_results), os.path.expanduser(args.out))
    print(f"saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
