# -*- coding: utf-8 -*-
"""soup_any.py - weight arithmetic over merged 16-bit models with the SAME architecture, single-file or sharded.
  --mode avg   out = alpha*A + (1-alpha)*B
  --mode ta    out = A + B - BASE          (task arithmetic: both fine-tune deltas of one parent, kept at full size)
Mirrors A's file layout; copies A's non-weight files (config, tokenizer, processor, chat template).
Tensors identical in A, B (and BASE) - e.g. frozen vision/audio towers - pass through unchanged.
Usage: python soup_any.py --mode ta --a g4_rl\\merged_16bit --b g4_math\\merged_16bit --base g4_sft1\\merged_16bit --out X"""
import argparse, json, os, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["avg", "ta"], required=True)
ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
ap.add_argument("--base", default=""); ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--out", required=True)
args = ap.parse_args()
if args.mode == "ta" and not args.base:
    raise SystemExit("--mode ta needs --base")


def layout(root):
    idx = os.path.join(root, "model.safetensors.index.json")
    if os.path.exists(idx):
        return json.load(open(idx))["weight_map"]
    with safe_open(os.path.join(root, "model.safetensors"), "pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


handles = {}
def get(root, lay, name):
    key = (root, lay[name])
    if key not in handles:
        handles[key] = safe_open(os.path.join(root, lay[name]), "pt")
    return handles[key].get_tensor(name)


la, lb = layout(args.a), layout(args.b)
lc = layout(args.base) if args.mode == "ta" else None
assert set(la) == set(lb) and (lc is None or set(lc) == set(la)), "tensor names differ"
os.makedirs(args.out, exist_ok=True)
changed = 0
for sh in sorted(set(la.values())):
    out = {}
    for n in [k for k, f in la.items() if f == sh]:
        ta, tb = get(args.a, la, n), get(args.b, lb, n)
        assert ta.shape == tb.shape, n
        if args.mode == "avg":
            r = args.alpha * ta.float() + (1 - args.alpha) * tb.float()
        else:
            r = ta.float() + tb.float() - get(args.base, lc, n).float()
        r = r.to(ta.dtype)
        changed += int(not torch.equal(r, ta))
        out[n] = r
    save_file(out, os.path.join(args.out, sh), metadata={"format": "pt"})
    print("wrote", sh, len(out), "tensors", flush=True)
if os.path.exists(os.path.join(args.a, "model.safetensors.index.json")):
    json.dump({"metadata": {}, "weight_map": la}, open(os.path.join(args.out, "model.safetensors.index.json"), "w"))
for f in os.listdir(args.a):
    p = os.path.join(args.a, f)
    if os.path.isfile(p) and not f.endswith(".safetensors") and f != "model.safetensors.index.json":
        shutil.copy2(p, os.path.join(args.out, f))
json.dump(vars(args), open(os.path.join(args.out, "soup_args.json"), "w"), indent=1)
print(f"SOUP DONE {args.out} mode={args.mode} tensors_changed_vs_A={changed}", flush=True)
