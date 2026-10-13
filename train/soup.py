# -*- coding: utf-8 -*-
"""soup.py - weight-average two merged 16-bit models that share the Fanar-1-9B ancestry:
out = alpha*A + (1-alpha)*B, tensor by tensor (float32 math, bf16 storage).
Used once, with a FIXED alpha chosen before looking at any score, to see whether the
pilot's everyday-word reliability and the GRPO model's unseen-root gains can coexist.
Usage: python soup.py --a <dirA> --b <dirB> --alpha 0.5 --out <dir>"""
import argparse, json, os, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)      # alpha weight
ap.add_argument("--b", required=True)
ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
ia = json.load(open(os.path.join(args.a, "model.safetensors.index.json")))["weight_map"]
ib = json.load(open(os.path.join(args.b, "model.safetensors.index.json")))["weight_map"]
assert set(ia) == set(ib), "the two models must have identical tensor names"
handles = {}
def get(root, idx, name):
    f = idx[name]
    key = (root, f)
    if key not in handles:
        handles[key] = safe_open(os.path.join(root, f), "pt")
    return handles[key].get_tensor(name)

shards = sorted(set(ia.values()))
new_map = {}
for sh in shards:
    names = [n for n, f in ia.items() if f == sh]
    out = {}
    for n in names:
        ta, tb = get(args.a, ia, n), get(args.b, ib, n)
        assert ta.shape == tb.shape, n
        out[n] = (args.alpha * ta.float() + (1 - args.alpha) * tb.float()).to(ta.dtype)
        new_map[n] = sh
    save_file(out, os.path.join(args.out, sh), metadata={"format": "pt"})
    print("wrote", sh, len(out), "tensors", flush=True)
json.dump({"metadata": {}, "weight_map": new_map},
          open(os.path.join(args.out, "model.safetensors.index.json"), "w"))
for f in os.listdir(args.a):
    if f.endswith(".safetensors") or f == "model.safetensors.index.json":
        continue
    p = os.path.join(args.a, f)
    if os.path.isfile(p):
        shutil.copy2(p, os.path.join(args.out, f))
print("SOUP DONE", args.out)
