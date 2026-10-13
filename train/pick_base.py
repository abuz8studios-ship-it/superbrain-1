# -*- coding: utf-8 -*-
"""pick_base.py - choose the GRPO (Run 2b) base by held-out ROOT accuracy.

"Passed its gates" is not "better": on 2026-10-07 the full Run 2a (600k items)
PASSED but regressed held-out roots vs the pilot (Packet B root few-shot 58.3 vs
67.9, bare 58.1 vs 66.8, and it no longer beat the Fanar base on bare roots).
Roots are the core goal, so rank PASSing runs by mean(Packet B root few-shot,
root bare); morph accuracy only breaks ties. Prints the run dir name on stdout."""
import json, os, sys

T = os.path.dirname(os.path.abspath(__file__))
rows = []
for name in ("run2a_pilot", "run2a"):
    p = os.path.join(T, name, "verdict.json")
    if not os.path.exists(p):
        continue
    v = json.load(open(p, encoding="utf-8"))
    if not v.get("PASS"):
        print(f"{name}: verdict FAIL - skipped", file=sys.stderr)
        continue
    pb = v["packet_b"]
    root = (pb["root/fewshot"]["acc"] + pb["root/bare"]["acc"]) / 2
    rows.append((root, pb["morph"]["acc"], name))
    print(f"{name}: PB root mean {root:.4f}  morph {pb['morph']['acc']:.4f}", file=sys.stderr)
if not rows:
    sys.exit("no PASSing run to use as GRPO base")
print(max(rows)[2])
