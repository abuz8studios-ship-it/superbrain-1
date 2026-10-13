# -*- coding: utf-8 -*-
"""rl_to_grpo.py - turn the held-out RL pool (roots the SFT model never saw)
into train_grpo2.py input: {"prompt", "task", "answer"} per line.

No difficulty filter here: GRPO samples 8 completions per prompt and a group
that is all-right or all-wrong simply contributes no gradient, so on UNSEEN
roots (held-out accuracy ~65-70%) a large share of groups still teach.
Class-balanced so the weak classes (defective/assimilated/hamzated/doubled)
are not drowned by sound roots.
Usage: python rl_to_grpo.py --pool rl_pool_a5.jsonl --out grpo_rl.jsonl [--n 12000]"""
import argparse, collections, json, os, random, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"E:\ABU\02_PROJECTS\AHMAD-1\train"
ap = argparse.ArgumentParser()
ap.add_argument("--pool", default=os.path.join(ROOT, "rl_pool_a5.jsonl"))
ap.add_argument("--out", default=os.path.join(ROOT, "grpo_rl.jsonl"))
ap.add_argument("--n", type=int, default=12000)
ap.add_argument("--seed", type=int, default=11)
args = ap.parse_args()
rng = random.Random(args.seed)
rows = [json.loads(l) for l in open(args.pool, encoding="utf-8")]
by = collections.defaultdict(list)
for r in rows:
    by[r["cls"]].append(r)
# equal share per class, capped by what the class has
per = args.n // len(by)
out = []
for cls, items in sorted(by.items()):
    rng.shuffle(items)
    out += items[:per]
rng.shuffle(out)
with open(args.out, "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps({"prompt": r["prompt"], "task": "root",
                            "answer": r["completion"].strip(), "cls": r["cls"]},
                           ensure_ascii=False) + "\n")
print(f"wrote {len(out)} -> {args.out} | by class "
      f"{dict(collections.Counter(r['cls'] for r in out))} | roots "
      f"{len({r['rk'] for r in out})}")
