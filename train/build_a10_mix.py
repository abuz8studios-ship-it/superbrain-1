# -*- coding: utf-8 -*-
"""build_a10_mix.py - SFT mix for run7_a10: all of packet_a10_train + a replay slice of sft_a9_mix
(the last Fanar-line SFT mix: roots / morph / reason / conj / form / derive) so the champion does not
forget Packet B roots/morph or math while it learns science.

replay = --replay-frac of the FINAL mix (default 0.25 -> replay = n_a10 / 3), stratified by task
(each task keeps its share of sft_a9_mix). Any A10 train row whose qhash or prompt is in the exam is
dropped (leak guard). Output columns: prompt, completion, task, src. Seeded, reproducible.
Usage: python build_a10_mix.py [--replay-frac 0.25] [--out train\\sft_a10_mix.jsonl]"""
import argparse, json, os, random, sys
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
T = r"E:\ABU\02_PROJECTS\AHMAD-1\train"


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a10", default=os.path.join(T, "packet_a10_train.jsonl"))
    ap.add_argument("--exam", default=os.path.join(T, "packet_a10_exam.jsonl"))
    ap.add_argument("--replay", default=os.path.join(T, "sft_a9_mix.jsonl"))
    ap.add_argument("--replay-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(T, "sft_a10_mix.jsonl"))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    exam = load(a.exam)
    ex_h = {x.get("qhash") for x in exam}; ex_p = {x["prompt"] for x in exam}
    a10 = [x for x in load(a.a10) if x.get("qhash") not in ex_h and x["prompt"] not in ex_p]
    n_drop = len(load(a.a10)) - len(a10)
    rep = load(a.replay)
    n_rep = round(len(a10) * a.replay_frac / (1 - a.replay_frac))
    by = defaultdict(list)
    for x in rep:
        by[x.get("task", "?")].append(x)
    picked = []
    for task, xs in sorted(by.items()):
        k = round(n_rep * len(xs) / len(rep))
        picked += rng.sample(xs, min(k, len(xs)))
    rows = [{"prompt": x["prompt"], "completion": x["completion"], "task": "a10_" + x.get("subject", "sci"), "src": "a10"} for x in a10]
    rows += [{"prompt": x["prompt"], "completion": x["completion"], "task": x.get("task", "?"), "src": "a9_replay"} for x in picked]
    rng.shuffle(rows)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cnt = defaultdict(int)
    for r in rows:
        cnt[r["task"]] += 1
    print(f"MIX a10={len(a10)} (dropped {n_drop} exam-overlap) replay={len(picked)} total={len(rows)} -> {a.out}")
    print("MIX_TASKS", dict(sorted(cnt.items())))


if __name__ == "__main__":
    main()
