# -*- coding: utf-8 -*-
"""build_grpo2_data.py - SUPERBRAIN 1 Run 2b data: the Packet A2 items the
post-SFT model gets right only SOMETIMES.

GRPO learns nothing from a group that is all-right or all-wrong, so each
candidate is sampled --k times (temperature 1.0) on a llama-server running
the Run 2a GGUF; items with 1..k-1 correct are kept. Candidates are drawn
with extra weight on the weak buckets named in a verdict.json (morph forms
and root classes, weight = 1 - accuracy + 0.1).

Usage (server up, e.g. train\\run2a\\gguf\\serve.bat q5_k_m 0 8098 with -np 8):
  python build_grpo2_data.py --url http://127.0.0.1:8098 \\
      --verdict train\\run2a\\verdict.json --candidates 12000 --keep 3000"""
import argparse, collections, json, os, random, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"E:\ABU\02_PROJECTS\AHMAD-1"
sys.path.insert(0, os.path.join(ROOT, "src"))
import a2_reward as R  # noqa: E402

# Share of candidates per task before bucket weighting.
MIX = {"morph": 0.35, "root": 0.30, "conj": 0.15, "form": 0.07,
       "derive": 0.08, "reason": 0.05}


def sample(url, prompt, seed):
    body = json.dumps({"prompt": prompt, "n_predict": 48, "temperature": 1.0,
                       "top_k": 0, "top_p": 1.0, "seed": seed,
                       "cache_prompt": True}).encode("utf-8")
    req = urllib.request.Request(url + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))["content"]


def bucket_weights(verdict_path):
    w = collections.defaultdict(lambda: 1.0)
    if not verdict_path:
        return w
    v = json.load(open(verdict_path, encoding="utf-8"))
    for form, acc in v.get("morph_by_form", {}).items():
        w[f"morph/{form}"] = w[f"form/{form}"] = 1.0 - acc + 0.1
    pb = v.get("packet_b_root_class_bare", {})
    for cls, acc in pb.items():
        w[f"root/{cls}"] = 1.0 - acc + 0.1
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8098")
    ap.add_argument("--verdict", default="")
    ap.add_argument("--data", default=os.path.join(ROOT, "train",
                                                   "packet_a2.jsonl"))
    ap.add_argument("--candidates", type=int, default=12000)
    ap.add_argument("--keep", type=int, default=3000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(ROOT, "train",
                                                  "grpo2_train.jsonl"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    by_task = collections.defaultdict(list)
    for line in open(args.data, encoding="utf-8"):
        x = json.loads(line)
        by_task[x["task"]].append(x)
    w = bucket_weights(args.verdict)
    cands = []
    present = {t: s for t, s in MIX.items() if by_task[t]}   # e.g. an RL pool is root-only
    scale = sum(MIX.values()) / sum(present.values())
    for task, share in present.items():
        share *= scale
        pool = by_task[task]
        weights = [w[R.bucket(x)] for x in pool]
        cands += rng.choices(pool, weights=weights,
                             k=int(args.candidates * share))
    rng.shuffle(cands)
    print("candidates", len(cands),
          dict(collections.Counter(R.bucket(x) for x in cands)), flush=True)

    def score(x):
        gold = R.gold_answer(x)
        return sum(R.correct(x["task"], sample(args.url, x["prompt"], s), gold)
                   for s in range(args.k))

    with ThreadPoolExecutor(args.workers) as ex:
        hits = []
        for i, h in enumerate(ex.map(score, cands), 1):
            hits.append(h)
            if i % 500 == 0:
                print(f"  [{i}/{len(cands)}] scored", flush=True)

    hist = collections.Counter(hits)
    by_b = collections.defaultdict(lambda: [0, 0])
    keep, seen = [], set()
    for x, h in zip(cands, hits):
        b = R.bucket(x)
        by_b[b][0] += h
        by_b[b][1] += args.k
        if 0 < h < args.k and x["prompt"] not in seen:
            seen.add(x["prompt"])
            keep.append({"prompt": x["prompt"], "task": x["task"],
                         "answer": R.gold_answer(x), "bucket": b,
                         "hits": h})
    rng.shuffle(keep)
    keep = keep[:args.keep]
    with open(args.out, "w", encoding="utf-8") as f:
        for x in keep:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("hits histogram (of k=%d):" % args.k, dict(sorted(hist.items())))
    print("sampled accuracy by bucket:")
    for b, (c, n) in sorted(by_b.items()):
        print(f"  {b:18s} {c / n:6.1%}  (n={n // args.k})")
    print("kept", len(keep), dict(collections.Counter(x["bucket"]
                                                      for x in keep)))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
