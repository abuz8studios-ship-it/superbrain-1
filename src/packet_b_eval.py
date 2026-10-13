#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""packet_b_eval.py — score a GGUF (via llama-server) on Packet B, the
root-disjoint held-out test built by train\\build_packet_b.py.

Greedy, raw completion. Root answers are compared by root identity (all
hamza/alef carriers folded, ى->ي), morphology answers by bake-off letters.
Root items are scored twice: trained few-shot prompt and bare question.
Writes proof\\packetb_<name>.json (per item) and appends a summary line to
proof\\packetb_results.jsonl.

Usage: python packet_b_eval.py --url http://127.0.0.1:8098 --name <name>
"""

import argparse
import collections
import datetime
import json
import os
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(PROJECT_ROOT, "train"))
import bakeoff_arabic as B  # noqa: E402
from build_packet_b import root_key  # noqa: E402

PACKET = os.path.join(PROJECT_ROOT, "train", "packet_b.jsonl")


def complete(url, prompt, n=24):
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0.0,
                       "top_k": 1, "cache_prompt": True}).encode("utf-8")
    req = urllib.request.Request(url.rstrip("/") + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8098")
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel requests (start llama-server with -np N)")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(PACKET, encoding="utf-8")]
    jobs = []
    for it in items:
        jobs.append((it, "fewshot", it["prompt"]))
        if it["section"].startswith("root"):
            jobs.append((it, "bare", it["prompt_bare"]))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(args.workers) as ex:
        raws = []
        for i, raw in enumerate(ex.map(lambda j: complete(args.url, j[2]),
                                       jobs), 1):
            raws.append(raw)
            if i % 200 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] generated", flush=True)

    tally = collections.defaultdict(lambda: [0, 0])   # key -> [correct, n]
    out = []
    for (it, mode, _), raw in zip(jobs, raws):
        if it["section"].startswith("root"):
            pred = B.parse_root_output(raw)
            ok = root_key(pred) == it["root_key"]
            keys = [f"root/{mode}", f"{it['section']}/{mode}",
                    f"root_class/{mode}/{it['root_class']}"]
        else:
            pred = B.parse_verb_output(raw)
            ok = pred == B.ar_letters(it["answer"])
            keys = ["morph", it["section"], f"morph_form/{it['form']}"]
        for k in keys:
            tally[k][0] += ok
            tally[k][1] += 1
        out.append({**{k: v for k, v in it.items()
                       if not k.startswith("prompt")},
                    "mode": mode, "pred": pred, "ok": ok, "raw": raw[:120]})

    summary = {k: {"correct": c, "n": n, "acc": round(c / n, 4)}
               for k, (c, n) in sorted(tally.items())}
    print("\n===== PACKET B =====")
    for k, s in summary.items():
        print(f"{k:32s} {s['correct']:4d}/{s['n']:<4d} = {s['acc']:.2%}")
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M EDT")
    with open(os.path.join(PROJECT_ROOT, "proof", f"packetb_{args.name}.json"),
              "w", encoding="utf-8") as f:
        json.dump({"model": args.name, "date": date, "items": out}, f,
                  ensure_ascii=False, indent=1)
    import hashlib
    digest = hashlib.sha256(open(PACKET, "rb").read()).hexdigest()[:16]
    with open(os.path.join(PROJECT_ROOT, "proof", "packetb_results.jsonl"),
              "a", encoding="utf-8") as f:
        f.write(json.dumps({"model": args.name, "date": date,
                            "packet_sha256_16": digest, "engine": "llama.cpp",
                            "summary": summary}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
