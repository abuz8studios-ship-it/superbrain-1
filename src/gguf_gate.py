#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gguf_gate.py — run the bake-off gate (same 85 prompts, same parsers) against
a GGUF served by llama.cpp's llama-server, so the shipped quant is scored, not
just the 16-bit merge.

Greedy, n_predict 256, raw completion (no chat template) — mirrors
bakeoff_arabic.run_model. Appends one JSON line to proof/bakeoff_results.jsonl.

Usage: python gguf_gate.py --url http://127.0.0.1:8099 --name run1_fanar-q4_k_m
"""

import argparse
import datetime
import json
import os
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bakeoff_arabic as B  # noqa: E402


def complete(url, prompt):
    body = json.dumps({"prompt": prompt, "n_predict": B.MAX_NEW_TOKENS,
                       "temperature": 0.0, "top_k": 1,
                       "cache_prompt": True}).encode("utf-8")
    req = urllib.request.Request(url.rstrip("/") + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    root_items = B.load_items()
    prompts = B.build_prompts(root_items, instruct=False)
    outputs = []
    for i, (_, p) in enumerate(prompts, 1):
        outputs.append(complete(args.url, p))
        if i % 10 == 0 or i == len(prompts):
            print(f"  [{i}/{len(prompts)}] generated", flush=True)

    per_task = {t: {"n": 0, "correct": 0, "details": []}
                for t in ("root", "reasoning", "morphology")}
    it = iter(outputs)
    for w, gt in root_items:
        raw = next(it); pred = B.parse_root_output(raw); ok = pred == gt
        per_task["root"]["n"] += 1; per_task["root"]["correct"] += ok
        per_task["root"]["details"].append(
            {"word": w, "gt": gt, "pred": pred, "ok": ok, "raw": raw[:200]})
    for q, gt in B.REASONING_ITEMS:
        raw = next(it); pred = B.parse_answer_output(raw)
        ok = pred == B.norm_ar(gt)
        per_task["reasoning"]["n"] += 1; per_task["reasoning"]["correct"] += ok
        per_task["reasoning"]["details"].append(
            {"q": q, "gt": gt, "pred": pred, "ok": ok, "raw": raw[:300]})
    for root, form, expected in B.CURATED_MORPHOLOGY:
        raw = next(it); pred = B.parse_verb_output(raw)
        exp = B.ar_letters(expected); ok = pred == exp
        per_task["morphology"]["n"] += 1; per_task["morphology"]["correct"] += ok
        per_task["morphology"]["details"].append(
            {"root": root, "form": form, "gt": exp, "pred": pred, "ok": ok,
             "raw": raw[:200]})

    scores = {t: {"n": v["n"], "correct": v["correct"],
                  "acc": round(v["correct"] / v["n"], 4)}
              for t, v in per_task.items()}
    total_n = sum(v["n"] for v in per_task.values())
    overall = round(sum(v["correct"] for v in per_task.values()) / total_n, 4)

    print("\n===== RESULTS =====")
    for t, s in scores.items():
        print(f"{t:10s}: {s['correct']}/{s['n']} = {s['acc']:.2%}")
    print(f"{'overall':10s}: {overall:.2%}")
    print("===================")

    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M EDT")
    details_path = os.path.join(B.PROJECT_ROOT, "proof",
                                f"bakeoff_details_{args.name}.json")
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.name, "date": date, "details": per_task}, f,
                  ensure_ascii=False, indent=1)
    record = {"model": args.name, "path": args.url, "date": date,
              "engine": "llama.cpp llama-server", "instruct": False,
              "temperature": "greedy", "max_new_tokens": B.MAX_NEW_TOKENS,
              "answer_parse": "first", "eval_version": B.EVAL_VERSION,
              "status": "ok", **scores,
              "overall": overall}
    B.append_jsonl(record)
    print(f"details -> {details_path}\nappended -> {B.RESULTS_PATH}")


if __name__ == "__main__":
    main()
