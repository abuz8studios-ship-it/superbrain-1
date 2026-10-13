#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rescore_gate.py — re-score saved gate runs under the current EVAL_VERSION
without touching a GPU. Only root golds changed in v3 (GOLD_FIX), so each
stored root `pred` is compared to the corrected gold; reasoning and
morphology outcomes are kept as recorded.

Usage: python rescore_gate.py [--append] proof\\bakeoff_details_<name>.json ...
"""

import argparse
import datetime
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bakeoff_arabic as B  # noqa: E402


def rescore(path):
    d = json.load(open(path, encoding="utf-8"))
    det = {t: (v["details"] if isinstance(v, dict) else v)
           for t, v in d["details"].items()}
    gold = dict(B.load_items())
    root_ok = [x["pred"] == gold[x["word"]] for x in det["root"]]
    counts = {"root": (sum(root_ok), len(root_ok))}
    for t in ("reasoning", "morphology"):
        counts[t] = (sum(x["ok"] for x in det[t]), len(det[t]))
    scores = {t: {"n": n, "correct": c, "acc": round(c / n, 4)}
              for t, (c, n) in counts.items()}
    overall = round(sum(c for c, _ in counts.values())
                    / sum(n for _, n in counts.values()), 4)
    return d.get("model", os.path.basename(path)), scores, overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("details", nargs="+")
    ap.add_argument("--append", action="store_true",
                    help="append v3 records to proof/bakeoff_results.jsonl")
    args = ap.parse_args()
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M EDT")
    for path in args.details:
        model, scores, overall = rescore(path)
        print(f"{model:36s} root {scores['root']['acc']:.2%}  reason "
              f"{scores['reasoning']['acc']:.2%}  morph "
              f"{scores['morphology']['acc']:.2%}  overall {overall:.2%}")
        if args.append:
            B.append_jsonl({"model": model, "date": date,
                            "rescored_from": os.path.basename(path),
                            "eval_version": B.EVAL_VERSION, "status": "ok",
                            **scores, "overall": overall})


if __name__ == "__main__":
    main()
