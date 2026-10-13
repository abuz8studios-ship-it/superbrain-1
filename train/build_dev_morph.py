# -*- coding: utf-8 -*-
"""build_dev_morph.py - held-out DEV set for the verb-form skills (root+form -> verb,
verb -> form), built only from the 270 dev roots that no SFT/RL stage ever trains on.
v3-v5 dev sets measured ROOTS only, so nothing watched the morphology front while GRPO
and early-stopped SFT quietly lost 8 points of it. Runs in the sarf venv.
Usage: python build_dev_morph.py  ->  train\\dev_morph.jsonl"""
import json, os, sys, collections, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_packet_a5 as A

random.seed(31)
verbs, nouns = A.load_lexicon()
DEV = A.pick_dev_roots(verbs, nouns)
root_disp = {}
for v in verbs:
    root_disp.setdefault(v["rk"], v["root"])
form_seen, morph_rows, form_items = collections.defaultdict(set), [], []
for v in verbs:
    if v["rk"] not in DEV or len(v["rk"]) != 3 or v["rk"] in A.EVAL_ROOTS:
        continue
    n = A.infer_form(A.skeleton(v["voc"]), v["rk"])
    if n is None or n == 9 or (v["rk"], n) in A.FEWSHOT_PAIRS:
        continue
    shown = A.skeleton(v["voc"])
    ans = shown if n in (2, 5) else v["plain"]
    form_seen[shown].add(n)
    cls = A.root_class(v["rk"])
    form_items.append((shown, n, cls))
    morph_rows.append({"task": "morph", "form": n, "cls": cls,
                       "prompt": A.MORPH_T[0](root_disp[v["rk"]], n), "answer": ans})
form_rows = [{"task": "form", "form": n, "cls": cls, "prompt": A.FORM_T[0](shown),
              "answer": A.ORD[n]} for shown, n, cls in form_items if form_seen[shown] == {n}]
rows = morph_rows + form_rows
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dev_morph.jsonl"), "w",
          encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("dev morph/form items:", len(rows), "| morph", len(morph_rows), "form", len(form_rows),
      "| by form:", dict(sorted(collections.Counter(r["form"] for r in rows).items())))
