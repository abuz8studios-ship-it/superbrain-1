# -*- coding: utf-8 -*-
"""bakeoff_pick.py - pick the G2 body from RAW (untrained) scores, same harness for every body.

Rule fixed before any Oryx number existed (2026-10-12 14:30 ET):
  body score = mean(Packet B root/bare, Packet B root/fewshot, Packet B morph, Arabic GSM8K math)
  highest score wins; ties -> the Arabic-native body (Ahmad 2026-10-05: Arabic-native foundation first).
Writes train\\BAKEOFF.json and prints the table."""
import json, os, sys, time

ROOT = r"E:\ABU\02_PROJECTS\AHMAD-1"
PROOF = os.path.join(ROOT, "proof")
BODIES = [  # (body, packet-b/gate model name, math name, hf dir, arabic_native)
    ("Fanar-1-9B (base)", "fanar-base-q5_k_m", "fanar_base-q5_k_m", r"E:\ABU\MODELS\hf\Fanar-1-9B", True),
    ("Gemma-4-E4B (base)", "g4_base-q5_k_m", "g4_base-q5_k_m", r"E:\ABU\MODELS\hf\gemma-4-E4B", False),
    ("Fanar-2-Oryx-IVU", "oryx_base-q5_k_m", "oryx_base-q5_k_m", r"E:\ABU\MODELS\hf\Fanar-2-Oryx-IVU", True),
]


def last(path, name):
    rec = None
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("model") == name:
                rec = r
    return rec


rows = []
for body, pbname, mname, hf, native in BODIES:
    pb = last(os.path.join(PROOF, "packetb_results.jsonl"), pbname)
    gate = last(os.path.join(PROOF, "bakeoff_results.jsonl"), pbname)
    mp = os.path.join(PROOF, f"math_{mname}.json")
    math = json.load(open(mp, encoding="utf-8"))["acc"] if os.path.exists(mp) else None
    s = pb["summary"] if pb else {}
    m = {"bare": s.get("root/bare", {}).get("acc"), "fewshot": s.get("root/fewshot", {}).get("acc"),
         "morph": s.get("morph", {}).get("acc"), "math": math}
    ok = all(v is not None for v in m.values())
    rows.append({"body": body, "hf_dir": hf, "arabic_native": native, **m,
                 "gate_v3": gate.get("overall") if gate and gate.get("eval_version") == "v3" else None,
                 "score": round(sum(m.values()) / 4, 4) if ok else None, "complete": ok})

done = [r for r in rows if r["complete"]]
win = max(done, key=lambda r: (r["score"], r["arabic_native"])) if done else None
out = {"date": time.strftime("%Y-%m-%d %H:%M"),
       "rule": "mean(PB root/bare, PB root/fewshot, PB morph, math); ties -> Arabic-native",
       "rows": rows, "winner": win["body"] if win else None,
       "all_measured": len(done) == len(rows)}
json.dump(out, open(os.path.join(ROOT, "train", "BAKEOFF.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for r in rows:
    print(f"{r['body']:<22} bare={r['bare']} fewshot={r['fewshot']} morph={r['morph']} math={r['math']} gate={r['gate_v3']} score={r['score']}")
print("WINNER", out["winner"], "all_measured", out["all_measured"])
