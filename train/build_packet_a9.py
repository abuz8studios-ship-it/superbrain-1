# -*- coding: utf-8 -*-
"""build_packet_a9.py - SUPERBRAIN-1 Packet A9 (Arabic math, train) + Packet B math (exam).

Source: Omartificial-Intelligence-Space/Arabic-gsm8k-v2 (Apache-2.0), Arabic GSM8K.
  main_train -> packet_a9.jsonl        (task "reason", gold answer = GSM8K '####' number)
  main_test  -> packet_b_math.jsonl    (EXAM ONLY - never trained, question-hash checked)
Also writes sft_a9_mix.jsonl = A9 + replay of A7 (same count) + all A8, so math is
learned without forgetting roots/morphology."""
import os, re, json, random, hashlib, collections, sys
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
T = r"E:\ABU\02_PROJECTS\AHMAD-1\train"
D = r"E:\ABU\02_PROJECTS\AHMAD-1\data\arabic_gsm8k_v2"
random.seed(1409)
HEAD = "المهمة: حل السؤال التالي خطوة بخطوة ثم اكتب الجواب النهائي رقمًا.\nسؤال: "


def clean(sol):
    sol = re.sub(r"<<[^>]*>>", "", sol)
    body, _, ans = sol.rpartition("####")
    ans = ans.strip().replace(",", "").replace("٬", "")
    return body.strip(), ans


def to_item(q, sol):
    body, ans = clean(sol)
    if not re.fullmatch(r"-?\d+(\.\d+)?", ans):
        return None
    return {"task": "reason", "src": "gsm8k_ar", "prompt": HEAD + q.strip() + "\nالحل:",
            "completion": f" {body}\nالجواب: {ans}\n", "answer": ans,
            "qhash": hashlib.sha1(q.strip().encode("utf-8")).hexdigest()[:16]}


def main():
    tr = pd.read_parquet(os.path.join(D, "main_train.parquet"))
    te = pd.read_parquet(os.path.join(D, "main_test.parquet"))
    a9 = [x for x in (to_item(q, a) for q, a in zip(tr.question, tr.answer)) if x]
    pb = [x for x in (to_item(q, a) for q, a in zip(te.question, te.answer)) if x]
    test_h = {x["qhash"] for x in pb}
    a9 = [x for x in a9 if x["qhash"] not in test_h]
    leak = sum(x["qhash"] in test_h for x in a9)
    with open(os.path.join(T, "packet_a9.jsonl"), "w", encoding="utf-8") as f:
        for x in a9: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(os.path.join(T, "packet_b_math.jsonl"), "w", encoding="utf-8") as f:
        for x in pb: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    a7 = [json.loads(l) for l in open(os.path.join(T, "packet_a7.jsonl"), encoding="utf-8")]
    a8 = [json.loads(l) for l in open(os.path.join(T, "packet_a8.jsonl"), encoding="utf-8")]
    mix = a9 + random.sample(a7, len(a9)) + a8
    random.shuffle(mix)
    with open(os.path.join(T, "sft_a9_mix.jsonl"), "w", encoding="utf-8") as f:
        for x in mix: f.write(json.dumps({"prompt": x["prompt"], "completion": x["completion"],
                                          "task": x["task"]}, ensure_ascii=False) + "\n")
    msg = (f"A9 train={len(a9)} | Packet B math={len(pb)} | LEAKAGE={leak} | "
           f"sft_a9_mix={len(mix)} {dict(collections.Counter(x['task'] for x in mix))} | "
           f"pb_math sha256[:16]={hashlib.sha256(open(os.path.join(T,'packet_b_math.jsonl'),'rb').read()).hexdigest()[:16]}")
    print(msg); open(os.path.join(T, "packet_a9_build.log"), "w", encoding="utf-8").write(msg + "\n")


if __name__ == "__main__":
    main()
