# -*- coding: utf-8 -*-
"""build_g4_rl_pool.py - never-seen root pool for the Gemma G-line (g4_sft1 was SFT'd on sft_a9_mix, whose replay
contains 477 root items from rl_pool_a7 and 125 from dev_root_a7). Keeps rl_pool_a7 items whose root never appears
as an answer in sft_a9_mix and is not a dev / Packet B root. Writes train\\rl_pool_g4.jsonl."""
import json

T = "E:\\ABU\\02_PROJECTS\\AHMAD-1\\train\\"


def rows(n):
    for l in open(T + n, encoding="utf-8"):
        if l.strip():
            yield json.loads(l)


def key(s):
    for a, b in (("أ", "ء"), ("إ", "ء"), ("ؤ", "ء"), ("ئ", "ء"), ("آ", "ء"), ("ا", "ء"), ("ى", "ي")):
        s = s.replace(a, b)
    return s.strip()


seen = {key(r["completion"].split("الجذر:")[-1].split("\n")[0])
        for r in rows("sft_a9_mix.jsonl") if str(r.get("task", "")).startswith("root")}
ban = {l.strip() for l in open(T + "packet_b_roots.txt", encoding="utf-8") if l.strip() and not l.startswith("#")}
ban |= {r["rk"] for r in rows("dev_root_a7.jsonl")}
pool = [r for r in rows("rl_pool_a7.jsonl") if r["rk"] not in seen and r["rk"] not in ban]
with open(T + "rl_pool_g4.jsonl", "w", encoding="utf-8") as f:
    for r in pool:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("rl_pool_g4 items", len(pool), "roots", len({r["rk"] for r in pool}),
      "| dropped roots seen in G1 SFT:", len({r["rk"] for r in rows("rl_pool_a7.jsonl")} & seen))
