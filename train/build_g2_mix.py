# -*- coding: utf-8 -*-
"""build_g2_mix.py - G2 SFT curriculum for a fresh body, the run6 lineage compressed into one SFT pass:
  60,000 rows of packet_a7 (roots/traces/morph; rl_pool_a7 + dev roots already held out by its builder)
  + all of sft_a9_mix (math + A7/A8 replay, 16,645)  + all of packet_a8 (Quran/QAC forms, 2,919)
Why 60k not 153k: every SFT recipe on Fanar plateaued at 65-73% unseen-root accuracy within ~16-50k
samples (2026-10-07); the gain past that came from GRPO on never-seen roots, which G2 does next.
Guards: drops any row whose prompt contains a Packet B word (packet_b.jsonl is public now) and any row
whose root is in packet_b_roots.txt or rl_pool_a7. Writes train\\g2_sft_mix.jsonl (prompt, completion)."""
import json, os, random, re

T = r"E:\ABU\02_PROJECTS\AHMAD-1\train"
random.seed(42)


def rows(name):
    with open(os.path.join(T, name), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


ban_roots = {l.strip() for l in open(os.path.join(T, "packet_b_roots.txt"), encoding="utf-8")
             if l.strip() and not l.startswith("#")}
ban_roots |= {r.get("rk") for r in rows("rl_pool_a7.jsonl") if r.get("rk")}
# dev_root_a7 roots stay unseen too (sft_a9_mix replay was built before A7's holdouts: 477 pool + 125 dev leaks)
ban_roots |= {r.get("rk") for r in rows("dev_root_a7.jsonl") if r.get("rk")}
pb_words = {r["word"] for r in rows("packet_b.jsonl") if r.get("word")}
quoted = re.compile(r"«([^»]+)»|كلمة:\s*(\S+)")


def key(s):
    # root answers write hamza as a bare alef (اتي = key ءتي); hollow roots use و/ي, so ا here is always hamza
    for a, b in (("أ", "ء"), ("إ", "ء"), ("ؤ", "ء"), ("ئ", "ء"), ("آ", "ء"), ("ا", "ء"), ("ى", "ي")):
        s = s.replace(a, b)
    return s.strip()


def clean(r):
    rk = r.get("rk") or r.get("root")
    if rk and rk in ban_roots:
        return False
    if str(r.get("task", "")).startswith("root"):   # answer root of root/root_trace items
        if key(r["completion"].split("الجذر:")[-1].split("\n")[0]) in ban_roots:
            return False
    m = quoted.search(r["prompt"])
    w = (m.group(1) or m.group(2)) if m else None
    return not (w and w in pb_words)


a7 = [r for r in rows("packet_a7.jsonl") if clean(r)]
random.shuffle(a7)
mix = a7[:60000] + [r for r in rows("sft_a9_mix.jsonl") if clean(r)] + [r for r in rows("packet_a8.jsonl") if clean(r)]
random.shuffle(mix)
with open(os.path.join(T, "g2_sft_mix.jsonl"), "w", encoding="utf-8") as f:
    for r in mix:
        f.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"]}, ensure_ascii=False) + "\n")
print(f"g2_sft_mix rows={len(mix)} (a7 pool after guards {len(a7)}, packet_b words {len(pb_words)}, banned roots {len(ban_roots)})")
