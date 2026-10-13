# -*- coding: utf-8 -*-
"""build_grpo_data.py - SUPERBRAIN 1 GRPO dataset (Packet A).
Verifiable rewards only: every answer is checked by arabic_root.py oracle or
by grammar rule / arithmetic. Every bake-off eval word, eval root and few-shot
example is EXCLUDED (Packet A never sees Packet B)."""
import sys, os, json, random
ROOT = r"E:\ABU\02_PROJECTS\AHMAD-1"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from arabic_root import find_root_of_root, _ROOTS
import bakeoff_arabic as B
random.seed(42)

EVAL_WORDS = {B.norm_ar(w) for w in B.ROOT_WORDS} | {
    B.norm_ar(w) for w in ["مكتبات", "مدرسة", "شاهد", "محفوظ", "استخراج"]}
EVAL_ROOTS = {B.ar_letters(r) for r, _, _ in B.CURATED_MORPHOLOGY}
FEWSHOT_PAIRS = {("سلم", 2), ("ذكر", 5), ("نصر", 6), ("دخل", 7), ("علم", 3)}

WEAK = set("اويءئؤأإآ")
SOUND = sorted(r for r in _ROOTS if len(r) == 3 and not set(r) & WEAK
               and r[0] != r[1] and r[1] != r[2])

# ---- ROOT task: pattern words, kept only if the oracle agrees ----
PATTERNS = ["م{0}{1}و{2}", "{0}ا{1}{2}", "م{0}{1}{2}", "م{0}{1}{2}ه",
            "ت{0}{1}ي{2}", "{0}{1}و{2}", "ي{0}{1}{2}", "ت{0}{1}{2}",
            "م{0}ا{1}{2}", "است{0}{1}ا{2}", "{0}{1}{2}ه", "{0}{1}ا{2}",
            "ال{0}ا{1}{2}", "{0}ا{1}{2}ون", "م{0}{1}{2}ات", "ي{0}{1}{2}ون",
            "ت{0}{1}{2}ين", "م{0}{1}و{2}ه", "{0}{1}{2}ت", "ن{0}{1}{2}"]
root_items = []
for r in sorted(_ROOTS):
    if len(r) != 3:
        continue
    for p in PATTERNS:
        w = p.format(*r)
        if B.norm_ar(w) in EVAL_WORDS:
            continue
        o = find_root_of_root(w)
        if o and B.ar_letters(o["root"]) == B.ar_letters(r):
            root_items.append({"task": "root", "prompt": B.root_prompt(w),
                               "answer": B.ar_letters(r)})

# ---- MORPHOLOGY task: Forms I-X (no IX) of sound roots, rule-built ----
def form(r, n):
    f, a, l = r
    return {1: r, 2: r, 3: f + "ا" + a + l, 4: "ا" + r, 5: "ت" + r,
            6: "ت" + f + "ا" + a + l, 7: "ان" + r,
            8: None if f in "صضطظدذزث" else "ا" + f + "ت" + a + l,
            10: "است" + r}[n]
morph_items = []
for r in SOUND:
    if r in EVAL_ROOTS:
        continue
    for n in (1, 2, 3, 4, 5, 6, 7, 8, 10):
        if (r, n) in FEWSHOT_PAIRS:
            continue
        v = form(r, n)
        if v:
            morph_items.append({"task": "morphology",
                                "prompt": B.morph_prompt(r, n), "answer": v})

# ---- REASONING task: generated Arabic arithmetic (keeps 4B's 100%) ----
reason_items = []
names = ["سعيد", "هدى", "يوسف", "مريم", "علي", "نور", "زيد", "سارة"]
things = ["كتب", "أقلام", "تفاحات", "كرات", "دفاتر"]
for _ in range(300):
    n, t = random.choice(names), random.choice(things)
    a, b = random.randint(2, 40), random.randint(2, 12)
    k = random.choice(["add", "sub", "mul"])
    if k == "add":
        q, ans = f"لدى {n} {a} {t} واشترى {b} أخرى. كم {t} لديه؟", a + b
    elif k == "sub":
        a = a + b
        q, ans = f"لدى {n} {a} {t} وأعطى {b} منها. كم بقي معه؟", a - b
    else:
        q, ans = f"لدى {n} {b} صناديق، في كل صندوق {a} {t}. كم المجموع؟", a * b
    reason_items.append({"task": "reasoning", "prompt": B.reasoning_prompt(q),
                         "answer": str(ans)})

random.shuffle(root_items); random.shuffle(morph_items)
data = root_items[:600] + morph_items[:600] + reason_items
random.shuffle(data)
out = os.path.join(ROOT, "train", "grpo_train.jsonl")
with open(out, "w", encoding="utf-8") as f:
    for d in data:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
print(f"sound roots={len(SOUND)} root_items={len(root_items)} "
      f"morph_items={len(morph_items)} reasoning={len(reason_items)} "
      f"-> wrote {len(data)} to {out}")
leak = [d for d in data if d["task"] == "root" and
        d["prompt"].rsplit("كلمة: ", 1)[1].split("\n")[0] in EVAL_WORDS]
print("eval-word leakage:", len(leak))
