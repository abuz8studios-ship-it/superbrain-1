#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bakeoff_arabic.py — AHMAD-1 GRPO base-model bake-off harness.

Evaluates a base LLM (or instruct model via --instruct) on three Arabic
probes and appends one JSON line per run to proof/bakeoff_results.jsonl.

Tasks:
  ROOT       — 50 Arabic surface words -> trilateral root, few-shot (5 worked
               examples). Ground truth = find_root_of_root oracle from
               arabic_root.py (never invented: items are validated against
               the oracle at startup and any oracle-unknown word aborts).
  REASONING  — 20 short MSA arithmetic/syllogism problems, few-shot CoT,
               exact-match on the final answer.
  MORPHOLOGY — 15 root + form-number (I-X) conjugations. The paradigm table
               below is CURATED by hand from standard Arabic grammar
               (Wright/Haywood): 3rd-person masculine singular perfect, no
               diacritics, alef-folded.

Generation: greedy (or --temperature), max_new_tokens 256, left-padded batching.
Base models get plain-text completion prompts (NO chat template). Instruct
models (--instruct) get a single user-turn chat template (for Falcon-H1-Arabic
when access lands).

Honest contract: no invented numbers. A load/run failure is logged to the
JSONL with status "error" and the process exits non-zero.
"""

import argparse
import datetime
import json
import os
import sys
import unicodedata

# GPU1 is the quiet training GPU; honor an explicit user setting.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)  # for arabic_root.py

RESULTS_PATH = os.path.join(PROJECT_ROOT, "proof", "bakeoff_results.jsonl")

MAX_NEW_TOKENS = 256
# v1: original 85 items. v2 (2026-10-05): سمع item corrected to Form VIII.
# v3 (2026-10-06): 4 oracle root golds corrected (GOLD_FIX) — hollow roots
# keep their weak radical, as in QAC/Arramooz and Packet B.
# Scores from different eval versions are not comparable.
EVAL_VERSION = "v3"

# arabic_root.py returns the past-tense verb for hollow roots (ماتت -> مات)
# and splits عقول as ع+قول. Packet A2 and Packet B use true roots.
GOLD_FIX = {"عقول": "عقل", "ماتت": "موت", "مقام": "قوم", "قيام": "قوم"}

import re

# ---------------------------------------------------------------------------
# normalization (shared with arabic_root.py semantics)
# ---------------------------------------------------------------------------

_DIAC_RE = re.compile(
    "[\u064b-\u0652\u0640\u0670\u06d6-\u06dc\u06df-\u06e4\u06e7\u06e8\u06ea-\u06ed]")
_FOLD = {"\u0622": "\u0627", "\u0623": "\u0627", "\u0625": "\u0627",
         "\u0629": "\u0647", "\u0649": "\u064a"}
_AR_DIGITS = {ord("\u0660") + i: str(i) for i in range(10)}
_AR_DIGITS.update({ord("\u06f0") + i: str(i) for i in range(10)})


def norm_ar(s):
    s = unicodedata.normalize("NFC", s or "")
    s = _DIAC_RE.sub("", s)
    s = "".join(_FOLD.get(ch, ch) for ch in s)
    return s.translate(_AR_DIGITS).strip()


def ar_letters(s):
    return "".join(ch for ch in norm_ar(s) if "\u0621" <= ch <= "\u06d3")


# ---------------------------------------------------------------------------
# TASK 1 — ROOT (50 words; GT from oracle)
# ---------------------------------------------------------------------------
# Surface words chosen from verified root lexicon + affix patterns. The script
# asserts at startup that find_root_of_root returns a root for every one.
ROOT_WORDS = [  # exactly 50 surface words (oracle-verified at startup)
    "مكتوب", "كاتب", "مكتبة", "مكاتب",
    "معلم", "تعليم", "متعلم", "معلومات",
    "مفهوم", "تفهم",
    "رحمة", "مرحوم",
    "قلوب", "مقلوب",
    "منور",
    "محقق",
    "صبور",
    "عاقل", "عقول",
    "روحاني",
    "عامل", "عاملون",
    "دخول", "داخل", "مدخل",
    "خروج", "مخرج", "تخرج",
    "اكلت", "ماكل",
    "شربت", "شارب", "مشروب",
    "لباس", "يلبس", "ملبوس",
    "مدرس", "دراسة", "تدريس",
    "جامع", "مجمع", "تجمع",
    "محمد", "حامد",
    "تولد",
    "ماتت",
    "مقام", "قيام",
    "قاعد",
    "عبود",
]

_ROOT_HEADER = (
    "المهمة: استخرج الجذر الثلاثي للكلمة العربية.\n"
    "أمثلة:\n"
    "كلمة: مكتبات\nالجذر: كتب\n"
    "كلمة: مدرسة\nالجذر: درس\n"
    "كلمة: شاهد\nالجذر: شهد\n"
    "كلمة: محفوظ\nالجذر: حفظ\n"
    "كلمة: استخراج\nالجذر: خرج\n"
    "الآن اشتق الجذر لكل كلمة:\n"
)


def root_prompt(word):
    return f"{_ROOT_HEADER}كلمة: {word}\nالجذر:"


def parse_root_output(text):
    # generation starts right after "الجذر:" — take first non-empty line
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return ar_letters(line)
    return ""


# ---------------------------------------------------------------------------
# TASK 2 — REASONING (20 MSA arithmetic + syllogisms; authored here)
# ---------------------------------------------------------------------------
REASONING_ITEMS = [
    ("خالد لديه 7 تفاحات وأعطى أخاه 3 تفاحات. كم تفاحة بقيت معه؟", "4"),
    ("سلمى اشترت 5 أقلام بسعر 2 ريال للقلم الواحد. كم ريالاً دفعت؟", "10"),
    ("في صف 12 طالباً، غاب 4 طلاب. كم طالباً حضر؟", "8"),
    ("قطار يقطع 60 كيلومتراً في الساعة. كم كيلومتراً يقطع في 3 ساعات؟", "180"),
    ("عمر أحمد 10 سنوات وعمر أخيه ضعف عمره. كم عمر أخيه؟", "20"),
    ("لدى نعمان 20 كرة، أعطى نصفها لصديقه. كم كرة أعطى؟", "10"),
    ("بستان فيه 8 أشجار، زرع العامل 5 أشجار جديدة. كم شجرة في البستان الآن؟", "13"),
    ("قرأت فاطمة 15 صفحة في اليوم الأول و25 صفحة في اليوم الثاني. كم صفحة قرأت في اليومين؟", "40"),
    ("لكل قطة 4 أرجل. كم رجلاً لـ 5 قطط؟", "20"),
    ("لدى سالم 3 صناديق، في كل صندوق 6 كتب. كم كتاباً لديه؟", "18"),
    ("كانت مع مريم 90 ريالاً، أنفقت 50 ريالاً. كم ريالاً بقي معها؟", "40"),
    ("جميع القطط حيوانات. مُشتاق قطة. هل مُشتاق حيوان؟", "نعم"),
    ("لا يملك أي طائر أربع أرجل. النسر طائر. هل للنسر أربع أرجل؟", "لا"),
    ("بعض التلاميذ مجتهدون. عمر تلميذ. هل عمر مجتهد بالضرورة؟", "لا"),
    ("كل المعادن أصلب من الماء. الحديد معدن. هل الحديد أصلب من الماء؟", "نعم"),
    ("جميع طلاب الفصل الثالث مجتهدون. ليلى من طلاب الفصل الثالث. هل ليلى مجتهدة؟", "نعم"),
    ("إذا كان اليوم السبت فغداً الأحد. اليوم السبت. هل غداً الأحد؟", "نعم"),
    ("بعض الفواكه حمراء. التفاح فاكهة. هل التفاح أحمر بالضرورة؟", "لا"),
    ("إذا أمطرت السماء خرج الناس بمظلات. لم تُمطر السماء. هل خرج الناس بمظلات بالضرورة؟", "لا"),
    ("رقم إذا ضربته في نفسه كان الناتج 49. ما الرقم؟", "7"),
]

_REASON_HEADER = (
    "المهمة: حل الأسئلة التالية خطوة بخطوة ثم كتابة الجواب النهائي.\n"
    "أمثلة:\n"
    "سؤال: لدى خالد 5 تفاحات واشترى 2 أخرى. كم تفاحة لديه؟\n"
    "الحل: 5 + 2 = 7\nالجواب: 7\n"
    "سؤال: كل الطيور تطير. الببغاء طائر. هل الببغاء يطير؟\n"
    "الحل: الببغاء طائر، وكل الطيور تطير، إذن الببغاء يطير.\nالجواب: نعم\n"
    "سؤال: ناقص من 10 العدد 6. كم الباقي؟\n"
    "الحل: 10 - 6 = 4\nالجواب: 4\n"
    "سؤال: بعض الأزهار بيضاء. الياسمين زهرة. هل الياسمين أبيض بالضرورة؟\n"
    "الحل: بعض الأزهار فقط بيضاء، فليس بالضرورة أن الياسمين أبيض.\nالجواب: لا\n"
    "سؤال: بائع لديه 3 سلال، في كل سلة 5 برتقالات. كم برتقالة لديه؟\n"
    "الحل: 3 × 5 = 15\nالجواب: 15\n"
    "الآن حل:\n"
)


def reasoning_prompt(q):
    return f"{_REASON_HEADER}سؤال: {q}\nالحل:"


def parse_answer_output(text):
    """Extract the final answer after 'الجواب:'; fallback: last number/نعم/لا."""
    marker = "الجواب:"
    if marker in text:
        tail = text.split(marker, 1)[1]  # first answer = answer to posed question
        line = tail.strip().splitlines()[0] if tail.strip() else ""
        ans = norm_ar(line).rstrip(".,!؟?").strip()
        if ans:
            return ans
    # fallback: last Arabic/Latin digit run or نعم/لا anywhere
    m = re.findall(r"\d+", norm_ar(text))
    if m:
        return m[-1]
    for tok in ("نعم", "لا"):
        if tok in norm_ar(text):
            return tok
    return ""


# ---------------------------------------------------------------------------
# TASK 3 — MORPHOLOGY (15 items; CURATED paradigm table, standard grammar)
# ---------------------------------------------------------------------------
# Curated by hand from standard Arabic morphology (Wright, Haywood & Nahmad):
# Form I-X of strong triliteral roots, 3rd-person masculine singular perfect,
# undiacritized, alef-folded. Labeled CURATED — not model-derived.
FORM_NUM_TO_AR = {1: "الأولى", 2: "الثانية", 3: "الثالثة", 4: "الرابعة",
                  5: "الخامسة", 6: "السادسة", 7: "السابعة", 8: "الثامنة",
                  9: "التاسعة", 10: "العاشرة"}

CURATED_MORPHOLOGY = [
    # (root, form, expected 3ms perfect)
    ("كتب", 2, "كتّب"),
    ("كتب", 8, "اكتتب"),
    ("كتب", 10, "استكتب"),
    ("علم", 2, "علّم"),
    ("علم", 4, "أعلم"),
    ("علم", 10, "استعلم"),
    ("فهم", 2, "فهّم"),
    ("فهم", 4, "أفهم"),
    ("عمل", 3, "عامل"),
    ("عمل", 10, "استعمل"),
    ("سمع", 8, "استمع"),  # v2 fix: استمع is Form VIII (v1 mislabeled it Form X)
    ("خرج", 2, "خرّج"),
    ("خرج", 10, "استخرج"),
    ("نصر", 3, "ناصر"),
    ("ذكر", 5, "تذكّر"),
]

_MORPH_HEADER = (
    "المهمة: صرف الفعل الثلاثي حسب صيغة الفعل المطالب بها، بصيغة الماضي للغائب.\n"
    "أمثلة:\n"
    "الجذر: سلم — الصيغة الثانية\nالفعل: سلّم\n"
    "الجذر: ذكر — الصيغة الخامسة\nالفعل: تذكّر\n"
    "الجذر: نصر — الصيغة السادسة\nالفعل: تناصر\n"
    "الجذر: دخل — الصيغة السابعة\nالفعل: اندخل\n"
    "الجذر: علم — الصيغة الثالثة\nالفعل: عالم\n"
    "الآن اصرف:\n"
)


def morph_prompt(root, form):
    return (f"{_MORPH_HEADER}الجذر: {root} — الصيغة {FORM_NUM_TO_AR[form]}"
            f"\nالفعل:")


def parse_verb_output(text):
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return ar_letters(line)
    return ""


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
def load_items():
    """Validate ROOT_WORDS against the oracle; abort if any is unknown."""
    from arabic_root import find_root_of_root
    root_items = []  # (word, gt_root_letters)
    bad = []
    for w in ROOT_WORDS:
        r = find_root_of_root(w)
        if r is None:
            bad.append(w)
        else:
            root_items.append((w, ar_letters(GOLD_FIX.get(w, r["root"]))))
    if bad:
        raise RuntimeError(f"oracle rejected ROOT_WORDS entries: {bad}")
    return root_items


def build_prompts(root_items, instruct):
    """Return list of (task, prompt) tuples."""
    prompts = []
    for w, _ in root_items:
        prompts.append(("root", root_prompt(w)))
    for q, _ in REASONING_ITEMS:
        prompts.append(("reasoning", reasoning_prompt(q)))
    for root, form, _ in CURATED_MORPHOLOGY:
        prompts.append(("morphology", morph_prompt(root, form)))
    return prompts


def wrap_instruct(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)


def run_model(model_path, instruct, temperature, batch_size):
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer)

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # for batched causal LM generation

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.cuda().eval()

    root_items = load_items()
    prompts = build_prompts(root_items, instruct)
    if instruct:
        prompts = [(t, wrap_instruct(tok, p)) for t, p in prompts]

    greedy = temperature is None or temperature <= 0.0
    gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=not greedy,
                      pad_token_id=tok.pad_token_id)
    if not greedy:
        gen_kwargs.update(temperature=temperature, top_p=0.9)

    outputs = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            enc = tok([p for _, p in chunk], return_tensors="pt",
                      padding=True, truncation=True,
                      max_length=1024).to("cuda")
            gen = model.generate(**enc, **gen_kwargs)
            text = tok.batch_decode(
                gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            outputs.extend(text)
            print(f"  [{i + len(chunk)}/{len(prompts)}] generated", flush=True)

    # score
    per_task = {t: {"n": 0, "correct": 0, "details": []}
                for t in ("root", "reasoning", "morphology")}
    it = iter(outputs)

    for w, gt in root_items:
        raw = next(it)
        pred = parse_root_output(raw)
        ok = pred == gt
        per_task["root"]["n"] += 1
        per_task["root"]["correct"] += ok
        per_task["root"]["details"].append(
            {"word": w, "gt": gt, "pred": pred, "ok": ok, "raw": raw[:200]})

    for q, gt in REASONING_ITEMS:
        raw = next(it)
        pred = parse_answer_output(raw)
        ok = pred == norm_ar(gt)
        per_task["reasoning"]["n"] += 1
        per_task["reasoning"]["correct"] += ok
        per_task["reasoning"]["details"].append(
            {"q": q, "gt": gt, "pred": pred, "ok": ok, "raw": raw[:300]})

    for root, form, expected in CURATED_MORPHOLOGY:
        raw = next(it)
        pred = parse_verb_output(raw)
        exp = ar_letters(expected)
        ok = pred == exp
        per_task["morphology"]["n"] += 1
        per_task["morphology"]["correct"] += ok
        per_task["morphology"]["details"].append(
            {"root": root, "form": form, "gt": exp, "pred": pred, "ok": ok,
             "raw": raw[:200]})

    total_n = sum(v["n"] for v in per_task.values())
    total_c = sum(v["correct"] for v in per_task.values())
    scores = {t: {"n": v["n"], "correct": v["correct"],
                  "acc": round(v["correct"] / v["n"], 4)}
              for t, v in per_task.items()}
    overall = round(total_c / total_n, 4)
    return {"scores": scores, "overall": overall, "details": per_task}


def append_jsonl(record):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="AHMAD-1 base-model bake-off")
    ap.add_argument("--model", required=True, help="HF path or local snapshot")
    ap.add_argument("--instruct", action="store_true",
                    help="use chat template (instruct models, e.g. Falcon-H1)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 = greedy (default); >0 e.g. 0.2 for sampling")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    norm_path = args.model.replace("\\", "/").rstrip("/")
    m = re.search(r"models--[^/]+--([^/]+)", norm_path)
    parts = norm_path.split("/")
    model_name = m.group(1) if m else parts[-1]
    # train\<run>\merged_16bit -> name the run, so runs don't overwrite each
    # other's details file
    if not m and model_name in ("merged_16bit", "lora") and len(parts) > 1:
        model_name = parts[-2]
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M EDT")
    temp = args.temperature if args.temperature > 0 else None

    print(f"bakeoff: {model_name} | greedy={temp is None} | {date}",
          flush=True)
    record = {"model": model_name, "path": args.model, "date": date,
              "instruct": args.instruct,
              "temperature": temp if temp else "greedy",
              "max_new_tokens": MAX_NEW_TOKENS, "answer_parse": "first",
              "eval_version": EVAL_VERSION, "status": "ok"}

    try:
        result = run_model(args.model, args.instruct, temp, args.batch_size)
        record.update(result["scores"])
        record["overall"] = result["overall"]
        print("\n===== RESULTS =====")
        for t, s in result["scores"].items():
            print(f"{t:10s}: {s['correct']}/{s['n']} = {s['acc']:.2%}")
        print(f"{'overall':10s}: {result['overall']:.2%}")
        print("===================")
        # dump per-item details next to the JSONL for audit
        details_path = os.path.join(
            PROJECT_ROOT, "proof",
            f"bakeoff_details_{model_name}.json")
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump({"model": model_name, "date": date,
                       "details": result["details"]}, f,
                      ensure_ascii=False, indent=1)
        print(f"details -> {details_path}")
    except Exception as e:  # honest failure logging
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {e}"
        print(f"FAILED: {record['error']}", flush=True)

    append_jsonl(record)
    print(f"appended -> {RESULTS_PATH}")
    sys.exit(0 if record["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
