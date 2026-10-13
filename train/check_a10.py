# -*- coding: utf-8 -*-
"""check_a10.py - numeric checker for SUPERBRAIN-1 Packet A10 (science).

score(pred_text, item) -> bool : takes the LAST number in pred_text (Arabic-Indic digits, Arabic
decimal/thousands separators, thousands commas and exponent notation handled) and compares it to
item["answer_value"] (fallback float(item["answer"])) with relative tolerance item["tol_rel"] (default 1e-2).

Run directly for the self-test: gold completions must score 100%, shuffled answers must score < 5%.
"""
import os, re, json, math, random, sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
T = os.environ.get("A10_DIR", r"E:\ABU\02_PROJECTS\AHMAD-1\train")
_DIG = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫", "01234567890123456789.")
_NUM = re.compile(r"[-−]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def last_number(text):
    if not text:
        return None
    s = text.translate(_DIG).replace("٬", "")
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)  # thousands commas
    s = re.sub(r"(\d)\s*[×x]\s*10\^\s*([-+−]?\d+)", r"\1e\2", s)  # 1.2 × 10^3
    m = _NUM.findall(s)
    if not m:
        return None
    try:
        return float(m[-1].replace("−", "-"))
    except ValueError:
        return None


def score(pred_text, item):
    p = last_number(pred_text)
    if p is None or not math.isfinite(p):
        return False
    g = float(item["answer_value"] if "answer_value" in item else item["answer"])
    tol = float(item.get("tol_rel", 1e-2))
    if g == 0:
        return abs(p) <= 1e-9
    return abs(p - g) <= tol * abs(g)


def _load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def self_test():
    ok_all = True
    rng = random.Random(10)
    for split in ("train", "exam"):
        items = _load(os.path.join(T, f"packet_a10_{split}.jsonl"))
        n = len(items)
        gold_c = sum(score(x["completion"], x) for x in items)
        gold_a = sum(score(f"الجواب: {x['answer']} {x['unit']}", x) for x in items)
        perm = list(range(n))
        rng.shuffle(perm)
        shuf_c = sum(score(items[perm[i]]["completion"], items[i]) for i in range(n))
        shuf_a = sum(score(items[perm[i]]["answer"], items[i]) for i in range(n))
        # unit-format robustness: Arabic-Indic digits for gold
        ar = sum(score(x["answer"].translate(str.maketrans("0123456789.", "٠١٢٣٤٥٦٧٨٩٫")), x) for x in items)
        gc, ga, sc, sa = (100 * gold_c / n, 100 * gold_a / n, 100 * shuf_c / n, 100 * shuf_a / n)
        print(f"[{split}] n={n} gold_completion={gold_c}/{n} ({gc:.2f}%) gold_answer={gold_a}/{n} ({ga:.2f}%) "
              f"gold_arabic_digits={ar}/{n} | shuffled_completion={shuf_c}/{n} ({sc:.2f}%) "
              f"shuffled_answer={shuf_a}/{n} ({sa:.2f}%)")
        ok = gold_c == n and gold_a == n and ar == n and sc < 5 and sa < 5
        ok_all &= ok
    unit = [("الجواب: ١٢٫٥ م/ث", {"answer_value": 12.5}), ("so 1,234.5 J", {"answer_value": 1234.5}),
            ("= 2.5 × 10^3", {"answer_value": 2500}), ("−4.2 درجة", {"answer_value": -4.2}),
            ("لا أعرف", {"answer_value": 3.0})]
    exp = [True, True, True, True, False]
    u = [score(t, it) == e for (t, it), e in zip(unit, exp)]
    print(f"[unit] parser cases passed {sum(u)}/{len(u)}")
    ok_all &= all(u)
    print("SELF-TEST", "PASS" if ok_all else "FAIL")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if self_test() else 1)
