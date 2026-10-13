# -*- coding: utf-8 -*-
"""build_packet_a11.py - SUPERBRAIN-1 Packet A11 (Arabic-instruction Python coding, train + exam).

Fully programmatic: every prompt, reference solution, hidden test and gold output is produced by
the templates in this file (random.Random(11)) and by EXECUTING code. No LLM text, no teacher outputs.

Kinds (27 generators):
  predict_output    (10) - "ما الذي يطبعه البرنامج؟"  gold = stdout of the snippet, obtained by running
                           it in a fresh `python -I` subprocess (timeout) at build time.
  write_function    (11) - "اكتب دالة ..."            hidden asserts; expected values computed from the
                           reference, then reference+asserts re-run in a subprocess and must pass.
  fix_bug            (3) - buggy function shown; reference must pass, buggy version must FAIL the tests.
  complete_function  (3) - skeleton with ____ to fill; reference must pass the tests.

Schema (A10-style): task, src, prompt (HEAD + question + "\\nالجواب:"), completion, answer, qhash
+ kind, topic, template, lang; predict: code; code kinds: entry, reference, tests, test_code (+ buggy).

Split: each generator has 3 phrasings; phrasings 0/1 -> train, phrasing 2 -> exam ONLY, and exam
parameter tuples are disjoint from all train parameter tuples of that generator.
Leakage check: exact / normalized prompt, normalized question, question hash, param combos,
snippet code and reference code overlap train<->exam.
Scoring: a11_score.py (exact output match, or hidden tests in a subprocess with 5 s timeout).
"""
import os, re, sys, json, copy, time, random, hashlib, collections
from concurrent.futures import ThreadPoolExecutor

os.environ["CUDA_VISIBLE_DEVICES"] = ""
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a11_score import run_py, program_with_tests, SENTINEL  # noqa: E402

T = os.environ.get("A11_DIR", r"E:\ABU\02_PROJECTS\AHMAD-1\train")
OUT = os.environ.get("A11_OUT", "packet_a11")
SEED = 11
N_TRAIN_PER_GEN = int(os.environ.get("A11_NTRAIN", "223"))
N_EXAM_PER_GEN = int(os.environ.get("A11_NEXAM", "38"))
WORKERS = int(os.environ.get("A11_WORKERS", "6"))
BUILD_TIMEOUT = 10.0
HEAD = "المهمة: حل مهمة البرمجة التالية بلغة بايثون.\nسؤال: "
TAIL = "\nالجواب:"


def ch(r, xs):
    return xs[r.randrange(len(xs))]


AR_NAMES = ["أحمد", "خالد", "يوسف", "عمر", "حسن", "سلمى", "زينب", "مريم", "ليلى", "فيصل",
            "علي", "سعيد", "هدى", "نور", "باسل", "هشام", "رنا", "وليد", "منى", "راشد"]
AR_WORDS = ["كتاب", "مدرسة", "قلم", "سماء", "بحر", "شمس", "قمر", "نهر", "جبل", "وردة", "باب", "بيت",
            "سيارة", "طائرة", "حاسوب", "برمجة", "مكتبة", "علم", "سلام", "مدينة", "شجرة", "طالب", "معلم",
            "لغة", "عربية", "قهوة", "ماء", "نور", "حديقة", "صديق", "دفتر", "سوق", "عمل", "ليل", "نهار",
            "فجر", "مطر", "رمل", "صحراء", "جمل", "حصان", "سفينة", "ميناء", "تفاحة", "برتقال", "خبز"]
AR_LETTERS = ["ا", "ب", "ت", "ر", "س", "ل", "م", "ن", "ه", "و", "ي", "ع", "ك", "د", "ق", "ح", "ج", "ف", "ص", "ش"]
EN_WORDS = ["python", "data", "brain", "code", "table", "apple", "river", "stone", "light", "music",
            "garden", "planet", "window", "rocket", "matrix", "signal", "vector", "pixel", "cloud",
            "engine", "banana", "orange", "summer", "winter", "falcon", "desert", "camel", "coffee"]
SHORT_EN = ["a", "an", "of", "to", "is", "in", "on", "at", "by", "we", "it", "go", "up"]


def ar_sentence(r, lo=0, hi=7):
    return " ".join(ch(r, AR_WORDS) for _ in range(r.randint(lo, hi)))


def en_sentence(r, lo=0, hi=8):
    pool = EN_WORDS + SHORT_EN
    return " ".join(ch(r, pool) for _ in range(r.randint(lo, hi)))


def ints(r, nlo, nhi, lo, hi):
    return [r.randint(lo, hi) for _ in range(r.randint(nlo, nhi))]


def body(*lines):
    return "\n".join(lines)


# ---------------------------------------------------------------- predict-the-output generators
# each returns dict(kind="predict_output", key, topic, code)
PQ = ["ما الذي يطبعه برنامج بايثون التالي؟\n```python\n{code}\n```",
      "توقّع ناتج تنفيذ الشيفرة التالية واكتبه كما يظهر على الشاشة تمامًا:\n```python\n{code}\n```",
      "نفّذ المقطع البرمجي الآتي في ذهنك، ثم اكتب بدقة كل ما سيُطبع سطرًا بسطر:\n```python\n{code}\n```"]


def g_p_slice(r):
    v = ch(r, ["a", "nums", "xs", "data", "lst", "items"])
    a = ints(r, 5, 9, -9, 40)
    i = r.randint(0, 3)
    j = r.randint(i + 1, len(a))
    s = ch(r, [-1, 2, 3, -2])
    agg = ch(r, ["len", "sum", "max", "min"])
    code = body(f"{v} = {a}", f"print({v}[{i}:{j}])", f"print({v}[::{s}])", f"print({agg}({v}[{i}:{j}]))")
    return dict(kind="predict_output", key=(v, tuple(a), i, j, s, agg), topic="list_slicing", code=code)


def g_p_loop(r):
    init, a = r.randint(0, 20), r.randint(-5, 10)
    b, s, k = a + r.randint(3, 25), r.randint(1, 4), r.randint(1, 9)
    op = ch(r, ["+=", "-="])
    v, it = ch(r, ["total", "acc", "s", "result"]), ch(r, ["i", "n", "j", "x"])
    code = body(f"{v} = {init}", f"for {it} in range({a}, {b}, {s}):", f"    {v} {op} {it} * {k}",
                f"print({v})", f"print({it})")
    return dict(kind="predict_output", key=(init, a, b, s, k, op, v, it), topic="for_loop", code=code)


def g_p_str(r):
    w = ch(r, EN_WORDS) + ch(r, ["", "", "_" + ch(r, EN_WORDS)])
    v = ch(r, ["s", "word", "text", "name"])
    c = ch(r, list(w.replace("_", "")))
    d = ch(r, ["*", "#", "X", "0"])
    i = r.randint(0, 3)
    j = r.randint(i + 1, len(w))
    pool = [f"{v}.upper()", f"{v}[::-1]", f"{v}.count({c!r})", f"{v}.replace({c!r}, {d!r})",
            f"{v}.find({c!r})", f"{v}[{i}:{j}]", f"len({v})", f"{v}.capitalize()", f"{v}.split('_')",
            f"{v}[-{r.randint(1, 4)}:]", f"{v}.index({c!r}) + len({v})"]
    ops = r.sample(pool, 3)
    code = body(f"{v} = {w!r}", *[f"print({o})" for o in ops])
    return dict(kind="predict_output", key=(w, v, tuple(ops)), topic="strings", code=code)


def g_p_dict(r):
    keys = r.sample(AR_NAMES, r.randint(3, 5))
    d = {k: r.randint(0, 50) for k in keys}
    k1 = ch(r, keys)
    add = r.randint(1, 20)
    new = ch(r, [n for n in AR_NAMES if n not in keys])
    val = r.randint(0, 60)
    v = ch(r, ["scores", "d", "marks", "points"])
    last = ch(r, [f"print(sorted({v}.items(), key=lambda kv: (kv[1], kv[0]))[:2])",
                  f"print([k for k in {v} if {v}[k] > {r.randint(10, 40)}])",
                  f"print({v}.get({ch(r, AR_NAMES)!r}, -1))"])
    code = body(f"{v} = {d!r}", f"{v}[{k1!r}] += {add}", f"{v}[{new!r}] = {val}", f"print(len({v}))",
                f"print(max({v}, key={v}.get))", f"print(sum({v}.values()))", last)
    return dict(kind="predict_output", key=(tuple(d.items()), k1, add, new, val, v, last), topic="dict", code=code)


def g_p_while(r):
    v = ch(r, ["n", "x", "m"])
    c = ch(r, ["steps", "count", "c"])
    if r.random() < 0.5:
        n, d, lim = r.randint(50, 100000), r.randint(2, 9), r.randint(0, 20)
        code = body(f"{v} = {n}", f"{c} = 0", f"while {v} > {lim}:", f"    {v} = {v} // {d}", f"    {c} += 1",
                    f"print({v}, {c})")
        key = ("div", n, d, lim, v, c)
    else:
        n = r.randint(2, 400)
        code = body(f"{v} = {n}", f"{c} = 0", f"while {v} != 1:", f"    if {v} % 2 == 0:", f"        {v} = {v} // 2",
                    "    else:", f"        {v} = 3 * {v} + 1", f"    {c} += 1", f"print({c})")
        key = ("collatz", n, v, c)
    return dict(kind="predict_output", key=key, topic="while_loop", code=code)


def g_p_comp(r):
    a = r.randint(-6, 6)
    b = a + r.randint(5, 15)
    q = r.randint(2, 5)
    rr = r.randint(0, q - 1)
    m, c = r.randint(-3, 5), r.randint(-10, 10)
    expr = ch(r, [f"x * {m} + {c}", f"x ** 2 - {abs(c)}", f"(x + {abs(c)}) // {q}"])
    v = ch(r, ["res", "out", "vals"])
    code = body(f"{v} = [{expr} for x in range({a}, {b}) if x % {q} == {rr}]", f"print({v})",
                f"print(len({v}), sum({v}))")
    return dict(kind="predict_output", key=(a, b, q, rr, expr, v), topic="list_comprehension", code=code)


def g_p_func(r):
    A, D = r.randint(-5, 9), r.randint(0, 9)
    p, q = r.randint(-10, 20), r.randint(-10, 20)
    f = ch(r, ["f", "calc", "g", "compute", "h"])
    code = body(f"def {f}(x, y={D}):", "    if x > y:", f"        return x * {A} - y", f"    return x + y * {A}",
                "", f"print({f}({p}))", f"print({f}({p}, {q}))", f"print({f}({f}({q}), {p}))")
    return dict(kind="predict_output", key=(A, D, p, q, f), topic="functions", code=code)


def g_p_rec(r):
    f = ch(r, ["f", "rec", "go", "calc"])
    fam = r.randrange(4)
    if fam == 0:
        b, base, m, c = r.randint(0, 3), r.randint(1, 5), r.randint(1, 3), r.randint(-3, 4)
        N = b + r.randint(1, 10)
        code = body(f"def {f}(n):", f"    if n <= {b}:", f"        return {base}", f"    return {m} * {f}(n - 1) + {c}",
                    "", f"print({f}({N}))")
        key = (fam, b, base, m, c, N, f)
    elif fam == 1:
        N = r.randint(10, 10 ** 9)
        code = body(f"def {f}(n):", "    if n < 10:", "        return n", f"    return n % 10 + {f}(n // 10)", "",
                    f"print({f}({N}))", f"print({f}({f}({N})))")
        key = (fam, N, f)
    elif fam == 2:
        n, k = r.randint(-5, 9), r.randint(0, 8)
        code = body(f"def {f}(n, k):", "    if k == 0:", "        return 1", f"    return n * {f}(n, k - 1)", "",
                    f"print({f}({n}, {k}))")
        key = (fam, n, k, f)
    else:
        c, N = r.randint(0, 3), r.randint(2, 18)
        code = body(f"def {f}(n):", "    if n < 2:", f"        return n + {c}", f"    return {f}(n - 1) + {f}(n - 2)", "",
                    f"print({f}({N}))")
        key = (fam, c, N, f)
    return dict(kind="predict_output", key=key, topic="recursion", code=code)


def g_p_sort(r):
    mode = r.randrange(3)
    if mode == 0:
        ws = r.sample(EN_WORDS, 5)
        code = body(f"words = {ws!r}", "print(sorted(words, key=len))",
                    "print(sorted(words, key=lambda w: (len(w), w), reverse=True)[:3])")
        key = (mode, tuple(ws))
    elif mode == 1:
        ns = ints(r, 5, 8, -20, 40)
        c = r.randint(-10, 30)
        code = body(f"nums = {ns}", f"print(sorted(nums, key=lambda x: (abs(x - {c}), x)))",
                    "print(sorted(nums, reverse=True)[:3])")
        key = (mode, tuple(ns), c)
    else:
        ps = [(n, r.randint(0, 100)) for n in r.sample(AR_NAMES, 4)]
        code = body(f"pairs = {ps!r}", "print(sorted(pairs, key=lambda p: (p[1], p[0])))",
                    "print([p[0] for p in sorted(pairs, key=lambda p: (-p[1], p[0]))][:2])")
        key = (mode, tuple(ps))
    return dict(kind="predict_output", key=key, topic="sorting", code=code)


def g_p_arabic(r):
    w = ch(r, AR_WORDS) + ch(r, ["", " " + ch(r, AR_WORDS)])
    L = ch(r, list(w.replace(" ", "")) + AR_LETTERS[:5])
    v = ch(r, ["w", "kalima", "text", "s"])
    pool = [f"len({v})", f"{v}[::-1]", f"{v}.count({L!r})", f"{v}[0]", f"{v}[-1]", f"{v}.split()",
            f"{v}.replace(' ', '-')", f"{L!r} in {v}", f"{v}.find({L!r})", f"len({v}.replace(' ', ''))"]
    ops = r.sample(pool, 3)
    code = body(f"{v} = {w!r}", *[f"print({o})" for o in ops])
    return dict(kind="predict_output", key=(w, L, v, tuple(ops)), topic="arabic_text", code=code)


# ---------------------------------------------------------------- write-a-function generators
# each returns dict(kind="write_function", key, topic, entry, ref, args (list of arg tuples), qs[3])
W = "write_function"


def g_w_sum_mult(r):
    a = r.randint(2, 9)
    b = r.randint(a + 1, 15)
    f = ch(r, ["sum_multiples", "multiples_total", "sum_div", "total_multiples", "add_multiples", "mult_sum"])
    ref = body(f"def {f}(n):", f"    return sum(x for x in range(1, n) if x % {a} == 0 or x % {b} == 0)")
    args = [(n,) for n in [r.randint(20, 300)] + [r.randint(0, 300) for _ in range(5)] + [1]]
    qs = [f"اكتب دالة بايثون باسم `{f}(n)` تُرجع مجموع الأعداد الصحيحة الموجبة الأصغر من n التي تقبل القسمة على {a} أو على {b}.",
          f"أنشئ دالة `{f}(n)` تحسب مجموع كل عدد x حيث 1 ≤ x < n ويكون x من مضاعفات {a} أو من مضاعفات {b}، ثم تُرجع هذا المجموع.",
          f"المطلوب دالة بايثون اسمها `{f}` تستقبل عددًا صحيحًا n، وتعيد حاصل جمع الأعداد من 1 إلى n−1 التي يقسمها {a} أو {b} دون باقٍ."]
    return dict(kind=W, key=(a, b, f), topic="math", entry=f, ref=ref, args=args, qs=qs)


def g_w_count_letter(r):
    L = ch(r, AR_LETTERS)
    mode = r.randrange(3)
    f = ch(r, ["count_letter", "letter_count", "count_ar", "how_many", "count_in_text", "tally_letter", "count_hits", "letter_hits"])
    expr = [f"text.count({L!r})", f"sum(1 for w in text.split() if {L!r} in w)",
            f"sum(1 for w in text.split() if w.startswith({L!r}))"][mode]
    what = [f"عدد مرات ظهور الحرف «{L}» في النص",
            f"عدد الكلمات (المفصولة بمسافات) التي تحتوي على الحرف «{L}» مرة واحدة على الأقل",
            f"عدد الكلمات (المفصولة بمسافات) التي تبدأ بالحرف «{L}»"][mode]
    ref = body(f"def {f}(text):", f"    return {expr}")
    args = [(ar_sentence(r, 3, 8),)] + [(ar_sentence(r, 0, 8),) for _ in range(6)] + [("",)]
    qs = [f"اكتب دالة بايثون `{f}(text)` تأخذ نصًا عربيًا وتُرجع {what}.",
          f"أنشئ الدالة `{f}(text)` بحيث تُعيد {what}، حيث text سلسلة نصية بالعربية.",
          f"عرّف دالة اسمها `{f}` تستقبل سلسلة نصية عربية text، والمطلوب أن تحسب وتُرجع {what}."]
    return dict(kind=W, key=(L, mode, f), topic="arabic_text", entry=f, ref=ref, args=args, qs=qs)


def g_w_reverse_words(r):
    L = r.randint(1, 6)
    sep = ch(r, [" ", "-", "_", "/", "|", ","])
    f = ch(r, ["reverse_words", "flip_words", "rev_long_words", "backwards", "words_reversed", "invert_words", "rev_join", "mirror_words"])
    ref = body(f"def {f}(s):", f"    return {sep!r}.join([w for w in s.split() if len(w) >= {L}][::-1])")
    args = [(en_sentence(r, 3, 8),)] + [(en_sentence(r, 0, 8),) for _ in range(6)]
    qs = [f"اكتب دالة `{f}(s)` تأخذ جملة وتُرجع الكلمات التي طولها {L} أحرف أو أكثر بترتيب معكوس، مفصولًا بينها بالفاصل {sep!r}.",
          f"أنشئ دالة بايثون `{f}(s)`: قسّم الجملة s إلى كلمات (حسب المسافات)، احتفظ فقط بالكلمات التي لا يقل طولها عن {L}، اعكس ترتيبها ثم اجمعها في نص واحد بالفاصل {sep!r}.",
          f"المطلوب دالة اسمها `{f}` تستقبل نصًا s، وتعيد نصًا يحتوي الكلمات التي عدد أحرفها ≥ {L} من آخرها إلى أولها، موصولة بالفاصل {sep!r}."]
    return dict(kind=W, key=(L, sep, f), topic="strings", entry=f, ref=ref, args=args, qs=qs)


def g_w_filter_range(r):
    lo = r.randint(-20, 20)
    hi = lo + r.randint(5, 40)
    k = ch(r, [i for i in range(-5, 6) if i != 0])
    f = ch(r, ["filter_range", "pick_between", "in_range_scaled", "select_range", "clip_scale", "between_times"])
    ref = body(f"def {f}(xs):", f"    return sorted(x * {k} for x in xs if {lo} <= x <= {hi})")
    args = [(ints(r, 4, 10, lo - 10, hi + 10),)] + [(ints(r, 0, 10, -30, 60),) for _ in range(6)] + [([],)]
    qs = [f"اكتب دالة `{f}(xs)` تأخذ قائمة أعداد صحيحة وتُرجع قائمة مرتبة تصاعديًا تحتوي حاصل ضرب كل عنصر يقع بين {lo} و{hi} (شاملًا الطرفين) في {k}.",
          f"أنشئ دالة بايثون `{f}(xs)`: اختر من القائمة xs العناصر x التي تحقق {lo} ≤ x ≤ {hi}، اضرب كلًا منها في {k}، ثم أعِد النتائج مرتبة تصاعديًا.",
          f"عرّف الدالة `{f}` التي تستقبل قائمة أعداد xs وتعيد قائمة تصاعدية بقيم x × {k} لكل عنصر x في المدى المغلق [{lo}, {hi}]."]
    return dict(kind=W, key=(lo, hi, k, f), topic="lists", entry=f, ref=ref, args=args, qs=qs)


def poly_str(a, b, c):
    s = f"{a}x²"
    s += f" {'+' if b >= 0 else '-'} {abs(b)}x"
    s += f" {'+' if c >= 0 else '-'} {abs(c)}"
    return s


def g_w_poly(r):
    a = ch(r, [i for i in range(-9, 10) if i != 0])
    b, c = r.randint(-9, 9), r.randint(-9, 9)
    f = ch(r, ["poly", "evaluate", "p", "quad", "eval_poly", "value_at"])
    ref = body(f"def {f}(x):", f"    return {a} * x ** 2 + ({b}) * x + ({c})")
    args = [(x,) for x in [r.randint(1, 10)] + [r.randint(-20, 20) for _ in range(6)] + [0]]
    ps = poly_str(a, b, c)
    qs = [f"اكتب دالة بايثون `{f}(x)` تُرجع قيمة كثيرة الحدود {ps} عند العدد الصحيح x.",
          f"أنشئ دالة `{f}(x)` تحسب المقدار {ps} لقيمة x المعطاة وتُرجع الناتج عددًا صحيحًا.",
          f"عرّف الدالة `{f}` التي تستقبل عددًا صحيحًا x وتعيد قيمة الدالة التربيعية p(x) = {ps}."]
    return dict(kind=W, key=(a, b, c, f), topic="math", entry=f, ref=ref, args=args, qs=qs)


def g_w_digit_pow(r):
    p, m = r.randint(1, 5), r.randint(2, 97)
    f = ch(r, ["digit_power", "digits_mod", "sum_digit_pow", "dig_pow_mod", "digit_score", "pow_digits"])
    ref = body(f"def {f}(n):", f"    return sum(int(d) ** {p} for d in str(n)) % {m}")
    args = [(n,) for n in [r.randint(100, 99999)] + [r.randint(0, 10 ** 6) for _ in range(6)] + [0]]
    qs = [f"اكتب دالة `{f}(n)` تأخذ عددًا صحيحًا غير سالب n وتُرجع باقي قسمة (مجموع أرقام n كلٌّ مرفوع للأس {p}) على {m}.",
          f"أنشئ دالة بايثون `{f}(n)`: ارفع كل رقم من أرقام n للقوة {p}، اجمع النتائج، ثم أعِد باقي قسمة المجموع على {m}.",
          f"عرّف الدالة `{f}` التي تستقبل عددًا طبيعيًا n وتعيد (Σ رقم^{p} لكل أرقام n) mod {m}."]
    return dict(kind=W, key=(p, m, f), topic="math", entry=f, ref=ref, args=args, qs=qs)


FIZZ = [("Fizz", "Buzz"), ("Ping", "Pong"), ("Foo", "Bar"), ("Tic", "Tac"), ("Hi", "Yo"), ("Red", "Blue"), ("Sun", "Moon")]


def g_w_fizz(r):
    a = r.randint(2, 7)
    b = r.randint(a + 1, 11)
    w1, w2 = ch(r, FIZZ)
    f = ch(r, ["fizz", "game", "fizzbuzz", "label_numbers", "play"])
    ref = body(f"def {f}(n):", "    out = []", "    for i in range(1, n + 1):", "        s = ''",
               f"        if i % {a} == 0:", f"            s += {w1!r}", f"        if i % {b} == 0:", f"            s += {w2!r}",
               "        out.append(s or str(i))", "    return out")
    args = [(n,) for n in [r.randint(b, min(a * b + 2, 30))] + [r.randint(0, 25) for _ in range(4)]]
    qs = [f"اكتب دالة `{f}(n)` تُرجع قائمة نصوص للأعداد من 1 إلى n: العدد الذي يقبل القسمة على {a} يُستبدل بـ {w1!r}، وعلى {b} بـ {w2!r}، وعلى كليهما بـ {(w1 + w2)!r}، وإلا يُكتب العدد نفسه نصًا.",
          f"أنشئ دالة بايثون `{f}(n)` تبني قائمة لكل i من 1 إلى n: إن قَبِل i القسمة على {a} أضف {w1!r}، وإن قَبِلها على {b} أضف {w2!r} (فيصبح {(w1 + w2)!r} عند الاثنين)، وإن لم يتحقق أيٌّ منهما ضع str(i).",
          f"عرّف الدالة `{f}` التي تستقبل n وتعيد قائمة بطول n: مضاعفات {a} تصبح {w1!r}، مضاعفات {b} تصبح {w2!r}، المضاعفات المشتركة تصبح {(w1 + w2)!r}، وبقية الأعداد تبقى أرقامًا بصيغة نص."]
    return dict(kind=W, key=(a, b, w1, f), topic="loops", entry=f, ref=ref, args=args, qs=qs)


def g_w_caesar(r):
    k = r.randint(1, 25)
    enc = r.random() < 0.5
    K = k if enc else -k
    f = ch(r, ["caesar", "shift_text", "encode", "decode", "rotate_letters", "cipher", "shift_letters", "rot"])
    ref = body(f"def {f}(s):", "    out = []", "    for c in s:", "        if 'a' <= c <= 'z':",
               f"            out.append(chr((ord(c) - 97 + ({K})) % 26 + 97))", "        else:", "            out.append(c)",
               "    return ''.join(out)")
    def samp():
        t = en_sentence(r, 1, 5)
        if r.random() < 0.4:
            t += " " + ch(r, ["Hello", "OK", "2024", "A1", "!"])
        return t
    args = [(samp(),) for _ in range(6)] + [("",)]
    dirw = f"إزاحة كل حرف إنجليزي صغير (a-z) {k} مواقع إلى الأمام في الأبجدية (مع الالتفاف من z إلى a)" if enc else \
        f"إزاحة كل حرف إنجليزي صغير (a-z) {k} مواقع إلى الخلف في الأبجدية (مع الالتفاف من a إلى z)"
    keep = "وتبقى بقية المحارف (الأحرف الكبيرة والأرقام والمسافات والرموز) كما هي"
    qs = [f"اكتب دالة `{f}(s)` تُرجع النص الناتج عن {dirw}، {keep}.",
          f"أنشئ دالة بايثون `{f}(s)` لتشفير قيصر: المطلوب {dirw}، {keep}.",
          f"عرّف الدالة `{f}` التي تستقبل نصًا s وتعيد نصًا جديدًا بعد {dirw}، {keep}."]
    return dict(kind=W, key=(k, enc, f), topic="strings", entry=f, ref=ref, args=args, qs=qs)


def g_w_char_freq(r):
    S = "".join(sorted(r.sample("abcdeghilmnorstu", r.randint(2, 4))))
    f = ch(r, ["char_freq", "count_chars", "freq_of", "letter_stats", "char_counts", "tally"])
    ref = body(f"def {f}(s):", f"    return {{c: s.count(c) for c in {S!r}}}")
    args = [(en_sentence(r, 1, 6),) for _ in range(6)] + [("",)]
    lst = "، ".join(f"'{c}'" for c in S)
    qs = [f"اكتب دالة `{f}(s)` تُرجع قاموسًا مفاتيحه المحارف {lst} وقيمة كل مفتاح عدد مرات ظهوره في النص s (حتى لو كان صفرًا).",
          f"أنشئ دالة بايثون `{f}(s)` تعيد dict يربط كل محرف من المحارف {lst} بعدد تكراره في s، مع إدراج المحارف غير الموجودة بقيمة 0.",
          f"عرّف الدالة `{f}` التي تستقبل سلسلة s وتعيد قاموس تكرارات للمحارف {lst} فقط (كل محرف ← عدد مرات وجوده، ولو كان 0)."]
    return dict(kind=W, key=(S, f), topic="dict", entry=f, ref=ref, args=args, qs=qs)


def g_w_running(r):
    Tm = r.randint(10, 200)
    f = ch(r, ["running_sum", "capped_cumsum", "reset_sum", "cumulative", "acc_reset", "rolling_total"])
    ref = body(f"def {f}(xs):", "    out, s = [], 0", "    for x in xs:", "        s += x", f"        if s > {Tm}:",
               "            s = 0", "        out.append(s)", "    return out")
    args = [(ints(r, 5, 10, 1, Tm // 2 + 5),)] + [(ints(r, 0, 10, -10, Tm),) for _ in range(5)] + [([],)]
    qs = [f"اكتب دالة `{f}(xs)` تُرجع قائمة المجاميع التراكمية لعناصر xs، مع قاعدة: إذا أصبح المجموع الجاري أكبر من {Tm} بعد إضافة عنصر فإنه يُصفَّر قبل تسجيله.",
          f"أنشئ دالة بايثون `{f}(xs)`: ابدأ بمجموع s = 0، ولكل عنصر x أضِفه إلى s، فإن صار s > {Tm} اجعل s = 0، ثم أضف s إلى قائمة النتيجة. أعِد القائمة.",
          f"عرّف الدالة `{f}` التي تستقبل قائمة أعداد وتعيد مجاميعها التراكمية، بحيث يُعاد المجموع إلى الصفر كلما تجاوز {Tm} (القيمة المسجلة حينها 0)."]
    return dict(kind=W, key=(Tm, f), topic="lists", entry=f, ref=ref, args=args, qs=qs)


def g_w_coprime(r):
    k = r.randint(2, 90)
    f = ch(r, ["count_coprime", "coprimes", "phi_upto", "rel_prime_count", "coprime_count", "count_gcd1"])
    ref = body(f"def {f}(n):", "    from math import gcd", f"    return sum(1 for i in range(1, n + 1) if gcd(i, {k}) == 1)")
    args = [(n,) for n in [r.randint(10, 150)] + [r.randint(0, 200) for _ in range(5)] + [1]]
    qs = [f"اكتب دالة `{f}(n)` تُرجع عدد الأعداد الصحيحة i في المدى من 1 إلى n (شاملًا) التي تكون أولية نسبيًا مع {k}، أي أن القاسم المشترك الأكبر لها مع {k} يساوي 1.",
          f"أنشئ دالة بايثون `{f}(n)` تعدّ قيم i حيث 1 ≤ i ≤ n و gcd(i, {k}) = 1، ثم تُرجع العدد.",
          f"عرّف الدالة `{f}` التي تستقبل n وتعيد كم عددًا بين 1 وn لا يشترك مع العدد {k} في أي قاسم أكبر من 1."]
    return dict(kind=W, key=(k, f), topic="math", entry=f, ref=ref, args=args, qs=qs)


# ---------------------------------------------------------------- fix-the-bug generators
# return dict(kind="fix_bug", ..., buggy=<code>); the buggy code must fail >=1 hidden test
FX = "fix_bug"
FIXQ = ["الدالة التالية يُفترض أن {spec}، لكنها تحتوي على خطأ برمجي. أصلح الخطأ واكتب الدالة الصحيحة كاملة.\n```python\n{code}\n```",
        "في الشيفرة الآتية خطأ يجعل الدالة لا تعمل كما يجب. المطلوب منها أن {spec}. اكتشف الخطأ وأعد كتابة الدالة بعد إصلاحه.\n```python\n{code}\n```",
        "صحّح الدالة التالية لكي {spec}؛ فهي حاليًا تُعطي نتائج خاطئة في بعض الحالات. اكتب النسخة المصحّحة كاملة.\n```python\n{code}\n```"]


def fix_qs(spec, buggy):
    return [t.format(spec=spec, code=buggy) for t in FIXQ]


def g_f_power_sum(r):
    p, c = r.randint(1, 5), r.randint(1, 12)
    f = ch(r, ["power_sum", "weighted_sum", "sum_powers", "series", "total_pow", "pow_total", "sum_terms", "scaled_powers"])
    bug = r.randrange(3)
    rng = ["range(1, n + 1)", "range(1, n)", "range(0, n)", "range(1, n + 1)"]
    upd = ["+=", "+=", "+=", "="]
    def mk(i):
        return body(f"def {f}(n):", "    total = 0", f"    for i in {rng[i]}:", f"        total {upd[i]} {c} * i ** {p}",
                    "    return total")
    ref, buggy = mk(0), mk(bug + 1)
    spec = f"تُرجع مجموع {c} × i^{p} لكل عدد صحيح i من 1 إلى n (شاملًا n)"
    args = [(n,) for n in [r.randint(3, 8)] + [r.randint(0, 15) for _ in range(4)] + [1, 0]]
    return dict(kind=FX, key=(p, c, f), topic="loops", entry=f, ref=ref, buggy=buggy, args=args,
                qs=fix_qs(spec, buggy))


def g_f_max_div(r):
    d = r.randint(2, 12)
    v = ch(r, ["xs", "nums", "values", "arr"])
    f = ch(r, ["max_multiple", "largest_div", "best_multiple", "top_divisible", "max_div", "biggest_mult", "max_of_div", "find_max_div"])
    bug = r.randrange(3)
    def mk(i):
        ini = ["None", "0", "None", "None"][i]
        cond = [f"x % {d} == 0 and (best is None or x > best)", f"x % {d} == 0 and (best is None or x > best)",
                f"x % {d} == 0 and (best is None or x < best)", f"x % {d} != 0 and (best is None or x > best)"][i]
        return body(f"def {f}({v}):", f"    best = {ini}", f"    for x in {v}:", f"        if {cond}:", "            best = x",
                    "    return best")
    ref, buggy = mk(0), mk(bug + 1)
    spec = f"تُرجع أكبر عنصر في القائمة {v} يقبل القسمة على {d}، أو None إذا لم يوجد مثل هذا العنصر"
    args = [(ints(r, 4, 9, -30, 60),), ([-d * r.randint(1, 5) for _ in range(3)],), ([d * k + 1 for k in range(r.randint(1, 4))],),
            (ints(r, 0, 9, -40, 40),), (ints(r, 3, 9, -50, 50),), ([],)]
    return dict(kind=FX, key=(d, v, f), topic="lists", entry=f, ref=ref, buggy=buggy, args=args, qs=fix_qs(spec, buggy))


def g_f_starts_with(r):
    L = ch(r, AR_LETTERS)
    f = ch(r, ["count_start", "starts_count", "words_starting", "count_prefix", "begin_count", "count_initial", "prefix_hits", "lead_count"])
    v = ch(r, ["text", "s", "sentence", "line"])
    bug = r.randrange(3)
    def mk(i):
        loop = f"    for w in {v}:" if i == 3 else f"    for w in {v}.split():"
        cond = [f"w.startswith({L!r})", f"w.endswith({L!r})", f"{L!r} in w", f"w.startswith({L!r})"][i]
        return body(f"def {f}({v}):", "    count = 0", loop, f"        if {cond}:", "            count += 1", "    return count")
    ref, buggy = mk(0), mk(bug + 1)
    spec = f"تُرجع عدد الكلمات (المفصولة بمسافات) في النص العربي {v} التي تبدأ بالحرف «{L}»"
    starts = [w for w in AR_WORDS if w.startswith(L)] or [L + "مل"]
    def sent():
        ws = [ch(r, AR_WORDS) for _ in range(r.randint(2, 6))] + [ch(r, starts) for _ in range(r.randint(0, 2))]
        r.shuffle(ws)
        return " ".join(ws)
    args = [(sent(),) for _ in range(6)] + [("",)]
    return dict(kind=FX, key=(L, v, f), topic="arabic_text", entry=f, ref=ref, buggy=buggy, args=args, qs=fix_qs(spec, buggy))


# ---------------------------------------------------------------- complete-the-function generators
CP = "complete_function"
COMPQ = ["أكمل الدالة التالية بوضع التعبير المناسب مكان ____ بحيث {spec}، ثم اكتب الدالة كاملة.\n```python\n{code}\n```",
         "في الدالة الآتية جزء ناقص مكتوب ____. املأه لكي {spec}، واكتب الدالة بعد إكمالها.\n```python\n{code}\n```",
         "المطلوب إكمال الشيفرة التالية (استبدل ____ بالتعبير الصحيح) حتى {spec}. اكتب الحل كاملًا.\n```python\n{code}\n```"]


def comp_qs(spec, skel):
    return [t.format(spec=spec, code=skel) for t in COMPQ]


def g_c_count_div(r):
    d, e = r.sample(range(2, 10), 2)
    f = ch(r, ["count_div", "count_special", "only_div", "count_matching", "div_not", "count_sel", "tally_div", "how_many_div"])
    def mk(cond):
        return body(f"def {f}(xs):", "    count = 0", "    for x in xs:", f"        if {cond}:", "            count += 1",
                    "    return count")
    ref, skel = mk(f"x % {d} == 0 and x % {e} != 0"), mk("____")
    spec = f"تُرجع الدالة عدد عناصر xs التي تقبل القسمة على {d} ولا تقبل القسمة على {e}"
    args = [(ints(r, 6, 12, 0, 90),)] + [(ints(r, 0, 12, -40, 90),) for _ in range(5)] + [([],)]
    return dict(kind=CP, key=(d, e, f), topic="loops", entry=f, ref=ref, args=args, qs=comp_qs(spec, skel))


def g_c_rotate(r):
    k = r.randint(1, 20)
    left = r.random() < 0.5
    f = ch(r, ["rotate", "shift_list", "rotate_list", "cycle", "spin", "roll", "turn_list", "rotate_by"])
    v = ch(r, ["xs", "items", "lst", "arr"])
    def mk(ret):
        return body(f"def {f}({v}):", f"    if not {v}:", "        return []", f"    k = {k} % len({v})", f"    return {ret}")
    ref = mk(f"{v}[k:] + {v}[:k]" if left else f"{v}[len({v}) - k:] + {v}[:len({v}) - k]")
    skel = mk("____")
    spec = (f"تُرجع الدالة القائمة بعد تدويرها {k} مواقع إلى اليسار (العنصر الأول ينتقل إلى النهاية في كل خطوة)" if left else
            f"تُرجع الدالة القائمة بعد تدويرها {k} مواقع إلى اليمين (العنصر الأخير ينتقل إلى البداية في كل خطوة)")
    args = [(ints(r, 3, 8, 0, 50),)] + [(ints(r, 1, 12, -9, 50),) for _ in range(5)] + [([],)]
    return dict(kind=CP, key=(k, left, v, f), topic="lists", entry=f, ref=ref, args=args, qs=comp_qs(spec, skel))


PRIMES = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 997, 1009, 10007]


def g_c_rec_mod(r):
    s, M = r.randint(1, 4), ch(r, PRIMES)
    f = ch(r, ["multi_fact", "fact_mod", "step_product", "prod_mod", "mfact", "rec_prod"])
    def mk(ret):
        return body(f"def {f}(n):", "    if n <= 0:", "        return 1", f"    return {ret}")
    ref, skel = mk(f"(n * {f}(n - {s})) % {M}"), mk("____")
    spec = (f"تحسب الدالة تعاوديًا (بالاستدعاء الذاتي) حاصل الضرب n × (n−{s}) × (n−{2 * s}) × … للحدود الموجبة، "
            f"مع أخذ باقي القسمة على {M} في كل خطوة (وتُرجع 1 عندما n ≤ 0)")
    args = [(n,) for n in [r.randint(3, 12)] + [r.randint(0, 60) for _ in range(5)] + [0]]
    return dict(kind=CP, key=(s, M, f), topic="recursion", entry=f, ref=ref, args=args, qs=comp_qs(spec, skel))


GENS = [g_p_slice, g_p_loop, g_p_str, g_p_dict, g_p_while, g_p_comp, g_p_func, g_p_rec, g_p_sort, g_p_arabic,
        g_w_sum_mult, g_w_count_letter, g_w_reverse_words, g_w_filter_range, g_w_poly, g_w_digit_pow, g_w_fizz,
        g_w_caesar, g_w_char_freq, g_w_running, g_w_coprime,
        g_f_power_sum, g_f_max_div, g_f_starts_with,
        g_c_count_div, g_c_rotate, g_c_rec_mod]


# ---------------------------------------------------------------- normalisation / leakage (as A10)
_DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_AR_DIG = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def norm(s):
    s = _DIAC.sub("", s or "").translate(_AR_DIG)
    s = re.sub("[إأآٱ]", "ا", s).replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    s = re.sub(r"[^\w]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def code_norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def question_of(prompt):
    p = prompt or ""
    if p.startswith(HEAD):
        p = p[len(HEAD):]
    return p.rsplit(TAIL, 1)[0].strip()


EXS = ["\nمثال: `{call}` يجب أن تُرجع `{res}`.",
       "\nللتوضيح: الاستدعاء `{call}` يُرجع `{res}`.",
       "\nمثال: `{call} == {res}`"]


def call_str(entry, args):
    return f"{entry}({', '.join(repr(a) for a in args)})"


def _passes(src, entry, args, results):
    """in-process screen (build-time only, our own template code): does `src` reproduce `results`?"""
    try:
        ns = {}
        exec(src, ns)
        fn = ns[entry]
        return all(fn(*copy.deepcopy(a)) == res for a, res in zip(args, results))
    except Exception:
        return False


def materialize(gname, spec, ti):
    """Build an item (without subprocess gold) for phrasing ti; None if the instance is unusable."""
    kind = spec["kind"]
    base = {"task": "code", "src": "a11_code_gen", "lang": "python", "kind": kind, "topic": spec["topic"],
            "template": f"{gname}#{ti}"}
    if kind == "predict_output":
        q = PQ[ti].format(code=spec["code"])
        base.update(code=spec["code"])
    else:
        ns = {}
        try:
            exec(spec["ref"], ns)
            fn = ns[spec["entry"]]
            results = [fn(*copy.deepcopy(a)) for a in spec["args"]]
        except Exception:
            return None
        for res in results:  # must round-trip through repr
            try:
                if eval(repr(res)) != res:
                    return None
            except Exception:
                return None
        if kind == "fix_bug" and _passes(spec["buggy"], spec["entry"], spec["args"], results):
            return None  # bug not exposed by the tests -> reject instance
        tests = [f"assert {call_str(spec['entry'], a)} == {res!r}" for a, res in zip(spec["args"], results)]
        q = spec["qs"][ti] + EXS[ti].format(call=call_str(spec["entry"], spec["args"][0]), res=repr(results[0]))
        base.update(entry=spec["entry"], reference=spec["ref"], tests=tests, test_code="\n".join(tests),
                    answer=spec["ref"], completion=f" ```python\n{spec['ref']}\n```\n")
        if kind == "fix_bug":
            base["buggy"] = spec["buggy"]
    base["prompt"] = HEAD + q + TAIL
    base["qhash"] = hashlib.sha1(q.strip().encode("utf-8")).hexdigest()[:16]
    return base


def verify(item):
    """Subprocess gold/verification. Returns (ok, message)."""
    k = item["kind"]
    if k == "predict_output":
        rc, out, err = run_py(item["code"], timeout=BUILD_TIMEOUT)
        if rc != 0:
            return False, f"snippet rc={rc} err={err[-200:]}"
        out = out.rstrip("\n")
        if not out.strip() or len(out) > 600:
            return False, f"bad output len={len(out)}"
        item["answer"] = out
        item["completion"] = f" الناتج:\n```text\n{out}\n```\n"
        return True, ""
    rc, out, err = run_py(program_with_tests(item["reference"], item["test_code"]), timeout=BUILD_TIMEOUT)
    if not (rc == 0 and SENTINEL in out):
        return False, f"reference failed rc={rc} err={err[-200:]}"
    if k == "fix_bug":
        rc, out, err = run_py(program_with_tests(item["buggy"], item["test_code"]), timeout=BUILD_TIMEOUT)
        if rc == 0 and SENTINEL in out:
            return False, "buggy version passes the tests"
    return True, ""


def main():
    t0 = time.time()
    r = random.Random(SEED)
    train, exam = [], []
    used_keys = collections.defaultdict(set)
    seen_norm = set()
    for g in GENS:
        name = g.__name__[2:]
        for split, need, phr in (("train", N_TRAIN_PER_GEN, None), ("exam", N_EXAM_PER_GEN, 2)):
            tries = cnt = 0
            while cnt < need:
                tries += 1
                if tries > 200000:
                    raise SystemExit(f"cannot fill {split} for {name} ({cnt}/{need})")
                spec = g(r)
                key = spec["key"]
                if split == "exam" and key in used_keys[name]:
                    continue
                ti = r.randrange(2) if phr is None else phr
                it = materialize(name, spec, ti)
                if it is None:
                    continue
                n = norm(question_of(it["prompt"]))
                if n in seen_norm:
                    continue
                seen_norm.add(n)
                used_keys[name if split == "train" else name + "#exam"].add(key)
                (train if split == "train" else exam).append(it)
                cnt += 1
        print(f"[gen] {name}: train={N_TRAIN_PER_GEN} exam={N_EXAM_PER_GEN} keys_train={len(used_keys[name])}", flush=True)
    print(f"[gen] done in {time.time() - t0:.1f}s; verifying {len(train) + len(exam)} items in subprocesses "
          f"(workers={WORKERS})", flush=True)

    # ---- subprocess verification / gold outputs
    allitems = train + exam
    t1 = time.time()
    fails = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, (ok, msg) in enumerate(ex.map(verify, allitems)):
            if not ok:
                fails.append((allitems[i]["template"], msg))
            if (i + 1) % 1000 == 0:
                print(f"[verify] {i + 1}/{len(allitems)} fails={len(fails)} {time.time() - t1:.0f}s", flush=True)
    vstats = collections.Counter(x["kind"] for x in allitems)
    if fails:
        for f_ in fails[:20]:
            print("[FAIL]", f_)
        raise SystemExit(f"VERIFY FAILED: {len(fails)} items")
    r.shuffle(train)
    r.shuffle(exam)

    # ---- leakage
    def sets(items):
        ps = [x["prompt"] for x in items]
        qs = [question_of(p) for p in ps]
        return {"exact": set(ps), "norm": {norm(p) for p in ps}, "qnorm": {norm(q) for q in qs},
                "qhash": {hashlib.sha1(q.encode("utf-8")).hexdigest()[:16] for q in qs},
                "code": {code_norm(x.get("code") or x.get("reference")) for x in items}}

    tr_s, ex_s = sets(train), sets(exam)
    leak = {k: len(tr_s[k] & ex_s[k]) for k in tr_s}
    leak["param_combos"] = sum(len(used_keys[g.__name__[2:]] & used_keys[g.__name__[2:] + "#exam"]) for g in GENS)
    dup_train = len(train) - len({norm(question_of(x["prompt"])) for x in train})
    total = sum(leak.values())

    trp, exp_ = os.path.join(T, f"{OUT}_train.jsonl"), os.path.join(T, f"{OUT}_exam.jsonl")
    with open(trp, "w", encoding="utf-8", newline="\n") as f:
        for x in train:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(exp_, "w", encoding="utf-8", newline="\n") as f:
        for x in exam:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    lines = [
        f"A11 code | seed={SEED} | generators={len(GENS)} | phrasings train=0,1 exam=2 | python={sys.version.split()[0]}",
        f"train={len(train)} exam={len(exam)}",
        f"train by kind {dict(sorted(collections.Counter(x['kind'] for x in train).items()))}",
        f"exam  by kind {dict(sorted(collections.Counter(x['kind'] for x in exam).items()))}",
        f"train by topic {dict(sorted(collections.Counter(x['topic'] for x in train).items()))}",
        f"exam  by topic {dict(sorted(collections.Counter(x['topic'] for x in exam).items()))}",
        f"verified in subprocess: {dict(vstats)} | fails=0 | fix_bug buggy-versions all fail tests | "
        f"verify_time={time.time() - t1:.0f}s",
        f"leakage train_vs_exam {leak} | train_internal_dups={dup_train}",
        f"LEAKAGE={total}",
        f"{OUT}_train sha256[:16]={sha(trp)} | {OUT}_exam sha256[:16]={sha(exp_)}",
    ]
    msg = "\n".join(lines)
    print(msg)
    open(os.path.join(T, f"{OUT}_build.log"), "w", encoding="utf-8").write(msg + "\n")
    return 0 if total == 0 and dup_train == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
