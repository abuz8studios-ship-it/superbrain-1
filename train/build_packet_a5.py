# -*- coding: utf-8 -*-
"""build_packet_a3.py - SUPERBRAIN 1 training data v3 (Packet A v3).

What v3 changes vs v2, each from a measured failure of the Run 2a pilot on the
held-out Packet B (2026-10-07 error analysis, 598 wrong root items):
  1. DEV ROOTS  300-ish roots are held out of EVERY training task and become a
                dev set (dev_root.jsonl + dev_sft.jsonl). Training stops on dev
                loss, iteration decisions use dev accuracy, Packet B only GRADES.
                (v2 had no held-out signal during training: the full Run 2a
                trained 12 h while held-out accuracy fell 9 points.)
  2. PRONOUNS   attached pronoun suffixes on nouns (كتابه، مدرستها) and verbs
                (كتبه، أتاه) were absent; 17% of pilot errors left affixes on
                and clitic words scored 59% vs 71%.
  3. BALANCE    v2 root items were 64% sound roots (already ~80% held-out) and
                5-10% per weak class (44-60%). v3 re-weights toward the weak
                classes and caps items per root (less memorisation per root).
Everything else (gold-only sources, Packet B ban, probe-root ban, ambiguity
drop, Quran-contradiction veto, eval-word exclusion) is unchanged from v2.

Runs in E:\\ABU\\venvs\\sarf.
Usage: python build_packet_a3.py [--n 120000] [--cap 45]"""
import sys, os, re, json, random, sqlite3, unicodedata, collections, argparse
ROOT = r"E:\ABU\02_PROJECTS\AHMAD-1"
sys.path[:0] = [ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "train")]
import bakeoff_arabic as B
from build_packet_b import root_key, root_class, ARRAMOOZ, QAC, uthmani_candidates
from sarf_engine import (banned_roots, conjugate, infer_form, skeleton, plain,
                         TENSES)
from libqutrub import verb_const as VC

random.seed(2026)
ORD = B.FORM_NUM_TO_AR

EVAL_WORDS = {B.norm_ar(w) for w in B.ROOT_WORDS} | {
    B.norm_ar(w) for w in ["مكتبات", "مدرسة", "شاهد", "محفوظ", "استخراج"]}
EVAL_ROOTS = {root_key(r) for r, _, _ in B.CURATED_MORPHOLOGY}
FEWSHOT_PAIRS = {(root_key(r), n) for r, n in
                 [("سلم", 2), ("ذكر", 5), ("نصر", 6), ("دخل", 7), ("علم", 3)]}
PROBE_ROOTS = {root_key("خدم")}
BAN = banned_roots() | PROBE_ROOTS

# roots held out per class (dev). Weak classes get proportionally more.
DEV_N = {"sound": 110, "hollow": 40, "defective": 35, "doubled": 25,
         "assimilated": 25, "hamzated": 35}
# share of root-task items per class (v2 was sound .64, others .05-.10)
CLASS_Q = {"sound": 0.30, "hollow": 0.17, "defective": 0.15, "doubled": 0.10,
           "assimilated": 0.10, "hamzated": 0.14, "quad": 0.04}

NOUN_PRON = ["ه", "ها", "هم", "هما", "هن", "ك", "كم", "كما", "كن", "نا", "ي"]
VERB_PRON = ["ه", "ها", "هم", "هما", "هن", "ك", "كم", "نا", "ني"]

ROOT_T = [B.root_prompt,
          lambda w: f"ما جذر كلمة «{w}»؟\nالجذر:",
          lambda w: f"استخرج جذر كلمة: {w}\nالجذر:",
          lambda w: f"كلمة: {w}\nالجذر:",
          lambda w: f"الكلمة «{w}» جذرها:"]
MORPH_T = [lambda r, n: B.morph_prompt(r, n),
           lambda r, n: f"صغ الفعل الماضي من الجذر «{r}» على الصيغة {ORD[n]}.\nالفعل:",
           lambda r, n: f"الجذر: {r} — الصيغة {ORD[n]}\nالفعل:"]
FORM_T = [lambda v: f"ما صيغة الفعل «{v}»؟\nالصيغة:",
          lambda v: f"الفعل: {v}\nالصيغة:"]
CONJ_T = [lambda v, t, p: f"صرّف الفعل «{v}» في {t} مع الضمير «{p}».\nالجواب:",
          lambda v, t, p: f"الفعل: {v} | الزمن: {t} | الضمير: {p}\nالتصريف:"]
DERIVE = {"اسم فاعل": "اسم الفاعل", "اسم مفعول": "اسم المفعول",
          "مصدر": "المصدر"}
DERIVE_T = [lambda v, k: f"ما {k} من الفعل «{v}»؟\nالجواب:",
            lambda v, k: f"{k} من «{v}»:"]
# ---- derivation traces (v5): prefixes | pronoun suffix | dictionary form | root
# Every piece is known exactly from how the surface was generated.
TRACE_T = [lambda w: f"كلمة: {w}\nالتحليل:",
           lambda w: f"حلل الكلمة «{w}» واستخرج جذرها.\nالتحليل:"]


def split_pre(pre):
    toks = []
    while pre:
        if pre.startswith("لل"):
            toks += ["ل", "ال"]; pre = pre[2:]
        elif pre.startswith("ال"):
            toks.append("ال"); pre = pre[2:]
        else:
            toks.append(pre[0]); pre = pre[1:]
    return toks


def pre_of(x, base):
    """the proclitic string that was put in front of `base` to make `x`"""
    return x[:len(x) - len(base)] if x.endswith(base) else "لل"


def trace_text(pre, suf, lem, root):
    return (f"السوابق: {'+'.join(split_pre(pre)) or '-'} | "
            f"اللواحق: {('ـ' + suf) if suf else '-'} | "
            f"الأصل: {lem} | الجذر: {root}")


# dev prompts mirror the two Packet B root modes exactly
DEV_MODES = {"fewshot": B.root_prompt,
             "bare": lambda w: f"ما جذر كلمة «{w}»؟\nالجذر:"}


def item(task, prompt, answer, **meta):
    return {"task": task, "prompt": prompt, "completion": " " + answer + "\n",
            **meta}


def nfc(s):
    return unicodedata.normalize("NFC", s or "")


def radicals_in_order(word, rk):
    strong = [c for c in rk if c not in "ويء"]
    w = B.ar_letters(word)
    i = 0
    for c in strong:
        j = w.find(c, i)
        if j < 0:
            return False
        i = j + 1
    return True


# ---- attached pronouns (conservative: skip orthographic edge cases) ---------
def attach_noun(s, p):
    if s.endswith("ة"):
        s = s[:-1] + "ت"                      # مدرسة + ها -> مدرستها
    elif s.endswith(("ى", "ا", "و")):
        return None
    elif s.endswith("ي") and p == "ي":
        return None
    elif s.endswith(("ين", "ون", "ان")) and p != "ي":
        return None
    return s + p


def attach_verb(s, p):
    if s.endswith("ن") and p in ("ني", "نا"):
        return None
    if s.endswith("ى"):
        s = s[:-1] + "ا"                      # أتى + ه -> أتاه
    elif s.endswith("وا"):
        s = s[:-1]                            # كتبوا + ه -> كتبوه
    return s + p


def load_lexicon():
    d = sqlite3.connect(os.path.join(ARRAMOOZ, "arabicdictionary.sqlite"))
    verbs = []
    for voc, r, ft, tr in d.execute(
            "select vocalized, root, future_type, transitive from verbs"):
        if voc and r and root_key(r) not in BAN and len(root_key(r)) in (3, 4):
            verbs.append({"voc": nfc(voc), "plain": plain(voc),
                          "root": B.ar_letters(r), "rk": root_key(r),
                          "ft": ft or "فتحة", "tr": tr})
    cols = [c[1] for c in d.execute("pragma table_info(nouns)")]
    nouns = []
    for row in d.execute("select * from nouns"):
        n = dict(zip(cols, row))
        if not n["root"] or not n["unvocalized"]:
            continue
        rk = root_key(n["root"])
        if rk in BAN or len(rk) not in (3, 4):
            continue
        n["rk"], n["w"] = rk, nfc(n["unvocalized"])
        n["kind"] = (n["wordtype"] or "").split(":")[-1]
        nouns.append(n)
    return verbs, nouns


def qac_word_roots():
    out = collections.defaultdict(set)
    cur, segs = None, []

    def flush():
        rs = {root_key(r) for _, r in segs if r}
        if len(rs) == 1:
            for w in uthmani_candidates("".join(f for f, _ in segs)):
                out[w] |= rs
    for ln in open(QAC, encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        if len(p) < 4 or not p[0][:1].isdigit():
            continue
        loc = p[0].rsplit(":", 1)[0]
        if loc != cur:
            if segs:
                flush()
            segs.clear(); cur = loc
        m = re.search(r"ROOT:([^|]+)", p[3])
        segs.append((p[1], m.group(1) if m else None))
    if segs:
        flush()
    return out


def noun_forms(n):
    w = n["w"]
    forms = {w}
    if n.get("mankous"):
        return forms
    stem = w[:-1] if w.endswith("ة") else w
    if n.get("feminable") and not w.endswith("ة"):
        forms.add(w + "ة")
    if n.get("masculin_plural") and not w.endswith("ة"):
        forms |= {w + "ون", w + "ين"}
    if n.get("feminin_plural"):
        forms.add(stem + "ات")
    if n.get("dualable"):
        forms |= ({stem + "تان", stem + "تين"} if w.endswith("ة")
                  else {w + "ان", w + "ين"})
    for bp in (n.get("broken_plural") or "").split(";"):
        bp = plain(bp).strip()
        if len(bp) >= 2:
            forms.add(bp)
    return forms


def nominative_forms(n):
    w = n["w"]
    stem = w[:-1] if w.endswith("ة") else w
    return {w + "ون", w + "ان", stem + "تان"}


def with_proclitics(w, noun, nominative=False):
    out = {w}
    out |= {"و" + w, "ف" + w}
    if noun and nominative:
        if not w.startswith("ال"):
            out |= {"ال" + w, "وال" + w, "فال" + w}
        return out
    if noun:
        if not w.startswith("ال"):
            al = "ال" + w
            out |= {al, "و" + al, "ب" + al, "ف" + al,
                    ("لل" + w[1:] if w.startswith("ل") else "لل" + w)}
            out |= {"ب" + w, "ل" + w, "ك" + w}
    return out


def pick_dev_roots(verbs, nouns):
    allrk = {v["rk"] for v in verbs} | {n["rk"] for n in nouns}
    by = collections.defaultdict(list)
    for rk in sorted(allrk):
        by[root_class(rk)].append(rk)
    rng = random.Random(777)
    dev = set()
    for cls, k in DEV_N.items():
        pool = sorted(by[cls])
        rng.shuffle(pool)
        dev |= set(pool[:k])
    print("roots per class (all):", {c: len(v) for c, v in sorted(by.items())})
    print("dev roots held out   :", {c: sum(1 for r in dev if root_class(r) == c)
                                     for c in DEV_N}, "total", len(dev), flush=True)
    return dev


# roots held out of SFT for GRPO (--rl-pool): the SFT model has never seen
# them, so its pass rate on them is the held-out rate and GRPO trains on the
# real failure mode (v2's GRPO filter drew from trained roots: 97% pass).
RL_N = {"sound": 150, "hollow": 110, "defective": 100, "doubled": 70,
        "assimilated": 60, "hamzated": 100}


def pick_rl_roots(verbs, nouns, dev):
    allrk = ({v["rk"] for v in verbs} | {n["rk"] for n in nouns}) - dev
    by = collections.defaultdict(list)
    for rk in sorted(allrk):
        by[root_class(rk)].append(rk)
    rng = random.Random(888)
    rl = set()
    for cls, k in RL_N.items():
        pool = sorted(by[cls])
        rng.shuffle(pool)
        rl |= set(pool[:k])
    print("RL-pool roots held out:", {c: sum(1 for r in rl if root_class(r) == c)
                                      for c in RL_N}, "total", len(rl), flush=True)
    return rl


def balanced_root(pool, want, cap):
    """class-weighted, at most `cap` items per root (topped up if a class is short)."""
    by_root = collections.defaultdict(list)
    for it in pool:
        by_root[root_key(it["completion"].strip())].append(it)
    kept, spare = [], []
    for rk, its in by_root.items():
        random.shuffle(its)
        kept += its[:cap]
        spare += its[cap:cap * 2]
    by_cls, spare_cls = collections.defaultdict(list), collections.defaultdict(list)
    for it in kept:
        by_cls[it["cls"]].append(it)
    for it in spare:
        spare_cls[it["cls"]].append(it)
    out, report = [], {}
    for cls, q in CLASS_Q.items():
        w = int(want * q)
        items = by_cls[cls]
        random.shuffle(items)
        take = items[:w]
        if len(take) < w:                         # relax the cap for this class
            sp = spare_cls[cls]
            random.shuffle(sp)
            take += sp[:w - len(take)]
        out += take
        report[cls] = (len(take), w, len({root_key(t['completion'].strip()) for t in take}))
    print("root items by class (got/wanted/distinct roots):", report, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120000)
    ap.add_argument("--cap", type=int, default=45)
    ap.add_argument("--morph-boost", type=float, default=1.0,
                    help="multiply the morph/form share of the mix (root share shrinks to keep 1.0)")
    ap.add_argument("--trace-frac", type=float, default=0.5,
                    help="share of root items that also get a derivation-trace twin")
    ap.add_argument("--out", default=os.path.join(ROOT, "train", "packet_a5.jsonl"))
    ap.add_argument("--dev-out", default=os.path.join(ROOT, "train", "dev_root_a5.jsonl"))
    ap.add_argument("--dev-sft", default=os.path.join(ROOT, "train", "dev_sft_a5.jsonl"))
    ap.add_argument("--rl-pool", action="store_true",
                    help="also hold out RL_N roots from SFT and write --rl-out")
    ap.add_argument("--rl-out", default=os.path.join(ROOT, "train", "rl_pool_a5.jsonl"))
    ap.add_argument("--rl-per-root", type=int, default=40)
    ap.add_argument("--rl-roots", default="",
                    help="override roots held out per class for RL, e.g. sound:350,hollow:250,"
                         "defective:220,doubled:160,assimilated:100,hamzated:220")
    args = ap.parse_args()
    if args.rl_roots:
        RL_N.update({k: int(v) for k, v in (kv.split(":") for kv in args.rl_roots.split(","))})

    verbs, nouns = load_lexicon()
    qac = qac_word_roots()
    DEV = pick_dev_roots(verbs, nouns)
    RL = pick_rl_roots(verbs, nouns, DEV) if args.rl_pool else set()
    HOLD = DEV | RL
    print(f"lexicon after Packet-B ban: verbs {len(verbs)}, nouns {len(nouns)}", flush=True)

    surf_roots = collections.defaultdict(set)
    pron_roots = {}                       # pronoun-attached surface -> root, "*" = conflict

    def padd(f, rk):
        cur = pron_roots.get(f)
        if cur is None:
            pron_roots[f] = rk
        elif cur != rk:
            pron_roots[f] = "*"
    root_disp = {}
    verb_conj, verb_pron = [], []
    fails = 0
    for i, v in enumerate(verbs):
        try:
            c = conjugate(v["voc"], v["ft"], v["tr"])
        except Exception:
            fails += 1
            continue
        verb_conj.append((v, c))
        root_disp.setdefault(v["rk"], v["root"])
        for s in c:
            for x in with_proclitics(s, noun=False):
                surf_roots[x].add(v["rk"])
        for s in c:                       # EVERY pronoun form, for ambiguity only
            for p in VERB_PRON:
                f = attach_verb(s, p)
                if f:
                    padd(f, v["rk"])
        for s in random.sample(sorted(c), min(5, len(c))):
            for p in random.sample(VERB_PRON, 3):
                f = attach_verb(s, p)
                if f:
                    verb_pron.append((v, f, p))
        if i % 2000 == 0:
            print(f"  conjugated {i}/{len(verbs)}", flush=True)
    noun_surf, noun_pron = [], []
    vetoed = 0
    for n in nouns:
        if n["w"] in qac and n["rk"] not in qac[n["w"]]:
            vetoed += 1
            continue
        if not radicals_in_order(n["w"], n["rk"]):
            vetoed += 1
            continue
        root_disp.setdefault(n["rk"], B.ar_letters(n["root"]))
        noms = nominative_forms(n) - {n["w"]}
        forms = sorted(noun_forms(n))
        for f in forms:
            for x in with_proclitics(f, noun=True, nominative=f in noms):
                noun_surf.append((n, x, f))
                surf_roots[x].add(n["rk"])
        for f in forms:                   # EVERY pronoun form, for ambiguity only
            for p in NOUN_PRON:
                g = attach_noun(f, p)
                if g:
                    padd(g, n["rk"])
        for f in random.sample(forms, min(2, len(forms))):
            for p in random.sample(NOUN_PRON, 3):
                g = attach_noun(f, p)
                if g:
                    for x in (g, "و" + g, "ف" + g, "ب" + g, "ل" + g):
                        noun_pron.append((n, x, g, p))
    print(f"qutrub failures {fails} | nouns vetoed {vetoed} | distinct surfaces "
          f"{len(surf_roots)} | ambiguous {sum(len(r) > 1 for r in surf_roots.values())}"
          f" | pronoun forms: verbs {len(verb_pron)} nouns {len(noun_pron)}", flush=True)

    def root_ok(s, rk, pron=False):
        if B.norm_ar(s) in EVAL_WORDS or len(B.ar_letters(s)) < 2:
            return False
        # short word+pronoun strings can be a common word the lexicon lacks
        # (اختك = أخت+ك, not ختو): only trust long pronoun-attached forms
        if pron and len(B.ar_letters(s)) < 6:
            return False
        sr = surf_roots.get(s, set())
        if not ((sr <= {rk}) if pron else (sr == {rk})):
            return False
        # a surface must not also be a pronoun-form of ANY other root
        # (checked with and without one leading proclitic): اختك = أخت+ك, not ختو
        for v in {s, s[1:] if s[:1] in "وفبل" else s}:
            pr = pron_roots.get(v)
            if pr is not None and pr != rk:
                return False
        return True

    pools = collections.defaultdict(list)
    dev_cand = collections.defaultdict(list)       # rk -> [(surface, src)]
    rl_cand = collections.defaultdict(list)

    def emit_root(src, rk, x, pre, suf, lem):
        if rk in DEV:
            dev_cand[rk].append((x, src, pre, suf, lem))
        elif rk in RL:
            rl_cand[rk].append((x, src, pre, suf, lem))
        else:
            pools["root"].append(item("root", random.choice(ROOT_T)(x),
                                      root_disp[rk], src=src, cls=root_class(rk),
                                      w=x, pre=pre, suf=suf, lem=lem))

    seen = set()
    for v, c in verb_conj:
        surfs = sorted(c)
        random.shuffle(surfs)
        for s in surfs[:14]:
            for x in random.sample(sorted(with_proclitics(s, False)), 2):
                if x not in seen and root_ok(x, v["rk"]):
                    seen.add(x)
                    emit_root("verb", v["rk"], x, pre_of(x, s), "", v["plain"])
    for v, f, p in verb_pron:
        for x in random.sample(sorted({f, "و" + f, "ف" + f}), 2):
            if x not in seen and root_ok(x, v["rk"], pron=True):
                seen.add(x)
                emit_root("verb_pron", v["rk"], x, pre_of(x, f), p, v["plain"])
    for n, x, f in noun_surf:
        if x not in seen and root_ok(x, n["rk"]):
            seen.add(x)
            emit_root("noun", n["rk"], x, pre_of(x, f), "", n["w"])
    for n, x, g, p in noun_pron:
        if x not in seen and root_ok(x, n["rk"], pron=True):
            seen.add(x)
            emit_root("noun_pron", n["rk"], x, pre_of(x, g), p, n["w"])

    # ---- morphology / form / conj / derive: dev roots excluded -------------
    form_seen, form_items = collections.defaultdict(set), []
    for v, _ in verb_conj:
        if v["rk"] in HOLD or len(v["rk"]) != 3 or v["rk"] in EVAL_ROOTS:
            continue
        n = infer_form(skeleton(v["voc"]), v["rk"])
        if n is None or n == 9 or (v["rk"], n) in FEWSHOT_PAIRS:
            continue
        r = root_disp[v["rk"]]
        shown = skeleton(v["voc"])
        ans = shown if n in (2, 5) else v["plain"]
        form_seen[shown].add(n)
        form_items.append((shown, n, root_class(v["rk"])))
        if n == 1 and root_class(v["rk"]) == "sound":
            continue
        for t in MORPH_T:
            pools["morph"].append(item("morph", t(r, n), ans, form=n,
                                       cls=root_class(v["rk"])))
    for shown, n, cls in form_items:
        if form_seen[shown] != {n}:
            continue
        if n == 1 and cls == "sound" and random.random() > 0.25:
            continue
        pools["form"].append(item("form", random.choice(FORM_T)(shown),
                                  ORD[n], form=n, cls=cls))
    for v, c in verb_conj:
        if v["rk"] in HOLD:
            continue
        pairs = [(s, tp) for s, tps in c.items() for tp in tps]
        for s, (t, p) in random.sample(pairs, min(6, len(pairs))):
            pools["conj"].append(item("conj", random.choice(CONJ_T)(
                v["plain"], t, p), s, cls=root_class(v["rk"])))
    answers = collections.defaultdict(set)
    for n in nouns:
        k = DERIVE.get(n["kind"])
        if (k and n.get("original") and n["rk"] not in EVAL_ROOTS
                and n["rk"] not in HOLD and n.get("number") == "مفرد"
                and radicals_in_order(n["w"], n["rk"])):
            answers[(plain(n["original"]), k, n["rk"])].add(n["w"])
    for (verb, k, rk), ws in answers.items():
        pools["derive"].append(item("derive", random.choice(DERIVE_T)(verb, k),
                                    "، ".join(sorted(ws)), cls=root_class(rk)))

    names = ["سعيد", "هدى", "يوسف", "مريم", "علي", "نور", "زيد", "سارة"]
    things = ["كتب", "أقلام", "تفاحات", "كرات", "دفاتر"]
    for _ in range(30000):
        nm, t = random.choice(names), random.choice(things)
        a, b = random.randint(2, 60), random.randint(2, 15)
        k = random.choice(["add", "sub", "mul"])
        if k == "add":
            q, ans = f"لدى {nm} {a} {t} واشترى {b} أخرى. كم {t} لديه؟", a + b
        elif k == "sub":
            a += b
            q, ans = f"لدى {nm} {a} {t} وأعطى {b} منها. كم بقي معه؟", a - b
        else:
            q, ans = f"لدى {nm} {b} صناديق، في كل صندوق {a} {t}. كم المجموع؟", a * b
        pools["reason"].append({"task": "reason", "prompt": B.reasoning_prompt(q),
                                "completion": f" {a} {'+-×'[['add','sub','mul'].index(k)]} {b} = {ans}\nالجواب: {ans}\n"
                                if k != "sub" else f" {a} - {b} = {ans}\nالجواب: {ans}\n"})

    have = {k: len(v) for k, v in pools.items()}
    print("available:", have, "| root by src:",
          dict(collections.Counter(i["src"] for i in pools["root"])), flush=True)
    b = args.morph_boost       # v6: verb-form items were 11% of v2-v5 data; E3 SFT lost 8 pts of forms
    quota = {"root": 0.66 - 0.11 * (b - 1), "morph": 0.08 * b, "form": 0.03 * b,
             "conj": 0.15, "derive": 0.03, "reason": 0.05}
    data = []
    for k, q in quota.items():
        want = int(args.n * q)
        pool = pools[k]
        if k == "root":
            rt = balanced_root(pool, want, args.cap)
            data += rt
            # trace twin: same word, but the answer shows the derivation
            for it in rt:
                if random.random() < args.trace_frac:
                    data.append({"task": "root_trace", "src": it["src"],
                                 "cls": it["cls"],
                                 "prompt": random.choice(TRACE_T)(it["w"]),
                                 "completion": " " + trace_text(
                                     it["pre"], it["suf"], it["lem"],
                                     it["completion"].strip()) + "\n"})
            continue
        random.shuffle(pool)
        data += pool[:want] if len(pool) >= want else \
            pool * (want // len(pool)) + pool[:want % len(pool)]
    random.shuffle(data)

    # ---- dev set (held-out roots only) ------------------------------------
    dev_rows, dev_sft = [], []
    rng = random.Random(4242)
    for rk in sorted(dev_cand):
        cands = sorted(set(dev_cand[rk]))
        pron = [c for c in cands if c[1].endswith("_pron")]
        rest = [c for c in cands if not c[1].endswith("_pron")]
        rng.shuffle(pron); rng.shuffle(rest)
        pick = pron[:2] + rest[:6 - min(2, len(pron))]
        for x, src, pre, suf, lem in pick:
            for mode, fn in DEV_MODES.items():
                row = {"word": x, "answer": root_disp[rk], "rk": rk,
                       "cls": root_class(rk), "src": src, "mode": mode,
                       "prompt": fn(x)}
                dev_rows.append(row)
                dev_sft.append({"prompt": row["prompt"],
                                "completion": " " + root_disp[rk] + "\n"})
            dev_sft.append({"prompt": TRACE_T[0](x), "completion": " " + trace_text(
                pre, suf, lem, root_disp[rk]) + "\n"})          # trace-mode dev loss
    with open(args.dev_out, "w", encoding="utf-8") as f:
        for r in dev_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.dev_sft, "w", encoding="utf-8") as f:
        for r in dev_sft:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- RL pool (held-out roots, GRPO format = same fields as v2 items) ---
    if args.rl_pool:
        rrng = random.Random(999)
        rl_rows = []
        for rk in sorted(rl_cand):
            cands = sorted(set(rl_cand[rk]))
            rrng.shuffle(cands)
            for x, src, _pre, _suf, _lem in cands[:args.rl_per_root]:
                fn = DEV_MODES[rrng.choice(["fewshot", "bare"])]
                rl_rows.append(item("root", fn(x), root_disp[rk], src=src,
                                    cls=root_class(rk), rk=rk))
        with open(args.rl_out, "w", encoding="utf-8") as f:
            for r in rl_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"RL pool: {len(rl_rows)} items over {len({r['rk'] for r in rl_rows})} "
              f"roots -> {args.rl_out} | by class "
              f"{dict(collections.Counter(r['cls'] for r in rl_rows))}", flush=True)

    # ---- leakage audits ----------------------------------------------------
    train_rk = {root_key(d["completion"].strip()) for d in data if d["task"] == "root"}
    leak_b = sum(root_key(d["completion"].strip()) in BAN for d in data if d["task"] == "root")
    leak_dev = len(train_rk & DEV)
    leak_dev_other = sum(1 for d in data if d["task"] in ("morph", "form", "conj", "derive")
                         and any(rk and rk in d["prompt"] for rk in ()))   # roots not in prompt text; checked via cls pools above
    with open(args.out, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    tc = collections.Counter(d["task"] for d in data)
    cc = collections.Counter(d.get("cls") for d in data if d["task"] == "root")
    sc = collections.Counter(d.get("src") for d in data if d["task"] == "root")
    print(f"wrote {len(data)} -> {args.out}")
    print("by task:", dict(tc))
    print("root items by class:", dict(cc), "| by src:", dict(sc))
    print(f"distinct roots in root task: {len(train_rk)}")
    print(f"dev: {len(dev_rows)} items over {len({r['rk'] for r in dev_rows})} roots -> {args.dev_out}")
    print(f"Packet-B root leakage into train: {leak_b} (must be 0)")
    print(f"dev-root leakage into train root task: {leak_dev} (must be 0)")
    print(f"RL-root leakage into train root task: {len(train_rk & RL)} (must be 0)")
    probe = [d for d in data if "خدم" in d["prompt"] or
             root_key(d["completion"].strip()) in PROBE_ROOTS]
    print("probe root خدم items in training (must be 0):", len(probe))


if __name__ == "__main__":
    main()
