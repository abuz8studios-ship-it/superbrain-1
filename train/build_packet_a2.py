# -*- coding: utf-8 -*-
"""build_packet_a2.py - SUPERBRAIN 1 Run-2 training data (Packet A v2).

Goal: ingrain Arabic morphology at dictionary scale instead of 99 roots.
Every answer comes from gold sources, never from a guess:
  lexicon      Arramooz verbs + nouns (5,550 roots), with its own inflection
               flags (feminine, dual, sound plurals, broken plurals) and its
               participle/masdar -> source-verb links
  paradigms    Qutrub conjugations of every verb (12 tenses x 14 persons)
  forms I-X    sarf_engine.infer_form (99.0% precision vs Quranic gold)
Tasks (prompt -> completion):
  root     word -> root           (verbs in every tense/person, nouns with
                                   inflections and proclitics)
  morph    root + form -> verb    and   verb -> form
  conj     verb + tense + person -> conjugated form
  derive   verb -> active/passive participle, masdar
  reason   Arabic arithmetic (keeps reasoning from regressing)
Safety:
  - Packet B roots (train\\packet_b_roots.txt) are banned everywhere.
  - v2 gate words / few-shot words are excluded from the root task, gate
    morphology roots from the morph task (same rules as Packet A v1).
  - A surface word that maps to more than one root anywhere in the lexicon
    is dropped from the root task (one right answer per item).
  - Arramooz root entries the Quranic corpus contradicts are dropped.

Runs in E:\\ABU\\venvs\\sarf.
Usage: python build_packet_a2.py [--n 600000]"""
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

# ---- exclusions (mirror build_grpo_data.py) ------------------------------
EVAL_WORDS = {B.norm_ar(w) for w in B.ROOT_WORDS} | {
    B.norm_ar(w) for w in ["مكتبات", "مدرسة", "شاهد", "محفوظ", "استخراج"]}
EVAL_ROOTS = {root_key(r) for r, _, _ in B.CURATED_MORPHOLOGY}
FEWSHOT_PAIRS = {(root_key(r), n) for r, n in
                 [("سلم", 2), ("ذكر", 5), ("نصر", 6), ("دخل", 7), ("علم", 3)]}
# خدم is the root of the step-2 probe word مستخدمين: kept out of training so
# the probe stays a generalization test, not a memorization test.
PROBE_ROOTS = {root_key("خدم")}
BAN = banned_roots() | PROBE_ROOTS

# ---- prompt templates ----------------------------------------------------
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


def item(task, prompt, answer, **meta):
    return {"task": task, "prompt": prompt, "completion": " " + answer + "\n",
            **meta}


def nfc(s):
    return unicodedata.normalize("NFC", s or "")


def radicals_in_order(word, rk):
    """Sanity: the strong radicals of the root appear in order in the word."""
    strong = [c for c in rk if c not in "ويء"]
    w = B.ar_letters(word)
    i = 0
    for c in strong:
        j = w.find(c, i)
        if j < 0:
            return False
        i = j + 1
    return True


# ---- lexicon -------------------------------------------------------------
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
    """Modern-spelling QAC words -> gold root keys (used to veto Arramooz)."""
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


# ---- noun inflection from Arramooz flags ---------------------------------
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
    """Inflections that only occur in the nominative (sound plural -ون,
    dual -ان/-تان): they never follow a preposition."""
    w = n["w"]
    stem = w[:-1] if w.endswith("ة") else w
    return {w + "ون", w + "ان", stem + "تان"}


def with_proclitics(w, noun, nominative=False):
    out = {w}
    out |= {"و" + w, "ف" + w}
    if noun and nominative:
        # sound plural/dual in -ون/-ان is nominative: ال and و/ف only,
        # never a preposition (كمنتخبون is ungrammatical; كمنتخبين is right)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600000)
    args = ap.parse_args()

    verbs, nouns = load_lexicon()
    qac = qac_word_roots()
    print(f"lexicon after Packet-B ban: verbs {len(verbs)}, nouns {len(nouns)}",
          flush=True)

    # ---- surface -> roots, across everything (ambiguity detection) -------
    surf_roots = collections.defaultdict(set)
    root_disp = {}
    verb_conj = []                 # (verb, {surface: [(tense, pronoun)]})
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
        if i % 2000 == 0:
            print(f"  conjugated {i}/{len(verbs)}", flush=True)
    noun_surf = []                 # (noun, surface)
    vetoed = 0
    for n in nouns:
        if n["w"] in qac and n["rk"] not in qac[n["w"]]:
            vetoed += 1            # Quranic gold contradicts Arramooz
            continue
        if not radicals_in_order(n["w"], n["rk"]):
            vetoed += 1
            continue
        root_disp.setdefault(n["rk"], B.ar_letters(n["root"]))
        noms = nominative_forms(n) - {n["w"]}
        for f in noun_forms(n):
            for x in with_proclitics(f, noun=True, nominative=f in noms):
                noun_surf.append((n, x))
                surf_roots[x].add(n["rk"])
    print(f"qutrub failures {fails} | nouns vetoed {vetoed} | distinct surfaces "
          f"{len(surf_roots)} | ambiguous {sum(len(r) > 1 for r in surf_roots.values())}",
          flush=True)

    def root_ok(s, rk):
        return (surf_roots[s] == {rk} and B.norm_ar(s) not in EVAL_WORDS
                and len(B.ar_letters(s)) >= 2)

    pools = collections.defaultdict(list)

    # ---- root task ------------------------------------------------------
    seen = set()
    for v, c in verb_conj:
        surfs = sorted(c)
        random.shuffle(surfs)
        for s in surfs[:14]:
            for x in random.sample(sorted(with_proclitics(s, False)), 2):
                if x not in seen and root_ok(x, v["rk"]):
                    seen.add(x)
                    pools["root"].append(item("root", random.choice(ROOT_T)(x),
                                              root_disp[v["rk"]], src="verb",
                                              cls=root_class(v["rk"])))
    for n, x in noun_surf:
        if x not in seen and root_ok(x, n["rk"]):
            seen.add(x)
            pools["root"].append(item("root", random.choice(ROOT_T)(x),
                                      root_disp[n["rk"]], src="noun",
                                      cls=root_class(n["rk"])))

    # ---- morphology: root+form -> verb, verb -> form ---------------------
    form_seen, form_items = collections.defaultdict(set), []
    for v, _ in verb_conj:
        if len(v["rk"]) != 3 or v["rk"] in EVAL_ROOTS:
            continue
        n = infer_form(skeleton(v["voc"]), v["rk"])
        if n is None or n == 9 or (v["rk"], n) in FEWSHOT_PAIRS:
            continue
        r = root_disp[v["rk"]]
        shown = skeleton(v["voc"])          # keeps shadda: كبّد is II, كبد is I
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
            continue                        # same spelling, two forms
        if n == 1 and cls == "sound" and random.random() > 0.25:
            continue
        pools["form"].append(item("form", random.choice(FORM_T)(shown),
                                  ORD[n], form=n, cls=cls))

    # ---- conjugation -----------------------------------------------------
    for v, c in verb_conj:
        pairs = [(s, tp) for s, tps in c.items() for tp in tps]
        for s, (t, p) in random.sample(pairs, min(6, len(pairs))):
            pools["conj"].append(item("conj", random.choice(CONJ_T)(
                v["plain"], t, p), s, cls=root_class(v["rk"])))

    # ---- derivation (Arramooz participle/masdar -> verb links) -----------
    answers = collections.defaultdict(set)
    for n in nouns:
        k = DERIVE.get(n["kind"])
        if (k and n.get("original") and n["rk"] not in EVAL_ROOTS
                and n.get("number") == "مفرد"
                and radicals_in_order(n["w"], n["rk"])):
            answers[(plain(n["original"]), k, n["rk"])].add(n["w"])
    for (verb, k, rk), ws in answers.items():
        pools["derive"].append(item("derive", random.choice(DERIVE_T)(verb, k),
                                    "، ".join(sorted(ws)), cls=root_class(rk)))

    # ---- reasoning (same generator as Packet A v1) ------------------------
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

    # ---- mix -------------------------------------------------------------
    have = {k: len(v) for k, v in pools.items()}
    print("available:", have, flush=True)
    # small gold pools (morph/form/derive) capped near 2x repetition
    quota = {"root": 0.66, "morph": 0.08, "form": 0.03, "conj": 0.15,
             "derive": 0.03, "reason": 0.05}
    data = []
    for k, q in quota.items():
        want = int(args.n * q)
        pool = pools[k]
        random.shuffle(pool)
        data += pool[:want] if len(pool) >= want else \
            pool * (want // len(pool)) + pool[:want % len(pool)]
    random.shuffle(data)

    # ---- leakage audit ---------------------------------------------------
    leak_b = 0
    for d in data:
        if d["task"] == "root" and root_key(d["completion"].strip()) in BAN:
            leak_b += 1
    out = os.path.join(ROOT, "train", "packet_a2.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    tc = collections.Counter(d["task"] for d in data)
    cc = collections.Counter(d.get("cls") for d in data if d["task"] == "root")
    print(f"wrote {len(data)} -> {out}")
    print("by task:", dict(tc))
    print("root items by root class:", dict(cc))
    print(f"distinct roots in root task: "
          f"{len({root_key(d['completion'].strip()) for d in data if d['task']=='root'})}")
    print(f"Packet-B root leakage: {leak_b}")
    probe = [d for d in data if "خدم" in d["prompt"] or
             root_key(d["completion"].strip()) in PROBE_ROOTS]
    print("probe root خدم items in training (must be 0):", len(probe))


if __name__ == "__main__":
    main()
