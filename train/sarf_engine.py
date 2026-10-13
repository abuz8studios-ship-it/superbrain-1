# -*- coding: utf-8 -*-
"""sarf_engine.py - SUPERBRAIN 1 morphology engine (Packet A v2 source).

Gold lexicon : Arramooz (13,942 vocalized verbs, 30,215 nouns, 5,550 roots).
Paradigms    : Qutrub (libqutrub) conjugates every verb, weak roots included
               (hollow, defective, assimilated, doubled, hamzated).
Forms I-X    : infer_form() reads the derived form off (verb, root) with
               weak-aware templates.
Gold check   : validate() measures all three against the Quranic Arabic
               Corpus (human-annotated) BEFORE any training item is generated.

Runs in E:\\ABU\\venvs\\sarf (libqutrub, pyarabic). Packet B roots
(train\\packet_b_roots.txt) are banned everywhere.

Usage: E:\\ABU\\venvs\\sarf\\Scripts\\python.exe sarf_engine.py validate"""
import sys, os, re, sqlite3, unicodedata, collections, json
ROOT = r"E:\ABU\02_PROJECTS\AHMAD-1"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "train"))
from build_packet_b import (root_key, root_class, uthmani_candidates,
                            load_attested, QAC, ARRAMOOZ)
import pyarabic.araby as A
from libqutrub import classverb, verb_const as VC

SHADDA = "\u0651"
_HARAKAT = re.compile("[\u064b-\u0650\u0652\u0670]")


def banned_roots():
    p = os.path.join(ROOT, "train", "packet_b_roots.txt")
    return {l.strip() for l in open(p, encoding="utf-8")
            if l.strip() and not l.startswith("#")}


def skeleton(vocalized):
    """Unvocalized spelling that keeps shadda (it is a letter for form II/V)."""
    s = unicodedata.normalize("NFC", vocalized).replace("\u0671", "ا")
    return _HARAKAT.sub("", s)


def plain(word):
    """How the word is actually written: no harakat, no shadda."""
    return A.strip_tashkeel(unicodedata.normalize("NFC", word))


# ---- Forms I-X from (verb, root) -----------------------------------------
_HZ = "[اأإآءؤئى]"


def infer_form(skel, rk):
    """Return the form number (1-10) of a 3ms-perfect verb, or None when no
    template or more than one template fits equally well."""
    if len(rk) != 3:
        return None
    f, a, l = rk

    def rx(x, opt=True):
        if x in "وي":
            return "([وياى]?)" if opt else "([وياى])"
        if x == "ء":
            return "(" + _HZ + ")"
        return "(" + re.escape(x) + ")"

    F, Aa, L = rx(f), rx(a), rx(l)
    AL = Aa + L
    if a == l:                                   # doubled: مدّ, استمرّ
        AL = "(?:" + Aa + L + "|(" + re.escape(a) + ")" + SHADDA + ")"
    t = {
        1: F + AL,
        2: F + Aa + SHADDA + L,
        3: F + "ا" + AL,
        4: "[اأ]" + F + AL,
        5: "ت" + F + Aa + SHADDA + L,
        6: "ت" + F + "ا" + AL,
        7: "ان" + F + AL,
        8: "ا(?:" + F + "[تطد]|[تطدذزصضظ]" + SHADDA + "?)" + SHADDA + "?" + AL,
        9: "ا" + F + Aa + L + SHADDA,
        10: "است" + F + AL,
    }
    if f == "ء":
        t[4] = "(?:[اأ]" + F + AL + "|آ" + AL + ")"
    best, best_drops = [], 99
    for n, pat in t.items():
        m = re.fullmatch(pat, skel)
        if not m:
            continue
        drops = sum(1 for g in m.groups() if g == "")
        if drops < best_drops:
            best, best_drops = [n], drops
        elif drops == best_drops:
            best.append(n)
    return best[0] if len(best) == 1 else None


# ---- paradigms (Qutrub) --------------------------------------------------
TENSES = [VC.TensePast, VC.TenseFuture, VC.TenseJussiveFuture,
          VC.TenseSubjunctiveFuture, VC.TenseConfirmedFuture,
          VC.TenseImperative, VC.TenseConfirmedImperative,
          VC.TensePassivePast, VC.TensePassiveFuture,
          VC.TensePassiveJussiveFuture, VC.TensePassiveSubjunctiveFuture,
          VC.TensePassiveConfirmedFuture]


def conjugate(vocalized, future_type, transitive):
    """{plain surface: [(tense, pronoun), ...]} for one verb."""
    out = collections.defaultdict(list)
    v = classverb.VerbClass(vocalized, bool(transitive), future_type)
    v.conjugate_all_tenses()
    for t in TENSES:
        for p in VC.PronounsTable:
            try:
                w = v.get_conj(t, p)
            except Exception:
                continue
            if w:
                out[plain(w)].append((t, p))
    return out


# ---- lexicon -------------------------------------------------------------
def load_verbs(ban=frozenset()):
    d = sqlite3.connect(os.path.join(ARRAMOOZ, "arabicdictionary.sqlite"))
    rows = []
    for voc, unv, r, ft, tri, tr in d.execute(
            "select vocalized, unvocalized, root, future_type, triliteral, "
            "transitive from verbs"):
        if voc and r and root_key(r) not in ban:
            rows.append({"voc": voc, "unv": unv, "root": B_letters(r),
                         "rk": root_key(r), "future_type": ft or "فتحة",
                         "transitive": tr})
    return rows


def B_letters(r):
    import bakeoff_arabic as B
    return B.ar_letters(r)


# ---- gold check against the Quranic Arabic Corpus --------------------------
def qac_tokens(att):
    """Single-segment QAC verbs: (std surface candidates, root_key, VF, feats)."""
    toks, cur, segs = [], None, []

    def flush():
        if len(segs) == 1 and segs[0][1] == "V":
            form, tag, feats = segs[0]
            m, r = re.search(r"VF:(\d+)", feats), re.search(r"ROOT:([^|]+)", feats)
            if m and r:
                toks.append((uthmani_candidates(form), root_key(r.group(1)),
                             int(m.group(1)), feats,
                             uthmani_candidates(form, keep_shadda=True)))

    for ln in open(QAC, encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        if len(p) < 4 or not p[0][:1].isdigit():
            continue
        loc = p[0].rsplit(":", 1)[0]
        if loc != cur:
            flush(); segs.clear(); cur = loc
        segs.append((p[1], p[2], p[3]))
    flush()
    return toks


def validate():
    att = load_attested()
    toks = qac_tokens(att)
    verbs = load_verbs()
    by_root = collections.defaultdict(list)
    for v in verbs:
        by_root[v["rk"]].append(v)
    report = {}

    # (1) form inference on QAC perfect 3ms verbs (gold VF)
    ok = wrong = none = 0
    errs = collections.Counter()
    for cands, rk, vf, feats, cands_sh in toks:
        if "PERF" not in feats or "|3MS" not in feats or "PASS" in feats:
            continue
        cs = ([c for c in cands_sh if c.replace(SHADDA, "") in att]
              or sorted(cands_sh))
        got = infer_form(cs[0], rk)
        if got is None:
            none += 1
        elif got == vf:
            ok += 1
        else:
            wrong += 1; errs[(vf, got, root_class(rk))] += 1
    report["form_inference_vs_QAC"] = {
        "correct": ok, "wrong": wrong, "undecided": none,
        "precision": round(ok / max(1, ok + wrong), 4),
        "coverage": round((ok + wrong) / max(1, ok + wrong + none), 4),
        "top_errors(gold,got,class)": [[*k, n] for k, n in errs.most_common(8)]}

    # (2) form inference on Arramooz verbs (vocalized skeleton keeps shadda)
    dist = collections.Counter(infer_form(skeleton(v["voc"]), v["rk"])
                               for v in verbs)
    report["arramooz_forms"] = {str(k): n for k, n in sorted(
        dist.items(), key=lambda x: (x[0] is None, x[0] or 0))}

    # (3) paradigm recall: does Qutrub (over Arramooz verbs of the same root)
    #     produce the real QAC verb token?
    roots_needed = {t[1] for t in toks}
    gen = collections.defaultdict(set)
    fail = 0
    for rk in roots_needed:
        for v in by_root.get(rk, []):
            try:
                gen[rk] |= set(conjugate(v["voc"], v["future_type"],
                                         v["transitive"]))
            except Exception:
                fail += 1
    distinct = {}
    for cands, rk, vf, feats, _ in toks:
        distinct[(frozenset(plain(c) for c in cands), rk)] = (vf, feats)
    hit = miss = noroot = 0
    by_cls = collections.defaultdict(lambda: [0, 0])
    misses = []
    for (cands, rk), (vf, feats) in distinct.items():
        if rk not in gen:
            noroot += 1
            continue
        h = bool(cands & gen[rk])
        hit += h; miss += not h
        by_cls[root_class(rk)][0] += h; by_cls[root_class(rk)][1] += 1
        if not h and len(misses) < 25:
            misses.append([sorted(cands)[0], rk, vf])
    report["paradigm_recall_vs_QAC"] = {
        "distinct_verb_tokens": len(distinct), "root_not_in_arramooz": noroot,
        "hit": hit, "miss": miss, "recall": round(hit / max(1, hit + miss), 4),
        "by_class": {k: f"{a}/{b} = {a / b:.1%}" for k, (a, b) in
                     sorted(by_cls.items())},
        "qutrub_failures": fail, "sample_misses": misses}

    # (4) root agreement QAC vs Arramooz on shared words
    d = sqlite3.connect(os.path.join(ARRAMOOZ, "arabicdictionary.sqlite"))
    arr_root = collections.defaultdict(set)
    for w, r in list(d.execute("select unvocalized, root from nouns")) + \
            list(d.execute("select unvocalized, root from verbs")):
        if w and r:
            arr_root[unicodedata.normalize("NFC", w)].add(root_key(r))
    agree = dis = 0
    dis_ex = []
    dis_kind = collections.Counter()
    for cands, rk, vf, feats, _ in toks:
        for c in cands:
            if c in arr_root:
                if rk in arr_root[c]:
                    agree += 1
                else:
                    dis += 1
                    alts = arr_root[c]
                    fold = lambda x: x.replace("ي", "و")
                    dis_kind["weak_letter_convention" if any(
                        fold(a) == fold(rk) for a in alts) else "homograph_or_other"] += 1
                    if len(dis_ex) < 12:
                        dis_ex.append([c, rk, sorted(arr_root[c])])
                break
    report["root_agreement_QAC_vs_Arramooz"] = {
        "agree": agree, "disagree": dis,
        "rate": round(agree / max(1, agree + dis), 4),
        "disagree_kinds": dict(dis_kind), "examples": dis_ex}

    out = os.path.join(ROOT, "proof", "sarf_engine_validation.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print("->", out)


if __name__ == "__main__":
    if sys.argv[1:] == ["validate"]:
        validate()
