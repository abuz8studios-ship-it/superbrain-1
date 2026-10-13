# -*- coding: utf-8 -*-
"""build_packet_b.py - SUPERBRAIN 1 held-out test (Packet B). NEVER TRAIN ON IT.

Root-disjoint: a fixed set of held-out roots is chosen, and EVERY word of those
roots is banned from all training data (train\packet_b_roots.txt is the ban
list every training builder must load). So Packet B measures generalization to
roots the model has never been trained on, not memorization.

Sections (scored separately):
  root_qac   - real Quranic words, gold root from the Quranic Arabic Corpus
               (human-annotated). Uthmani spelling is converted and a word is
               kept only if the converted spelling is attested in modern
               Arabic (Arramooz word lists), so no Uthmani artifacts.
  root_msa   - modern words from the Arramooz dictionary (participles incl.
               مستفعل, masdars, verbs), gold root from Arramooz.
  morph_qac  - real Quranic verbs, 3rd masc. sing. perfect, gold form (VF)
               from the corpus: (root, form) -> verb.
  morph_msa  - modern verbs from Arramooz, sound roots, Forms II-X; the form
               is read off the vocalized spelling (shadda kept), and kept
               only when exactly one form template matches.
Each root/morph item is asked in the trained few-shot format; root items are
also asked bare (zero-shot), the way a person would actually ask.

Gold sources only (QAC + Arramooz) - nothing here is produced by our own
generator, so the test cannot inherit the generator's mistakes."""
import sys, os, re, json, random, sqlite3, unicodedata, collections, hashlib
ROOT = r"E:\ABU\02_PROJECTS\AHMAD-1"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
import bakeoff_arabic as B
random.seed(1405)

QAC = r"E:\ABU\NEOCORTEX\data\corpus\quran-morphology.txt"
ARRAMOOZ = r"E:\ABU\venvs\sarf\Lib\site-packages\arramooz\data"
N_HELD_ROOTS = 250
OUT = os.path.join(ROOT, "train")

# ---- normalization -------------------------------------------------------
HAMZAS = "اأإآءؤئ"


def root_key(r):
    """Root identity for comparison: all hamza/alef carriers -> ء, ى -> ي."""
    r = B.ar_letters(r)
    return "".join("ء" if ch in HAMZAS else ("ي" if ch == "ى" else ch)
                   for ch in r)


def root_class(k):
    if len(k) != 3:
        return "quad" if len(k) == 4 else "other"
    if k[1] == k[2]:
        return "doubled"
    if "ء" in k:
        return "hamzated"
    if k[0] in "وي":
        return "assimilated"
    if k[1] in "وي":
        return "hollow"
    if k[2] in "وي":
        return "defective"
    return "sound"


_QMARKS = re.compile("[ؐ-ًؚ-ْٕٓ-ٟ"
                     "ۖ-ۤۧ-ۭ࣓-ࣿ]")
_QMARKS_KEEP_SHADDA = re.compile("[ؐ-ًؚ-ِْٓ"
                                 "ٕ-ٟۖ-ۤۧ-ۭ"
                                 "࣓-ࣿ]")


def uthmani_candidates(s, keep_shadda=False):
    """Standard-spelling candidates for an Uthmani word. Uthmani marks are
    ambiguous in standard spelling, so each gets its possible readings and
    the caller keeps only attested spellings:
      dagger alef ٰ   -> alef or nothing    (كتاب but هذا)
      small yeh/waw     -> letter or nothing   (يحيي but به)
      medial bare hamza -> any hamza seat      (تسأل)
      medial hamza+alef -> also alef madda     (القرآن)
      alef+maddah sign  = lengthening only     (شاء)
      initial hamza+alef -> alef madda         (آمن)
      alef maqsura is word-final only; medially it is yeh."""
    import itertools
    s = s.replace("ٓ", "")                # before NFC fuses alef+maddah
    s = unicodedata.normalize("NFC", s)
    s = s.replace("ٱ", "ا").replace("آ", "ا")
    for a, b in (("ىٔ", "ئ"), ("ئ", "ئ"),
                 ("ؤ", "ؤ"), ("أ", "أ"),
                 ("ٔ", "ء")):
        s = s.replace(a, b)
    s = (_QMARKS_KEEP_SHADDA if keep_shadda else _QMARKS).sub(
        "", s.replace("ـ", ""))
    s = re.sub("^ءا", "آ", s)
    choices, first, i = [], True, 0
    while i < len(s):
        ch = s[i]
        i += 1
        if ch == "ء" and not first and s[i:i + 1] == "ا":
            choices.append(("ءا", "آ", "أا"))
            i += 1
        elif ch == "ٰ":
            choices.append(("ا", ""))
        elif ch == "ۦ":
            choices.append(("ي", ""))
        elif ch == "ۥ":
            choices.append(("و", ""))
        elif ch == "ء" and not first:
            choices.append(("ء", "أ", "ؤ", "ئ"))
        else:
            choices.append((ch,))
        if ch != "ّ":
            first = False
    out = set()
    for combo in itertools.islice(itertools.product(*choices), 256):
        w = "".join(combo)
        w = w[:-1].replace("ى", "ي") + w[-1:]
        out.add(w)
        if w.endswith("ى"):     # Uthmani final ى is often modern ي
            out.add(w[:-1] + "ي")
    return out


# ---- attested modern spellings (Arramooz) --------------------------------
def load_attested():
    att = set()
    c = sqlite3.connect(os.path.join(ARRAMOOZ, "wordfreq.sqlite"))
    att |= {w for (w,) in c.execute("select unvocalized from wordfreq")}
    d = sqlite3.connect(os.path.join(ARRAMOOZ, "arabicdictionary.sqlite"))
    att |= {w for (w,) in d.execute("select unvocalized from nouns")}
    att |= {w for (w,) in d.execute("select unvocalized from verbs")}
    return {unicodedata.normalize("NFC", w) for w in att if w}


PROCLITICS = ("وال", "فال", "بال", "كال", "لل", "ال", "و", "ف", "ب", "ل")


def attested(w, att):
    if w in att:
        return True
    for p in PROCLITICS:
        if w.startswith(p) and len(w) - len(p) >= 3 and w[len(p):] in att:
            return True
    return False


# ---- Quranic Arabic Corpus -----------------------------------------------
def load_qac(att):
    words = collections.defaultdict(set)      # std word -> {root_key}
    root_disp = {}                            # root_key -> display root
    morph = []                                # (root_key, form, verb)
    cur, segs = None, []

    def flush():
        if not segs:
            return
        rs = [r for _, _, r, _ in segs if r]
        if len(set(rs)) != 1:
            return
        rk = rs[0]
        cands = [w for w in uthmani_candidates("".join(f for f, _, _, _ in segs))
                 if attested(w, att)]
        if len(cands) == 1 and len(cands[0]) >= 3:
            words[cands[0]].add(rk)
        # morphology: a bare verb, perfect, 3ms, known form
        if len(segs) == 1 and segs[0][1] == "V":
            feats = segs[0][3]
            m = re.search(r"VF:(\d+)", feats)
            if m and "PERF" in feats and "|3MS" in feats and "PASS" not in feats:
                vc = [w for w in uthmani_candidates(segs[0][0])
                      if attested(w, att)]
                if len(vc) == 1:
                    morph.append((rk, int(m.group(1)), vc[0]))

    for ln in open(QAC, encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        if len(p) < 4 or not p[0][:1].isdigit():
            continue
        loc = p[0].rsplit(":", 1)[0]
        if loc != cur:
            flush(); segs.clear(); cur = loc
        m = re.search(r"ROOT:([^|]+)", p[3])
        rk = None
        if m:
            rk = root_key(m.group(1))
            root_disp.setdefault(rk, B.ar_letters(m.group(1)))
        segs.append((p[1], p[2], rk, p[3]))
    flush()
    return words, root_disp, morph


def load_arramooz():
    d = sqlite3.connect(os.path.join(ARRAMOOZ, "arabicdictionary.sqlite"))
    items = collections.defaultdict(set)     # root_key -> {(word, kind)}
    for w, r, t in d.execute("select unvocalized, root, wordtype from nouns"):
        if w and r:
            items[root_key(r)].add((unicodedata.normalize("NFC", w),
                                    (t or "").split(":")[-1] or "noun"))
    for w, r in d.execute("select unvocalized, root from verbs"):
        if w and r:
            items[root_key(r)].add((unicodedata.normalize("NFC", w), "فعل"))
    return items


def load_arramooz_forms():
    """Gold verbs of SOUND roots with an unambiguous form, read off the
    vocalized spelling (shadda kept): root_key -> {(form, verb)}."""
    d = sqlite3.connect(os.path.join(ARRAMOOZ, "arabicdictionary.sqlite"))
    out = collections.defaultdict(set)
    for voc, r in d.execute("select vocalized, root from verbs"):
        if not voc or not r:
            continue
        rk = root_key(r)
        if root_class(rk) != "sound":
            continue
        f, a, l = rk
        s = re.sub("[ً-ِْ]", "", unicodedata.normalize("NFC", voc))
        s = s.replace("ٱ", "ا")
        tmpl = {1: f + a + l, 2: f + a + "ّ" + l, 3: f + "ا" + a + l,
                4: "أ" + f + a + l, 5: "ت" + f + a + "ّ" + l,
                6: "ت" + f + "ا" + a + l, 7: "ان" + f + a + l,
                8: None if f in "صضطظدذزثت" else "ا" + f + "ت" + a + l,
                10: "است" + f + a + l}
        hits = [n for n, t in tmpl.items() if t and s in (t, t.replace("أ", "ا"))]
        if len(hits) == 1:
            out[rk].add((hits[0], s.replace("ّ", "") if hits[0] != 2
                         and hits[0] != 5 else s))
    return out


def bare_root_prompt(word):
    return f"ما جذر كلمة «{word}»؟\nالجذر:"


def main():
    att = load_attested()
    qac_words, root_disp, qac_morph = load_qac(att)
    arr = load_arramooz()
    by_root = collections.defaultdict(list)
    for w, rks in qac_words.items():
        if len(rks) == 1:
            by_root[next(iter(rks))].append(w)

    # v2 gate roots + few-shot roots stay out of Packet B (independent tests)
    gate = {root_key(r) for r, _, _ in B.CURATED_MORPHOLOGY}
    from arabic_root import find_root_of_root
    gate |= {root_key(find_root_of_root(w)["root"]) for w in B.ROOT_WORDS}
    gate |= {root_key(r) for r in ("كتب", "درس", "شهد", "حفظ", "خرج", "سلم",
                                   "ذكر", "نصر", "دخل", "علم")}

    # one right answer per item: drop homographs (a word Arramooz files under
    # a different root too, e.g. قل = قول or قلل)
    d = sqlite3.connect(os.path.join(ARRAMOOZ, "arabicdictionary.sqlite"))
    arr_roots_of = collections.defaultdict(set)
    for w, r in list(d.execute("select unvocalized, root from nouns")) + \
            list(d.execute("select unvocalized, root from verbs")):
        if w and r:
            arr_roots_of[unicodedata.normalize("NFC", w)].add(root_key(r))
    for rk in list(by_root):
        by_root[rk] = [w for w in by_root[rk]
                       if not (arr_roots_of.get(w, set()) - {rk})]
        if not by_root[rk]:
            del by_root[rk]

    eligible = sorted(rk for rk in by_root
                      if rk in arr and rk not in gate and len(rk) == 3)
    strata = collections.defaultdict(list)
    for rk in eligible:
        strata[root_class(rk)].append(rk)
    # proportional stratified sample, weak classes at least 20 roots each
    held = []
    for cls, rks in sorted(strata.items()):
        k = max(round(N_HELD_ROOTS * len(rks) / len(eligible)),
                min(20, len(rks)))
        held += random.sample(rks, min(k, len(rks)))
    held = sorted(set(held))
    held_set = set(held)

    items = []
    for rk in held:
        gt = root_disp.get(rk, rk)
        cls = root_class(rk)
        for w in random.sample(sorted(by_root[rk]), min(2, len(by_root[rk]))):
            items.append({"section": "root_qac", "word": w, "answer": gt,
                          "root_key": rk, "root_class": cls})
        msa = sorted(arr[rk])
        pref = [x for x in msa if x[0].startswith("مست")] + \
               [x for x in msa if x[1] in ("اسم فاعل", "اسم مفعول", "مصدر")]
        rest = [x for x in msa if x not in pref]
        pick = []
        for x in pref + random.sample(rest, len(rest)):
            if arr_roots_of.get(x[0], set()) - {rk}:
                continue                      # homograph: >1 valid root
            if x[0] not in pick and x[0] not in by_root[rk] and len(x[0]) >= 3:
                pick.append(x[0])
            if len(pick) == 2:
                break
        for w in pick:
            items.append({"section": "root_msa", "word": w, "answer": gt,
                          "root_key": rk, "root_class": cls})
    seen = set()
    for rk, n, v in qac_morph:
        if rk in held_set and n in B.FORM_NUM_TO_AR and (rk, n) not in seen:
            seen.add((rk, n))
            items.append({"section": "morph_qac", "root": root_disp.get(rk, rk),
                          "form": n, "answer": v, "root_key": rk,
                          "root_class": root_class(rk)})
    arr_forms = load_arramooz_forms()
    for rk in held:
        for n, v in sorted(arr_forms.get(rk, ())):
            if n != 1 and (rk, n) not in seen:
                seen.add((rk, n))
                items.append({"section": "morph_msa",
                              "root": root_disp.get(rk, rk), "form": n,
                              "answer": v, "root_key": rk,
                              "root_class": "sound"})
    for it in items:
        if it["section"].startswith("root"):
            it["prompt"] = B.root_prompt(it["word"])
            it["prompt_bare"] = bare_root_prompt(it["word"])
        else:
            it["prompt"] = B.morph_prompt(it["root"], it["form"])

    with open(os.path.join(OUT, "packet_b.jsonl"), "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "packet_b_roots.txt"), "w",
              encoding="utf-8") as f:
        f.write("# Packet B held-out roots (root_key form: hamza->ء, ى->ي).\n"
                "# BANNED from all training data. Never train on these.\n")
        f.write("\n".join(held) + "\n")
    digest = hashlib.sha256(open(os.path.join(OUT, "packet_b.jsonl"),
                                 "rb").read()).hexdigest()[:16]

    sec = collections.Counter(it["section"] for it in items)
    cls = collections.Counter(root_class(rk) for rk in held)
    forms = collections.Counter(it["form"] for it in items
                                if it["section"].startswith("morph"))
    print(f"attested modern spellings: {len(att)} | QAC words kept: "
          f"{len(qac_words)} | QAC roots: {len(by_root)} | eligible roots: "
          f"{len(eligible)}")
    print(f"held-out roots: {len(held)} by class {dict(cls)}")
    print(f"items: {len(items)} {dict(sec)} | morph forms {dict(sorted(forms.items()))}")
    print(f"mst- words in root_msa: "
          f"{sum(1 for it in items if it['section']=='root_msa' and it['word'].startswith('مست'))}")
    print(f"packet_b.jsonl sha256[:16] = {digest}")


if __name__ == "__main__":
    main()
