#!/usr/bin/env python3
"""arabic_root.py — AHMAD-1 brick one: find_root_of_root(word, max_depth=5).

Stdlib only, no deps. Derives the trilateral (or doubled-biliteral like
hq-q from hqq) root of an Arabic word by layered affix stripping, then
verifies against a built-in root lexicon so it never invents a root.

Honest contract: returns None when no verified root is found — an unknown
word is reported unknown, never smoothed over. That no-hallucination rule
is the whole point of the triangular-root honesty core.
"""

import re
import sys
import unicodedata

# -- normalization (same fold as MUTHALLATH-1 arabic_lever.py) --
_DIACRITICS = re.compile(
    "[\u064b-\u0652\u0640\u0670\u06d6-\u06dc\u06df-\u06e4\u06e7\u06e8\u06ea-\u06ed]")
_FOLD = {"\u0622": "\u0627", "\u0623": "\u0627", "\u0625": "\u0627",
         "\u0629": "\u0647", "\u0649": "\u064a"}


def normalize(word):
    word = unicodedata.normalize("NFC", word or "")
    word = _DIACRITICS.sub("", word)
    return "".join(_FOLD.get(ch, ch) for ch in word.strip())


# -- stripping tables (ordered longest-first) --
_PREFIXES = ["\u0627\u0644", "\u0648\u0627\u0644", "\u0641\u0627\u0644",
             "\u0627\u0633\u062a", "\u0627\u0633", "\u0627\u0646",
             "\u0627\u0641\u062a", "\u0627\u062a", "\u062a\u0633\u062a",
             "\u0645\u0633\u062a", "\u0645", "\u062a", "\u064a", "\u0646",
             "\u0627", "\u0648", "\u0641", "\u0628", "\u0643", "\u0644",
             "\u0633"]
_SUFFIXES = ["\u0648\u0646", "\u064a\u0646", "\u0627\u062a", "\u0647\u0645",
             "\u0647\u0627", "\u0643\u0645", "\u0646\u0627", "\u0648\u0627",
             "\u062a\u0645", "\u062a\u0646", "\u0648\u0627", "\u064a\u0629",
             "\u064a\u0629", "\u0629", "\u0647", "\u0643", "\u064a",
             "\u0646\u0627", "\u0648", "\u0627", "\u0646", "\u062a",
             "\u0645"]

# A working lexicon of verified roots (seed: probe set + high-frequency
# Quranic/MSA triliterals + doubled biliterals; extended only with cited
# additions, never guessed). Stored as joined strings, alef-folded.
_ROOTS = set("""
كتب علم فهم رحم قلب نور حقق صبر عقل روح
فعل عمل دخل خرج اكل شرب لبس درس قرأ جمع حمد
وجد ولد مات حيا نام قام قعد عبد صلح سلم امن كفر
شهد عهد حبب صدق عفا ظلم عدل حكم شور نصح دعا جهد
غفر فقه هدا دلل رشد نصر فتح نزل قتل رزق حسب وصف
سمع بصر نظر سال سأل ضرب كسر قطع وصل حمل وضع رفع
طلب منع مدد ردد شدد عدد مرر فرر ضلل قلل
كبر صغر طول قصر قرب بعد سبق لحق ذكر نسي شكر كفر
حفظ فهم جهل عرف درس علم تعلم قرأ كتب خطب
غفر قول بيع سير عود زور
""".split())


def _letters(word):
    return [ch for ch in word if "\u0621" <= ch <= "\u06d3"]


# Explicit hollow-verb table (surface form -> verified root). Certain,
# classical, and closed: anything outside this table stays honestly unknown
# rather than guessed by blind middle-drop.
_HOLLOW = {
    "\u0642\u0627\u0644": "\u0642\u0648\u0644",  # qaal -> qwl
    "\u0628\u0627\u0639": "\u0628\u064a\u0639",  # baaE -> byE
    "\u0633\u0627\u0631": "\u0633\u064a\u0631",  # Saar -> syr
    "\u0639\u0627\u062f": "\u0639\u0648\u062f",  # Eaad -> Ewd
    "\u0632\u0627\u0631": "\u0632\u0648\u0631",  # zaar -> zwr
    "\u0646\u0627\u0645": "\u0646\u0648\u0645",  # naam -> nwm
    "\u0642\u0627\u0645": "\u0642\u0648\u0645",  # qaam -> qwm
    "\u0645\u0627\u062a": "\u0645\u0648\u062a",  # maat -> mwt
}


def _candidate_roots(core):
    """Yield (root, method) candidates from a stripped core, longest first."""
    L = _letters(core)
    if len(L) == 3:
        yield "".join(L), "triliteral"
    if len(L) == 2:
        # doubled biliteral: Hq (hq-q) -> Hqq
        yield L[0] + L[1] + L[1], "doubledBiliteral"
        yield L[0] + L[0] + L[1], "doubledBiliteralAlt"
    if len(L) == 4:
        # quadriliteral or pattern residue: try inner triple both ways
        yield "".join(L[1:]), "patternInner"
        yield "".join(L[:3]), "patternOuter"
        # hollow-verb/noun patterns (kitaab->ktb, ghafaar->ghfr): drop a
        # middle weak letter. Len-3 words never take this path, so qaal-type
        # verbs stay honestly unknown instead of mapping to a wrong root.
        _weak = {"\u0627", "\u0648", "\u064a"}
        if L[1] in _weak:
            yield L[0] + L[2] + L[3], "hollowDrop"
        if L[2] in _weak:
            yield L[0] + L[1] + L[3], "hollowDrop"
    if len(L) > 4:
        yield "".join(L[1:4]), "patternInner"
        yield "".join(L[:3]), "patternOuter"


def find_root_of_root(word, max_depth=5):
    """Return {'root','depth','method','core'} or None (unknown = unknown)."""
    w = normalize(word)
    if not w:
        return None
    if w in _HOLLOW:
        return {"root": _HOLLOW[w], "depth": 0, "method": "hollowTable",
                "core": w}
    # depth 0: the word as-is
    for depth in range(0, max(1, max_depth) + 1):
        for root, method in _candidate_roots(w):
            if root in _ROOTS:
                return {"root": root, "depth": depth, "method": method,
                        "core": w}
        if depth >= max_depth:
            break
        # strip one layer: longest matching prefix, else longest suffix
        stripped = None
        for pre in sorted(_PREFIXES, key=len, reverse=True):
            if w.startswith(pre) and len(w) - len(pre) >= 2:
                stripped = w[len(pre):]
                break
        if stripped is None:
            for suf in sorted(_SUFFIXES, key=len, reverse=True):
                if w.endswith(suf) and len(w) - len(suf) >= 2:
                    stripped = w[:len(w) - len(suf)]
                    break
        if not stripped or stripped == w:
            break
        w = stripped
    return None


if __name__ == "__main__":
    print(find_root_of_root(sys.argv[1] if len(sys.argv) > 1 else ""))
