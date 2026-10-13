#!/usr/bin/env python3
"""probe_arabic_root.py — AHMAD-1 brick-one gate.
10 words -> exact trilateral roots. Exit 0 = PASS, 1 = FAIL.
"""
import sys

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))

from arabic_root import find_root_of_root

PROBE = [
    ("كتب", "كتب"), ("علم", "علم"), ("فهم", "فهم"), ("رحم", "رحم"),
    ("قلب", "قلب"), ("نور", "نور"), ("حق", "حقق"), ("صبر", "صبر"),
    ("عقل", "عقل"), ("روح", "روح"),
]

fails = []
for word, want in PROBE:
    got = find_root_of_root(word)
    ok = got is not None and got["root"] == want
    print(("PASS " if ok else "FAIL ") + word + " -> " +
          (got["root"] if got else "None") + " (want " + want + ")")
    if not ok:
        fails.append(word)

# honesty check: garbage in -> None out, never a guessed root
for junk in ("xyz", "123", ""):
    r = find_root_of_root(junk)
    ok = r is None
    print(("PASS " if ok else "FAIL ") + "junk(%r) -> %r" % (junk, r))
    if not ok:
        fails.append("junk:" + junk)

print("PROBE %s (%d fails)" % ("GREEN" if not fails else "RED", len(fails)))
sys.exit(1 if fails else 0)
