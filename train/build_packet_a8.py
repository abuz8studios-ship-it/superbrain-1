# -*- coding: utf-8 -*-
"""build_packet_a8.py - SUPERBRAIN-1 Packet A8: Quranic Arabic Corpus training items.

Every QAC word with one gold root (homographs dropped), every bare 3MS perfect
verb with its gold form. Packet B held-out roots (packet_b_roots.txt), Packet B
words and the 85-item gate roots are BANNED and checked to zero.

Tasks use the exact A7 formats so a2_reward / build_grpo2_data / train_grpo2 work
unchanged: root (word->root), morph (root+form->verb), form (verb->form).

Usage: python build_packet_a8.py   -> train\\packet_a8.jsonl + packet_a8_build.log"""
import os, sys, json, random, sqlite3, unicodedata, collections, hashlib
ROOT = r"E:\ABU\02_PROJECTS\AHMAD-1"
T = os.path.join(ROOT, "train")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, T)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import bakeoff_arabic as B
import build_packet_b as PB          # guarded by __main__, import is safe
random.seed(1408)

ORD = {1: "الأولى", 2: "الثانية", 3: "الثالثة", 4: "الرابعة", 5: "الخامسة",
       6: "السادسة", 7: "السابعة", 8: "الثامنة", 9: "التاسعة", 10: "العاشرة"}


def main():
    banned = {l.strip() for l in open(os.path.join(T, "packet_b_roots.txt"), encoding="utf-8")
              if l.strip() and not l.startswith("#")}
    pb_words = set()
    for l in open(os.path.join(T, "packet_b.jsonl"), encoding="utf-8"):
        x = json.loads(l)
        if "word" in x: pb_words.add(x["word"])
        if x.get("section", "").startswith("morph"): pb_words.add(x["answer"])
    gate = {PB.root_key(r) for r, _, _ in B.CURATED_MORPHOLOGY}
    try:
        from arabic_root import find_root_of_root
        gate |= {PB.root_key(find_root_of_root(w)["root"]) for w in B.ROOT_WORDS}
    except Exception as e:
        print("gate root_words skipped:", e)
    ban = banned | gate

    att = PB.load_attested()
    qac_words, root_disp, qac_morph = PB.load_qac(att)
    d = sqlite3.connect(os.path.join(PB.ARRAMOOZ, "arabicdictionary.sqlite"))
    arr_roots_of = collections.defaultdict(set)
    for w, r in list(d.execute("select unvocalized, root from nouns")) + \
            list(d.execute("select unvocalized, root from verbs")):
        if w and r:
            arr_roots_of[unicodedata.normalize("NFC", w)].add(PB.root_key(r))

    items, drop = [], collections.Counter()
    for w, rks in sorted(qac_words.items()):
        if len(rks) != 1: drop["multi_root"] += 1; continue
        rk = next(iter(rks))
        if rk in ban: drop["banned_root"] += 1; continue
        if w in pb_words: drop["packet_b_word"] += 1; continue
        if arr_roots_of.get(w, set()) - {rk}: drop["homograph"] += 1; continue
        cls = PB.root_class(rk)
        items.append({"task": "root", "src": "qac", "cls": cls, "w": w,
                      "prompt": f"استخرج جذر كلمة: {w}\nالجذر:",
                      "completion": f" {root_disp.get(rk, rk)}\n"})
    seen = set()
    for rk, n, v in qac_morph:
        if rk in ban or v in pb_words or n not in ORD or (rk, n) in seen:
            continue
        seen.add((rk, n)); cls = PB.root_class(rk); r = root_disp.get(rk, rk)
        items.append({"task": "morph", "src": "qac", "cls": cls, "form": n,
                      "prompt": f"صغ الفعل الماضي من الجذر «{r}» على الصيغة {ORD[n]}.\nالفعل:",
                      "completion": f" {v}\n"})
        items.append({"task": "form", "src": "qac", "cls": cls, "form": n,
                      "prompt": f"ما صيغة الفعل «{v}»؟\nالصيغة:",
                      "completion": f" {ORD[n]}\n"})

    # leakage proof: recompute from the written items
    leak = 0
    for it in items:
        if it["task"] == "root":
            if PB.root_key(it["completion"].strip()) in ban or it["w"] in pb_words: leak += 1
    random.shuffle(items)
    out = os.path.join(T, "packet_a8.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for it in items: f.write(json.dumps(it, ensure_ascii=False) + "\n")
    dig = hashlib.sha256(open(out, "rb").read()).hexdigest()[:16]
    msg = (f"packet_a8 items={len(items)} by task={dict(collections.Counter(i['task'] for i in items))} "
           f"by cls={dict(collections.Counter(i['cls'] for i in items))} dropped={dict(drop)} "
           f"banned_roots={len(ban)} LEAKAGE={leak} sha256[:16]={dig}")
    print(msg)
    open(os.path.join(T, "packet_a8_build.log"), "w", encoding="utf-8").write(msg + "\n")
    if leak: sys.exit(3)


if __name__ == "__main__":
    main()
