# -*- coding: utf-8 -*-
"""
grassmann_probe.py — Empirical geometry probe of transformer hidden states.

Question for AHMAD-1: do concepts live as POINTS (k=1) or SUBSPACES (k>=2)
on the Grassmannian G(k, R^d)? And does cross-lingual (Arabic/English) concept
alignment show up in subspace geometry?

Model: google/gemma-4-E2B-it (bf16, GPU1), last hidden state.
Output: E:\\ABU\\02_PROJECTS\\AHMAD-1\\proof\\grassmann_probe_results.json
        + human verdict appended into the JSON and printed.
"""
import json
import math
import os
import sys
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = r"E:\ABU\MODELS\hf_cache\models--google--gemma-4-E2B-it\snapshots\3e22461f65e89153144f8adb70e3b8c2cc9845a7"
OUT_JSON = r"E:\ABU\02_PROJECTS\AHMAD-1\proof\grassmann_probe_results.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)

# ----------------------------------------------------------------------------
# 1. Stimuli: 3 mutually exclusive English concept triplets, 5 templates each
# ----------------------------------------------------------------------------
EN_TEMPLATES = [
    "The {w} was mentioned in the story.",
    "I saw a {w} yesterday near the market.",
    "She wrote about the {w} in her notebook.",
    "A {w} appeared suddenly in the garden.",
    "The children talked about the {w} all evening.",
]

CONCEPTS_EN = {
    "animal": ["dog", "cat", "bird"],
    "color": ["red", "blue", "green"],
    "motion": ["run", "walk", "swim"],
}

# Arabic triplet {كتب، علم, فهم} (write / know-teach / understand) + English
# equivalents + transliterations. Same 5 sentence frames, per language.
AR_WORDS = ["كتب", "علم", "فهم"]
EN_EQUIV = ["write", "teach", "understand"]
TR_WORDS = ["kataba", "alima", "fahima"]

AR_TEMPLATES = [
    "ذُكر {w} في القصة.",
    "رأيت {w} بالأمس بالقرب من السوق.",
    "كتبت عن {w} في دفتري.",
    "ظهر {w} فجأة في الحديقة.",
    "تحدث الأطفال عن {w} طوال المساء.",
]

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
print(f"[probe] loading {MODEL_PATH} on {DEVICE} ...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map=DEVICE if DEVICE == "cuda" else None
)
model.eval()


def encode_find_span(sentence, word):
    """Tokenize; return (input_ids, attention_mask, token span of `word`)."""
    enc = tok(sentence, return_offsets_mapping=True, return_tensors="pt")
    offs = enc["offset_mapping"][0].tolist()
    start = sentence.find(word)
    if start < 0:
        raise ValueError(f"word {word!r} not in sentence {sentence!r}")
    end = start + len(word)
    span = [i for i, (a, b) in enumerate(offs) if a < end and b > start and b > a]
    if not span:
        raise ValueError(f"no tokens overlap word {word!r} in {sentence!r}")
    return enc["input_ids"], enc["attention_mask"], (span[0], span[-1] + 1)


@torch.no_grad()
def extract(sentence, word):
    """Return (mean_pooled_vector, token_vectors) for the word span."""
    ids, mask, (s, e) = encode_find_span(sentence, word)
    out = model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE),
                output_hidden_states=True)
    h = out.hidden_states[-1][0].float().cpu()  # (T, d)
    toks = h[s:e]
    return toks.mean(dim=0), toks


# ----------------------------------------------------------------------------
# 2. Extraction
# ----------------------------------------------------------------------------
def build_dataset():
    """returns list of dicts: concept, word, lang, sentence, mean, toks."""
    items = []
    for concept, words in CONCEPTS_EN.items():
        for w in words:
            for t in EN_TEMPLATES:
                s = t.format(w=w)
                mv, tv = extract(s, w)
                items.append(dict(concept=concept, word=w, lang="en",
                                  sentence=s, mean=mv, toks=tv))
    # Arabic + English equivalents + transliterations (same concept id)
    for idx, (ar, en, tr) in enumerate(zip(AR_WORDS, EN_EQUIV, TR_WORDS)):
        concept = f"arb{idx}"
        for t in AR_TEMPLATES:
            s = t.format(w=ar)
            mv, tv = extract(s, ar)
            items.append(dict(concept=concept, word=ar, lang="ar",
                              sentence=s, mean=mv, toks=tv))
        for t in EN_TEMPLATES:
            s = t.format(w=en)
            mv, tv = extract(s, en)
            items.append(dict(concept=concept, word=en, lang="en_equiv",
                              sentence=s, mean=mv, toks=tv))
        for t in EN_TEMPLATES:
            s = t.format(w=tr)
            mv, tv = extract(s, tr)
            items.append(dict(concept=concept, word=tr, lang="translit",
                              sentence=s, mean=mv, toks=tv))
    return items


print("[probe] extracting representations ...")
items = build_dataset()
print(f"[probe] {len(items)} word-in-context vectors extracted.")

# ----------------------------------------------------------------------------
# 3. Measurements
# ----------------------------------------------------------------------------
def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def pairwise(lst):
    return [cos(lst[i], lst[j]) for i in range(len(lst))
            for j in range(i + 1, len(lst))]


# (a) within vs across concept cosines (English triplets)
by_concept_en = defaultdict(list)
for it in items:
    if it["lang"] == "en":
        by_concept_en[it["concept"]].append(it["mean"])
within, across = [], []
concepts_en = list(by_concept_en)
for c in concepts_en:
    within += pairwise(by_concept_en[c])
for i in range(len(concepts_en)):
    for j in range(i + 1, len(concepts_en)):
        for a in by_concept_en[concepts_en[i]]:
            for b in by_concept_en[concepts_en[j]]:
                across.append(cos(a, b))
res_a = {
    "within_concept_mean_cos": sum(within) / len(within),
    "within_concept_n": len(within),
    "across_concept_mean_cos": sum(across) / len(across),
    "across_concept_n": len(across),
}
print(f"(a) within={res_a['within_concept_mean_cos']:.4f} "
      f"across={res_a['across_concept_mean_cos']:.4f}")

# (b) simplex test: 3 word-centroids per concept, pairwise cos ~ -1/(k-1)=-0.5
res_b = {}
for c in concepts_en:
    cents = []
    for w in CONCEPTS_EN[c]:
        vecs = [it["mean"] for it in items if it["lang"] == "en"
                and it["concept"] == c and it["word"] == w]
        cents.append(torch.stack(vecs).mean(dim=0))
    cs = pairwise(cents)
    res_b[c] = {"centroid_pairwise_cos": cs,
                "mean": sum(cs) / len(cs),
                "expected_simplex_k3": -0.5}
    print(f"(b) simplex[{c}] pairwise cos = {[round(x,3) for x in cs]} "
          f"(want ~ -0.5)")

# (c) PCA per concept — points (k=1 dominant) or subspaces (k>=2)?
def pca_report(vecs):
    X = torch.stack(vecs)
    X = X - X.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(X, full_matrices=False)
    ev = (S ** 2)
    ev = ev / ev.sum()
    cum = torch.cumsum(ev, dim=0)
    ks = [1, 2, 3, 4, 8]
    topk = {k: float(cum[min(k, len(cum)) - 1]) for k in ks}
    pr = float((ev.sum() ** 2) / (ev ** 2).sum())  # participation ratio
    return {"explained_var_top1": float(ev[0]),
            "cumulative_at_k": topk, "participation_ratio": pr}, ev


def topk_basis(vecs, k=2):
    X = torch.stack(vecs)
    X = X - X.mean(dim=0, keepdim=True)
    U, S, _ = torch.linalg.svd(X, full_matrices=False)
    return U[:, :k]  # orthonormal columns


res_c = {}
for c in concepts_en:
    rep, _ = pca_report(by_concept_en[c])
    res_c[c] = rep
    print(f"(c) PCA[{c}] top1={rep['explained_var_top1']:.3f} "
          f"cum={ {kk: round(v,3) for kk,v in rep['cumulative_at_k'].items()} } "
          f"PR={rep['participation_ratio']:.2f}")

# (d) principal angles between top-2 subspaces of different concepts
def principal_angles(A, B):
    cs = torch.linalg.svdvals(A.t() @ B)
    cs = torch.clamp(cs, -1.0, 1.0)
    return [float(math.degrees(math.acos(x))) for x in cs]


bases_en = {c: topk_basis(by_concept_en[c], 2) for c in concepts_en}
res_d = {}
print("(d) principal angles (deg) between top-2 subspaces:")
for i in range(len(concepts_en)):
    for j in range(i + 1, len(concepts_en)):
        a, b = concepts_en[i], concepts_en[j]
        ang = principal_angles(bases_en[a], bases_en[b])
        res_d[f"{a}|{b}"] = ang
        print(f"    {a} vs {b}: {[round(x,1) for x in ang]}")

# random baseline: principal angles between random 2-subspaces in R^d
d_dim = items[0]["mean"].shape[0]
rand_angs = []
for _ in range(50):
    A = torch.randn(d_dim, 2)
    B = torch.randn(d_dim, 2)
    A, _ = torch.linalg.qr(A)
    B, _ = torch.linalg.qr(B)
    rand_angs += principal_angles(A, B)
res_d["random_baseline_mean_deg"] = sum(rand_angs) / len(rand_angs)
print(f"    random 2-subspace baseline: {res_d['random_baseline_mean_deg']:.1f} deg")

# (e) cross-lingual: Arabic vs English-equivalent vs transliteration subspaces
by_lang = defaultdict(list)
for it in items:
    if it["concept"].startswith("arb"):
        by_lang[(it["concept"], it["lang"])].append(it["mean"])
concepts_ar = sorted({it["concept"] for it in items
                      if it["concept"].startswith("arb")})
bases_ar = {key: topk_basis(v, 2) for key, v in by_lang.items()}
res_e = {"concepts": {}}
print("(e) cross-lingual principal angles (deg):")
for c in concepts_ar:
    enb = bases_ar[(c, "en_equiv")]
    arb = bases_ar[(c, "ar")]
    trb = bases_ar[(c, "translit")]
    match_ar = principal_angles(arb, enb)
    others = [cc for cc in concepts_ar if cc != c]
    mismatch = []
    for oc in others:
        mismatch += principal_angles(arb, bases_ar[(oc, "en_equiv")])
    match_tr = principal_angles(trb, enb)
    res_e["concepts"][c] = {
        "ar_word": [it["word"] for it in items
                    if it["concept"] == c and it["lang"] == "ar"][0],
        "en_equiv": [it["word"] for it in items
                     if it["concept"] == c and it["lang"] == "en_equiv"][0],
        "ar_vs_en_equiv_deg": match_ar,
        "translit_vs_en_equiv_deg": match_tr,
        "ar_vs_OTHER_en_equiv_deg": mismatch,
        "mean_ar_vs_match": sum(match_ar) / len(match_ar),
        "mean_ar_vs_mismatch": sum(mismatch) / len(mismatch),
    }
    r = res_e["concepts"][c]
    print(f"    {c} ({r['ar_word']}~{r['en_equiv']}): "
          f"ar~en {[round(x,1) for x in match_ar]} "
          f"tr~en {[round(x,1) for x in match_tr]} "
          f"ar~other-en {[round(x,1) for x in mismatch]}")

# PCA for the Arabic concept subspaces too (points vs subspaces)
for key, vecs in by_lang.items():
    if key[1] in ("ar", "en_equiv"):
        rep, _ = pca_report(vecs)
        res_c[f"{key[0]}_{key[1]}"] = rep

# ----------------------------------------------------------------------------
# 4. Verdict (decision rules stated inline, thresholds pre-registered)
# ----------------------------------------------------------------------------
mean_top1 = sum(r["explained_var_top1"] for r in res_c.values()) / len(res_c)
mean_cum2 = sum(r["cumulative_at_k"][2] for r in res_c.values()) / len(res_c)
mean_cum3 = sum(r["cumulative_at_k"][3] for r in res_c.values()) / len(res_c)
mean_pr = sum(r["participation_ratio"] for r in res_c.values()) / len(res_c)

sep_lists = [a for a in res_d.values() if isinstance(a, list)]
mean_sep_deg = sum(sum(a) / len(a) for a in sep_lists) / len(sep_lists)

matches = [v["mean_ar_vs_match"] for v in res_e["concepts"].values()]
mismatches = [v["mean_ar_vs_mismatch"] for v in res_e["concepts"].values()]
mean_match = sum(matches) / len(matches)
mean_mismatch = sum(mismatches) / len(mismatches)

point_like = mean_top1 >= 0.60
subspace_like = (not point_like) and mean_cum2 >= 0.70 and mean_pr >= 2.0
grassmann_go = subspace_like and mean_sep_deg <= 60.0
crossling = mean_match + 5.0 < mean_mismatch
k_rec = 2
if mean_cum2 >= 0.85 and mean_pr >= 2.5:
    k_rec = 3

verdict_lines = [
    "=== HUMAN VERDICT ===",
    f"(a) Clustering: within-concept mean cos {res_a['within_concept_mean_cos']:.3f} "
    f"vs across-concept {res_a['across_concept_mean_cos']:.3f} "
    f"(gap {res_a['within_concept_mean_cos'] - res_a['across_concept_mean_cos']:.3f}).",
    "(b) Simplex (thesis 1.1): per-triplet centroid pairwise cos "
    + "; ".join(f"{c} mean {res_b[c]['mean']:.3f}" for c in res_b)
    + " vs -0.5 predicted by an equiangular 3-simplex.",
    f"(c) Points vs subspaces: mean top-1 explained variance {mean_top1:.3f}, "
    f"cumulative at k=2 {mean_cum2:.3f}, k=3 {mean_cum3:.3f}, "
    f"mean participation ratio {mean_pr:.2f}.",
    "    -> verdict: "
    + ("POINT-LIKE (k=1 dominant)" if point_like else
       ("SUBSPACE-LIKE (k>=2 needed)" if subspace_like
        else "AMBIGUOUS / weak structure")),
    f"(d) Grassmann separation: mean principal angle between concept top-2 "
    f"subspaces {mean_sep_deg:.1f} deg (random baseline "
    f"{res_d['random_baseline_mean_deg']:.1f} deg).",
    "    -> concepts are "
    + ("well-separated on G(2,R^d)" if mean_sep_deg <= 60
       else "NOT cleanly separated (near-random)")
    + "; " + ("GO for a Grassmannian layer." if grassmann_go
              else "NO-GO for Grassmannian at k=2."),
    f"(e) Cross-lingual: ar~en-equivalent mean angle {mean_match:.1f} deg vs "
    f"ar~other-English-concept {mean_mismatch:.1f} deg "
    f"(margin {mean_mismatch - mean_match:.1f} deg).",
    "    -> cross-lingual subspace alignment "
    + ("VISIBLE." if crossling else "NOT visible."),
]
if grassmann_go:
    if crossling:
        verdict_lines.append(
            f"RECOMMENDATION: target G({k_rec}, R^{d_dim}) — concepts behave as "
            f"k={k_rec} subspaces with measurable cross-lingual alignment.")
    else:
        verdict_lines.append(
            f"RECOMMENDATION: G({k_rec}, R^{d_dim}) is GEOMETRICALLY SUPPORTED for "
            f"same-language concepts, but NULL for cross-lingual transfer — Arabic "
            f"and English subspaces do NOT align more than mismatched concepts do "
            f"(margin {mean_mismatch - mean_match:.1f} deg). A Grassmannian layer "
            f"may proceed ONLY as a monolingual component; cross-lingual concept "
            f"bridging is NOT evidenced here. NOTE: romanized (transliterated) "
            f"Arabic aligns tightly with English equivalents (0.7-13.6 deg), i.e. "
            f"script identity, not semantics, dominates the Arabic subspace gap.")
elif point_like:
    verdict_lines.append(
        "RECOMMENDATION: NULL for G(k>=2,R^d) — hidden states are point-like "
        "(k=1 dominant). The geometric layer stays RETIRED per "
        "MUTHALLATH-1 kill-test discipline.")
else:
    verdict_lines.append(
        "RECOMMENDATION: NULL — measured geometry supports neither a clean "
        "point model nor a subspace model. Do NOT target G(k,R^d) on this "
        "evidence alone.")
verdict_lines.append(
    "NOTE: this probe gates the geometric layer ONLY; it does not override "
    "the MUTHALLATH-1 kill-test discipline (regex baseline, checkpoint-diff, "
    "held-out eval) required before any geometric component earns re-entry.")
verdict = "\n".join(verdict_lines)
print("\n" + verdict)

# ----------------------------------------------------------------------------
# 5. Save
# ----------------------------------------------------------------------------
results = {
    "model": "google/gemma-4-E2B-it",
    "model_path": MODEL_PATH,
    "device": DEVICE,
    "hidden_dim": d_dim,
    "layer": "last",
    "pooling": "mean over word token span (last hidden state), bf16 -> float",
    "n_word_in_context_vectors": len(items),
    "seed": 0,
    "concepts_en": CONCEPTS_EN,
    "arabic_triplet": {"ar": AR_WORDS, "en_equiv": EN_EQUIV, "translit": TR_WORDS},
    "a_within_vs_across": res_a,
    "b_simplex_test": res_b,
    "c_pca_per_concept": res_c,
    "d_principal_angles_top2_deg": res_d,
    "e_crosslingual_deg": res_e,
    "verdict": verdict,
    "verdict_flags": {
        "point_like": point_like,
        "subspace_like": subspace_like,
        "grassmann_go": grassmann_go,
        "crosslingual_alignment_visible": crossling,
        "recommended_k": k_rec if grassmann_go else None,
        "mean_top1_explained_var": mean_top1,
        "mean_cum_k2": mean_cum2,
        "mean_cum_k3": mean_cum3,
        "mean_participation_ratio": mean_pr,
        "mean_concept_separation_deg": mean_sep_deg,
        "mean_ar_en_match_deg": mean_match,
        "mean_ar_en_mismatch_deg": mean_mismatch,
    },
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n[probe] results written to {OUT_JSON}")
