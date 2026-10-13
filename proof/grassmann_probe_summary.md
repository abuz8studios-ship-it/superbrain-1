# Grassmann Probe Summary — AHMAD-1 geometric-layer go/no-go

Date: 2026-09-19 · Script: `src/grassmann_probe.py` · Raw data: `proof/grassmann_probe_results.json`
Model: google/gemma-4-E2B-it (bf16, GPU1), last hidden state, d=1536.
Stimuli: 3 English concept triplets ({dog,cat,bird}, {red,blue,green}, {run,walk,swim}),
5 templates each (45 sentences); Arabic triplet {كتب, علم, فهم} with 5 Arabic-script,
5 English-equivalent, 5 transliteration contexts. 90 word-in-context mean-pooled vectors.

## Measured numbers

| Question | Result | Reading |
|---|---|---|
| (a) within vs across concept cosine | 0.730 vs 0.704 (gap 0.026) | Raw vectors cluster weakly; context-template direction dominates point geometry |
| (b) simplex test (thesis 1.1: pairwise cos ≈ −0.5) | animal 0.952, color 0.926, motion 0.933 | **NULL** — centroid geometry is a tight cone (~0.93–0.95), nowhere near an equiangular simplex. Founder thesis 1.1 (point simplex) is rejected at 2B-it scale |
| (c) points vs subspaces | mean top-1 EV 0.545; cum@k=2 0.763, k=3 0.879, k=8 0.972; participation ratio 2.73 | **SUBSPACE-LIKE**: no single dominant direction; 2–3 dims needed. Not point-like (rule: point-like needs top-1 ≥ 0.60) |
| (d) principal angles between concept top-2 subspaces | animal–color 9.2°/24.2°, animal–motion 11.4°/35.1°, color–motion 13.5°/33.4° (mean 21.1°); random baseline 88.1° | Concepts are **strongly separated on G(2,R¹⁵³⁶)** — far from random, far from identical |
| (e) cross-lingual alignment | ar~en-equivalent 48.9° vs ar~other-English 47.3° (margin −1.7°) | **NULL** — Arabic subspaces do NOT align with their English equivalents more than with mismatched concepts. However, transliterated Arabic (kataba/alima/fahima) aligns at 0.7–13.6° with English equivalents: script identity, not semantics, dominates the Arabic gap |

## Answers to the three gating questions

1. **Points or subspaces?** Subspaces. Top-1 direction explains only ~54% of within-concept
   variance; the participation ratio (~2.7) says a concept is a 2–3 dimensional object,
   not a point. The simplex-point alternative (thesis 1.1) is independently rejected:
   word centroids sit at cosine ~0.93, not −0.5.
2. **Is cross-lingual concept alignment visible in subspace geometry?** No. Arabic and
   English top-2 subspaces of the same concept are no closer (48.9°) than mismatched
   concept pairs (47.3°). The only alignment signal is script-level: romanized Arabic
   matches English almost exactly.
3. **Target G(k,R^d), and at what k?** G(2,R¹⁵³⁶) is geometrically supported **for
   same-language concepts** (k=3 if headroom is wanted — cum@3 = 0.879). A Grassmannian
   layer may proceed only as a **monolingual** component; cross-lingual bridging is
   not evidenced by subspace geometry.

## Verdict

**CONDITIONAL GO** — the simplex/point layer stays dead (thesis 1.1 NULL again, now at
2B-it), but a subspace layer on G(2,R^d) has real measured structure behind it
(21° mean separation vs 88° random; k=2 captures 76% of concept variance). The
cross-lingual motivation for the geometric layer is NULL and must not be claimed.

This probe gates geometry only. Per the PRD and MUTHALLATH-1 kill-test discipline, no
geometric component earns re-entry without a pre-registered kill test (regex baseline,
checkpoint-diff, held-out eval) — this measurement says "the substrate exists," not
"the layer will help."
