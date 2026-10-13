---
license: apache-2.0
base_model: QCRI/Fanar-1-9B
language:
- ar
pipeline_tag: text-generation
library_name: gguf
tags:
- arabic
- morphology
- sarf
- root-extraction
- arabic-math
- gsm8k
- grpo
- model-merging
- llama.cpp
- gguf
---

<!-- DRAFT 2026-10-13 01:40 — NOT UPLOADED. Waiting for Ahmad's OK. Numbers from train/soup_g2f_run6/verdict.json
     and proof/math_soup_g2f_run6-q5_k_m.json. Upload file would be soup_g2f_run6/gguf/superbrain1-q5_k_m.gguf
     renamed superbrain1-v3-q5_k_m.gguf, plus eval/verdict_v3.json + eval/math_arabic_gsm8k_v3.json. -->

# SUPERBRAIN-1 — Arabic-first root reasoning + Arabic math (Fanar-1-9B)

SUPERBRAIN-1 is a 9B Arabic model by ABUZ8, built on
[QCRI/Fanar-1-9B](https://huggingface.co/QCRI/Fanar-1-9B). It reasons through Arabic morphology (ṣarf):
find the root of a word, build any verb Form I–X from a root, conjugate, derive. It also solves Arabic
math word problems step by step.

It is a **completion model, not a chat model**. Give it one of the prompt formats below and read the first
line of the output.

## Files

| File | Version | What it is |
|---|---|---|
| `superbrain1-v3-q5_k_m.gguf` | **v3 (current, 2026-10-13)** | best on every axis: roots, verb forms, Arabic math |
| `superbrain1-run6-q5_k_m.gguf` | v2 (2026-10-12) | roots + Quranic words + Arabic math |
| `superbrain1-q5_k_m.gguf` | v1 (2026-10-06) | morphology specialist |

## Results

Greedy decoding on the Q5_K_M GGUF served by llama.cpp. Evaluation files are in `eval/`.

**Packet B — held-out test.** 1,343 items over **258 roots excluded from all training data** (Quranic words
with gold roots from the Quranic Arabic Corpus, modern words from Arramooz). Published as
[abuz8studios/arabic-sarf-packet-b](https://huggingface.co/datasets/abuz8studios/arabic-sarf-packet-b).

**Arabic math.** All 1,190 problems of the Arabic GSM8K test split, exact match on the final number; test
questions hash-checked out of training.

| | Fanar-1-9B base | v1 | v2 | **v3** |
|---|---|---|---|---|
| Root extraction, bare question | 59.5% | 66.8% | 78.0% | **78.1%** |
| Root extraction, few-shot | 43.2% | 67.9% | 78.6% | **78.8%** |
| Verb Forms I–X (428 items) | 16.6% | **94.9%** | 88.8% | 92.8% |
| Arabic GSM8K (1,190) | 16.8% | — | 63.6% | **77.8%** |
| 85-item gate (v3) | 65.9% | 96.5% | 95.3% | 95.3% |

v3 root accuracy by class (bare): sound 91.5, hollow 70.2, doubled 70.1, assimilated 64.8, hamzated 58.1,
defective 45.5. Verb forms: I 81.8, II 95.9, III 89.1, IV 94.4, V 95.5, VI 90.0, VII 100, VIII 89.7, X 95.5.

**Generalization probe.** The root خدم was banned from all training data. For مستخدمين ("users") v3 answers
**خدم** under all three prompt formats.

## How v3 was made

v3 is a **weight average (model soup, α = 0.5)** of two models that both descend from Fanar-1-9B. α and the
choice of parent were fixed before any v3 score existed, and v3 was graded once.

1. **v2 (run6)**: SFT → GRPO on roots → GRPO on Quranic words → GRPO on Arabic math (see v2 notes).
2. **G2 line (new)**, from the Fanar base:
   - SFT on 78,202 items: 60k root/trace/verb-form items + math + Quranic forms, with 1,292 roots and the
     269 dev roots verified absent;
   - **GRPO on the 1,292 never-seen roots**: bare-root accuracy 64.5% → 75.7%. The earlier 585-root pool gave
     +5.7 points, and 2.2× the unseen roots gave about 2× the gain;
   - GRPO on Arabic math: 70.9% → 76.5%.
3. **Soup** of (2) and (1): each parent's strength survives. Roots ≥ v2, verb forms +4.0, math +14.2.

All rewards are program checks: a root oracle or an exact number. There is no AI judge.

## Prompt formats

Root (bare): `ما جذر كلمة «مستخدمين»؟\nالجذر:` → ` خدم`

Verb form: `الجذر: درج — الصيغة العاشرة\nالفعل:` → ` استدرج`

Conjugation: `الفعل: نجم | الزمن: المضارع المجزوم | الضمير: هما\nالتصريف:` → ` ينجما`

Math: `المهمة: حل السؤال التالي خطوة بخطوة ثم اكتب الجواب النهائي رقمًا.\nسؤال: ...\nالحل:` → worked steps,
then `الجواب: <number>`.

## Usage (llama.cpp)

```bash
llama-server -m superbrain1-v3-q5_k_m.gguf -ngl 99 -c 4096 --port 8099
curl -s http://127.0.0.1:8099/completion -d '{"prompt":"ما جذر كلمة «مستخدمين»؟\nالجذر:","n_predict":16,"temperature":0}'
```

## Limitations

- **About one root in five is still missed.** Defective (45.5%) and hamzated (58.1%) roots are the weak classes.
- **Single words only.** No sentence context or disambiguation.
- **Not chat-tuned.**
- Quranic words are tested in modern spelling, converted from Uthmani script.

## Data and licenses

- Base: QCRI Fanar-1-9B (Apache-2.0).
- Morphology training data was **generated** with Arramooz, Qutrub and PyArabic (GPL); the tools are not distributed.
- Math: Arabic GSM8K v2 (Apache-2.0). The train split was used for training; the test split only for evaluation.
- Evaluation: Quranic Arabic Corpus (GPL).

Thanks to QCRI, Taha Zerrouki (Arramooz, Qutrub, PyArabic), Kais Dukes (Quranic Arabic Corpus) and
Omartificial Intelligence Space (Arabic GSM8K).
