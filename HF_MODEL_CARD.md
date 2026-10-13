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
- llama.cpp
- gguf
---

<!-- DRAFT 2026-10-06. Numbers below are the Run 2a PILOT (150k items).
     Replace with the final release run's verdict.json before upload.
     Previous card archived at F:\archive\AHMAD-1\HF_MODEL_CARD_pre-2026-10-06.md -->

# SUPERBRAIN-1 — Arabic morphology (ṣarf) on Fanar-1-9B

SUPERBRAIN-1 is a 9B Arabic model by ABUZ8, fine-tuned from
[QCRI/Fanar-1-9B](https://huggingface.co/QCRI/Fanar-1-9B) to do Arabic
morphology: extract the root of a word, build any verb Form I–X from a
root, conjugate, derive masdars and participles, and name a verb's form.

It is a **completion model, not a chat model**: give it one of the prompt
formats below and read the first line of the output.

## Results

All scores: greedy decoding, Q5_K_M GGUF served by llama.cpp.

**Packet B — held-out test.** 1,343 items over **258 roots that were
excluded from all training data**: real Quranic words (gold roots from the
Quranic Arabic Corpus) and modern words from the Arramooz lexicon. Root items
are asked both with a few-shot prompt and as a bare question.

| | Fanar-1-9B base | Run 1 (GRPO) | **SUPERBRAIN-1** |
|---|---|---|---|
| Root extraction, few-shot | 43.2% | 53.8% | **67.9%** |
| Root extraction, bare question | 59.5% | 48.9% | **66.8%** |
| Verb Forms I–X (428 items) | 16.6% | 61.2% | **94.9%** |
| — Form III | 1.8% | 0.0% | **94.5%** |
| — Form VIII | 6.9% | 6.9% | **96.5%** |
| — Form X | 0.0% | 0.0% | **100%** |

Root accuracy by root class (bare): sound 79%, hollow 60%, doubled 56%,
hamzated 51%, defective 45%, assimilated 44%.

**85-item gate (v3)** — 50 root words, 20 arithmetic/logic questions, 15
verb forms: **96.5%** (base 61.2%, Run 1 84.7%).

**Generalization probe.** The root خدم was banned from all training data.
Asked for the root of مستخدمين ("users"), SUPERBRAIN-1 answers **خدم** under
all three prompt formats (Run 1 answered استخدم).

## Prompt formats

Root (bare): `ما جذر كلمة «مستخدمين»؟\nالجذر:` → ` خدم`

Root (few-shot, as in the evaluation):

```
المهمة: استخرج الجذر الثلاثي للكلمة العربية.
أمثلة:
كلمة: مكتبات
الجذر: كتب
...
الآن اشتق الجذر لكل كلمة:
كلمة: المقاومة
الجذر:
```
→ ` قوم` (hollow roots keep their weak radical)

Verb form: `الجذر: درج — الصيغة العاشرة\nالفعل:` → ` استدرج`

Conjugation: `الفعل: نجم | الزمن: المضارع المجزوم | الضمير: هما\nالتصريف:` → ` ينجما`

Masdar: `ما المصدر من الفعل «أحرج»؟\nالجواب:` → ` إحراج`

Form: `ما صيغة الفعل «حرّف»؟\nالصيغة:` → ` الثانية`

## Usage (llama.cpp)

```bash
llama-server -m superbrain1-q5_k_m.gguf -ngl 99 -c 4096 --port 8099
curl -s http://127.0.0.1:8099/completion -d '{"prompt":"ما جذر كلمة «مستخدمين»؟\nالجذر:","n_predict":16,"temperature":0}'
```

Q5_K_M scores identically to the 16-bit model on the gate; Q4_K_M lost one
gate item in Run 1 testing.

## How it was trained

1. **Run 1 — GRPO** (verifiable exact-match reward) on 99 roots, 200 steps,
   QLoRA r32.
2. **Run 2a — supervised fine-tuning** on **600,000 gold items from 5,955
   roots** (root extraction 396k, conjugation 90k, verb forms 48k, form
   naming 18k, derivation 18k, arithmetic 30k). Items were generated from the
   Arramooz lexicon and the Qutrub conjugator, after validating the
   generator against the Quranic Arabic Corpus (Form inference precision
   99.0%, paradigm recall 94.8%). Packet B roots and the probe root خدم were
   excluded. QLoRA r64 on all linear layers, loss on the answer only, 1 epoch.

## Limitations

- **Roots of unseen words are right about two times in three.** Main error
  types: clitic ambiguity out of context (فساق read as ف + ساق → سوق, a
  valid parse that the test marks wrong), waw-vs-yeh final radicals (رضو /
  رضي; dictionaries disagree on several), and affixes left on (تمذهب →
  مذهب).
- Single words without context; no disambiguation from sentence meaning.
- Not instruction- or chat-tuned; free-form questions get base-model
  behaviour.
- Quranic words are tested in modern spelling (Uthmani script converted).

## Data and licenses

- Base model: QCRI Fanar-1-9B, Apache-2.0.
- Training data was **generated** with Arramooz (GPL), Qutrub / libqutrub
  (GPL) and PyArabic (GPL). These tools and their databases are not
  distributed with this model; only generated question–answer pairs were
  used for training.
- Evaluation only: Quranic Arabic Corpus (GPL).

Thanks to QCRI for Fanar, Taha Zerrouki for Arramooz, Qutrub and PyArabic,
and Kais Dukes for the Quranic Arabic Corpus.
