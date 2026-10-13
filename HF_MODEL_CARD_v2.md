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
- llama.cpp
- gguf
---

# SUPERBRAIN-1 — Arabic-first root reasoning + Arabic math (Fanar-1-9B)

SUPERBRAIN-1 is a 9B Arabic model by ABUZ8, built on
[QCRI/Fanar-1-9B](https://huggingface.co/QCRI/Fanar-1-9B). It is trained to
reason through Arabic morphology (ṣarf): find the root of a word, build
any verb Form I–X from a root, conjugate, derive. On top of that, it solves
Arabic math word problems step by step.

It is a **completion model, not a chat model**. Give it one of the prompt
formats below and read the first line of the output.

## Files

| File | Version | What it is |
|---|---|---|
| `superbrain1-run6-q5_k_m.gguf` | **v2 (current, 2026-10-12)** | roots + Quranic words + Arabic math |
| `superbrain1-q5_k_m.gguf` | v1 (2026-10-06) | morphology specialist (best verb forms) |

## Results

All scores use greedy decoding on the Q5_K_M GGUF served by llama.cpp. Evaluation files are in `eval/`.

**Packet B — held-out test.** 1,343 items over **258 roots that were
excluded from all training data**: real Quranic words (gold roots from the
Quranic Arabic Corpus) and modern words from the Arramooz lexicon. The exam is
published as a dataset: [abuz8studios/arabic-sarf-packet-b](https://huggingface.co/datasets/abuz8studios/arabic-sarf-packet-b).

**Arabic math.** All 1,190 problems of the Arabic GSM8K test split
([Omartificial-Intelligence-Space/Arabic-gsm8k-v2](https://huggingface.co/datasets/Omartificial-Intelligence-Space/Arabic-gsm8k-v2)),
with exact match on the final number. The test questions were hash-checked so that none of them reached training.

| | Fanar-1-9B base | v1 (SFT) | **v2 (run6)** |
|---|---|---|---|
| Root extraction, bare question | 59.5% | 66.8% | **78.0%** |
| Root extraction, few-shot | 43.2% | 67.9% | **78.6%** |
| Verb Forms I–X (428 items) | 16.6% | **94.9%** | 88.8% |
| Arabic GSM8K (1,190) | — | — | **63.6%** |
| 85-item gate (v3) | 61.2% | 96.5% | 95.3% |

Math rose from 21.9% to 63.6%. The checkpoint before the math stage
(Quranic-word round) scored 21.9% on the same 1,190 problems. 300 GRPO
steps with an exact-answer reward took it to 63.6%.

v2 root accuracy by root class (bare question):

| Class | Accuracy |
|---|---|
| sound | 90.7% |
| doubled | 68.8% |
| hollow | 68.6% |
| assimilated | 64.8% |
| hamzated | 63.5% |
| defective | 48.1% |

v1 scored 79/56/60/44/51/45% on the same classes.

**Generalization probe.** The root خدم was banned from all training data.
Asked for the root of مستخدمين ("users"), both v1 and v2 answer **خدم**
under all three prompt formats.

## Prompt formats

Root (bare): `ما جذر كلمة «مستخدمين»؟\nالجذر:` → ` خدم`

Verb form: `الجذر: درج — الصيغة العاشرة\nالفعل:` → ` استدرج`

Conjugation: `الفعل: نجم | الزمن: المضارع المجزوم | الضمير: هما\nالتصريف:` → ` ينجما`

Math:

```
المهمة: حل السؤال التالي خطوة بخطوة ثم اكتب الجواب النهائي رقمًا.
سؤال: ...
الحل:
```
→ The model writes worked steps, then `الجواب: <number>`.

## Usage (llama.cpp)

```bash
llama-server -m superbrain1-run6-q5_k_m.gguf -ngl 99 -c 4096 --port 8099
curl -s http://127.0.0.1:8099/completion -d '{"prompt":"ما جذر كلمة «مستخدمين»؟\nالجذر:","n_predict":16,"temperature":0}'
```

## How v2 was trained

Every stage is kept only if it beats the previous champion on Packet B
without losing more than 2 points elsewhere.

1. **Supervised fine-tuning** (v1) on 600,000 generated gold items from
   5,955 roots: root extraction, conjugation, verb forms, form naming,
   derivation. QLoRA r64, loss on the answer only.
2. **GRPO rounds on roots and morphology** (champion "rl1f"), with an
   exact-match reward checked by a program.
3. **GRPO on Quranic words** (355 hard items, 400 steps).
4. **GRPO on Arabic math** (662 Arabic GSM8K train problems the model
   sometimes got right, 300 steps, 8 samples per prompt), with an
   exact-number reward.

Every reward is a program check, either a root oracle or an exact number. No AI judge is used.

## Limitations

- **v2 traded some verb-form accuracy for roots and math** (94.9% → 88.8%).
  Form VI is the weak spot at 55%. Use v1 if you only need verb forms.
- **About one root in five is still missed.** Defective roots (final و/ي) are
  the hardest class at 48%. Errors also come from clitic ambiguity out of
  context.
- **Single words only.** There is no sentence context and no disambiguation.
- **Not chat-tuned.** Free-form questions get base-model behaviour.
- **Quranic spelling.** Quranic words are tested in modern spelling, converted from Uthmani script.

## Data and licenses

- **Base model:** QCRI Fanar-1-9B, Apache-2.0.
- **Morphology training data** was **generated** with Arramooz, Qutrub and
  PyArabic (all GPL). Those tools are not distributed here. Only generated question–answer pairs were used for training.
- **Math:** Arabic GSM8K v2 (Apache-2.0). Only the train split was used for training; the test split was used only for evaluation.
- **Evaluation:** Quranic Arabic Corpus (GPL).

Thanks to QCRI for Fanar, Taha Zerrouki for Arramooz, Qutrub and PyArabic,
Kais Dukes for the Quranic Arabic Corpus, and Omartificial Intelligence
Space for Arabic GSM8K.
