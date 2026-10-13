---
license: gpl-3.0
language:
- ar
task_categories:
- text-generation
tags:
- arabic
- morphology
- sarf
- root-extraction
- quran
- benchmark
pretty_name: Arabic Sarf Packet B (held-out root and verb-form exam)
size_categories:
- 1K<n<10K
---

# Arabic Sarf Packet B — a held-out exam for Arabic root reasoning

Packet B contains 1,343 exam items that test whether a language model can reason through Arabic
morphology (ṣarf) on **roots it has never seen**. It is the held-out test
used for [abuz8studios/SUPERBRAIN-1](https://huggingface.co/abuz8studios/SUPERBRAIN-1).
The items cover 258 roots, and all of them were excluded from every training packet.

| Section | Items | What it asks |
|---|---|---|
| `root_qac` | 415 | Root of a real Quranic word. Gold roots come from the Quranic Arabic Corpus, and words are given in modern spelling. |
| `root_msa` | 500 | Root of a modern Arabic word, taken from the Arramooz lexicon. |
| `morph_qac` | 68 | Build verb Form I–X from a Quranic root. |
| `morph_msa` | 360 | Build verb Form I–X from a modern root. |

## Fields

- `section`: one of the four sections above.
- `word`: the word whose root is asked (root items).
- `root`, `form`: the root and Form number (morph items).
- `answer`: the gold answer.
- `root_key`: the canonical root, with hamza written as ء.
- `root_class`: sound, hollow, doubled, hamzated, defective or assimilated.
- `prompt`: few-shot completion prompt.
- `prompt_bare`: bare question prompt (root items).

## How to score

Use greedy decoding and raw completion, with no chat template. Read the first line of the output.
Root answers are compared by root identity, with all hamza and alef carriers folded together and
ى read as ي. Verb forms are compared letter by letter after removing diacritics.
Root items are scored twice, once with the few-shot `prompt` and once with `prompt_bare`.

## Reference scores (Q5_K_M, llama.cpp)

| Model | Root (bare) | Root (few-shot) | Verb forms |
|---|---|---|---|
| QCRI Fanar-1-9B | 59.5% | 43.2% | 16.6% |
| SUPERBRAIN-1 v1 | 66.8% | 67.9% | 94.9% |
| SUPERBRAIN-1 v2 | 78.0% | 78.6% | 88.8% |

## Rules

**Never train on this set.** It only works as an exam while no model has seen it.

## License and sources

GPL-3.0, because the gold answers come from GPL sources:

- **Quranic Arabic Corpus** (Kais Dukes), for Quranic words and roots.
- **Arramooz** (Taha Zerrouki), for modern words and roots.
- **Qutrub / libqutrub** (Taha Zerrouki), for gold verb forms.

Built by ABUZ8 (Dayton, Ohio).
