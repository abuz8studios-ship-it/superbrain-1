# SUPERBRAIN-1 — code, trainers, evals, ledger

Arabic-first brain work by ABUZ8. This repo is the **code and research ledger**;
the **weights and official scores live on Hugging Face**:

- 🤗 Model (current v3, scores, prompt formats): https://huggingface.co/abuz8studios/SUPERBRAIN-1
- 🤗 Every training run (LoRAs, GGUFs, verdicts): https://huggingface.co/abuz8studios/SUPERBRAIN-1-lab
- 🤗 Held-out exam (258 roots excluded from training): https://huggingface.co/datasets/abuz8studios/arabic-sarf-packet-b
- 🤗 Training/eval packets: [arabic-science-packet-a10](https://huggingface.co/datasets/abuz8studios/arabic-science-packet-a10) · [arabic-code-packet-a11](https://huggingface.co/datasets/abuz8studios/arabic-code-packet-a11)
- Related experiment log (honest negative result): https://huggingface.co/abuz8studios/MUTHALLATH-1-m0

## Status — active research, not finished

SUPERBRAIN-1 is a 9B completion model on QCRI/Fanar-1-9B (Apache-2.0) trained for Arabic
morphology (root extraction, verb Forms I–X, conjugation) and Arabic math. The current
champion is **v3 run8b_soup (2026-10-16)** — full numbers and known weaknesses
(about 1 root in 5 still missed; defective roots hardest) are on the model card above.
Every stage is kept only if it beats the previous champion on Packet B under the
promotion rule recorded in this repo; rewards are program checks, never an AI judge.

## In this repo

- `arabic_root.py`, `probe_arabic_root.py` — root tools and probes
- `src/`, `train/` — training and evaluation code
- `proof/`, `LEDGER.md`, `PACKET_B_README.md` — evidence and the running research ledger
- `HF_MODEL_CARD*.md`, `HF_TODO.md` — model-card drafts/working notes (the live card is on HF)
- `BUILD_STATUS.md`, `GOAL.md`, `AUDIT_2026-09-19.md` — dated status/audit snapshots (history, not current state)

Contact: support@abuz8ai.com · Site: https://abuz8ai.com · X: https://x.com/AbuZ8Studios
