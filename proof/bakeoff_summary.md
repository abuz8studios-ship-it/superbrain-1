# Arabic Base Bake-Off — 2026-09-19

**Harness:** `src/bakeoff_arabic.py` | 85 prompts total: 50 root extraction (arabic_root.py oracle), 20 MSA reasoning, 15 verb-form morphology | greedy, max 256 new tokens, base models (no instruct).

## Results

| Model | Root (n=50) | Reasoning (n=20) | Morphology (n=15) | Overall |
|---|---|---|---|---|
| Qwen3-0.6B-Base | 16/50 = 32.0% | 6/20 = 30.0% | 4/15 = 26.7% | **30.59%** |
| Qwen3-4B-Base | 31/50 = 62.0% | 1/20 = 5.0% | 5/15 = 33.3% | **43.53%** |

Raw: `bakeoff_results.jsonl` | Per-item: `bakeoff_details_<model>.json` | Logs: `bakeoff_0p6b.log`, `bakeoff_4b.log`

## Readings (pre-verdict, pending Falcon)

1. **4B root extraction is 2× the 0.6B** (62% vs 32%) — scale clearly buys Arabic lexical competence. Both are far from oracle-grade; RL (GRPO with root reward) has huge headroom on the 4B.
2. **4B reasoning 5% is a red flag, likely extraction artifact** — Qwen3-4B-Base rambles/thinks long past 256 tokens, so the answer marker gets buried. The 0.6B answers tersely and scores 30%. Needs a decode-fix pass (stop strings / higher max_tokens / first-line extraction) before treating this number as real model competence. Do NOT kill the 4B on this number alone.
3. **Morphology flat (27–33%)** across sizes — verb-form morphology is not emerging from scale; it's a training-data/reward-shape gap, not a capacity gap. Good GRPO target.
4. **Falcon still missing from the board** — repos gated (HTTP 401), awaiting Ahmad's HF license acceptance + token. Falcon-H1-Arabic-7B (native 600B Arabic-token pretrain) is the most likely candidate to beat Qwen3-4B-Base on root accuracy; must be scored on this same harness before backbone selection.

## Next
- [ ] Ahmad: accept TII license + provide HF token → run Falcon-Arabic-7B / Falcon-H1-Arabic-7B through harness
- [ ] Decode-fix pass on 4B reasoning (stop strings, 512 tokens) → re-score before any verdict
- [ ] Then: backbone vote → GRPO pipeline on winner


## Update 2026-09-19 — parser fix + re-run (measured)

The 4B reasoning red flag was confirmed as an extraction artifact, not model
failure: the scorer took the *last* "الجواب:" in the completion, but base
models continue generating new few-shot Q&A pairs after answering, so the
last marker belonged to an invented question. Fixed to first-answer parse
(`answer_parse: "first"` in JSONL) and re-ran both Qwen bases (greedy, 256
max new tokens, GPU1, same 85 prompts). Corrected results:

| Model | Root (n=50) | Reasoning (n=20) | Morphology (n=15) | Overall |
|---|---|---|---|---|
| Qwen3-0.6B-Base | 16/50 = 32.0% | 17/20 = 85.0% | 4/15 = 26.7% | **43.53%** |
| Qwen3-4B-Base | 31/50 = 62.0% | 20/20 = 100% | 5/15 = 33.3% | **65.88%** |

The pre-fix lines in `bakeoff_results.jsonl` (reasoning 30% / 5%) are stale
parser artifacts; the lines with `answer_parse: "first"` are authoritative.
Read: 4B-Base dominates everywhere — double the root accuracy of the 0.6B
and perfect arithmetic/syllogism CoT — while morphology stays flat (~27–33%)
across scale, confirming morphology is a reward-shape gap, not a capacity gap
(the ideal GRPO target). Falcon-H1-Arabic-7B-Instruct remains gated pending
Ahmad's HF access; the harness accepts it via `--model <path> --instruct`.
Verdict candidate: Qwen3-4B-Base unless Falcon beats 65.88% overall on this
harness.
