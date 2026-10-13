# AHMAD-1 — exact next steps for Muse / Kimi
Read `GOAL.md` and `BUILD_STATUS.md` in this same folder first — they already
say most of this. This file exists to turn "Brick One" into copy-paste commands
so no interpretation is needed.

**Do not skip ahead.** GOAL.md is explicit: nothing else moves until Brick One passes.

---

## Step 1 — Build Brick One: `arabic_root.py`

**File to create:** `E:\ABU\02_PROJECTS\AHMAD-1\arabic_root.py`

**Requirement:** a function `find_root_of_root(word, max_depth=5)` that returns
the correct Arabic trilateral root for a given word.

**Note:** `E:\ABU\02_PROJECTS\AHMAD1_WORLDVIEW\` already has related work —
check `AHMAD1_WORLDVIEW\miner.py` and `miner_v2.py` before writing from scratch;
they may already contain a usable root-extraction approach. `GOAL.md` states
plainly: `arabic_lever.py EXISTS — normalizer only, NOT a root extractor` — so
that specific file is a known dead end, don't start there.

**The probe it must pass (10 words, exact test from GOAL.md):**
```
كتب، علم، فهم، رحم، قلب، نور، حق، صبر، عقل، روح
```
Each must resolve to its correct 3-letter root. Example: كتب (kataba, "he
wrote") → root ك-ت-ب (k-t-b).

**Pass condition:** file exists, runs without error, all 10 return the correct
root. **Fail condition:** anything else — including "close" or "8 of 10."
GOAL.md's own words: "Pass = file exists and probe runs clean. Fail = not done."

**When it passes:** log it in `LEDGER.md` in this folder — date, the 10 words,
the 10 roots it returned, pass/fail. That log entry IS the proof; a claim
without it does not count per this project's own rule.

---

## Step 2 — run in parallel with Step 1 (no dependency between them)

**2a. Extract MUTHALLATH-1's zip if not already done:**
```powershell
Test-Path "E:\ABU\02_PROJECTS\MUTHALLATH-1\DESIGN_v0.2.md"
```
If that returns `True`, it's already extracted — skip this step (BUILD_STATUS.md
was written 2026-09-14; check whether this already happened since).
If `False`:
```powershell
Expand-Archive -Path "C:\Users\wirec\Downloads\MUTHALLATH-1_v0.1.0.zip" -DestinationPath "E:\ABU\02_PROJECTS\MUTHALLATH-1" -Force
```

**2b. AHMAD_WORLDVIEW.md — only Ahmad can do this.** Do not attempt to answer
the 21 GAPS questions on his behalf. Flag it to him directly: file is at
`E:\ABU\02_PROJECTS\AHMAD1_WORLDVIEW\AHMAD_WORLDVIEW.draft.md`, needs his
signed answers, unsigned = does not exist for training purposes.

---

## Step 3 — after Brick One passes: verify the corpus is still what BUILD_STATUS.md says

```powershell
$path = "C:\Users\wirec\.claude\projects"
$files = Get-ChildItem $path -Recurse -Filter "*.jsonl" | Measure-Object
$size = (Get-ChildItem $path -Recurse -Filter "*.jsonl" | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host "JSONL files: $($files.Count)  |  Size: $([math]::Round($size,2)) GB"
```
**Compare against BUILD_STATUS.md's claim of 2,449 files / 2.26 GB.** If the
count is lower, that's expected — some transcripts were pruned by a Claude Code
retention bug on 2026-09-15/16 (see `session-ledger-2026-09-16.md` in the
memory directory) and partially recovered; the corpus may be smaller than
BUILD_STATUS.md recorded. **Do not treat a mismatch as your own error — report
the actual number and move on.** Training can proceed on whatever count exists;
it does not need to hit exactly 2,449.

---

## Step 4 — do not proceed past this without checking `GOAL.md`'s delivery checklist

Backbone model selection, tokenizer, 8-layer architecture, training run,
eval, quantization, HuggingFace upload — all still open, all listed with
their own pass/fail criteria in `GOAL.md`. This file only covers unblocking
Brick One; once it passes, re-read `GOAL.md` fresh rather than assuming this
file's job continues.

## Rules that apply to every step above (from GOAL.md, do not violate)
- Read GOAL.md + check actual disk state before any claim.
- No unverified superlatives in any log — a "first" or a "best" needs disk proof.
- Fixed model size until size is proven necessary — do not grow the backbone
  past what GOAL.md specifies without a measured reason written in the ledger.
