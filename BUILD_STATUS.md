# AHMAD-1 BUILD STATUS
**As of Sep 14, 2026 — READY TO BEGIN TRAINING**

## What's Ready NOW

### ✅ Training Corpus
- Location: `C:\Users\wirec\.claude\projects\`
- Size: 2,449 JSONL files (2.26 GB estimated)
- Status: LIVE & READY TO USE
- Format: Claude Code session events (type:"user", message.content structure)

### ✅ Claude Desktop Configuration
- Location: `C:\Users\wirec\AppData\Roaming\Claude\claude_desktop_config.json`
- Updated: Sep 14, 2026 09:59 UTC
- Changes: Precache enabled, minimal context boot, emotional blocker settings removed
- Restart required: YES (close and reopen Claude Desktop)

### ✅ Project Folder Structure
```
E:\ABU\02_PROJECTS\AHMAD-1\
├── CLAUDE.md                    ← Session law (this build)
├── BUILD_STATUS.md              ← This file
├── GOAL/
│   └── 01_WHAT_IT_IS.md
├── ARTIFACTS/                   ← Will hold build outputs
└── LEDGER.md                    ← (to be created)
```

### ✅ QADIR Core Infrastructure
- Location: `E:\ABU\QADIR_CORE\`
- Status: OPERATIONAL (87 API operations verified)
- MCP Server: Running on Pegasus (qadir.mcp_server)
- Available: As fallback brain during AHMAD-1 training

---

## What's Blocked (Needs Action)

### 🔴 BRICK ONE: arabic_root.py
**Status: NOT BUILT**
- Required: Function `find_root_of_root(word, max_depth=5)`
- Probe: 10 Arabic words → correct trilateral roots
- Blocker: None — can be built immediately
- Estimated effort: 1-2 hours

### 🟡 MUTHALLATH-1 Extraction
**Status: DOWNLOADED BUT ZIPPED**
- File: `C:\Users\wirec\Downloads\MUTHALLATH-1_v0.1.0.zip`
- Target: `E:\ABU\02_PROJECTS\MUTHALLATH-1\`
- Command: `Expand-Archive -Path C:\Users\wirec\Downloads\MUTHALLATH-1_v0.1.0.zip -DestinationPath E:\ABU\02_PROJECTS\MUTHALLATH-1 -Force`
- Effort: 1 minute

### 🟡 AHMAD_WORLDVIEW.md
**Status: DRAFT EXISTS, NEEDS AHMAD'S ANSWERS**
- File: `E:\ABU\02_PROJECTS\AHMAD1_WORLDVIEW\AHMAD_WORLDVIEW.draft.md`
- What it needs: Ahmad's signed response to 21 GAPS questions
- Questions cover: Fractal, Frequency, Color, Smell, Arabic Root, Triangle, Never-Stop (7 principles × 3 questions each)
- Effort: 1-2 hours (only Ahmad can answer authentically)

### 🟡 Backbone Model Selection
**Status: CANDIDATE IDENTIFIED, NOT DOWNLOADED**
- Recommendation: Qwen3-4B (4B dense model fits on edge devices)
- Alternative: Qwen3-Coder-Next 8B (if coding reasoning needed)
- Storage needed: 8-16 GB on Pegasus

---

## The Path Forward (3 Steps to Training)

### Step 1: Unblock Brick One
**Build arabic_root.py** — this is the critical path blocker
- Estimated time: 1-2 hours
- Test: Run on 10 Arabic words, all return correct roots
- Gate: Nothing moves until this passes

### Step 2: Extract & Worldview
**Unzip MUTHALLATH-1** (1 min) + **Ahmad answers GAPS** (1-2 hrs)
- These unblock the training pipeline
- Parallel execution OK

### Step 3: Train AHMAD-1
**Run training loop on corpus**
- Estimated time: 8-24 hours (depends on backbone model size, quantization, GPU load)
- Eval: Ahmad-style reasoning probe (10 test questions)
- Export: Quantize to GGUF + upload to HuggingFace

---

## Quick-Start Commands

### Unzip MUTHALLATH-1 (Windows PowerShell)
```powershell
Expand-Archive -Path "C:\Users\wirec\Downloads\MUTHALLATH-1_v0.1.0.zip" -DestinationPath "E:\ABU\02_PROJECTS\MUTHALLATH-1" -Force
```

### Verify training corpus
```powershell
$path = "C:\Users\wirec\.claude\projects"
$files = Get-ChildItem $path -Filter "*.jsonl" | Measure-Object
$size = (Get-ChildItem $path -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host "Total JSONL files: $($files.Count)"
Write-Host "Total size: $([math]::Round($size, 2)) GB"
```

### Start Claude Desktop (after config update)
```powershell
taskkill /IM "Claude.exe" /F  # Kill existing process
Start-Sleep -Seconds 2
& "C:\Users\wirec\AppData\Local\Programs\Claude\Claude.exe"  # Relaunch
```

---

## Success Metrics

**Build phase complete when:**
1. ✅ arabic_root.py passes probe on all 10 Arabic words
2. ✅ MUTHALLATH-1 extracted and folder structure verified
3. ✅ AHMAD_WORLDVIEW.md signed by Ahmad (all 21 GAPS answered)
4. ✅ Backbone model selected + downloaded
5. ✅ Training pipeline initialized on first 100 JSONL files (smoke test)

**Training complete when:**
1. ✅ Model trained to final checkpoint
2. ✅ Eval probe passes (Ahmad-style reasoning)
3. ✅ Quantized to GGUF format
4. ✅ HuggingFace model card written + uploaded
5. ✅ Demo video recorded (30 sec: "This is AHMAD-1, trained on Ahmad Odeh's reasoning")

---

## Critical Paths

**DO NOT:**
- Download new models or add complexity until Brick One passes
- Skip AHMAD_WORLDVIEW.md (training will hallucinate without this spec)
- Train on full corpus before smoke-testing on 100 files
- Publish model without eval passing

**DO:**
- Unblock Brick One immediately (1-2 hours, then everything else flows)
- Run in parallel: MUTHALLATH extract + Ahmad's worldview answering
- Test each stage before moving to the next
- Document every decision in LEDGER.md

---

**Status:** READY FOR BRICK ONE  
**Next immediate action:** Build arabic_root.py  
**Owner:** Ahmad Odeh  
**Last updated:** 2026-09-14 09:59 UTC
