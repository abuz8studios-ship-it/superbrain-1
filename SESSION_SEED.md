# SESSION SEED — AHMAD-1 Program (read this first, then act)
Written 2026-09-19 ~16:45 EDT by Qadir (session 1 of the program). Density over prose. Everything verified at write time.

## MISSION
AHMAD-1: sub-4GB Apache-2.0 Arabic-first reasoning model. Grassmannian concept layer (simplex approach ablated NULL — do not resurrect it). ISNAD-graded calibrated uncertainty. 8200 anomaly detection as product. North star: `docs\PRD_BRIEF.md` in this dir. Standard: the 1.2%.

## DONE (verified, do not redo)
- Brick One: `arabic_root.py` PASSES (10/10 roots + junk rejection). LEDGER.md entry 2026-09-19, proof `proof\probe_2026-09-19.txt`.
- Town mesh (Phases 0–2 of fusion-city plan): 267-agent swarm live, 24 turns/tick, ~$0 local brains, execute ON, AHMAD-1 tasks seeded. Proofs in `E:\ABU\ABUZ8_OS_DIST\proof\2026-09-19-phase0/1/2-*.md`.
- Town board has 6 AHMAD-1 workstream tasks (W1 backbone, W2 anomaly spec, W3 curation plan, W4 risks, W5 HF card, W6 root regression).

## WAVE-1 COMPLETE (all landed 2026-09-19 ~16:45)
- **AUDIT** (`AUDIT_2026-09-19.md`): corpus 3,029 jsonl/1.51GiB usable. No small base existed → **Qwen3-0.6B-Base (1.20GB) + Qwen3-4B-Base (8.06GB) downloaded + size-verified** at E:\ABU\MODELS\hf_cache. trl 1.13 + datasets 5.0 installed. MUSE headless: `muse.cmd exec --json --prompt-file <f>`, `MUSE_NO_AUTO_UPDATE=1`.
- **PROBE = CONDITIONAL GO** (`proof\grassmann_probe_results.json` + `_summary.md`): simplex thesis NULL AGAIN (cos 0.93–0.95, not −0.5). **Concepts are subspaces: k=2 captures 0.763 variance; principal angles 21.1° real vs 88.1° random — clean Grassmann separation on G(2,R^1536).** Cross-lingual alignment NULL (script identity dominates the Arabic gap — never claim semantic bridging). Layer proceeds MONOLINGUAL ONLY; kill-test discipline (regex-baseline gate) still gates re-entry.
- **LIT REVIEW** (`RESEARCH_LITREVIEW.md`, 150 lines): 20 verified papers + 14 extra. Recommendation k=3 semantic / k=8–16 detection (4,500 roots pack trivially by Welch bound). GRPO cost: 0.6B ≈ 90–150s/step → meaningful run 2–4 days LOCAL; 4B ≈ 8–15min/step → 2–4 weeks local — **spend the $500 rental on the 4B stage only, with Ahmad's go**. Honest odds: ~70% best-in-size-class Arabic reasoner + anomaly product; ~10% frontier-class narrow-domain sub-1GB.

## NEXT ACTIONS (priority order) — session 2 starts here
1. **GRPO pipeline on Qwen3-0.6B-Base** (fast loop, local): trl GRPOTrainer on GPU1; reward = arabic_root.py root-accuracy + ISNAD-format compliance. Prove reward design before scaling to 4B.
   - **[DONE 2026-09-19] Base bake-off** (`src\bakeoff_arabic.py`, 85 prompts): 0.6B 30.6% / 4B 43.5% overall (`proof\bakeoff_summary.md`). 4B root 62% = 2× the 0.6B. 4B reasoning 5% is a decode artifact (rambles past 256 tok) — re-test w/ stop strings before trusting. Morphology flat = GRPO reward target.
   - **Falcon slot still open**: gated repos, awaiting Ahmad's HF license + token; run Falcon-H1-Arabic-7B through same harness when access lands.
2. Town W1–W6 review when tasks close (board at **:61500** as of 2026-09-19 evening — port now pinned via data_root\config\settings.json `preferred_port: 61500`; env ABUZ8_PORT does NOT pin it, default auto-picks a free port. run_mu8uiwyh).
   - **[DONE 2026-09-19] KIMI SWARM LANE LIVE** — soloman + Kimi council seats route to Kimi K2.5 via OAuth (~13s/turn, proven in tick 200). Lane code: town.js MODEL_MAP + CLI_LANES; localbrain.js kimiReply MUST spawn via cliLaunch (raw execFile on kimi.cmd = EINVAL). Tick wall-time is 5–15 min — a silent runlog is NOT a hang; check `_tt()` trace in backend log first.
3. Bukhari curation → training packets → first real run → eval → sub-4GB GGUF (q8_0 convert → llama-quantize Q4_K_M).
4. **Falcon bake-off (Ahmad's pick, pending his action):** official TII Arabic Falcons exist (Falcon-H1-Arabic-7B-Instruct, Jan 2026, top Arabic-class OALL; Falcon-Arabic-7B, May 2025, 600B native Arabic tokens) but are GATED (HTTP 401). Ahmad must accept the TII license on huggingface.co AND provide HF auth (token or browser login on this box). NOTE: TII Falcon License 2.0 ≠ pure Apache-2.0 — PRD deviation he must approve. His claim 'trained from ground up on arabic root' is NOT accurate per verified sources (native Arabic yes — root-morphology-aware, no; plain Falcon-3 base is BAD at Arabic, fertility 4.54). Bake-off harness `src\bakeoff_arabic.py` (agent-uzovd55y) scores Qwen3-0.6B/4B-Base NOW; Falcon joins when access lands. Measurement decides the base.
5. Geometric layer: build ONLY the monolingual G(2) probe-validated component; PRE-REGISTER the kill-test before training.

## HARD-WON FACTS (do not rediscover)
- gemma-4-E2B is a REASONING model: needs max_tokens ≥2000 (town uses 8192); turn end is `<turn|>` NOT `<|im_end|>`.
- LFM2.5-2.6B hangs forever on ≥4k prompts. Retired. Qwen3.5-4B never stops thinking. Retired. Fleet = 3× gemma-E2B labs (:8025 lib+lodge+LoRA / :8022 mktg / :8023 shop) + sovereign :8011 (Ornith-35B, **PID 42152 do-not-kill**, judge/generator/escalation) + native :8902 + eval :8026.
- Windows loopback DUAL-BIND is real: wrong answers/hangs on a live port → netstat, kill ALL listeners with taskkill //PID //F (never MSYS kill).
- nvidia-smi lies about full GPUs (zombie WDDM); trust `torch.cuda.mem_get_info` — GPU1 usually has ~32GB free.
- llama.cpp lab rule: --parallel 2, ctx 8192 (4096/slot minimum for town prompts).
- `_assistantText` bug fixed in ABUZ8 OS sdk.js/brain.js — OpenAI-type providers work now.
- LoRA pipeline works end-to-end: scripts in `E:\ABU\ABUZ8_OS_DIST\scripts\p2_*.py`, venv `E:\ABU\ABUZ8_OS_DIST\venvs\lora`, adapter writer = p2_lora_to_gguf.py (NEOCORTEX converter is gemma4-blind). ~15 min/adapter.
- `convert_hf_to_gguf.py` (prismml) emits f32/f16/q8_0 only; K-quant via llama-quantize.exe.
- MUSE CLI: `C:\Users\wirec\AppData\Local\Programs\muse\muse.cmd` (powershell launcher, bins R3233/R3401). Non-interactive launch syntax = being documented by audit agent.
- Town project rules: LEDGER entry required for every claim (n, method, proof path). No unverified superlatives. Kill approaches on NULL results — that is success too.
- PRD_BRIEF.md was accidentally deleted once and rewritten verbatim same day (header notes it).

## MONEY/ESCALATION
Never spend/commit/move Ahmad's money — $0–500 rental budget needs his explicit go per run. Escalate via Telegram. No religion as product gate; never harm children; honesty in every ledger row; never delete prior work (backups before overwrites).

## FILE MAP
- Program: `E:\ABU\02_PROJECTS\AHMAD-1\` (GOAL.md, BUILD_STATUS.md, LEDGER.md, NEXT_AGENT_INSTRUCTIONS.md, docs\PRD_BRIEF.md, arabic_root.py, proof\)
- MUTHALLATH lineage: `E:\ABU\02_PROJECTS\MUTHALLATH-1\` (88-entry LEDGER, geometry.py, concept_layer.py — read before designing the Grassmannian layer)
- Worldview miner: `E:\ABU\02_PROJECTS\AHMAD1_WORLDVIEW\` (miner.py/miner_v2.py root-extraction ancestors)
- Town/fusion plan: `E:\ABU\ABUZ8_OS_DIST\sahara\05_*.md` + `06_HANDOFF_2026-09-19.md` (full fleet restart cmds)
- Transcripts corpus: `C:\Users\wirec\.claude\projects\` (claim: 2,449 jsonl/2.26GB — audit re-counts)

*Session 2: start by collecting the three wave-1 agents, then Next Actions #2. Keep it surgical.*
