# HUGGING FACE TODO — ABUZ8 models (written 2026-10-13 05:58)
Account: abuz8studios. Token already logged in on Pegasus (E:\ABU\venvs\sb1).
Uploader: E:\ABU\02_PROJECTS\AHMAD-1\upload_everything.py  (resumable: skips files already on HF at same size)
Queue: E:\ABU\02_PROJECTS\AHMAD-1\HF_UPLOAD_QUEUE.json (132 files, 144.9 GB)
Log:   E:\ABU\02_PROJECTS\AHMAD-1\upload2.log  (last line "ALL DONE n / 132" = finished)

## STEP 1 — Finish uploads
- [ ] Check log. If the python process died before "ALL DONE", re-run:
      E:\ABU\venvs\sb1\Scripts\python.exe E:\ABU\02_PROJECTS\AHMAD-1\upload_everything.py
- [ ] Repeat until "ALL DONE 132 / 132" and no FAIL lines.

## STEP 2 — Repos and contents
- [ ] abuz8studios/SUPERBRAIN-1 — WINNER soup_g2f_run6 Q5_K_M GGUF + verdict.
      Packet B: root few-shot 79%, root bare 78%, morphology 93%. Gate v3 95%.
- [ ] abuz8studios/SUPERBRAIN-1-lab — 16 rounds: runs/<run>/<run>-q5_k_m.gguf, lora/, verdict.json
      soup_g2f_run6, run6_math_grpo, run4_a8_quran, g2f_rl, g2f_math, g2f_sft, run7_a10,
      run7b_a10_gentle, soup50, rl3f, rl1f, g4_morph, g4_rl, g4_ta, g4_sft1, g4_math
- [ ] abuz8studios/MUTHALLATH-1-m0 — 0.5B triangle model: muthallath0_final, jabarr_finisher,
      jabarr_tri_ON, jabarr_tri_OFF + ablation.log, birth.log, LEDGER.md
- [ ] abuz8studios/FAIE-proofs — faie_30tps / honest / paging / presynth proof JSONs
      (GLM-5.2-REAM weights are third-party, NOT re-uploaded)
- [ ] abuz8studios/Qadir-v1-v2 — qadir-v1-c and qadir-v2-humble bf16 GGUFs (14.19 GB each) + Modelfiles

## STEP 3 — Model cards (README.md in each repo, numbers only from verdict/proof files)
- [ ] SUPERBRAIN-1: base QCRI/Fanar-1-9B, Apache-2.0, Packet B scores, "own test set" note.
- [ ] SUPERBRAIN-1-lab: one table row per run, copied from each verdict.json.
- [ ] MUTHALLATH-1-m0: ArabicMMLU n=200 base 0.300 / tri_OFF 0.290 / tri_ON 0.225 = no gain. Tool arena: finisher 77.5%.
- [ ] FAIE-proofs: 32.3 tok/s full residency, 1.31x kernel speedup, GLM-5.2-REAM Q2_K.
- [ ] Qadir-v1-v2: no test scores found on disk — mark "untested".

## STEP 4 — Verify
- [ ] For each repo: HfApi().list_repo_tree(repo, recursive=True) sizes == local sizes in HF_UPLOAD_QUEUE.json.

## STEP 5 — Delete (ONLY after Ahmad says "delete")
- [ ] D:\ABU\backups\AHMAD-1 (all 14 folders)
- [ ] E:\ABU\02_PROJECTS\AHMAD-1\train\* run folders except soup_g2f_run6\gguf\*q5_k_m.gguf
- [ ] D:\ABU\02_PROJECTS\MUTHALLATH-1\m0\checkpoints (30 x 1.1 GB)
- [ ] E:\ABU\MODELS\hf\gemma-4-E4B
- [ ] Log each deletion + freed GB in LEDGER.md and ABUZ8 Life board.
