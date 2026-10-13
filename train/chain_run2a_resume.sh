#!/usr/bin/env bash
# chain_run2a_resume.sh - SUPERBRAIN 1 Run 2a, finish after the 2026-10-06
# 17:06:33 GPU fault (nvlddmkm 153, GPU 1 / bus 03:00.0) killed the full run at
# step 6998/7032 with no final adapter. Resumes from the newest checkpoint
# (step, optimizer, scheduler, RNG restored) with the SAME args as chain_run2a.sh,
# retries on another fault, then post_train -> verdict. chain_run2b.sh waits on
# run2a/verdict.json and then runs GRPO.
set -u
T=/e/ABU/02_PROJECTS/AHMAD-1/train
W='E:\ABU\02_PROJECTS\AHMAD-1\train'
PY=E:/ABU/venvs/sb1/Scripts/python.exe
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
cd "$T" || exit 1
log() { echo "[$(date '+%F %T')] $*" | tee -a chain_run2a.log; }

for attempt in 1 2 3 4; do
  CK=$(ls -d run2a/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  CK=${CK%/}
  [ -n "$CK" ] || { log "no checkpoint to resume - FAILED"; exit 5; }
  log "RESUME attempt $attempt from $CK"
  CUDA_VISIBLE_DEVICES=1 $PY train_sft.py --init-lora "$W\\run2a_pilot\\lora" --skip 150000 \
    --resume "$(cygpath -w "$T/$CK")" --out "$W\\run2a" >> run2a_resume.log 2>> run2a_resume.err && break
  log "attempt $attempt died"; sleep 20
  [ $attempt -eq 4 ] && { log "FULL RUN CRASHED (4 attempts)"; exit 5; }
done
log "full run saved -> post_train"
$PY post_train.py --run "$W\\run2a" --gpu 0 --port 8097 > post_run2a.log 2>&1 \
  || { log "post_train(run2a) FAILED"; exit 6; }
grep -q '"PASS": true' run2a/verdict.json && log "RUN 2A VERDICT: PASS" || log "RUN 2A VERDICT: FAIL"
