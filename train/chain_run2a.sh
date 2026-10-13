#!/usr/bin/env bash
# chain_run2a.sh - SUPERBRAIN 1 Run 2a, unattended: wait for the pilot ->
# verdict -> (PASS only) full run continuing from the pilot adapter on the
# remaining 450k items -> verdict. Stops at the first failed gate.
set -u
T=/e/ABU/02_PROJECTS/AHMAD-1/train
W='E:\ABU\02_PROJECTS\AHMAD-1\train'
PY=E:/ABU/venvs/sb1/Scripts/python.exe
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
cd "$T" || exit 1
log() { echo "[$(date '+%F %T')] $*" | tee -a chain_run2a.log; }

until grep -q '^SAVED' run2a_pilot.log 2>/dev/null; do
  grep -qE 'Traceback|out of memory' run2a_pilot.err 2>/dev/null && { log "PILOT CRASHED"; exit 2; }
  sleep 30
done
log "pilot saved -> post_train"
$PY post_train.py --run "$W\\run2a_pilot" --gpu 0 --port 8097 > post_run2a_pilot.log 2>&1 \
  || { log "post_train(pilot) FAILED"; exit 3; }
grep -q '"PASS": true' run2a_pilot/verdict.json || { log "PILOT VERDICT: FAIL - stopping"; exit 4; }

log "PILOT PASS -> full run (continue from pilot, skip 150000)"
CUDA_VISIBLE_DEVICES=1 $PY train_sft.py --init-lora "$W\\run2a_pilot\\lora" --skip 150000 \
  --out "$W\\run2a" > run2a.log 2> run2a.err || { log "FULL RUN CRASHED"; exit 5; }
log "full run saved -> post_train"
$PY post_train.py --run "$W\\run2a" --gpu 0 --port 8097 > post_run2a.log 2>&1 \
  || { log "post_train(run2a) FAILED"; exit 6; }
grep -q '"PASS": true' run2a/verdict.json && log "RUN 2A VERDICT: PASS" || log "RUN 2A VERDICT: FAIL"
