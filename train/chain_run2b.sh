#!/usr/bin/env bash
# chain_run2b.sh - SUPERBRAIN 1 Run 2b, unattended: wait for the Run 2a
# verdict -> serve the best passing SFT model -> difficulty filter (items it
# gets right only sometimes, weighted to its weak buckets) -> GRPO -> verdict.
set -u
T=/e/ABU/02_PROJECTS/AHMAD-1/train
W='E:\ABU\02_PROJECTS\AHMAD-1\train'
PY=E:/ABU/venvs/sb1/Scripts/python.exe
LLAMA=/e/ABU/tools/llama.cpp/build-cuda/bin/llama-server.exe
PORT=8098
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
cd "$T" || exit 1
log() { echo "[$(date '+%F %T')] $*" | tee -a chain_run2b.log; }

until [ -f run2a/verdict.json ]; do
  grep -qE 'CRASHED|FAILED' chain_run2a.log 2>/dev/null && { log "run2a chain failed - stopping"; exit 2; }
  sleep 60
done
BASE=$($PY pick_base.py 2>> chain_run2b.log) || { log "no PASSing base - stopping"; exit 8; }
log "base for GRPO: $BASE"

# Run 2b is built ON E: (Ahmad's rule). GRPO + merge + f16/q5 GGUF export write
# ~46 GB: refuse to start, rather than die two steps from the end, if E: can't hold it.
FREE_GB=$(df -k /e | awk 'NR==2{print int($4/1048576)}')
[ "$FREE_GB" -ge 60 ] || { log "E: has only ${FREE_GB} GB free (<60) - NOT starting GRPO"; exit 7; }
log "E: preflight OK (${FREE_GB} GB free)"

CUDA_VISIBLE_DEVICES=0 "$LLAMA" -m "$BASE/gguf/superbrain1-q5_k_m.gguf" -ngl 99 -c 16384 -np 8 \
  --host 127.0.0.1 --port $PORT > "$BASE/gguf/server_grpo2_filter.log" 2>&1 &
SRV=$!
for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; sleep 2; done
kill -0 $SRV 2>/dev/null || { log "filter server failed to start"; exit 3; }
$PY build_grpo2_data.py --url http://127.0.0.1:$PORT --verdict "$BASE/verdict.json" \
  --candidates 16000 --keep 4000 --out "$W\\grpo2_train.jsonl" > grpo2_filter.log 2>&1
RC=$?; kill $SRV; sleep 3
[ $RC -eq 0 ] || { log "filter FAILED"; exit 4; }
log "filter done: $(grep '^kept' grpo2_filter.log | cut -c1-80)"

# GPU 1 (bus 03:00.0) threw nvlddmkm event 153 on 10/5 and 10/6 under load and
# killed jobs: retry from the newest checkpoint instead of dying.
for attempt in 1 2 3; do
  RES=""
  CK=$(ls -d run2b/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  CK=${CK%/}
  if [ -n "$CK" ]; then RES="--resume $(cygpath -w "$T/$CK")"; log "GRPO attempt $attempt resuming from $CK"
  else log "GRPO attempt $attempt fresh"; fi
  CUDA_VISIBLE_DEVICES=1 $PY train_grpo2.py --model "$W\\$BASE\\merged_16bit" --out "$W\\run2b" $RES \
    >> run2b.log 2>> run2b.err && break
  log "GRPO attempt $attempt died"; sleep 20
  [ $attempt -eq 3 ] && { log "GRPO CRASHED"; exit 5; }
done
log "run2b saved -> post_train"
$PY post_train.py --run "$W\\run2b" --gpu 0 --port 8097 > post_run2b.log 2>&1 \
  || { log "post_train(run2b) FAILED"; exit 6; }
grep -q '"PASS": true' run2b/verdict.json && log "RUN 2B VERDICT: PASS" || log "RUN 2B VERDICT: FAIL"
