#!/usr/bin/env bash
# chain_rl.sh NAME SFT_ADAPTER GPU [STEPS] - GRPO on roots the SFT model NEVER saw.
#   1 merge the SFT adapter -> NAME/merged_16bit (the GRPO base)
#   2 dev accuracy of that SFT model (the number GRPO has to beat)
#   3 GRPO on grpo_rl.jsonl (585 held-out roots, class-balanced), 3 auto-resume tries
#   4 dev accuracy of the GRPO result
# Built ON E:; launch with detach.ps1 so an app update cannot kill it.
set -u
NAME=$1; SFT=$2; GPU=${3:-1}; STEPS=${4:-200}; DATAF=${5:-grpo_rl.jsonl}
T=/e/ABU/02_PROJECTS/AHMAD-1/train
W='E:\ABU\02_PROJECTS\AHMAD-1\train'
PY=E:/ABU/venvs/sb1/Scripts/python.exe
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
cd "$T" || exit 1
log() { echo "[$(date '+%F %T')] [$NAME] $*" | tee -a "chain_$NAME.log"; }

FREE_GB=$(df -k /e | awk 'NR==2{print int($4/1048576)}')
NEED=50                                   # merging the SFT adapter writes ~18 GB
[ -f "$NAME/merged_16bit/config.json" ] && NEED=15   # base already merged: only adapter checkpoints (~4 GB)
[ "$FREE_GB" -ge "$NEED" ] || { log "E: has only ${FREE_GB} GB free (<${NEED}) - NOT starting"; exit 7; }
mkdir -p "$NAME"

if [ ! -f "$NAME/merged_16bit/config.json" ]; then
  log "merging $SFT"
  $PY merge_adapter.py --adapter "$W\\$(echo "$SFT" | tr '/' '\\')" --out "$W\\$NAME\\merged_16bit" --gpu "$GPU" \
      > "$NAME.merge.log" 2>&1 || { log "merge FAILED"; exit 2; }
fi
log "merged OK"

$PY fast_eval.py "$W\\$NAME\\merged_16bit" --sets dev --gpu "$GPU" --out "evals/${NAME}_sft_dev.json" \
    > "evals/${NAME}_sft_dev.txt" 2>&1
log "SFT-base dev: $(grep -aE '^  ALL' "evals/${NAME}_sft_dev.txt" | head -1 | tr -s ' ')"

for attempt in 1 2 3; do
  RES=""
  CK=$(ls -d "$NAME"/grpo/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1); CK=${CK%/}
  if [ -n "$CK" ]; then RES="--resume $(cygpath -w "$T/$CK")"; log "GRPO attempt $attempt resuming from $CK"
  else log "GRPO attempt $attempt fresh"; fi
  CUDA_VISIBLE_DEVICES=$GPU $PY train_grpo2.py --model "$W\\$NAME\\merged_16bit" \
      --data "$W\\$DATAF" --out "$W\\$NAME\\grpo" --max-steps "$STEPS" \
      --bs 32 --accum 4 --gens 8 --no-merge $RES >> "$NAME.grpo.log" 2>> "$NAME.grpo.err" && break
  log "GRPO attempt $attempt died"; sleep 20
  [ $attempt -eq 3 ] && { log "GRPO CRASHED"; exit 5; }
done
log "GRPO done"

$PY fast_eval.py "$W\\$NAME\\grpo\\lora" --sets dev --gpu "$GPU" --out "evals/${NAME}_grpo_dev.json" \
    > "evals/${NAME}_grpo_dev.txt" 2>&1
log "GRPO dev: $(grep -aE '^  ALL' "evals/${NAME}_grpo_dev.txt" | head -1 | tr -s ' ')"
log "CHAIN DONE"
