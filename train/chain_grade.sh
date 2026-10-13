#!/usr/bin/env bash
# chain_grade.sh NAME ADAPTER [GPU_MERGE] [GPU_SERVE] - full official grade of a LoRA candidate:
#   merge ADAPTER -> NAME/merged_16bit -> post_train.py (GGUF q5_k_m, 85-item gate v3,
#   Packet B roots + morphology, probe) -> NAME/verdict.json. Packet B is used here ONLY to
#   GRADE a finished candidate. Drops the regenerable f16 GGUF afterwards to save 17.6 GB.
# Launch with detach.ps1 (survives Claude-app updates).
set -u
NAME=$1; AD=$2; GM=${3:-1}; GS=${4:-0}
T=/e/ABU/02_PROJECTS/AHMAD-1/train
W='E:\ABU\02_PROJECTS\AHMAD-1\train'
PY=E:/ABU/venvs/sb1/Scripts/python.exe
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
cd "$T" || exit 1
log() { echo "[$(date '+%F %T')] [$NAME] $*" | tee -a "chain_$NAME.log"; }
FREE_GB=$(df -k /e | awk 'NR==2{print int($4/1048576)}')
NEED=45                                   # merging writes ~18 GB, then GGUFs ~24 GB
[ -f "$NAME/merged_16bit/config.json" ] && NEED=28   # already merged: only the GGUFs (~24 GB) remain
[ "$FREE_GB" -ge "$NEED" ] || { log "E: has only ${FREE_GB} GB free (<${NEED}) - NOT starting"; exit 7; }
mkdir -p "$NAME"
if [ ! -f "$NAME/merged_16bit/config.json" ]; then
  log "merging $AD"
  $PY merge_adapter.py --adapter "$W\\$(echo "$AD" | tr '/' '\\')" --out "$W\\$NAME\\merged_16bit" --gpu "$GM" \
      > "$NAME.merge.log" 2>&1 || { log "merge FAILED"; exit 2; }
fi
log "merged OK - grading"
$PY post_train.py --run "$W\\$NAME" --gpu "$GS" --port 8097 > "$NAME.post.log" 2>&1 \
    || { log "post_train FAILED (see $NAME.post.log)"; exit 3; }
rm -f "$NAME/gguf/superbrain1-f16.gguf"
log "VERDICT: $(grep -a '"PASS"' "$NAME/verdict.json" | tr -d ' ,')"
log "GRADE DONE"
