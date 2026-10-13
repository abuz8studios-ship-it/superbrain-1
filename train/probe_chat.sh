#!/usr/bin/env bash
# probe_chat.sh <name> <gguf> <port> - serve one chat/reasoning model on GPU 1 with
# its chat template (thinking off) and measure zero-shot held-out dev accuracy.
set -u
T=/e/ABU/02_PROJECTS/AHMAD-1/train
LLAMA=/e/ABU/tools/llama.cpp/build-cuda/bin/llama-server.exe
PY=E:/ABU/venvs/sb1/Scripts/python.exe
export PYTHONIOENCODING=utf-8
cd "$T" || exit 1
NAME=$1; GGUF=$2; PORT=$3
OUT=evals/probe_chat_$NAME.txt
: > "$OUT"
CUDA_VISIBLE_DEVICES=1 "$LLAMA" -m "$GGUF" -ngl 99 -c 4096 -np 4 --jinja --host 127.0.0.1 --port "$PORT" \
    > "evals/probe_chat_$NAME.server.log" 2>&1 &
SRV=$!
for i in $(seq 1 150); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; sleep 2; done
if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
  $PY probe_chat.py --url "http://127.0.0.1:$PORT" --name "$NAME" --limit 600 >> "$OUT" 2>&1
else
  echo "server failed to start" >> "$OUT"
fi
kill $SRV 2>/dev/null
echo "ALL DONE" >> "$OUT"
