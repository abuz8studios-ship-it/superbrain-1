#!/usr/bin/env bash
# probe_models.sh - serve each local model on GPU 1 in turn and measure how much
# of the root task it already knows on held-out dev roots (no training).
set -u
T=/e/ABU/02_PROJECTS/AHMAD-1/train
LLAMA=/e/ABU/tools/llama.cpp/build-cuda/bin/llama-server.exe
PY=E:/ABU/venvs/sb1/Scripts/python.exe
export PYTHONIOENCODING=utf-8
cd "$T" || exit 1
OUT=evals/probe_models.txt
: > "$OUT"
probe() {  # name gguf port
  echo "##### $1" >> "$OUT"
  CUDA_VISIBLE_DEVICES=1 "$LLAMA" -m "$2" -ngl 99 -c 4096 -np 4 --host 127.0.0.1 --port "$3" \
      > "evals/probe_$1.server.log" 2>&1 &
  SRV=$!
  for i in $(seq 1 150); do curl -sf "http://127.0.0.1:$3/health" > /dev/null && break; sleep 2; done
  if curl -sf "http://127.0.0.1:$3/health" > /dev/null; then
    $PY probe_server.py --url "http://127.0.0.1:$3" --name "$1" >> "$OUT" 2>&1
  else
    echo "  server failed to start" >> "$OUT"
  fi
  kill $SRV 2>/dev/null; sleep 4
}
probe qwen3.8-27b  "/e/ABU/MODELS/llm/qwen3.8-27b/Qwen3.8-27B-UD-Q5_K_XL.gguf" 8211
probe muse-30b     "/e/ABU/MODELS/llm/muse-glimmer-30b-Q5/Muse-Glimmer-30B-UD-Q5_K_L.gguf" 8212
probe nemotron-30b "/e/ABU/MODELS/llm/nemotron-3.5-lightning-30b/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-IQ4_NL.gguf" 8213
probe ornith-35b   "/e/ABU/MODELS/llm/ornith-1.5-35b-a3b/Ornith-1.5-35B-A3B-Q4_K_M.gguf" 8214
echo "ALL DONE" >> "$OUT"
