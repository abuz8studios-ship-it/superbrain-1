# -*- coding: utf-8 -*-
"""ship_v2.py - publish SUPERBRAIN-1 v2 (run6) + the Packet B exam dataset (public)."""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from huggingface_hub import HfApi
R = r"E:\ABU\02_PROJECTS\AHMAD-1"
RUN = os.path.join(R, "train", "run6_math_grpo")
api = HfApi()

# --- dataset: Packet B exam
DS = "abuz8studios/arabic-sarf-packet-b"
api.create_repo(DS, repo_type="dataset", private=False, exist_ok=True)
api.upload_file(path_or_fileobj=os.path.join(R, "PACKET_B_README.md"), path_in_repo="README.md",
                repo_id=DS, repo_type="dataset", commit_message="dataset card")
api.upload_file(path_or_fileobj=os.path.join(R, "train", "packet_b.jsonl"), path_in_repo="packet_b.jsonl",
                repo_id=DS, repo_type="dataset", commit_message="Packet B exam (1,343 items)")
print("DATASET", f"https://huggingface.co/datasets/{DS}", flush=True)

# --- model: v2 next to v1
M = "abuz8studios/SUPERBRAIN-1"
info = api.model_info(M)
if info.private:
    sys.exit("model repo is private - refusing")
api.upload_file(path_or_fileobj=os.path.join(RUN, "verdict.json"), path_in_repo="eval/verdict_run6.json",
                repo_id=M, commit_message="v2 eval verdict (run6)")
api.upload_file(path_or_fileobj=os.path.join(R, "proof", "math_run6_math_grpo-q5_k_m.json"),
                path_in_repo="eval/math_arabic_gsm8k_run6.json", repo_id=M,
                commit_message="v2 Arabic GSM8K results (run6)")
g = os.path.join(RUN, "gguf", "superbrain1-q5_k_m.gguf")
print("uploading gguf", round(os.path.getsize(g) / 1e9, 2), "GB", flush=True)
api.upload_file(path_or_fileobj=g, path_in_repo="superbrain1-run6-q5_k_m.gguf", repo_id=M,
                commit_message="add v2 GGUF (run6: roots + Quran + Arabic math)")
api.upload_file(path_or_fileobj=os.path.join(R, "HF_MODEL_CARD_v2.md"), path_in_repo="README.md",
                repo_id=M, commit_message="model card v2")
print("SHIPPED", f"https://huggingface.co/{M}", flush=True)
