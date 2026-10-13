# -*- coding: utf-8 -*-
"""hf_make_public.py - flip abuz8studios/SUPERBRAIN-1 to public, but only if the
repo is complete (card + eval verdict + the q5_k_m GGUF fully uploaded)."""
import sys
from huggingface_hub import HfApi

REPO = "abuz8studios/SUPERBRAIN-1"
api = HfApi()
info = api.model_info(REPO, files_metadata=True)
files = {s.rfilename: (s.size or 0) for s in info.siblings}
print("files:", {k: round(v / 1e9, 2) for k, v in files.items()})
if (files.get("superbrain1-q5_k_m.gguf", 0) < 6.0e9
        or "README.md" not in files or "eval/verdict.json" not in files):
    sys.exit("repo incomplete - NOT making public")
api.update_repo_settings(REPO, private=False)
after = api.model_info(REPO)
print("private =", after.private, "->", f"https://huggingface.co/{REPO}")
