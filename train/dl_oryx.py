# -*- coding: utf-8 -*-
"""dl_oryx.py - fetch QCRI/Fanar-2-Oryx-IVU (Arabic VL, Qwen2.5-VL-7B base, Apache-2.0) into E:\\ABU\\MODELS\\hf.
Resumable: snapshot_download skips files already complete. Exit 0 only when all 4 shards have the hub size."""
import os, sys
from huggingface_hub import snapshot_download, HfApi

REPO = "QCRI/Fanar-2-Oryx-IVU"
DST = r"E:\ABU\MODELS\hf\Fanar-2-Oryx-IVU"

snapshot_download(REPO, local_dir=DST, max_workers=4)
info = HfApi().model_info(REPO, files_metadata=True)
bad = []
for s in info.siblings:
    p = os.path.join(DST, s.rfilename)
    if s.size and (not os.path.exists(p) or os.path.getsize(p) != s.size):
        bad.append((s.rfilename, s.size, os.path.getsize(p) if os.path.exists(p) else None))
if bad:
    sys.exit(f"INCOMPLETE {bad}")
print("DONE", DST, flush=True)
