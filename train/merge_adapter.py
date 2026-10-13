# -*- coding: utf-8 -*-
"""merge_adapter.py - merge a LoRA adapter into a 16-bit HF model so that
post_train.py (GGUF export + gates) and train_grpo2.py (GRPO base) can use it.

Usage: python merge_adapter.py --adapter <adapter_dir> --out <run_dir>\\merged_16bit [--gpu 0]
The adapter's own adapter_config.json names its base model."""
import argparse, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--gpu", default="0")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(
    args.adapter, max_seq_length=1024, load_in_4bit=True, fast_inference=False)
model.save_pretrained_merged(args.out, tok, save_method="merged_16bit")
if not os.path.exists(os.path.join(args.out, "config.json")):
    sys.exit("merge failed: no config.json in " + args.out)
print("MERGED", args.out, flush=True)
