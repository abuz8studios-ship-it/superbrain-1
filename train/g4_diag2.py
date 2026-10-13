# -*- coding: utf-8 -*-
import os, sys, torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
from transformers import AutoTokenizer, AutoModelForImageTextToText
M = sys.argv[1]
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForImageTextToText.from_pretrained(M, dtype=torch.bfloat16, device_map={"": 0})
p = "استخرج جذر كلمة: وعفا\nالجذر:"
c = " عفو\n"
ids = tok(p + c, return_tensors="pt").input_ids.to(0)
pl = len(tok(p).input_ids)
labels = ids.clone(); labels[:, :pl] = -100
model.eval()
with torch.no_grad():
    print("EVAL_LOSS", float(model(input_ids=ids, labels=labels).loss), flush=True)
    print("EVAL_LOSS_nocache", float(model(input_ids=ids, labels=labels, use_cache=False).loss), flush=True)
model.train()
print("TRAIN_LOSS_cache", float(model(input_ids=ids, labels=labels, use_cache=True).loss.detach()), flush=True)
print("TRAIN_LOSS_nocache", float(model(input_ids=ids, labels=labels, use_cache=False).loss.detach()), flush=True)
g = model.generate(tok(p, return_tensors="pt").input_ids.to(0), max_new_tokens=10, do_sample=False)
print("GEN", repr(tok.decode(g[0][pl:])), flush=True)
