# -*- coding: utf-8 -*-
import os, sys, torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM
M = r"E:\ABU\MODELS\hf\gemma-4-E4B-it"
tok = AutoTokenizer.from_pretrained(M)
mode = sys.argv[1] if len(sys.argv) > 1 else "itt"
cls = AutoModelForImageTextToText if mode == "itt" else AutoModelForCausalLM
model = cls.from_pretrained(M, dtype=torch.bfloat16, device_map={"": 0})
model.eval()
print("CLASS", type(model).__name__, flush=True)
p = "استخرج جذر كلمة: وعفا\nالجذر:"
c = " عفو\n"
ids = tok(p + c, return_tensors="pt").input_ids.to(0)
pl = len(tok(p).input_ids)
labels = ids.clone(); labels[:, :pl] = -100
with torch.no_grad():
    out = model(input_ids=ids, labels=labels)
print("LOSS", float(out.loss), flush=True)
lg = out.logits[0]
print("LOGIT_STATS", float(lg.abs().max()), float(lg.std()), flush=True)
g = model.generate(tok(p, return_tensors="pt").input_ids.to(0), max_new_tokens=12, do_sample=False)
print("GEN", repr(tok.decode(g[0][pl:])), flush=True)
model.train()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
out2 = model(input_ids=ids, labels=labels)
print("LOSS_TRAINMODE_GC", float(out2.loss), flush=True)
