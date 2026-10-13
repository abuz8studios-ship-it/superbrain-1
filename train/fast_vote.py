# -*- coding: utf-8 -*-
"""fast_vote.py - does sampling several answers and taking the majority beat greedy?
(test-time self-consistency; no new training data). Same dev items, parser and root
identity rule as fast_eval.py. Reports greedy, majority@K, and oracle@K (any sample right).
Usage: python fast_vote.py <model_dir> [--k 16] [--temp 0.7] [--limit 1200] [--gpu 1]"""
import argparse, collections, json, os, random, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"E:\ABU\02_PROJECTS\AHMAD-1"
sys.path.insert(0, os.path.join(ROOT, "train"))
from fast_eval import score, root_key
import bakeoff_arabic as B

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--limit", type=int, default=1200)
ap.add_argument("--gpu", default="1")
ap.add_argument("--batch", type=int, default=16)
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import torch
from unsloth import FastLanguageModel

rows = [json.loads(l) for l in open(os.path.join(ROOT, "train", "dev_root.jsonl"), encoding="utf-8")]
random.Random(9).shuffle(rows)
rows = rows[:args.limit]
model, tok = FastLanguageModel.from_pretrained(args.model, max_seq_length=1024,
                                               load_in_4bit=True, fast_inference=False)
FastLanguageModel.for_inference(model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

greedy_ok, vote_ok, oracle_ok, agree = [], [], [], []
t0 = time.time()
order = sorted(range(len(rows)), key=lambda i: len(rows[i]["prompt"]))
res = {}
for s in range(0, len(order), args.batch):
    idx = order[s:s + args.batch]
    enc = tok([rows[i]["prompt"] for i in idx], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=10, do_sample=False, pad_token_id=tok.pad_token_id)
        smp = model.generate(**enc, max_new_tokens=10, do_sample=True, temperature=args.temp,
                             top_p=1.0, top_k=0, num_return_sequences=args.k,
                             pad_token_id=tok.pad_token_id)
    L = enc["input_ids"].shape[1]
    for j, i in enumerate(idx):
        gold = rows[i]["answer"]
        graw = tok.decode(g[j][L:], skip_special_tokens=True)
        samples = [tok.decode(smp[j * args.k + q][L:], skip_special_tokens=True) for q in range(args.k)]
        keys = [root_key(B.parse_root_output(x)) for x in samples]
        cnt = collections.Counter(k for k in keys if k)
        top, topn = (cnt.most_common(1)[0] if cnt else ("", 0))
        res[i] = (score(graw, gold), top == root_key(gold), any(k == root_key(gold) for k in keys),
                  topn / args.k, rows[i]["cls"])
n = len(rows)
acc = lambda col, f=lambda r: True: sum(r[col] for r in res.values() if f(r)) / max(1, sum(1 for r in res.values() if f(r)))
print(f"=== {os.path.basename(args.model.rstrip(chr(92)))}: {n} dev items, K={args.k}, T={args.temp}, {time.time()-t0:.0f}s ===")
print(f"  greedy            {100*acc(0):5.1f}%")
print(f"  majority vote@{args.k}  {100*acc(1):5.1f}%   (gain {100*(acc(1)-acc(0)):+.1f})")
print(f"  oracle@{args.k} (any right) {100*acc(2):5.1f}%   <- ceiling if a perfect verifier existed")
for c in sorted({r[4] for r in res.values()}):
    f = lambda r, c=c: r[4] == c
    print(f"  {c:<12} greedy {100*acc(0, f):5.1f}%  vote {100*acc(1, f):5.1f}%  oracle {100*acc(2, f):5.1f}%")
# confidence: when the vote is unanimous-ish, how often is it right?
hi = [r for r in res.values() if r[3] >= 0.75]
print(f"  votes with >=75% agreement: {len(hi)}/{n} ({100*len(hi)/n:.0f}%) -> accuracy {100*sum(r[1] for r in hi)/max(1,len(hi)):.1f}%")
