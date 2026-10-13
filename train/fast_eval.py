# -*- coding: utf-8 -*-
"""fast_eval.py - held-out ROOT accuracy of a LoRA adapter / merged model,
scored with the project's own parser (bakeoff_arabic.parse_root_output) and
root-identity rule (hamza/alef folding, ى->ي), WITHOUT a GGUF export.

  dev      train\\dev_root.jsonl   roots held out of training: the signal used to
                                   ITERATE (pick recipes, stop training)
  packetb  train\\packet_b.jsonl   root items, 258 banned roots: only to GRADE a
                                   finished candidate - never to tune on

Usage: python fast_eval.py <adapter_or_model_dir> [--sets dev,packetb]
                           [--gpu 0] [--out result.json] [--limit N]"""
import argparse, collections, json, os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"E:\ABU\02_PROJECTS\AHMAD-1"
sys.path.insert(0, os.path.join(ROOT, "src"))
import bakeoff_arabic as B

HAMZAS = "اأإآءؤئ"
TARGET = 0.90


def root_key(r):
    r = B.ar_letters(r)
    return "".join("ء" if ch in HAMZAS else ("ي" if ch == "ى" else ch) for ch in r)


def score(raw, answer):
    return root_key(B.parse_root_output(raw)) == root_key(answer)


def trace_root(raw):
    """root from a derivation trace: text after the last 'الجذر:' on line 1"""
    line = next((l for l in (raw or "").splitlines() if l.strip()), "")
    return line.rsplit("الجذر:", 1)[1] if "الجذر:" in line else ""


def score_trace(raw, answer):
    return root_key(trace_root(raw)) == root_key(answer)


def root_class(k):
    if len(k) != 3:
        return "quad" if len(k) == 4 else "other"
    if k[1] == k[2]:
        return "doubled"
    if "ء" in k:
        return "hamzated"
    if k[0] in "وي":
        return "assimilated"
    if k[1] in "وي":
        return "hollow"
    if k[2] in "وي":
        return "defective"
    return "sound"


TRACE_PROMPT = lambda w: f"كلمة: {w}\nالتحليل:"


def score_morph(task, raw, answer):
    """verb-form skills, scored exactly like the official gates (a2_reward.correct)"""
    if task == "morph":
        return B.parse_verb_output(raw) == B.ar_letters(answer)
    first = next((l for l in (raw or "").splitlines() if l.strip()), "").strip()
    return B.norm_ar(first) == B.norm_ar(answer)


def load(which, limit, trace=False):
    if which == "devmorph":
        rows = [json.loads(l) for l in open(os.path.join(ROOT, "train", "dev_morph.jsonl"),
                                            encoding="utf-8")]
        out = [(r["prompt"], r["answer"], r["task"], "form%d" % r["form"]) for r in rows]
        return out[:limit] if limit else out
    if which == "dev":
        rows = [json.loads(l) for l in open(os.path.join(ROOT, "train", "dev_root.jsonl"),
                                            encoding="utf-8")]
        if trace:      # one question per word (dev has fewshot+bare rows per word)
            rows = [r for r in rows if r["mode"] == "bare"]
            out = [(TRACE_PROMPT(r["word"]), r["answer"], "trace", r["cls"]) for r in rows]
        else:
            out = [(r["prompt"], r["answer"], r["mode"], r["cls"]) for r in rows]
    else:
        rows = [json.loads(l) for l in open(os.path.join(ROOT, "train", "packet_b.jsonl"),
                                            encoding="utf-8")]
        out = []
        for r in rows:
            if not r["section"].startswith("root"):
                continue
            cls = r.get("root_class") or root_class(root_key(r["answer"]))
            if trace:
                out.append((TRACE_PROMPT(r["word"]), r["answer"], "trace", cls))
            else:
                out.append((r["prompt"], r["answer"], "fewshot", cls))
                out.append((r["prompt_bare"], r["answer"], "bare", cls))
    return out[:limit] if limit else out


def selfcheck():
    """does this scorer reproduce the official 'ok' flags on recorded outputs?"""
    p = os.path.join(ROOT, "proof", "packetb_run2a_pilot-q5_k_m.json")
    items = [i for i in json.load(open(p, encoding="utf-8"))["items"]
             if i["section"].startswith("root")]
    agree = sum(score(i["raw"], i["answer"]) == i["ok"] for i in items)
    print(f"scorer parity with the official Packet B scorer: {agree}/{len(items)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--sets", default="dev,packetb")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--trace", action="store_true",
                    help="derivation-trace prompts (التحليل:), root parsed from the trace")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    selfcheck()
    import torch
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        args.model, max_seq_length=1024, load_in_4bit=True, fast_inference=False)
    FastLanguageModel.for_inference(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    result = {"model": args.model, "sets": {}}
    for which in args.sets.split(","):
        items = load(which, args.limit, args.trace)
        order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
        raws = [None] * len(items)
        t0 = time.time()
        for s in range(0, len(order), args.batch):
            idx = order[s:s + args.batch]
            enc = tok([items[i][0] for i in idx], return_tensors="pt",
                      padding=True).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=(64 if args.trace else 12),
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            for i, g in zip(idx, gen):
                raws[i] = tok.decode(g[enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        sc = score_trace if args.trace else score
        rows = [{"mode": m, "cls": c, "answer": a, "raw": r,
                 "ok": score_morph(m, r, a) if which == "devmorph" else sc(r, a)}
                for (p, a, m, c), r in zip(items, raws)]
        acc = lambda f: (sum(x["ok"] for x in rows if f(x)) /
                         max(1, sum(1 for x in rows if f(x))))
        n = lambda f: sum(1 for x in rows if f(x))
        summ = {"ALL": acc(lambda x: True), "fewshot": acc(lambda x: x["mode"] == "fewshot"),
                "bare": acc(lambda x: x["mode"] == "bare"), "n": len(rows)}
        for c in sorted({x["cls"] for x in rows}):
            summ["class:" + c] = acc(lambda x, c=c: x["cls"] == c)
            summ["n:" + c] = n(lambda x, c=c: x["cls"] == c)
        result["sets"][which] = {"summary": summ, "rows": rows}
        print(f"\n=== {which}: {len(rows)} items, {time.time()-t0:.0f}s ===", flush=True)
        for k, v in summ.items():
            if k == "n" or k.startswith("n:"):
                continue
            mark = "  " if v >= TARGET else "<<"
            extra = f"  (n={summ.get('n:' + k[6:])})" if k.startswith("class:") else ""
            print(f"  {k:<20}{100*v:6.1f}%  {mark}{extra}", flush=True)
    if args.out:
        json.dump(result, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
        print("saved", args.out)


if __name__ == "__main__":
    main()
