# -*- coding: utf-8 -*-
"""probe_chat.py - held-out dev accuracy of a CHAT/reasoning model served by
llama-server, with thinking disabled: how much of the root task does it know
zero-shot? Same dev items and scorer as fast_eval.py.
Usage: python probe_chat.py --url http://127.0.0.1:8211 --name qwen27 [--limit N]"""
import argparse, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"E:\ABU\02_PROJECTS\AHMAD-1"
sys.path.insert(0, os.path.join(ROOT, "train"))
from fast_eval import score


INSTR = "\n\n(أجب بالجذر فقط، ثلاثة أحرف، دون أي شرح أو كلمات إضافية.)"


def extract(text):
    """first non-empty line; drop any 'الجذر:' style label before the root"""
    line = next((l for l in (text or "").splitlines() if l.strip()), "")
    line = line.replace("*", "").replace("«", "").replace("»", "").strip()
    return line.rsplit(":", 1)[-1].strip() if ":" in line else line


def ask(url, prompt):
    body = json.dumps({"messages": [{"role": "user", "content": prompt + INSTR}],
                       "max_tokens": 16, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode("utf-8")
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"] or ""


ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--workers", type=int, default=4)
args = ap.parse_args()
rows = [json.loads(l) for l in open(os.path.join(ROOT, "train", "dev_root.jsonl"), encoding="utf-8")]
if args.limit:
    import random
    random.Random(5).shuffle(rows)
    rows = rows[:args.limit]
t0 = time.time()
with ThreadPoolExecutor(args.workers) as ex:
    raws = list(ex.map(lambda r: ask(args.url, r["prompt"]), rows))
ok = [score(extract(r), x["answer"]) for r, x in zip(raws, rows)]


def acc(f):
    s = [o for o, x in zip(ok, rows) if f(x)]
    return f"{100*sum(s)/max(1, len(s)):5.1f}% (n={len(s)})"


print(f"=== {args.name} (chat, thinking off): {len(rows)} dev items, {time.time()-t0:.0f}s ===")
print("  ALL     ", acc(lambda x: True))
print("  fewshot ", acc(lambda x: x["mode"] == "fewshot"))
print("  bare    ", acc(lambda x: x["mode"] == "bare"))
for c in sorted({x["cls"] for x in rows}):
    print(f"  {c:<12}", acc(lambda x, c=c: x["cls"] == c))
print("  sample outputs:", [(x["word"], x["answer"], r.strip()[:16]) for x, r in list(zip(rows, raws))[:6]])
