# -*- coding: utf-8 -*-
"""math_eval.py - Packet B math exam (Arabic GSM8K test, never trained) against a running llama-server.
Usage: python math_eval.py --url http://127.0.0.1:8097 --name rl1f-q5_k_m [--limit N] [--workers 8]
Writes proof\\math_<name>.json and prints accuracy."""
import argparse, json, os, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"E:\ABU\02_PROJECTS\AHMAD-1"
AR_DIG = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def ask(url, prompt):
    body = json.dumps({"prompt": prompt, "n_predict": 400, "temperature": 0,
                       "cache_prompt": True, "stop": ["\nسؤال:", "\n\n\n"]}).encode("utf-8")
    req = urllib.request.Request(url + "/completion", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))["content"]


def parse(text):
    t = (text or "").translate(AR_DIG).replace(",", "").replace("٬", "")
    seg = t.split("الجواب:", 1)[1] if "الجواب:" in t else t
    nums = re.findall(r"-?\d+(?:\.\d+)?", seg)
    if not nums: return None
    n = nums[0] if "الجواب:" in t else nums[-1]
    return str(float(n)).rstrip("0").rstrip(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8097"); ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(os.path.join(ROOT, "train", "packet_b_math.jsonl"), encoding="utf-8")]
    if a.limit: items = items[:a.limit]
    with ThreadPoolExecutor(a.workers) as ex:
        outs = list(ex.map(lambda x: ask(a.url, x["prompt"]), items))
    ok = [parse(o) == str(float(x["answer"])).rstrip("0").rstrip(".") for o, x in zip(outs, items)]
    acc = sum(ok) / len(ok)
    res = {"name": a.name, "n": len(ok), "correct": sum(ok), "acc": round(acc, 4),
           "samples": [{"q": x["prompt"][-200:], "gold": x["answer"], "out": o[:400], "ok": k}
                       for x, o, k in list(zip(items, outs, ok))[:20]]}
    os.makedirs(os.path.join(ROOT, "proof"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "proof", f"math_{a.name}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"MATH {a.name}: {sum(ok)}/{len(ok)} = {acc:.2%}")


if __name__ == "__main__":
    main()
