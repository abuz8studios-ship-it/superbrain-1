# -*- coding: utf-8 -*-
"""a10_eval.py - Packet A10 science exam (packet_a10_exam.jsonl, never trained) against a running llama-server.
Scoring = train\\check_a10.score (numeric, relative tolerance tol_rel) applied to the text after the LAST
"الجواب:" line when present, else to the whole output (check_a10 takes the last number).
Usage: python a10_eval.py --url http://127.0.0.1:8095 --name run6_math_grpo [--limit N] [--workers 8]
Writes proof\\a10_<name>.json (acc + per-subject + per-topic) and prints accuracy."""
import argparse, json, os, sys, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"E:\ABU\02_PROJECTS\AHMAD-1"
sys.path.insert(0, os.path.join(ROOT, "train"))
from check_a10 import score  # noqa: E402

STOP = ["\nسؤال:", "\nالمهمة:", "\n\n\n"]


def ask(url, prompt, n_predict):
    body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0,
                       "cache_prompt": True, "stop": STOP}).encode("utf-8")
    req = urllib.request.Request(url + "/completion", data=body, headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read().decode("utf-8"))["content"]
        except Exception as e:  # transient server hiccup: retry, then count as wrong
            err = e
    return f"<ERROR {err}>"


def judge(out, item):
    if "الجواب:" in out:
        seg = out.rsplit("الجواب:", 1)[1].split("\n", 1)[0]
        if score(seg, item):
            return True
        return False
    return score(out, item)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8095"); ap.add_argument("--name", required=True)
    ap.add_argument("--exam", default=os.path.join(ROOT, "train", "packet_a10_exam.jsonl"))
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-predict", type=int, default=512)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.exam, encoding="utf-8") if l.strip()]
    if a.limit: items = items[:a.limit]
    with ThreadPoolExecutor(a.workers) as ex:
        outs = list(ex.map(lambda x: ask(a.url, x["prompt"], a.n_predict), items))
    ok = [judge(o, x) for o, x in zip(outs, items)]
    n_err = sum(o.startswith("<ERROR") for o in outs)
    by_s, by_t = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for x, k in zip(items, ok):
        for d, key in ((by_s, x.get("subject", "?")), (by_t, x.get("subject", "?") + "/" + x.get("topic", "?"))):
            d[key][0] += int(k); d[key][1] += 1
    acc = sum(ok) / len(ok)
    res = {"name": a.name, "exam": a.exam, "n": len(ok), "correct": sum(ok), "acc": round(acc, 4),
           "server_errors": n_err,
           "per_subject": {k: {"n": v[1], "correct": v[0], "acc": round(v[0] / v[1], 4)} for k, v in sorted(by_s.items())},
           "per_topic": {k: {"n": v[1], "correct": v[0], "acc": round(v[0] / v[1], 4)} for k, v in sorted(by_t.items())},
           "samples": [{"q": x["prompt"][-200:], "gold": x["answer"], "unit": x.get("unit"), "out": o[:500], "ok": k}
                       for x, o, k in list(zip(items, outs, ok))[:25]]}
    os.makedirs(os.path.join(ROOT, "proof"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "proof", f"a10_{a.name}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"A10 {a.name}: {sum(ok)}/{len(ok)} = {acc:.2%} (server errors {n_err})")
    for k, v in res["per_subject"].items():
        print(f"  {k}: {v['correct']}/{v['n']} = {v['acc']:.2%}")


if __name__ == "__main__":
    main()
