# -*- coding: utf-8 -*-
"""a11_eval.py - score a GGUF on Packet A11 (Arabic-instruction Python coding exam, packet_a11_exam.jsonl,
1,026 items, never trained) via llama.cpp llama-server (never Ollama).

Scoring = train\\a11_score.score() unchanged (exact output match for predict_output; hidden-assert subprocess
run for write_function / fix_bug / complete_function). Prompts already end in "الجواب:" (same raw
prompt -> completion format as training), so they go to llama-server's /completion endpoint, greedy, like
a10_eval.py / packet_b_eval.py. --chat switches to /v1/chat/completions instead.

Usage:
  python a11_eval.py --gguf train\\run6_math_grpo\\gguf\\superbrain1-q5_k_m.gguf --gpu 1 --port 8096
  python a11_eval.py --url http://127.0.0.1:8095 --name run6_math_grpo     (already-running server)
  python a11_eval.py --selftest            (CPU only, no server: gold completions + controls through score())
Writes JSON (default proof\\a11_<name>.json): overall acc + per task type
(predict-output, write-function, fix-bug, complete), n, model, date, server_errors, samples."""
import argparse, json, os, random, socket, subprocess, sys, time, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
ROOT = r"E:\ABU\02_PROJECTS\AHMAD-1"
BIN = r"E:\ABU\tools\llama.cpp\build-cuda\bin"
sys.path.insert(0, os.path.join(ROOT, "train"))
from a11_score import score  # noqa: E402  (scoring logic lives there; do not duplicate)

EXAM = os.path.join(ROOT, "train", "packet_a11_exam.jsonl")
TYPE = {"predict_output": "predict-output", "write_function": "write-function",
        "fix_bug": "fix-bug", "complete_function": "complete"}
STOP = ["\nسؤال:", "\nالمهمة:"]


def ask(url, prompt, n_predict, chat=False):
    if chat:
        path = "/v1/chat/completions"
        body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": n_predict,
                "temperature": 0, "stop": STOP}
    else:
        path = "/completion"
        body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "top_k": 1,
                "cache_prompt": True, "stop": STOP}
    req = urllib.request.Request(url + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    err = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read().decode("utf-8"))
            return d["choices"][0]["message"]["content"] if chat else d["content"]
        except Exception as e:  # transient hiccup: retry, then count as wrong
            err = e
            time.sleep(1)
    return f"<ERROR {err}>"


def grade(items, outs, workers, timeout):
    """Run a11_score.score over (item, output) pairs in parallel (each code item = one CPU subprocess)."""
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda p: bool(score(p[1], p[0], timeout=timeout)), zip(items, outs)))


def summarize(items, outs, ok):
    tot, cor = defaultdict(int), defaultdict(int)
    for x, k in zip(items, ok):
        t = TYPE.get(x["kind"], x["kind"])
        tot[t] += 1
        cor[t] += int(k)
    n = len(ok)
    return {"n": n, "correct": sum(ok), "acc": round(sum(ok) / max(n, 1), 4),
            "per_type": {t: {"n": tot[t], "correct": cor[t], "acc": round(cor[t] / tot[t], 4)}
                         for t in TYPE.values() if tot[t]},
            "server_errors": sum(o.startswith("<ERROR") for o in outs)}


def fmt(tag, s):
    lines = [f"{tag}: {s['correct']}/{s['n']} = {s['acc']:.2%}"]
    lines += [f"  {t}: {v['correct']}/{v['n']} = {v['acc']:.2%}" for t, v in s["per_type"].items()]
    return "\n".join(lines)


def selftest(items, a):
    """No server. Same grading path as a real run, fed (1) gold completions, (2) controls."""
    gold = [x["completion"] for x in items]
    rot = gold[1:] + gold[:1]                      # each item gets its neighbour's completion
    rng = random.Random(0)
    shuf = gold[:]
    rng.shuffle(shuf)
    res = {"mode": "selftest", "date": time.strftime("%Y-%m-%d %H:%M"), "exam": a.exam}
    for tag, outs in (("gold", gold), ("shuffled", shuf), ("rotated", rot), ("empty", [""] * len(items))):
        t0 = time.time()
        s = summarize(items, outs, grade(items, outs, a.workers, a.timeout))
        s["secs"] = round(time.time() - t0, 1)
        res[tag] = s
        print(fmt(f"selftest[{tag}]", s), flush=True)
    return res


def start_server(a, log_path):
    with socket.socket() as sk:  # a foreign server here would get scored
        if sk.connect_ex(("127.0.0.1", a.port)) == 0:
            sys.exit(f"port {a.port} already in use - pick another --port (or pass --url)")
    slog = open(log_path, "w", encoding="utf-8")
    srv = subprocess.Popen(
        [os.path.join(BIN, "llama-server.exe"), "-m", a.gguf, "-ngl", "99",
         "-c", str(a.ctx * a.workers), "-np", str(a.workers),
         "--host", "127.0.0.1", "--port", str(a.port)],
        stdout=slog, stderr=subprocess.STDOUT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(a.gpu)})
    url = f"http://127.0.0.1:{a.port}"
    for _ in range(240):
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as r:
                if r.status == 200:
                    print("server up", url, flush=True)
                    return srv, slog, url
        except Exception:
            pass
        if srv.poll() is not None:
            slog.close()
            sys.exit("llama-server died; see " + log_path)
        time.sleep(2)
    stop_server(srv, slog)
    sys.exit("llama-server never became healthy; see " + log_path)


def stop_server(srv, slog):
    """Stops ONLY the llama-server this script launched."""
    if srv is not None and srv.poll() is None:
        srv.terminate()
        try:
            srv.wait(30)
        except subprocess.TimeoutExpired:
            srv.kill()
    if slog is not None:
        slog.close()


def model_name(a):
    if a.name:
        return a.name
    if a.gguf:  # ...\train\run6_math_grpo\gguf\superbrain1-q5_k_m.gguf -> run6_math_grpo-q5_k_m
        p = os.path.abspath(a.gguf)
        run = os.path.basename(os.path.dirname(os.path.dirname(p)))
        q = os.path.splitext(os.path.basename(p))[0].replace("superbrain1-", "")
        return f"{run}-{q}"
    return "selftest" if a.selftest else "url_model"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", help="GGUF to serve with llama-server (required unless --url or --selftest)")
    ap.add_argument("--gpu", default="0", help="GPU index -> CUDA_VISIBLE_DEVICES for llama-server")
    ap.add_argument("--port", type=int, default=8096)
    ap.add_argument("--url", help="use an already-running llama-server instead of launching one")
    ap.add_argument("--limit", type=int, default=0, help="first N items only (default 0 = all 1026)")
    ap.add_argument("--out", help="output JSON (default proof\\a11_<name>.json)")
    ap.add_argument("--name", help="model label (default derived from --gguf path)")
    ap.add_argument("--exam", default=EXAM)
    ap.add_argument("--workers", type=int, default=8, help="parallel requests / server slots / scorer procs")
    ap.add_argument("--ctx", type=int, default=4096, help="context per server slot")
    ap.add_argument("--n-predict", type=int, default=768)
    ap.add_argument("--timeout", type=float, default=5.0, help="per-item test timeout (a11_score default)")
    ap.add_argument("--chat", action="store_true", help="use /v1/chat/completions instead of raw /completion")
    ap.add_argument("--selftest", action="store_true", help="CPU-only: gold + control answers, no server")
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.exam, encoding="utf-8") if l.strip()]
    if a.limit:
        items = items[:a.limit]
    name = model_name(a)
    out = a.out or os.path.join(ROOT, "proof", f"a11_{name}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    if a.selftest:
        res = selftest(items, a)
        json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("wrote", out)
        ok = res["gold"]["acc"] >= 0.99 and res["shuffled"]["acc"] < 0.10 and res["empty"]["acc"] == 0
        print("SELFTEST", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    if not a.url and not a.gguf:
        ap.error("need --gguf (launch llama-server), --url (running server) or --selftest")
    if a.gguf and not a.url and not os.path.exists(a.gguf):
        sys.exit(f"gguf not found: {a.gguf}")
    srv = slog = None
    url = a.url.rstrip("/") if a.url else None
    t0 = time.time()
    try:
        if not url:
            srv, slog, url = start_server(a, os.path.join(ROOT, "proof", f"server_a11_{name}.log"))
        with ThreadPoolExecutor(a.workers) as ex:
            outs = list(ex.map(lambda x: ask(url, x["prompt"], a.n_predict, a.chat), items))
    finally:
        stop_server(srv, slog)
    gen_secs = round(time.time() - t0, 1)
    ok = grade(items, outs, a.workers, a.timeout)
    res = {"model": name, "gguf": os.path.abspath(a.gguf) if a.gguf else None, "url": a.url,
           "date": time.strftime("%Y-%m-%d %H:%M"), "exam": a.exam,
           "endpoint": "chat" if a.chat else "completion", "n_predict": a.n_predict, "gen_secs": gen_secs,
           **summarize(items, outs, ok),
           "samples": [{"kind": x["kind"], "q": x["prompt"][-200:], "out": o[:600], "ok": k}
                       for x, o, k in list(zip(items, outs, ok))[:25]],
           "failures": [{"kind": x["kind"], "template": x.get("template"), "out": o[:400]}
                        for x, o, k in zip(items, outs, ok) if not k][:40]}
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(fmt(f"A11 {name}", res) + f"\n  server errors: {res['server_errors']}\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
