# -*- coding: utf-8 -*-
"""a11_eval_v2.py - Packet A11 (1,026 items) harness v2. a11_eval.py (v1) and train\\a11_score.py are unchanged.

v2 changes vs v1 (ported from src\\a11_zero_check.py V2/V3, which proved fix-bug 111/114, complete 73/114):
  * write_function / fix_bug / complete_function: prompt + "\\n```python\\n" prefill; the scored text is
    "```python\\n" + model output (a closing ``` is appended if the model never closed the fence).
  * complete_function: if the (prefilled) output contains no "def ", its first line (text before any ```)
    is spliced into the "____" blank of the prompt's last ```python block (zero_check splice()).
  * predict_output: prompt unchanged; format-only normalisation before the unchanged a11_score.score():
    an unclosed leading ``` fence line / leading "الناتج:" label is stripped. Answer must still match exactly.
  * every raw model output (+ the text actually scored, ok flag) is written to a jsonl.
Grader = train\\a11_score.score() unchanged.

Usage:
  python a11_eval_v2.py --url http://127.0.0.1:8121 --name run6_math_grpo-q5_k_m [--limit 50]
  python a11_eval_v2.py --gguf ...\\superbrain1-q5_k_m.gguf --gpu 0 --port 8121
Writes proof\\a11_v2_<name>.json (+ proof\\a11_v2_raw_<name>.jsonl)."""
import argparse, json, os, re, socket, subprocess, sys, time, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
ROOT = r"E:\ABU\02_PROJECTS\AHMAD-1"
BIN = r"E:\ABU\tools\llama.cpp\build-cuda\bin"
sys.path.insert(0, os.path.join(ROOT, "train"))
from a11_score import score  # noqa: E402  (grader unchanged)

EXAM = os.path.join(ROOT, "train", "packet_a11_exam.jsonl")
TYPE = {"predict_output": "predict-output", "write_function": "write-function",
        "fix_bug": "fix-bug", "complete_function": "complete"}
STOP = ["\nسؤال:", "\nالمهمة:"]
PREFILL = "\n```python\n"
CODE_KINDS = ("write_function", "fix_bug", "complete_function")
_BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def ask(url, prompt, n_predict):
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "top_k": 1,
            "cache_prompt": True, "stop": STOP}
    req = urllib.request.Request(url + "/completion", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    err = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read().decode("utf-8"))["content"]
        except Exception as e:
            err = e
            time.sleep(1)
    return f"<ERROR {err}>"


def prompt_of(x):
    return x["prompt"] + PREFILL if x["kind"] in CODE_KINDS else x["prompt"]


def close_fence(t):
    return t if t.count("```") % 2 == 0 else t.rstrip("\n") + "\n```\n"


def splice(item, out):
    """zero_check splice(): put the model's expression into the ____ blank of the prompt's code."""
    blocks = _BLOCK.findall(item["prompt"])
    if not blocks:
        return None
    code = blocks[-1]
    expr = out.strip().strip("`").strip()
    if "____" in code and expr:
        return "```python\n" + code.replace("____", expr.split("\n")[0]) + "```"
    return None


def to_scored(x, out):
    """Text handed to the unchanged a11_score.score()."""
    if out.startswith("<ERROR"):
        return out
    k = x["kind"]
    if k in CODE_KINDS:
        full = close_fence("```python\n" + out)
        if k == "complete_function" and "def " not in out:
            s = splice(x, out.split("```")[0])
            if s is not None:
                return s
        return full
    # predict_output: format-only cleanup (exact-match grading unchanged)
    t = out
    if t.count("```") % 2 == 1:           # unclosed fence (e.g. "```text\n...") -> close it
        t = t.rstrip("\n") + "\n```\n"
    if "```" not in t and t.lstrip().startswith("الناتج:"):
        t = t.lstrip()[len("الناتج:"):]
    return t


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


def start_server(a, log_path):
    with socket.socket() as sk:
        if sk.connect_ex(("127.0.0.1", a.port)) == 0:
            sys.exit(f"port {a.port} already in use - pick another --port (or pass --url)")
    slog = open(log_path, "w", encoding="utf-8")
    srv = subprocess.Popen(
        [os.path.join(BIN, "llama-server.exe"), "-m", a.gguf, "-ngl", "99",
         "-c", str(a.ctx * a.workers), "-np", str(a.workers), "--host", "127.0.0.1", "--port", str(a.port)],
        stdout=slog, stderr=subprocess.STDOUT,
        env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(a.gpu)})
    url = f"http://127.0.0.1:{a.port}"
    for _ in range(240):
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as r:
                if r.status == 200:
                    print("server up", url, "pid", srv.pid, flush=True)
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
    if a.gguf:
        p = os.path.abspath(a.gguf)
        run = os.path.basename(os.path.dirname(os.path.dirname(p)))
        q = os.path.splitext(os.path.basename(p))[0].replace("superbrain1-", "")
        return f"{run}-{q}"
    return "url_model"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--port", type=int, default=8121)
    ap.add_argument("--url")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--raw")
    ap.add_argument("--name")
    ap.add_argument("--exam", default=EXAM)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--n-predict", type=int, default=768)
    ap.add_argument("--timeout", type=float, default=5.0)
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.exam, encoding="utf-8") if l.strip()]
    if a.limit:  # stratified smoke: first N/4 of each kind so every type is exercised
        per = defaultdict(list)
        for x in items:
            per[x["kind"]].append(x)
        q = max(1, a.limit // len(per))
        items = [x for v in per.values() for x in v[:q]]
    name = model_name(a)
    suffix = f"-smoke{a.limit}" if a.limit else ""
    out = a.out or os.path.join(ROOT, "proof", f"a11_v2_{name}{suffix}.json")
    raw = a.raw or os.path.join(ROOT, "proof", f"a11_v2_raw_{name}{suffix}.jsonl")
    if not a.url and not a.gguf:
        ap.error("need --gguf or --url")
    srv = slog = None
    url = a.url.rstrip("/") if a.url else None
    t0 = time.time()
    done = [0]

    def job(x):
        o = ask(url, prompt_of(x), a.n_predict)
        done[0] += 1
        if done[0] % 50 == 0:
            print(f"gen {done[0]}/{len(items)} {time.time() - t0:.0f}s", flush=True)
        return o

    try:
        if not url:
            srv, slog, url = start_server(a, os.path.join(ROOT, "proof", f"server_a11_v2_{name}.log"))
        with ThreadPoolExecutor(a.workers) as ex:
            outs = list(ex.map(job, items))
    finally:
        stop_server(srv, slog)
    gen_secs = round(time.time() - t0, 1)
    scored = [to_scored(x, o) for x, o in zip(items, outs)]
    with ThreadPoolExecutor(a.workers) as ex:
        ok = list(ex.map(lambda p: bool(score(p[1], p[0], timeout=a.timeout)), zip(items, scored)))
    with open(raw, "w", encoding="utf-8") as f:
        for i, (x, o, s, k) in enumerate(zip(items, outs, scored, ok)):
            f.write(json.dumps({"i": i, "qhash": x.get("qhash"), "kind": x["kind"], "template": x.get("template"),
                                "prompt_sent": prompt_of(x), "expected": x.get("answer"), "raw_output": o,
                                "scored_text": s, "ok": k}, ensure_ascii=False) + "\n")
    res = {"model": name, "harness": "a11_eval_v2", "gguf": os.path.abspath(a.gguf) if a.gguf else None,
           "url": a.url, "date": time.strftime("%Y-%m-%d %H:%M"), "exam": a.exam, "endpoint": "completion",
           "prefill_code_kinds": PREFILL, "n_predict": a.n_predict, "gen_secs": gen_secs, "raw": raw,
           **summarize(items, outs, ok),
           "v1_reference": {"file": os.path.join(ROOT, "proof", "a11_run6_math_grpo-q5_k_m.json"),
                            "correct": 103, "n": 1026},
           "failures": [{"kind": x["kind"], "template": x.get("template"), "out": o[:400]}
                        for x, o, k in zip(items, outs, ok) if not k][:60]}
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(fmt(f"A11v2 {name}", res) + f"\n  server errors: {res['server_errors']}\nwrote {out}\nraw {raw}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
