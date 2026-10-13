# -*- coding: utf-8 -*-
"""a11_zero_check.py - diagnose the 0/228 on A11 fix-bug + complete (board item a11_harness_zero_check_2026-10-13).

NEW file; a11_eval.py and a11_score.py are untouched (imported read-only).
Runs ONLY fix_bug + complete_function items on llama-server (GPU0 only), saves EVERY raw output, and scores:
  V0  original prompt, original a11_score.score                      (reproduces the official number)
  V1  original prompt, lenient: if output has no 'def', splice it into the prompt's buggy/blank code
      (complete: replace ____ with the output expression; fix_bug: no splice possible -> same as V0)
  V2  prompt + "\\n```python\\n" prefill (forces a code answer), original scorer on "```python\\n"+out
"""
import json, os, re, sys, time, urllib.request, subprocess, socket
from concurrent.futures import ThreadPoolExecutor
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"E:\ABU\02_PROJECTS\AHMAD-1"
sys.path.insert(0, os.path.join(ROOT, "train")); sys.path.insert(0, os.path.join(ROOT, "src"))
from a11_score import score, extract_code  # noqa
from a11_eval import ask  # noqa

BIN = r"E:\ABU\tools\llama.cpp\build-cuda\bin"
GGUF = os.path.join(ROOT, r"train\run6_math_grpo\gguf\superbrain1-q5_k_m.gguf")
PORT = 8141
OUT = os.path.join(ROOT, "proof", "a11_zero_check_run6_math_grpo-q5_k_m.json")
RAW = os.path.join(ROOT, "proof", "a11_zero_check_raw_run6_math_grpo-q5_k_m.jsonl")
_BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def splice(item, out):
    if "def " in out:
        return out
    blocks = _BLOCK.findall(item["prompt"])
    if not blocks:
        return out
    code = blocks[-1]
    expr = out.strip().strip("`").strip()
    if item["kind"] == "complete_function" and "____" in code and expr:
        return "```python\n" + code.replace("____", expr.split("\n")[0]) + "```"
    return out


def main():
    items = [json.loads(l) for l in open(os.path.join(ROOT, "train", "packet_a11_exam.jsonl"), encoding="utf-8")]
    items = [x for x in items if x["kind"] in ("fix_bug", "complete_function")]
    with socket.socket() as sk:
        if sk.connect_ex(("127.0.0.1", PORT)) == 0:
            sys.exit("port busy")
    slog = open(os.path.join(ROOT, "proof", "server_a11_zero_check.log"), "w", encoding="utf-8")
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0"}
    srv = subprocess.Popen([os.path.join(BIN, "llama-server.exe"), "-m", GGUF, "-ngl", "99", "-c", str(4096 * 8),
                            "-np", "8", "--host", "127.0.0.1", "--port", str(PORT)],
                           stdout=slog, stderr=subprocess.STDOUT, env=env)
    print("server pid", srv.pid, flush=True)
    url = f"http://127.0.0.1:{PORT}"
    try:
        for _ in range(240):
            try:
                with urllib.request.urlopen(url + "/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if srv.poll() is not None:
                sys.exit("server died")
            time.sleep(2)
        print("server up", flush=True)
        with ThreadPoolExecutor(8) as ex:
            o0 = list(ex.map(lambda x: ask(url, x["prompt"], 768), items))
            o2 = list(ex.map(lambda x: ask(url, x["prompt"] + "\n```python\n", 768), items))
    finally:
        srv.terminate()
        try:
            srv.wait(30)
        except subprocess.TimeoutExpired:
            srv.kill()
        slog.close()
        print("server stopped", flush=True)
    with ThreadPoolExecutor(8) as ex:
        v0 = list(ex.map(lambda p: bool(score(p[1], p[0])), zip(items, o0)))
        s1 = [splice(x, o) for x, o in zip(items, o0)]
        v1 = list(ex.map(lambda p: bool(score(p[1], p[0])), zip(items, s1)))
        p2 = ["```python\n" + o for o in o2]
        v2 = list(ex.map(lambda p: bool(score(p[1], p[0])), zip(items, p2)))
    res = {"model": "run6_math_grpo-q5_k_m", "gguf": GGUF, "date": time.strftime("%Y-%m-%d %H:%M"),
           "gpu": "CUDA_VISIBLE_DEVICES=0 (PCI_BUS_ID)", "n_predict": 768, "per_variant": {}}
    with open(RAW, "w", encoding="utf-8") as f:
        for x, a, b, c, d, e in zip(items, o0, v0, v1, o2, v2):
            f.write(json.dumps({"qhash": x.get("qhash"), "kind": x["kind"], "template": x.get("template"),
                                "prompt": x["prompt"], "expected": x["answer"], "out_v0": a,
                                "extracted_v0": extract_code(a), "ok_v0": b, "ok_v1_splice": c,
                                "out_v2_prefill": d, "ok_v2": e}, ensure_ascii=False) + "\n")
    for name, v in (("V0_original", v0), ("V1_splice", v1), ("V2_prefill", v2)):
        d = {}
        for k in ("fix_bug", "complete_function"):
            m = [ok for x, ok in zip(items, v) if x["kind"] == k]
            d[k] = f"{sum(m)}/{len(m)}"
        d["fix+complete"] = f"{sum(v)}/{len(v)}"
        res["per_variant"][name] = d
        print(name, d, flush=True)
    res["no_def_in_v0"] = sum("def " not in o for o in o0)
    res["no_def_in_v2"] = sum("def " not in o for o in o2)
    res["raw"] = RAW
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False), "\nwrote", OUT)


if __name__ == "__main__":
    main()
