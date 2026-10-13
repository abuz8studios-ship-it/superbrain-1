# -*- coding: utf-8 -*-
"""post_train_g4.py - SUPERBRAIN 1: everything between "training finished" and
"ship or kill", in one unattended command.

  merged_16bit -> gguf\\superbrain1-f16.gguf -> superbrain1-q5_k_m.gguf
  + serve.bat -> llama-server (GPU --gpu) -> 85-item v3 gate + Packet B
  + the مستخدمين probe -> verdict.json (both gates must pass)

Usage: python post_train_g4.py --run train\\run2a_pilot [--gpu 0] [--port 8097]"""
import argparse, json, os, subprocess, sys, time, urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"E:\ABU\02_PROJECTS\AHMAD-1"
LLAMA = r"E:\ABU\tools\llama.cpp-src-2610"   # G4: needs Gemma4 converter
BIN = r"E:\ABU\MODELS\runtime\llama-b10566-cuda13"   # G4: gemma4-capable quantize/server
PY = sys.executable
PROOF = os.path.join(ROOT, "proof")

# Bars to beat (run1_fanar q5_k_m; gate v3 = oracle golds fixed, see
# src/bakeoff_arabic.GOLD_FIX — run1 was 87.06% under v2; Packet B
# 100bb6b45a5c72df).
GATE_V3 = 0.8471
PB_BARS = {"root/fewshot": 0.5377, "root/bare": 0.4885, "morph": 0.6121}
PB_BASE_BARE = 0.5945   # Fanar base, bare root - run1 fell below this

PROBES = [
    "ما جذر كلمة «مستخدمين»؟\nالجذر:",
    "كلمة: مستخدمين\nالجذر:",
    ("المهمة: استخرج الجذر الثلاثي للكلمة العربية.\nأمثلة:\nكلمة: مكتبات\n"
     "الجذر: كتب\nكلمة: مدرسة\nالجذر: درس\nكلمة: شاهد\nالجذر: شهد\n"
     "كلمة: محفوظ\nالجذر: حفظ\nكلمة: استخراج\nالجذر: خرج\n"
     "الآن اشتق الجذر لكل كلمة:\nكلمة: مستخدمين\nالجذر:"),
]

SERVE_BAT = r"""@echo off
rem SUPERBRAIN-1 {name} on native llama.cpp (MSVC+CUDA 13.1, sm_120). Usage: serve.bat [quant] [gpu] [port]
set Q=%1& if "%1"=="" set Q=q5_k_m
set G=%2& if "%2"=="" set G=1
set P=%3& if "%3"=="" set P=8099
set CUDA_VISIBLE_DEVICES=%G%
E:\ABU\tools\llama.cpp\build-cuda\bin\llama-server.exe -m "%~dp0superbrain1-%Q%.gguf" -ngl 99 -c 4096 --host 127.0.0.1 --port %P%
"""


def run(cmd, log):
    print("+", " ".join(cmd), "->", log, flush=True)
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if r.returncode:
        sys.exit(f"FAILED ({r.returncode}): see {log}")


def complete(url, prompt, n=24):
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0.0,
                       "top_k": 1}).encode("utf-8")
    req = urllib.request.Request(url + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))["content"]


def last_jsonl(path, name):
    rec = None
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("model") == name:
            rec = r
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--quant", default="q5_k_m")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run)
    tag = os.path.basename(run_dir.rstrip("\\/"))
    name = f"{tag}-{args.quant}"
    gg = os.path.join(run_dir, "gguf")
    os.makedirs(gg, exist_ok=True)
    f16 = os.path.join(gg, "superbrain1-f16.gguf")
    q = os.path.join(gg, f"superbrain1-{args.quant}.gguf")

    if not os.path.exists(f16):
        run([PY, os.path.join(LLAMA, "convert_hf_to_gguf.py"),
             os.path.join(run_dir, "merged_16bit"), "--outtype", "f16",
             "--outfile", f16], os.path.join(gg, "convert.log"))
    if not os.path.exists(q):
        run([os.path.join(BIN, "llama-quantize.exe"), f16, q,
             args.quant.upper()], os.path.join(gg, f"quantize_{args.quant}.log"))
    with open(os.path.join(gg, "serve.bat"), "w", newline="\r\n") as f:
        f.write(SERVE_BAT.format(name=tag))

    url = f"http://127.0.0.1:{args.port}"
    import socket
    with socket.socket() as sk:   # a foreign server here would get scored
        if sk.connect_ex(("127.0.0.1", args.port)) == 0:
            sys.exit(f"port {args.port} already in use - pick another --port")
    slog = open(os.path.join(PROOF, f"server_{name}.log"), "w",
                encoding="utf-8")
    srv = subprocess.Popen(
        [os.path.join(BIN, "llama-server.exe"), "-m", q, "-ngl", "99",
         "-c", str(2048 * args.workers), "-np", str(args.workers),
         "--host", "127.0.0.1", "--port", str(args.port)],
        stdout=slog, stderr=subprocess.STDOUT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu})
    try:
        for _ in range(240):
            try:
                with urllib.request.urlopen(url + "/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if srv.poll() is not None:
                sys.exit("llama-server died; see " + slog.name)
            time.sleep(2)
        else:
            sys.exit("llama-server never became healthy")
        print("server up", url, flush=True)

        run([PY, os.path.join(ROOT, "src", "gguf_gate.py"), "--url", url,
             "--name", name], os.path.join(PROOF, f"gate_v3_{name}.log"))
        run([PY, os.path.join(ROOT, "src", "packet_b_eval.py"), "--url", url,
             "--name", name, "--workers", str(args.workers)],
            os.path.join(PROOF, f"packetb_{name}.log"))
        probes = [{"prompt": p.splitlines()[-2:], "raw": complete(url, p)}
                  for p in PROBES]
    finally:
        srv.terminate()
        srv.wait(30)
        slog.close()

    gate = last_jsonl(os.path.join(PROOF, "bakeoff_results.jsonl"), name)
    pb = last_jsonl(os.path.join(PROOF, "packetb_results.jsonl"), name)
    s = pb["summary"]
    assert gate["eval_version"] == "v3", gate
    g1 = gate["overall"] >= GATE_V3
    g2 = {k: s[k]["acc"] > bar for k, bar in PB_BARS.items()}
    verdict = {
        "model": name, "date": time.strftime("%Y-%m-%d %H:%M"),
        "gate_v3": {"overall": gate["overall"], "bar": GATE_V3, "pass": g1,
                    **{t: gate[t]["acc"] for t in
                       ("root", "reasoning", "morphology")}},
        "packet_b": {k: {"acc": s[k]["acc"], "bar": PB_BARS[k], "pass": g2[k]}
                     for k in PB_BARS},
        "packet_b_bare_vs_base": {"acc": s["root/bare"]["acc"],
                                  "base": PB_BASE_BARE,
                                  "beats_base": s["root/bare"]["acc"]
                                  > PB_BASE_BARE},
        "morph_by_form": {k.split("/")[1]: v["acc"] for k, v in s.items()
                          if k.startswith("morph_form/")},
        "packet_b_root_class_bare": {k.split("/")[2]: v["acc"]
                                     for k, v in s.items()
                                     if k.startswith("root_class/bare/")},
        "probe_مستخدمين": probes,
        "PASS": g1 and all(g2.values()),
    }
    for path in (os.path.join(run_dir, "verdict.json"),
                 os.path.join(PROOF, f"verdict_{name}.json")):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(verdict, f, ensure_ascii=False, indent=1)
    print(json.dumps(verdict, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

