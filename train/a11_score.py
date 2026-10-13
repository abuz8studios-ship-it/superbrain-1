# -*- coding: utf-8 -*-
"""a11_score.py - scorer for SUPERBRAIN Packet A11 (Arabic-instruction Python coding).

score(pred_text, item, timeout=5.0) -> bool
  kind == "predict_output": the predicted output (last ``` block, else the text after the last
      "الناتج:", else the whole text) must equal item["answer"] exactly, after normalising line
      endings and trailing whitespace on each line (leading/trailing blank lines ignored).
  kind in {"write_function", "fix_bug", "complete_function"}: the last ```python block (else the
      whole text) is run in a fresh `python -I` subprocess followed by item["test_code"] (hidden
      asserts); pass iff exit code 0 and the sentinel line is printed within `timeout` seconds.

CLI:  python a11_score.py ITEMS.jsonl PREDS.jsonl
      PREDS.jsonl is aligned line-by-line with ITEMS.jsonl; each line is {"pred": text} (or "output"/
      "completion"), or a bare JSON string. Prints accuracy overall and by kind.
"""
import os, re, sys, json, subprocess, collections

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SENTINEL = "__A11_TESTS_PASSED__"
CODE_KINDS = ("write_function", "fix_bug", "complete_function")
_FENCE_ANY = re.compile(r"```[^\n`]*\n(.*?)```", re.S)
_FENCE_PY = re.compile(r"```(?:python|py|python3)?[ \t]*\n(.*?)```", re.S | re.I)


def run_py(code, timeout=5.0, python=None):
    """Run `code` in a fresh isolated CPU-only interpreter. Returns (returncode, stdout, stderr)."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
    try:
        p = subprocess.run([python or sys.executable, "-I", "-X", "utf8", "-"], input=code.encode("utf-8"),
                           capture_output=True, timeout=timeout, env=env, creationflags=flags)
    except subprocess.TimeoutExpired:
        return -9, "", "TIMEOUT"
    out = p.stdout.decode("utf-8", "replace").replace("\r\n", "\n")
    err = p.stderr.decode("utf-8", "replace").replace("\r\n", "\n")
    return p.returncode, out, err


def norm_out(s):
    lines = (s or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(l.rstrip() for l in lines).strip("\n").strip()


def extract_output(text):
    t = text or ""
    blocks = _FENCE_ANY.findall(t)
    if blocks:
        return blocks[-1]
    if "الناتج:" in t:
        return t.rsplit("الناتج:", 1)[1]
    return t


def extract_code(text):
    t = text or ""
    blocks = _FENCE_PY.findall(t)
    if blocks:
        return blocks[-1]
    blocks = _FENCE_ANY.findall(t)
    if blocks:
        return blocks[-1]
    t = t.strip()
    if t.startswith("الجواب:"):
        t = t[len("الجواب:"):]
    return t


def program_with_tests(code, test_code):
    return f"{code}\n\n{test_code}\nprint({SENTINEL!r})\n"


def score(pred_text, item, timeout=5.0):
    kind = item.get("kind")
    if kind == "predict_output":
        return norm_out(extract_output(pred_text)) == norm_out(item["answer"])
    if kind in CODE_KINDS:
        code = extract_code(pred_text)
        if not code.strip():
            return False
        rc, out, _ = run_py(program_with_tests(code, item["test_code"]), timeout=timeout)
        return rc == 0 and SENTINEL in out
    raise ValueError(f"unknown kind {kind!r}")


def _pred_of(line):
    v = json.loads(line)
    if isinstance(v, str):
        return v
    for k in ("pred", "output", "completion", "text"):
        if k in v:
            return v[k]
    return ""


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    items = [json.loads(l) for l in open(argv[1], encoding="utf-8") if l.strip()]
    preds = [_pred_of(l) for l in open(argv[2], encoding="utf-8") if l.strip()]
    if len(items) != len(preds):
        print(f"length mismatch items={len(items)} preds={len(preds)}")
        return 2
    tot, ok = collections.Counter(), collections.Counter()
    for it, p in zip(items, preds):
        k = it.get("kind")
        tot[k] += 1
        ok[k] += bool(score(p, it))
    n, c = sum(tot.values()), sum(ok.values())
    print(f"A11 accuracy {c}/{n} = {100 * c / max(n, 1):.2f}%")
    for k in sorted(tot):
        print(f"  {k}: {ok[k]}/{tot[k]} = {100 * ok[k] / tot[k]:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
