# -*- coding: utf-8 -*-
"""check_a11.py - independent re-verification of SUPERBRAIN-1 Packet A11 (Arabic-instruction Python coding).

Checks (exit 0 only if all pass):
  1. counts: train == 6021, exam == 1026 (A11_NTRAIN/A11_NEXAM x 27 generators), 27 generators present in both,
     schema fields present, no duplicate prompts inside a split.
  2. reference solutions: random sample (>=300 train + 100 exam code items) re-run through a11_score.score()
     on their gold completion (subprocess + hidden tests, 5 s timeout) -> must be 100%;
     fix_bug items in the sample: the buggy version must FAIL.
  3. predict_output: random sample (150 train + 50 exam) snippets re-executed in a subprocess -> stdout must
     equal the stored answer; the gold completion and a bare "الناتج: ..." answer must score True.
  4. negative control: completions of OTHER items must score < 5%.
  5. leakage train<->exam: exact / normalized prompt, normalized question, question hash, code -> all 0.
"""
import os, sys, json, random, hashlib, collections, time
from concurrent.futures import ThreadPoolExecutor

os.environ["CUDA_VISIBLE_DEVICES"] = ""
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a11_score import score, run_py, norm_out, CODE_KINDS  # noqa: E402
from build_packet_a11 import norm, question_of, code_norm, GENS  # noqa: E402

T = os.environ.get("A11_DIR", r"E:\ABU\02_PROJECTS\AHMAD-1\train")
NG = len(GENS)
EXP_TRAIN = NG * int(os.environ.get("A11_NTRAIN", "223"))
EXP_EXAM = NG * int(os.environ.get("A11_NEXAM", "38"))
N_REF_TRAIN, N_REF_EXAM, N_PRED_TRAIN, N_PRED_EXAM, N_NEG = 300, 100, 150, 50, 100
WORKERS = int(os.environ.get("A11_WORKERS", "6"))
FIELDS = ["task", "src", "kind", "topic", "template", "prompt", "completion", "answer", "qhash"]


def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    t0 = time.time()
    ok_all = True
    tr = load(os.path.join(T, "packet_a11_train.jsonl"))
    ex = load(os.path.join(T, "packet_a11_exam.jsonl"))

    # 1. counts / schema
    gtr = {x["template"].split("#")[0] for x in tr}
    gex = {x["template"].split("#")[0] for x in ex}
    miss = sum(1 for x in tr + ex for k in FIELDS if k not in x) + \
        sum(1 for x in tr + ex if x["kind"] in CODE_KINDS and not x.get("test_code"))
    dups = (len(tr) - len({x["prompt"] for x in tr})) + (len(ex) - len({x["prompt"] for x in ex}))
    exam_phr = collections.Counter(x["template"].split("#")[1] for x in ex)
    c_ok = (len(tr) == EXP_TRAIN and len(ex) == EXP_EXAM and len(gtr) == NG and len(gex) == NG
            and miss == 0 and dups == 0 and set(exam_phr) == {"2"})
    print(f"[counts] train={len(tr)} (expect {EXP_TRAIN}) exam={len(ex)} (expect {EXP_EXAM}) "
          f"generators train={len(gtr)} exam={len(gex)} (expect {NG}) missing_fields={miss} dup_prompts={dups} "
          f"exam_phrasings={dict(exam_phr)} -> {'OK' if c_ok else 'FAIL'}")
    print(f"[counts] train by kind {dict(sorted(collections.Counter(x['kind'] for x in tr).items()))}")
    print(f"[counts] exam  by kind {dict(sorted(collections.Counter(x['kind'] for x in ex).items()))}")
    ok_all &= c_ok

    rng = random.Random(1111)
    code_tr = [x for x in tr if x["kind"] in CODE_KINDS]
    code_ex = [x for x in ex if x["kind"] in CODE_KINDS]
    pred_tr = [x for x in tr if x["kind"] == "predict_output"]
    pred_ex = [x for x in ex if x["kind"] == "predict_output"]
    ref_sample = rng.sample(code_tr, N_REF_TRAIN) + rng.sample(code_ex, N_REF_EXAM)
    pred_sample = rng.sample(pred_tr, N_PRED_TRAIN) + rng.sample(pred_ex, N_PRED_EXAM)

    # 2. reference solutions pass hidden tests; buggy versions fail
    def ref_job(x):
        good = score(x["completion"], x, timeout=5.0)
        bad = None
        if x["kind"] == "fix_bug":
            bad = score(f"```python\n{x['buggy']}\n```", x, timeout=5.0)
        return good, bad

    with ThreadPoolExecutor(WORKERS) as pool:
        res = list(pool.map(ref_job, ref_sample))
    n_ref = len(res)
    ref_pass = sum(1 for g, _ in res if g)
    fix = [b for _, b in res if b is not None]
    fix_bad_pass = sum(1 for b in fix if b)
    by_kind = collections.Counter(x["kind"] for x in ref_sample)
    r_ok = ref_pass == n_ref and fix_bad_pass == 0
    print(f"[reference] sample={n_ref} {dict(by_kind)} pass={ref_pass}/{n_ref} ({100 * ref_pass / n_ref:.2f}%) | "
          f"fix_bug buggy versions passing={fix_bad_pass}/{len(fix)} (must be 0) -> {'OK' if r_ok else 'FAIL'}")
    ok_all &= r_ok

    # 3. predict_output gold re-execution
    def pred_job(x):
        rc, out, _ = run_py(x["code"], timeout=5.0)
        same = rc == 0 and norm_out(out) == norm_out(x["answer"])
        return same, score(x["completion"], x), score("الناتج: " + x["answer"], x)

    with ThreadPoolExecutor(WORKERS) as pool:
        pres = list(pool.map(pred_job, pred_sample))
    n_p = len(pres)
    rerun = sum(a for a, _, _ in pres)
    gold = sum(b for _, b, _ in pres)
    bare = sum(c for _, _, c in pres)
    p_ok = rerun == gold == bare == n_p
    print(f"[predict] sample={n_p} re-executed_match={rerun}/{n_p} gold_completion_scores={gold}/{n_p} "
          f"bare_answer_scores={bare}/{n_p} -> {'OK' if p_ok else 'FAIL'}")
    ok_all &= p_ok

    # 4. negative control
    neg_items = rng.sample(code_ex, N_NEG // 2) + rng.sample(pred_ex, N_NEG // 2)
    pairs = []  # pair each item with a completion from a different generator (drawn serially: rng not thread-safe)
    for x in neg_items:
        pool_ = code_tr if x["kind"] in CODE_KINDS else pred_tr
        while True:
            y = pool_[rng.randrange(len(pool_))]
            if y["template"].split("#")[0] != x["template"].split("#")[0]:
                pairs.append((x, y))
                break
    with ThreadPoolExecutor(WORKERS) as pool:
        negs = list(pool.map(lambda xy: score(xy[1]["completion"], xy[0], timeout=5.0), pairs))
    neg_rate = 100 * sum(negs) / len(negs)
    n_ok = neg_rate < 5
    print(f"[negative] other-item completions scored correct: {sum(negs)}/{len(negs)} ({neg_rate:.1f}%, must be <5%) "
          f"-> {'OK' if n_ok else 'FAIL'}")
    ok_all &= n_ok

    # 5. leakage
    def sets(items):
        ps = [x["prompt"] for x in items]
        qs = [question_of(p) for p in ps]
        return {"exact": set(ps), "norm": {norm(p) for p in ps}, "qnorm": {norm(q) for q in qs},
                "qhash": {hashlib.sha1(q.encode("utf-8")).hexdigest()[:16] for q in qs},
                "code": {code_norm(x.get("code") or x.get("reference")) for x in items}}
    A, B = sets(tr), sets(ex)
    leak = {k: len(A[k] & B[k]) for k in A}
    total = sum(leak.values())
    l_ok = total == 0
    print(f"[leakage] train_vs_exam {leak} LEAKAGE={total} -> {'OK' if l_ok else 'FAIL'}")
    ok_all &= l_ok

    print(f"CHECK_A11 {'PASS' if ok_all else 'FAIL'} ({time.time() - t0:.0f}s)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
