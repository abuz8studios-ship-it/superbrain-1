# -*- coding: utf-8 -*-
"""ship_hf.py - SUPERBRAIN 1: publish a gated run to the Hugging Face Hub.

Refuses unless <run>\\verdict.json says PASS. Uploads the model card
(HF_MODEL_CARD.md, draft comment stripped) as README.md, the chosen GGUF
quants, and the verdict + Packet B summary as eval evidence.

Usage: python ship_hf.py --run train\\run2a [--quants q5_k_m] [--private]
       [--repo abuz8studios/SUPERBRAIN-1]"""
import argparse, json, os, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"E:\ABU\02_PROJECTS\AHMAD-1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--repo", default="abuz8studios/SUPERBRAIN-1")
    ap.add_argument("--quants", default="q5_k_m",
                    help="comma list: q5_k_m,q4_k_m,f16")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--card", default=os.path.join(ROOT, "HF_MODEL_CARD.md"),
                    help="model card uploaded as README.md")
    args = ap.parse_args()

    run = os.path.abspath(args.run)
    v = json.load(open(os.path.join(run, "verdict.json"), encoding="utf-8"))
    if not v.get("PASS"):
        sys.exit("verdict is not PASS - refusing to ship")
    card = open(args.card, encoding="utf-8").read()
    if "DRAFT" in card:
        card = re.sub(r"<!--.*?-->\n*", "", card, flags=re.S)
    files = [(os.path.join(run, "gguf", f"superbrain1-{q}.gguf"),
              f"superbrain1-{q}.gguf") for q in args.quants.split(",")]
    for src, _ in files:
        if not os.path.exists(src):
            sys.exit(f"missing {src}")

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private,
                    exist_ok=True)
    # exist_ok never changes visibility: never publish into a repo whose
    # visibility differs from what was asked for.
    info = api.model_info(args.repo)
    if bool(info.private) != bool(args.private):
        sys.exit(f"repo private={info.private} but --private={args.private}"
                 " - refusing to upload")
    api.upload_file(path_or_fileobj=card.encode("utf-8"),
                    path_in_repo="README.md", repo_id=args.repo,
                    commit_message="model card")
    api.upload_file(path_or_fileobj=os.path.join(run, "verdict.json"),
                    path_in_repo="eval/verdict.json", repo_id=args.repo,
                    commit_message="eval verdict")
    for src, dst in files:
        print("uploading", dst, f"{os.path.getsize(src) / 1e9:.2f} GB",
              flush=True)
        api.upload_file(path_or_fileobj=src, path_in_repo=dst,
                        repo_id=args.repo, commit_message=f"add {dst}")
    print("SHIPPED", f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
