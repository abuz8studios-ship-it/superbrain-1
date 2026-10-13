# -*- coding: utf-8 -*-
"""train_grpo_vl.py - GRPO for a vision-language SUPERBRAIN body (Qwen2.5-VL / Fanar-2-Oryx-IVU, Gemma-4).
Same data + reward as train_grpo2.py (exact gold answer via src\\a2_reward.correct, data from
build_grpo_data_long.py: fields prompt/task/answer), but plain transformers + peft (unsloth's
FastLanguageModel can't load these bodies). LoRA on the language model only, towers frozen.

  python train_grpo_vl.py --model <merged_16bit> --data train\\grpo_x.jsonl --max-steps 2 --out train\\vl_grpo_smoke
"""
import os, sys, argparse, time, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
ROOT = r"E:\ABU\02_PROJECTS\AHMAD-1"
sys.path.insert(0, os.path.join(ROOT, "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--bs", type=int, default=16, help="completions per micro-step (= prompts x --gens)")
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--max-comp", type=int, default=48, help="48 for roots, ~384 for math")
    ap.add_argument("--gc", action="store_true", help="gradient checkpointing (NOT for Gemma-4)")
    ap.add_argument("--gemma4-cache", action="store_true")
    ap.add_argument("--resume", default="")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.gc and args.gemma4_cache:
        sys.exit("ABORT: --gc breaks Gemma-4 KV-shared layers")

    import torch
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model
    # trl 0.24 + this transformers: _is_package_available returns a tuple, so (False, None) is truthy and
    # is_mergekit_available() "finds" mergekit -> ImportError. unsloth patches this; plain-HF runs must too.
    import trl.import_utils as _tiu
    for _n in dir(_tiu):
        if _n.startswith("_") and _n.endswith("_available") and isinstance(getattr(_tiu, _n), tuple):
            setattr(_tiu, _n, bool(getattr(_tiu, _n)[0]))
    from trl import GRPOConfig, GRPOTrainer
    from datasets import load_dataset
    import a2_reward as R

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    lang = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, bias="none",
        target_modules=lang, task_type="CAUSAL_LM"))
    if args.gemma4_cache:
        model.config.use_cache = True
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = True
        inner = model.get_base_model()
        _fwd = inner.forward
        import functools
        @functools.wraps(_fwd)   # keep forward's signature: generate() introspects it (logits_to_keep...)
        def _fwd_cache(*a, **k):
            k["use_cache"] = True
            return _fwd(*a, **k)
        inner.forward = _fwd_cache
    if args.gc:
        model.enable_input_require_grads()
    # trl 0.24 writes model.warnings_issued[...]; newer transformers models no longer carry that dict
    if not hasattr(model.get_base_model(), "warnings_issued"):
        model.get_base_model().warnings_issued = {}
    bad = [n for n, p in model.named_parameters() if p.requires_grad and "language_model" not in n]
    if bad:
        sys.exit(f"ABORT: non-language params trainable: {bad[:3]}")
    print("TRAINABLE_PARAMS", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)

    ds = load_dataset("json", data_files=args.data, split="train").shuffle(seed=42)
    # trl tokenizes GRPO prompts with add_special_tokens=False; Gemma without <bos> emits junk (reward 0 on every
    # completion in the first smoke). llama-server adds BOS itself, so the filter step never saw this.
    if tok.bos_token:
        ds = ds.map(lambda x: {"prompt": x["prompt"] if x["prompt"].startswith(tok.bos_token) else tok.bos_token + x["prompt"]})
        print("PROMPT_BOS", repr(ds[0]["prompt"][:30]), flush=True)

    def exact_reward(prompts, completions, task, answer, **kw):
        out = []
        for c, t, a in zip(completions, task, answer):
            c = c if isinstance(c, str) else c[0]["content"]
            out.append(1.0 if R.correct(t, c, a) else 0.0)
        return out

    cfg = GRPOConfig(
        output_dir=args.out, max_steps=args.max_steps,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
        per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.accum,
        num_generations=args.gens, max_prompt_length=512, max_completion_length=args.max_comp,
        temperature=1.0, logging_steps=1, save_steps=50, save_total_limit=2, bf16=True,
        gradient_checkpointing=args.gc,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gc else None,
        report_to="none", seed=42)
    trainer = GRPOTrainer(model=model, processing_class=tok, reward_funcs=[exact_reward],
                          args=cfg, train_dataset=ds)
    t0 = time.time()
    trainer.train(resume_from_checkpoint=args.resume or None)
    secs = time.time() - t0
    print(f"TRAIN_DONE steps={args.max_steps} secs={secs:.0f}", flush=True)
    model.save_pretrained(os.path.join(args.out, "lora"))
    if not args.no_merge:
        merged = model.merge_and_unload()
        merged.config.use_cache = True
        out = os.path.join(args.out, "merged_16bit")
        merged.save_pretrained(out, safe_serialization=True)
        tok.padding_side = "right"
        tok.save_pretrained(out)
        import shutil
        for f in ("processor_config.json", "preprocessor_config.json", "video_preprocessor_config.json",
                  "chat_template.jinja", "chat_template.json", "generation_config.json"):
            src = os.path.join(args.model, f)
            if os.path.exists(src):
                shutil.copy(src, out)
    with open(os.path.join(args.out, "run_meta.json"), "w") as f:
        json.dump({**vars(args), "n_train": len(ds), "secs": round(secs),
                   "log_history": trainer.state.log_history}, f, indent=1)
    print("SAVED", args.out, flush=True)


if __name__ == "__main__":
    main()
