# -*- coding: utf-8 -*-
"""train_sft_vl.py - SUPERBRAIN G-line SFT for any vision-language body (Qwen2.5-VL / Fanar-2-Oryx-IVU,
Gemma-4). Generalises train_sft_g4.py (kept unchanged for G1 reproducibility):
  --gemma4-cache  keep use_cache ON and no grad checkpointing (Gemma-4 KV-shared layers need it)
  --gc            gradient checkpointing (fine for Qwen2.5-VL; lets batch grow)
  --init-lora     continue from an earlier LoRA (staged packets)
LoRA on the language model's linear layers only; vision (and audio) towers frozen and kept in the merge,
so the GGUF + mmproj still see. Loss on the answer only (completion_only_loss).

  python train_sft_vl.py --model E:\\ABU\\MODELS\\hf\\Fanar-2-Oryx-IVU --max-steps 8 --out train\\oryx_smoke
"""
import os, sys, argparse, time, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
ROOT = r"E:\ABU\02_PROJECTS\AHMAD-1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "train", "sft_a9_mix.jsonl"))
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--gc", action="store_true")
    ap.add_argument("--gemma4-cache", action="store_true")
    ap.add_argument("--init-lora", default="")
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.gc and args.gemma4_cache:
        sys.exit("ABORT: --gc breaks Gemma-4 KV-shared layers (use_cache=False)")

    import torch
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model, PeftModel
    from trl import SFTConfig, SFTTrainer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa")
    lang = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    if args.init_lora:
        model = PeftModel.from_pretrained(model, args.init_lora, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=args.rank, lora_dropout=0.0, bias="none",
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
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("TRAINABLE_PARAMS", n_train, flush=True)
    if not n_train:
        sys.exit("ABORT: no trainable parameters")
    # no vision/audio tower weight may train
    bad = [n for n, p in model.named_parameters() if p.requires_grad and "language_model" not in n]
    if bad:
        sys.exit(f"ABORT: non-language params trainable: {bad[:3]}")

    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.select_columns(["prompt", "completion"]).shuffle(seed=42)
    if args.samples:
        ds = ds.select(range(min(args.samples, len(ds))))

    cfg = SFTConfig(
        output_dir=args.out, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        max_length=args.max_len, completion_only_loss=True, packing=False,
        gradient_checkpointing=args.gc,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gc else None,
        logging_steps=1 if 0 < args.max_steps <= 50 else 10,
        save_steps=args.save_steps, save_total_limit=2, bf16=True, report_to="none",
        seed=42, dataset_num_proc=1)
    trainer = SFTTrainer(model=model, processing_class=tok, args=cfg, train_dataset=ds)

    tds = trainer.train_dataset
    print("TRAIN_COLUMNS", tds.column_names, flush=True)
    if "completion_mask" not in tds.column_names:
        sys.exit("ABORT: no completion_mask - loss would cover the prompt")
    for i in range(3):
        ids, m = tds[i]["input_ids"], tds[i]["completion_mask"]
        print("LOSS_ON", repr(tok.decode([t for t, k in zip(ids, m) if k])),
              "| prompt_toks", len(ids) - sum(m), flush=True)

    t0 = time.time()
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    secs = time.time() - t0
    print(f"TRAIN_DONE samples={len(ds)} secs={secs:.0f}", flush=True)
    model.save_pretrained(os.path.join(args.out, "lora"))
    if not args.no_merge:
        merged = model.merge_and_unload()
        merged.config.use_cache = True
        out = os.path.join(args.out, "merged_16bit")
        merged.save_pretrained(out, safe_serialization=True)
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
