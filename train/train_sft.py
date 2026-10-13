# -*- coding: utf-8 -*-
"""train_sft.py - SUPERBRAIN 1 Run 2a: supervised fine-tuning on Packet A v2.

GRPO (run 1) sharpens what the model already half-knows; it cannot install
knowledge the model never saw (run 1 trained on 99 roots). Run 2a teaches the
morphology itself from train\\packet_a2.jsonl (dictionary-scale, gold-only),
loss on the answer only. Run 2b (GRPO) then sharpens the residual errors.

E: is full: run dirs (train\\run2a*) are junctions to F:\\ABUZ8\\AHMAD-1\\train.

Usage:
  python train_sft.py --samples 1280 --max-steps 20 --out ...\\smoke_sft (smoke)
  python train_sft.py --samples 150000 --out train\\run2a_pilot          (pilot)
  python train_sft.py --out train\\run2a                                 (full)
then: python train\\post_train.py --run train\\run2a_pilot   (gguf + gates)"""
import os, sys, argparse, time, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
ROOT = r"E:\ABU\02_PROJECTS\AHMAD-1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "train", "run1_fanar",
                                                    "merged_16bit"))
    ap.add_argument("--data", default=os.path.join(ROOT, "train",
                                                   "packet_a2.jsonl"))
    ap.add_argument("--samples", type=int, default=0, help="0 = all")
    ap.add_argument("--skip", type=int, default=0,
                    help="drop the first N shuffled items (already trained)")
    ap.add_argument("--init-lora", default="",
                    help="continue training this adapter instead of a new one")
    ap.add_argument("--dev", default="",
                    help="held-out dev jsonl (prompt/completion): evaluate on it "
                         "every --eval-steps and KEEP THE BEST checkpoint")
    ap.add_argument("--eval-steps", type=int, default=250)
    ap.add_argument("--patience", type=int, default=3,
                    help="stop after this many evals without a new best dev loss")
    ap.add_argument("--resume", default="",
                    help="resume from this checkpoint dir (step, optimizer, "
                         "scheduler, RNG restored); pass the SAME other args")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke tests")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.0, help="weight decay")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=384)
    # Windows spawns dataset workers by re-importing this file; 1 is safe.
    ap.add_argument("--num-proc", type=int, default=1)
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "train", "run2a"))
    args = ap.parse_args()

    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer
    from datasets import load_dataset

    if args.init_lora:
        # adapter_config.json names run1's merge as base; unsloth loads both
        # and reopens the adapter trainable (PeftModel is_trainable=True).
        model, tok = FastLanguageModel.from_pretrained(
            args.init_lora, max_seq_length=args.max_len, load_in_4bit=True,
            fast_inference=False, use_gradient_checkpointing="unsloth")
    else:
        model, tok = FastLanguageModel.from_pretrained(
            args.model, max_seq_length=args.max_len, load_in_4bit=True,
            fast_inference=False)
        model = FastLanguageModel.get_peft_model(
            model, r=args.rank, lora_alpha=args.rank, lora_dropout=0,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth", random_state=42)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("TRAINABLE_PARAMS", n_train, flush=True)
    if not n_train:
        sys.exit("ABORT: no trainable parameters")

    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.select_columns(["prompt", "completion"]).shuffle(seed=42)
    end = args.skip + args.samples if args.samples else len(ds)
    ds = ds.select(range(args.skip, min(end, len(ds))))

    cfg = SFTConfig(
        output_dir=args.out, num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_ratio=0.02, weight_decay=args.wd,
        max_length=args.max_len, completion_only_loss=True, packing=False,
        logging_steps=10 if args.max_steps < 0 or args.max_steps > 50 else 1,
        save_steps=args.eval_steps if args.dev else 500, save_total_limit=3,
        bf16=True, report_to="none", seed=42, dataset_num_proc=args.num_proc,
        **(dict(eval_strategy="steps", eval_steps=args.eval_steps,
                per_device_eval_batch_size=32, load_best_model_at_end=True,
                metric_for_best_model="eval_loss", greater_is_better=False)
           if args.dev else {}))
    dev_ds = (load_dataset("json", data_files=args.dev, split="train")
              .select_columns(["prompt", "completion"]) if args.dev else None)
    trainer = SFTTrainer(model=model, processing_class=tok, args=cfg,
                         train_dataset=ds, eval_dataset=dev_ds)

    if args.dev:
        from transformers import EarlyStoppingCallback
        trainer.add_callback(EarlyStoppingCallback(
            early_stopping_patience=args.patience))

    # Prove the loss is on the answer only before burning GPU hours.
    tds = trainer.train_dataset
    print("TRAIN_COLUMNS", tds.column_names, flush=True)
    if "completion_mask" not in tds.column_names:
        sys.exit("ABORT: no completion_mask - loss would cover the prompt")
    for i in range(3):
        ids, m = tds[i]["input_ids"], tds[i]["completion_mask"]
        print("LOSS_ON", repr(tok.decode([t for t, k in zip(ids, m) if k])),
              "| prompt_toks", len(ids) - sum(m), flush=True)

    t0 = time.time()
    trainer.train(resume_from_checkpoint=args.resume or None)
    secs = time.time() - t0
    print(f"TRAIN_DONE samples={len(ds)} secs={secs:.0f}", flush=True)
    model.save_pretrained(os.path.join(args.out, "lora"))
    tok.save_pretrained(os.path.join(args.out, "lora"))
    if not args.no_merge:
        model.save_pretrained_merged(os.path.join(args.out, "merged_16bit"),
                                     tok, save_method="merged_16bit")
    with open(os.path.join(args.out, "run_meta.json"), "w") as f:
        json.dump({**vars(args), "n_train": len(ds), "secs": round(secs),
                   "log_history": trainer.state.log_history}, f, indent=1)
    print("SAVED", args.out, flush=True)


if __name__ == "__main__":
    main()
