# -*- coding: utf-8 -*-
"""train_grpo2.py - SUPERBRAIN 1 Run 2b: GRPO on the post-SFT model.

SFT (Run 2a) installed the morphology; GRPO sharpens the residual errors.
Data = train\\grpo2_train.jsonl from build_grpo2_data.py (items the Run 2a
model gets right only sometimes, weighted to its weakest forms/classes).
Reward = 1.0 only for the exact gold answer (src\\a2_reward.py, same parsers
as the gates). Run dirs go on F: (E: is full) - junction before running.

Usage: python train_grpo2.py --max-steps 2 --out ...\\smoke_grpo2   (smoke)
       python train_grpo2.py --out train\\run2b                      (run)
then:  python train\\post_train.py --run train\\run2b"""
import os, sys, argparse, time, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
ROOT = r"E:\ABU\02_PROJECTS\AHMAD-1"
sys.path.insert(0, os.path.join(ROOT, "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "train", "run2a",
                                                    "merged_16bit"))
    ap.add_argument("--data", default=os.path.join(ROOT, "train",
                                                   "grpo2_train.jsonl"))
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--bs", type=int, default=16,
                    help="completions per device per micro-step (= prompts x --gens)")
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--max-comp", type=int, default=48,
                    help="max completion tokens (48 for roots, ~384 for math)")
    ap.add_argument("--resume", default="",
                    help="resume from this checkpoint dir (pass the same args)")
    ap.add_argument("--out", default=os.path.join(ROOT, "train", "run2b"))
    args = ap.parse_args()

    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    from datasets import load_dataset
    import a2_reward as R

    model, tok = FastLanguageModel.from_pretrained(
        args.model, max_seq_length=1024, load_in_4bit=True,
        fast_inference=False)
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=42)

    ds = load_dataset("json", data_files=args.data,
                      split="train").shuffle(seed=42)

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
        temperature=1.0, logging_steps=1, save_steps=100, bf16=True,
        report_to="none", seed=42)
    trainer = GRPOTrainer(model=model, processing_class=tok,
                          reward_funcs=[exact_reward], args=cfg,
                          train_dataset=ds)
    t0 = time.time()
    trainer.train(resume_from_checkpoint=args.resume or None)
    secs = time.time() - t0
    print(f"TRAIN_DONE steps={args.max_steps} secs={secs:.0f}", flush=True)
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
