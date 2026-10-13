# -*- coding: utf-8 -*-
"""train_grpo.py - SUPERBRAIN 1: GRPO on Qwen3-4B-Base, verifiable rewards.
Reward = 1.0 only when the answer is exactly right (oracle/grammar/math),
parsed with the SAME parsers the bake-off gate uses. No keyword rewards.
Usage: python train_grpo.py --max-steps 2   (smoke)  |  --max-steps 250"""
import os, sys, argparse, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
ROOT = r"E:\ABU\02_PROJECTS\AHMAD-1"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
ap = argparse.ArgumentParser()
ap.add_argument("--max-steps", type=int, default=250)
ap.add_argument("--out", default=os.path.join(ROOT, "train", "run1"))
ap.add_argument("--model", default="QCRI/Fanar-1-9B")
ap.add_argument("--load-4bit", action="store_true")
args = ap.parse_args()

from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
import bakeoff_arabic as B

model, tok = FastLanguageModel.from_pretrained(
    args.model, max_seq_length=1024, load_in_4bit=args.load_4bit,
    fast_inference=False)
model = FastLanguageModel.get_peft_model(
    model, r=32, lora_alpha=64, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=42)

ds = load_dataset("json", data_files=os.path.join(ROOT, "train",
                  "grpo_train.jsonl"), split="train").shuffle(seed=42)

def exact_reward(prompts, completions, task, answer, **kw):
    out = []
    for c, t, a in zip(completions, task, answer):
        c = c if isinstance(c, str) else c[0]["content"]
        if t == "root":
            ok = B.parse_root_output(c) == a
        elif t == "morphology":
            ok = B.parse_verb_output(c) == B.ar_letters(a)
        else:
            ok = B.parse_answer_output(c) == B.norm_ar(a)
        out.append(1.0 if ok else 0.0)
    return out

cfg = GRPOConfig(
    output_dir=args.out, max_steps=args.max_steps, learning_rate=2e-5,
    lr_scheduler_type="cosine", warmup_ratio=0.05,
    per_device_train_batch_size=16, gradient_accumulation_steps=2,
    num_generations=8, max_prompt_length=512, max_completion_length=64,
    temperature=1.0, logging_steps=1, save_steps=50, bf16=True,
    report_to="none", seed=42)
trainer = GRPOTrainer(model=model, processing_class=tok,
                      reward_funcs=[exact_reward], args=cfg, train_dataset=ds)
t0 = time.time()
trainer.train()
print(f"TRAIN_DONE steps={args.max_steps} secs={time.time()-t0:.0f}", flush=True)
model.save_pretrained(os.path.join(args.out, "lora"))
tok.save_pretrained(os.path.join(args.out, "lora"))
model.save_pretrained_merged(os.path.join(args.out, "merged_16bit"), tok,
                             save_method="merged_16bit")
print("SAVED", os.path.join(args.out, "merged_16bit"), flush=True)
