#!/usr/bin/env python3
"""SFT of Qwen3-1.7B with the LoRA family (LoRA / QLoRA / DoRA / QDoRA / rsLoRA / PiSSA / OLoRA / LoRA+).

    python scripts/train_lora.py --list-methods
    python scripts/train_lora.py --method lora   --run-name lora_r16
    python scripts/train_lora.py --method qlora  --r 64 --alpha 128 --bs 8
    python scripts/train_lora.py --method dora   --targets attn --lr 1e-4
    python scripts/train_lora.py --method loraplus --loraplus-ratio 16
    python scripts/train_lora.py --method lora --extra-data data/feedback/feedback_sft.jsonl   # learn from feedback
    python scripts/train_lora.py --method lora --init-adapter runs/lora_r16/adapter            # continue an adapter
    python scripts/train_lora.py --method lora --max-samples 2000 --max-steps 50              # smoke test

Data: chat-format JSONL with a "messages" list (data/final/sft_train.jsonl from build_training_mix.py,
or data/feedback/feedback_sft.jsonl exported by chat_feedback.py). Loss is computed on assistant turns only.

Output (runs/<run-name>/):
    adapter/            final LoRA adapter (+ tokenizer)   <- use with eval_model.py / chat_feedback.py
    checkpoint-*/       intermediate checkpoints
    run_config.json     all arguments + dataset stats
    log_history.json    training / validation losses
    merged/             (with --merge) full merged bf16 model
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from ai_qa_train.common import (DEFAULT_BASE, LORA_METHODS, ROOT, ChatCollator, build_lora_config,  # noqa: E402
                                encode_chat, is_quantized, load_model, load_tokenizer, read_jsonl)


def load_records(paths: list[str], levels: set[str] | None, max_samples: int | None, seed: int,
                 reasoning_frac: float) -> list[dict]:
    recs = []
    for p in paths:
        for r in read_jsonl(p):
            if "messages" not in r:
                continue
            if levels and r.get("level") not in levels:
                continue
            recs.append(r)
    rng = random.Random(seed)
    if reasoning_frac < 1.0:  # optionally down-weight long <think> samples
        recs = [r for r in recs if not r.get("has_reasoning") or rng.random() < reasoning_frac]
    rng.shuffle(recs)
    return recs[:max_samples] if max_samples else recs


def tokenize(recs: list[dict], tok, max_len: int, num_proc: int):
    from datasets import Dataset

    ds = Dataset.from_list([{"messages": r["messages"]} for r in recs])

    def fn(ex):
        e = encode_chat(tok, ex["messages"], max_len)
        if e is None:
            return {"input_ids": [], "labels": [], "length": 0}
        return {**e, "length": len(e["input_ids"])}

    ds = ds.map(fn, remove_columns=["messages"], num_proc=num_proc if len(recs) > 2000 else None,
                desc="tokenize")
    return ds.filter(lambda x: x["length"] > 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-methods", action="store_true")
    # model / adapter
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--method", default="lora", choices=list(LORA_METHODS))
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--targets", default="all-linear", help="all-linear | attn | mlp | qv | comma list of module names")
    ap.add_argument("--loraplus-ratio", type=float, default=16.0, help="lr(B)/lr(A) for --method loraplus")
    ap.add_argument("--init-adapter", default=None, help="continue training an existing adapter")
    # data
    ap.add_argument("--data", nargs="+", default=[os.path.join(ROOT, "data", "final", "sft_train.jsonl")])
    ap.add_argument("--val", default=os.path.join(ROOT, "data", "final", "sft_val.jsonl"))
    ap.add_argument("--extra-data", nargs="*", default=[], help="additional chat jsonl files, e.g. feedback_sft.jsonl")
    ap.add_argument("--extra-repeat", type=int, default=3, help="upsample extra-data (feedback) N times")
    ap.add_argument("--levels", default=None, help="comma list: train only on these levels")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=1000)
    ap.add_argument("--reasoning-frac", type=float, default=1.0, help="keep this fraction of <think> samples")
    ap.add_argument("--max-len", type=int, default=2048)
    # optimisation
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=4, help="per-device batch size")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03, help="warmup fraction of total steps")
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--report-to", default="none", help="none | tensorboard | wandb")
    ap.add_argument("--num-proc", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    # output
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--output-root", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--resume", action="store_true", help="resume from the last checkpoint in the run dir")
    ap.add_argument("--merge", action="store_true", help="also save a merged full model (not for 4-bit)")
    args = ap.parse_args()

    if args.list_methods:
        for k, v in LORA_METHODS.items():
            print(f"{k:9s} {v}")
        return

    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(args.seed)
    run = args.run_name or f"{args.method}_r{args.r}_{time.strftime('%m%d_%H%M')}"
    out = os.path.join(args.output_root, run)
    os.makedirs(out, exist_ok=True)

    # ---------------- data ----------------
    tok = load_tokenizer(args.base)
    tok.padding_side = "right"
    levels = set(args.levels.split(",")) if args.levels else None
    train_recs = load_records(args.data, levels, args.max_samples, args.seed, args.reasoning_frac)
    for p in args.extra_data:
        extra = list(read_jsonl(p))
        train_recs += [r for r in extra if "messages" in r] * args.extra_repeat
        print(f"+ {len(extra)} extra records from {p} (x{args.extra_repeat})")
    random.Random(args.seed).shuffle(train_recs)
    val_recs = load_records([args.val], levels, args.max_val, args.seed, 1.0) if os.path.exists(args.val) else []
    if not train_recs:
        sys.exit("no training records - run build_training_mix.py first or pass --data")
    train_ds = tokenize(train_recs, tok, args.max_len, args.num_proc)
    val_ds = tokenize(val_recs, tok, args.max_len, args.num_proc) if val_recs else None
    stats = {
        "train_records": len(train_ds), "val_records": len(val_ds) if val_ds else 0,
        "train_tokens": int(sum(train_ds["length"])),
        "levels": dict(Counter(r.get("level", "?") for r in train_recs)),
    }
    print(json.dumps(stats, indent=2))

    # ---------------- model ----------------
    q4 = is_quantized(args.method)
    model = load_model(args.base, adapter=args.init_adapter, quantize_4bit=q4, for_training=True)
    model.config.use_cache = False
    if q4:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=not args.no_grad_ckpt)
    if args.init_adapter is None:
        from peft import get_peft_model
        lcfg = build_lora_config(args.method, args.r, args.alpha, args.dropout, args.targets)
        model = get_peft_model(model, lcfg)
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    optimizers = (None, None)
    if args.method == "loraplus":
        from peft.optimizers import create_loraplus_optimizer
        opt = create_loraplus_optimizer(model=model, optimizer_cls=torch.optim.AdamW, lr=args.lr,
                                        loraplus_lr_ratio=args.loraplus_ratio, weight_decay=args.weight_decay)
        optimizers = (opt, None)

    steps_per_epoch = max(1, len(train_ds) // (args.bs * args.grad_accum))
    total = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)
    targs = TrainingArguments(
        output_dir=out, run_name=run, seed=args.seed,
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.bs, per_device_eval_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type=args.scheduler,
        warmup_steps=max(1, int(total * args.warmup)), weight_decay=args.weight_decay, max_grad_norm=1.0,
        bf16=True, optim="paged_adamw_8bit" if q4 else "adamw_torch_fused",
        logging_steps=args.logging_steps, report_to=args.report_to,
        eval_strategy="steps" if val_ds else "no", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=3,
        train_sampling_strategy="group_by_length", length_column_name="length",
        remove_unused_columns=False, dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=ChatCollator(tok.pad_token_id), optimizers=optimizers)

    with open(os.path.join(out, "run_config.json"), "w") as f:
        json.dump({"args": vars(args), "data_stats": stats, "total_steps": total}, f, indent=2)

    resume = None
    if args.resume:
        ck = sorted([d for d in os.listdir(out) if d.startswith("checkpoint-")], key=lambda d: int(d.split("-")[1]))
        resume = os.path.join(out, ck[-1]) if ck else None
    trainer.train(resume_from_checkpoint=resume)

    final = os.path.join(out, "adapter")
    trainer.model.save_pretrained(final)
    tok.save_pretrained(final)
    if val_ds:
        m = trainer.evaluate()
        print("final eval:", m)
    with open(os.path.join(out, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    if args.merge and not q4:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(os.path.join(out, "merged"))
        tok.save_pretrained(os.path.join(out, "merged"))
    print(f"\nadapter saved to {final}\n"
          f"evaluate:  python scripts/eval_model.py --model {final}\n"
          f"chat:      python scripts/chat_feedback.py --model {final}")


if __name__ == "__main__":
    main()
