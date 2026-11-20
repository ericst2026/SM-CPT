#!/usr/bin/env python3
"""RETRO-fit + LoRA continued pre-training of Qwen3-0.6B-Base (docs/retro_fit_lora_cpt.md).

    python -m cpt_trainer.retro.train --set runs/_retro_cache/web1b --db cpt_data/retro_db/ai_v1 \\
        --retro-config configs/retro/retro_qwen06b_v1.yaml \\
        --lora-config configs/lora/cpt_qwen06b_rs_adaptive_v5_r16_128.yaml \\
        --run-name retro_qwen06b_v1_web1b                      # see runs/launch_retro_qwen06b_v1_web1b.sh
    ... --resume                                               # continue after an interruption
    ... --no-retro --run-name lora_qwen06b_v5_web1b             # control: the same LoRA on the same data, no retrieval

Model: the frozen base + a LoRA rank plan (PEFT) + the RETRO module (neighbour encoder and chunked cross-attention,
cpt_trainer/retro/model.py), which PEFT keeps as modules_to_save, so checkpoints / adapters contain both and
--resume restores both (plus optimizer, scheduler, RNG and data order, as cpt_trainer/train.py).
Optimisation: AdamW with two parameter groups: RETRO layers at --retro-lr (randomly initialised; RETRO-fitting used
2e-4 for <= 1.5 B models) and LoRA at --lr (5e-5 as the web5b run). Warmup + cosine for both; the LoRA group can
start later (--lora-start-frac): the RETRO layers first learn to read the neighbours before the LoRA moves the
decoder (the literature freezes the decoder for retro-fitting; we co-train a LoRA, so we give the new path a head start).
Activation warm-up (--oracle-rate / --oracle-until): during the first part of training a share of the chunks gets
an "oracle" neighbour in its last slot (the chunk itself plus its true continuation), annealed linearly to 0, which
teaches the cross-attention to align neighbour and input quickly; afterwards only real neighbours are used.
Evaluation points (--eval-at): adapter saved to epoch<x>/adapter; validation loss of every corpus with retrieval on,
off (decoder + LoRA only) and with random neighbours; then cpt_trainer/retro/eval.py (MMLU 5-shot incl. ai_large and
the AI MCQ, with online retrieval and without) in a subprocess. Scores -> eval_history.json and summary.md.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ai_qa_train.common import ROOT, load_model, load_tokenizer  # noqa: E402
from cpt_trainer import lora_plan  # noqa: E402
from cpt_trainer.retro.dataset import RetroCollator, RetroSplit  # noqa: E402
from cpt_trainer.retro.db import Database  # noqa: E402
from cpt_trainer.retro.model import RetroConfig, attach_retro, param_report, retro_loss, trainable_split, unwrap  # noqa: E402
from cpt_trainer.train import Logger, epoch_label, interval_train_loss  # noqa: E402

DEFAULT_BASE = os.path.join(ROOT, "models", "Qwen3-0.6B-Base")


def lr_lambdas(total: int, warmup: int, lora_start: int, scheduler: str):
    def cos(step, start):
        if step < start:
            return 0.0
        s = step - start
        if s < warmup:
            return (s + 1) / max(1, warmup)
        if scheduler == "constant":
            return 1.0
        prog = (s - warmup) / max(1, total - start - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return (lambda st: cos(st, 0)), (lambda st: cos(st, lora_start))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--set", required=True, help="training set folder (cpt_trainer.retro.data + neighbours)")
    ap.add_argument("--db", default=None, help="retrieval database folder (cpt_trainer.retro.db)")
    ap.add_argument("--nbr-name", default=None, help="neighbour set under <set>/nbr/ (default: the database name)")
    ap.add_argument("--retro-config", default=os.path.join(ROOT, "configs", "retro", "retro_qwen06b_v1.yaml"))
    ap.add_argument("--lora-config", default=os.path.join(ROOT, "configs", "lora", "cpt_qwen06b_rs_adaptive_v5_r16_128.yaml"))
    ap.add_argument("--no-retro", action="store_true", help="control run: LoRA only, no retrieval layers")
    ap.add_argument("--k", type=int, default=None, help="neighbours per chunk in training (default: retro config)")
    # optimisation
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--max-seqs", type=int, default=None, help="smoke tests: cap training sequences")
    ap.add_argument("--lr", type=float, default=5e-5, help="LoRA learning rate")
    ap.add_argument("--retro-lr", type=float, default=2e-4, help="learning rate of the RETRO layers")
    ap.add_argument("--lora-start-frac", type=float, default=0.05, help="LoRA lr is 0 until this share of the steps")
    ap.add_argument("--oracle-rate", type=float, default=0.5, help="initial share of chunks with an oracle neighbour")
    ap.add_argument("--oracle-until", type=float, default=0.05, help="oracle rate decays linearly to 0 by this share of steps")
    ap.add_argument("--nbr-dropout", type=float, default=0.0, help="share of chunks trained without neighbours")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--scheduler", default="cosine", choices=["cosine", "constant"])
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--attn", default="flex_attention", choices=["flex_attention", "sdpa"],
                    help="decoder attention: flex (document BlockMask, GPU) or sdpa with a dense mask (CPU tests)")
    ap.add_argument("--pad-multiple", type=int, default=2048)
    ap.add_argument("--loss-chunk", type=int, default=4096)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    # evaluation
    ap.add_argument("--eval-at", default="0.25,0.5,0.75,1")
    ap.add_argument("--val-modes", default="on,off,random", help="validation loss modes at every eval point")
    ap.add_argument("--max-val-seqs", type=int, default=32)
    ap.add_argument("--eval-tasks", default="mmlu,mcq", help="cpt_trainer/retro/eval.py tasks at every eval point")
    ap.add_argument("--eval-modes", default="on,off", help="retrieval modes of those tasks")
    ap.add_argument("--eval-bs", type=int, default=8)
    ap.add_argument("--base-eval-from", default=os.path.join(ROOT, "runs", "cpt_qwen06b_rs_adaptive_v4_web5b", "eval_history.json"),
                    help="eval_history.json with the base model's epoch-0 scores on the SAME validation sequences")
    ap.add_argument("--no-eval", action="store_true", help="validation losses only, no MMLU / MCQ subprocess")
    # output
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--output-root", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from peft import get_peft_model
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    from transformers.trainer_callback import PrinterCallback

    set_seed(args.seed)
    base = os.path.abspath(args.base)
    num_layers, shapes = lora_plan.model_shapes(base)
    plan = lora_plan.load_plan(args.lora_config, num_layers)
    rcfg = None if args.no_retro else RetroConfig.from_file(args.retro_config)
    k = args.k or (rcfg.neighbours if rcfg else 0)
    run = args.run_name or f"retro_{time.strftime('%m%d_%H%M')}"
    out = os.path.join(args.output_root, run)
    os.makedirs(out, exist_ok=True)
    log = Logger(os.path.join(out, "train.log"))
    log(f"run {run} -> {out}")
    log("args: " + json.dumps(vars(args)))
    log(f"LoRA rank plan {plan.get('name')}, {plan['method']} alpha={plan['alpha']}:\n"
        + lora_plan.format_table(plan, shapes, plan["method"], plan["alpha"]))
    if rcfg:
        log(f"RETRO config {os.path.relpath(os.path.abspath(args.retro_config), ROOT)}: {json.dumps(rcfg.__dict__)} "
            f"-> {param_report(rcfg)}")

    # ---------------- data ----------------
    tok = load_tokenizer(base)
    set_dir = os.path.abspath(args.set)
    with open(os.path.join(set_dir, "meta.json")) as f:
        set_meta = json.load(f)
    db = Database(os.path.abspath(args.db)) if (rcfg and args.db) else None
    if rcfg and db is None:
        sys.exit("--db is required unless --no-retro")
    if db is not None and db.chunk_len != rcfg.chunk_len:
        sys.exit(f"database chunk_len {db.chunk_len} != retro config {rcfg.chunk_len}")
    nbr_name = (args.nbr_name or db.meta.get("name")) if db else None
    train_ds = RetroSplit(set_dir, "train", nbr_name, max_seqs=args.max_seqs)
    val_names = sorted(c for c in set_meta["stats"]["val"])
    val_ds = {c: RetroSplit(set_dir, f"val_{c}", nbr_name, max_seqs=args.max_val_seqs) for c in val_names}
    train_tokens = int(train_ds.lengths().sum())
    log(f"  training set {os.path.relpath(set_dir, ROOT)}: {len(train_ds):,} sequences, {train_tokens / 1e6:.1f} M tokens; "
        f"val {', '.join(f'{c} {len(v)}' for c, v in val_ds.items())} sequences"
        + (f"; database {db.meta['name']}: {len(db):,} chunks, {db.meta['tokens'] / 1e9:.2f} B tokens; k={k}" if db else ""))
    collator = RetroCollator(db, k, rcfg.nbr_len if rcfg else 0, tok.pad_token_id, args.pad_multiple, seed=args.seed)
    collator.nbr_dropout = args.nbr_dropout

    # ---------------- history (epoch 0 = base model) ----------------
    hist_path = os.path.join(out, "eval_history.json")
    history: list[dict] = []
    if args.resume and os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)

    def save_history():
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

    if not any(h.get("epoch") == 0 for h in history) and args.base_eval_from and os.path.exists(args.base_eval_from):
        with open(args.base_eval_from) as f:
            prev = next((h for h in json.load(f) if h.get("epoch") == 0), None)
        if prev:
            # same validation documents / packing (cpt_trainer/retro/data.py), so the base val losses carry over
            history.append({**prev, "copied_from": os.path.relpath(os.path.abspath(args.base_eval_from), ROOT)})
            save_history()
            log(f"epoch 0: base-model scores copied from {args.base_eval_from}")

    # ---------------- model ----------------
    if args.attn == "flex_attention":
        torch._dynamo.config.recompile_limit = 64
        torch._dynamo.config.cache_size_limit = 64
    model = load_model(base, for_training=True, attn=args.attn)
    model.config.use_cache = False
    if rcfg:
        attach_retro(model, rcfg, dtype=next(model.parameters()).dtype)
    lcfg = lora_plan.build_config(plan, plan["method"], plan["alpha"], plan.get("dropout", 0.0))
    if rcfg:
        lcfg.modules_to_save = ["retro"]
    model = get_peft_model(model, lcfg)
    lora_plan.check_model(model, plan)
    grad_ckpt = not args.no_grad_ckpt
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    retro_p, lora_p, other_p = trainable_split(model)
    if other_p:
        sys.exit(f"unexpected trainable parameters: {len(other_p)} tensors")
    n_retro, n_lora = sum(p.numel() for p in retro_p), sum(p.numel() for p in lora_p)
    log(f"  trainable: RETRO {n_retro / 1e6:.1f} M + LoRA {n_lora / 1e6:.1f} M = {(n_retro + n_lora) / 1e6:.1f} M "
        f"(web5b adapter: 158.3 M)")

    steps_per_epoch = max(1, math.ceil(len(train_ds) / (args.bs * args.grad_accum)))
    total = args.max_steps if args.max_steps > 0 else int(math.ceil(steps_per_epoch * args.epochs))
    warm = max(1, int(total * args.warmup))
    lora_start = int(total * args.lora_start_frac)
    oracle_until = int(total * args.oracle_until)
    eval_points = {}
    for fr in sorted(float(x) for x in args.eval_at.split(",") if x.strip()):
        eval_points[min(total, max(1, int(round(fr * steps_per_epoch))))] = min(fr, total / steps_per_epoch)
    log(f"  {total} optimizer steps ({args.bs} x {args.grad_accum} sequences, ~{args.grad_accum * train_tokens / max(1, len(train_ds)) / 1e3:.0f} k tokens "
        f"per step); warmup {warm}; LoRA starts at step {lora_start}; oracle neighbours {args.oracle_rate:.2f} -> 0 by step "
        f"{oracle_until}; evaluation at steps {list(eval_points)}")

    # ---------------- validation (in-process) ----------------
    @torch.no_grad()
    def val_losses(m, modes: list[str]) -> dict:
        was = m.training
        m.eval()
        res = {}
        for mode in modes:
            if mode != "off" and not rcfg:
                continue
            col = RetroCollator(db, k, rcfg.nbr_len if rcfg else 0, tok.pad_token_id, args.pad_multiple, seed=args.seed + 1,
                                mode=mode)
            for c, ds in val_ds.items():
                tot = n = 0.0
                for i in range(len(ds)):
                    b = col([ds[i]])
                    b = {kk: v.to(m.device) if torch.is_tensor(v) else v for kk, v in b.items()}
                    y = b["labels"][0, 1:]
                    cnt = float((y != -100).sum())
                    tot += float(retro_loss(m, b, args.attn, retrieval=mode != "off", loss_chunk=args.loss_chunk)) * cnt
                    n += cnt
                res.setdefault(mode, {})[c] = tot / max(1.0, n)
        if was:
            m.train()
        return res

    # ---------------- trainer ----------------
    class RetroTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if model.training and num_items_in_batch is None:
                raise RuntimeError("Trainer did not pass num_items_in_batch; the loss would be mis-scaled")
            loss = retro_loss(model, inputs, args.attn, num_items_in_batch if model.training else None, args.loss_chunk,
                              retrieval=rcfg is not None, grad_ckpt=grad_ckpt and model.training)
            return (loss, {}) if return_outputs else loss

        def create_optimizer(self, model=None):
            if self.optimizer is None:
                groups = []
                if retro_p:
                    groups.append({"params": retro_p, "lr": args.retro_lr, "weight_decay": args.weight_decay})
                groups.append({"params": lora_p, "lr": args.lr, "weight_decay": 0.0})
                self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8,
                                                   fused=torch.cuda.is_available())
            return self.optimizer

        def create_scheduler(self, num_training_steps, optimizer=None):
            if self.lr_scheduler is None:
                opt = optimizer or self.optimizer
                f_retro, f_lora = lr_lambdas(total, warm, lora_start, args.scheduler)
                fns = ([f_retro] if retro_p else []) + [f_lora]
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, fns)
            return self.lr_scheduler

        def _save(self, output_dir=None, state_dict=None):
            super()._save(output_dir, state_dict)
            if rcfg:
                rcfg.save(output_dir or self.args.output_dir)

    def cca_weight_norms(m) -> float | None:
        if not rcfg:
            return None
        lm, _ = unwrap(m)
        mods = lm.retro.modules_to_save["default"] if hasattr(lm.retro, "modules_to_save") else lm.retro
        return float(np.mean([float(c.attn.to_out.weight.float().norm()) for c in mods.cca.values()]))

    def run_eval(adapter_dir: str, tag: str) -> dict:
        if args.no_eval or not args.eval_tasks or not rcfg:
            return {}
        cmd = [sys.executable, "-m", "cpt_trainer.retro.eval", "--model", adapter_dir, "--base", base, "--db", db.path,
               "--tasks", args.eval_tasks, "--modes", args.eval_modes, "--bs", str(args.eval_bs), "--tag", tag, "--k", str(k)]
        lp = os.path.join(out, "eval_logs", f"{tag}.log")
        os.makedirs(os.path.dirname(lp), exist_ok=True)
        log(f"  eval {' '.join(cmd[2:])}  (log {os.path.relpath(lp, ROOT)})")
        t0 = time.time()
        with open(lp, "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        sp = os.path.join(ROOT, "eval_results", tag, "summary.json")
        if rc != 0 or not os.path.exists(sp):
            return {"eval_error": f"exit {rc}, see {lp}"}
        with open(sp) as f:
            s = json.load(f)
        res = {"eval_seconds": round(time.time() - t0)}
        for mode, r in s.get("modes", {}).items():
            sfx = "" if mode == "on" else f"_{mode}"
            for key in ("accuracy", "macro_accuracy", "ai_subjects_accuracy", "ai_related_accuracy", "ai_large_accuracy",
                        "letter_prob_mass", "mcq_accuracy"):
                if r.get(key) is not None:
                    res[key + sfx] = r[key]
        return res

    class Callback(TrainerCallback):
        def on_step_begin(self, targs, state, control, **kw):
            st = state.global_step
            collator.step = st
            collator.oracle_rate = args.oracle_rate * max(0.0, 1 - st / oracle_until) if oracle_until > 0 else 0.0

        def on_log(self, targs, state, control, logs=None, model=None, **kw):
            if not logs:
                return
            parts = [f"step {state.global_step}/{state.max_steps}", f"epoch {logs.get('epoch', state.epoch or 0):.2f}"]
            for key in ("loss", "grad_norm"):
                if key in logs:
                    parts.append(f"{key} {logs[key]:.4g}")
            lrs = self_trainer.lr_scheduler.get_last_lr() if self_trainer.lr_scheduler else []
            if lrs:
                parts.append("lr " + "/".join(f"{x:.2e}" for x in lrs) + (" (retro/lora)" if len(lrs) == 2 else ""))
            if rcfg:
                parts.append(f"oracle {collator.oracle_rate:.2f}")
                w = cca_weight_norms(model if model is not None else self_trainer.model)
                parts.append(f"|W_o| {w:.3f}")
            if "num_input_tokens_seen" in logs:
                parts.append(f"{logs['num_input_tokens_seen'] / 1e6:.1f} M tok")
            log("  " + " | ".join(parts))

        def on_step_end(self, targs, state, control, **kw):
            if state.global_step in eval_points:
                control.should_save = True

        def on_save(self, targs, state, control, model=None, **kw):
            st = state.global_step
            if st not in eval_points or any(h.get("step") == st for h in history):
                return
            ep = epoch_label(eval_points[st])
            adapter_dir = os.path.join(out, f"epoch{ep}", "adapter")
            model.save_pretrained(adapter_dir)
            tok.save_pretrained(adapter_dir)
            if rcfg:
                rcfg.save(adapter_dir)
            prev = max([h.get("step", 0) for h in history] + [0])
            t0 = time.time()
            vl = val_losses(model, [m for m in args.val_modes.split(",") if m])
            entry = {"epoch": ep, "step": st, "train_loss": interval_train_loss(state.log_history, prev, st),
                     "val_loss": vl.get("on") or vl.get("off"), "val_loss_modes": vl,
                     "adapter": os.path.relpath(adapter_dir, ROOT), "cca_wo_norm": cca_weight_norms(model)}
            log(f"epoch {ep}: adapter -> {entry['adapter']}; train loss {entry['train_loss']}; val loss by mode "
                f"{json.dumps({m: {c: round(v, 4) for c, v in d.items()} for m, d in vl.items()})} ({time.time() - t0:.0f}s)")
            torch.cuda.empty_cache()
            entry.update(run_eval(adapter_dir, f"{run}_epoch{ep}"))
            history.append(entry)
            save_history()
            log("\n" + history_table() + "\n")

    def history_table() -> str:
        modes = [m for m in ("on", "off", "random") if any(m in (h.get("val_loss_modes") or {}) for h in history)]
        head = ["epoch", "step", "train loss"] + [f"val {c} ({m})" for c in val_names for m in modes or ["on"]] + \
               ["MMLU", "MMLU off", "AI subj.", "AI-rel.", "ai_large", "ai_large off", "MCQ", "MCQ off"]
        rows = ["| " + " | ".join(head) + " |", "|" + "---:|" * len(head)]
        for h in history:
            vm = h.get("val_loss_modes") or ({"on": h["val_loss"]} if h.get("val_loss") else {})
            cells = [str(h.get("epoch")), str(h.get("step", 0)), f"{h['train_loss']:.4f}" if h.get("train_loss") else ""]
            for c in val_names:
                for m in modes or ["on"]:
                    v = (vm.get(m) or {}).get(c) if h.get("epoch") != 0 else (h.get("val_loss") or {}).get(c)
                    cells.append(f"{v:.4f}" if v is not None else "")
            for key in ("accuracy", "accuracy_off", "ai_subjects_accuracy", "ai_related_accuracy", "ai_large_accuracy",
                        "ai_large_accuracy_off", "mcq_accuracy", "mcq_accuracy_off"):
                v = h.get(key)
                cells.append(f"{v:.4f}" if isinstance(v, (int, float)) else "")
            rows.append("| " + " | ".join(cells) + " |")
        return "\n".join(rows)

    targs = TrainingArguments(
        output_dir=out, run_name=run, seed=args.seed, num_train_epochs=args.epochs, max_steps=total,
        per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm,
        bf16=torch.cuda.is_available(), logging_steps=args.logging_steps, report_to="none", disable_tqdm=True,
        eval_strategy="no", save_strategy="steps", save_steps=args.save_steps or total + 1, save_total_limit=2,
        remove_unused_columns=False, dataloader_num_workers=args.num_workers, include_num_input_tokens_seen=True,
        use_cpu=not torch.cuda.is_available())
    trainer = RetroTrainer(model=model, args=targs, train_dataset=train_ds, data_collator=collator, callbacks=[Callback()])
    self_trainer = trainer
    trainer.loss_is_scaled_for_ga = True   # retro_loss divides by the token count of the accumulated batch
    trainer.remove_callback(PrinterCallback)

    with open(os.path.join(out, "run_config.json"), "w") as f:
        json.dump({"args": vars(args), "base": base, "set_meta": set_meta, "db_meta": {k_: v for k_, v in (db.meta if db else {}).items()
                   if k_ not in ("contamination_examples",)}, "retro": rcfg.__dict__ if rcfg else None,
                   "lora_plan": lora_plan.resolved_json(plan, plan["method"], plan["alpha"], plan.get("dropout", 0.0), shapes),
                   "trainable": {"retro": n_retro, "lora": n_lora}, "total_steps": total,
                   "eval_points": {str(k_): v for k_, v in eval_points.items()}}, f, indent=2, default=str)

    resume = None
    if args.resume:
        cks = sorted([d for d in os.listdir(out) if d.startswith("checkpoint-")], key=lambda d: int(d.split("-")[1]))
        resume = os.path.join(out, cks[-1]) if cks else None
        log(f"resuming from {resume}" if resume else "no checkpoint to resume from; starting fresh")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    minutes = (time.time() - t0) / 60

    final = os.path.join(out, "adapter")
    trainer.model.save_pretrained(final)
    tok.save_pretrained(final)
    if rcfg:
        rcfg.save(final)
    with open(os.path.join(out, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    lines = [f"# RETRO-fit run `{run}`", "",
             f"- base `{os.path.relpath(base, ROOT)}`; LoRA {plan.get('name')} ({n_lora / 1e6:.1f} M)"
             + (f" + RETRO layers ({n_retro / 1e6:.1f} M: {len(rcfg.cca_layers)} CCA layers at {rcfg.cca_layers}, "
                f"{rcfg.enc_layers}-layer encoder d={rcfg.enc_dim}; chunk {rcfg.chunk_len}, k={k}, neighbour {rcfg.nbr_len} tokens)"
                if rcfg else " (no retrieval: control run)"),
             f"- lr LoRA {args.lr} (from step {lora_start}), RETRO {args.retro_lr}; {args.scheduler}; {args.bs} x {args.grad_accum} "
             f"sequences per step; {trainer.state.global_step} steps, {minutes:.0f} min",
             f"- data: {os.path.relpath(set_dir, ROOT)} ({train_tokens / 1e6:.0f} M tokens)"
             + (f"; database {db.meta['name']} ({db.meta['tokens'] / 1e9:.2f} B tokens, {len(db):,} chunks)" if db else ""),
             "", "## Scores", "", history_table(), ""]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("\n".join(lines))


if __name__ == "__main__":
    main()
