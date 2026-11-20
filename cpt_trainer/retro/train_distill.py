#!/usr/bin/env python3
"""LoRA CPT of Qwen3-0.6B-Base with retrieval as a training-only teacher (docs/retro_distill_lora_cpt.md).

    python -m cpt_trainer.retro.train_distill --set runs/_retro_cache/web1b --db cpt_data/retro_db/ai_v1 \\
        --teacher context --run-name rkd_ctx_qwen06b_v1_web1b          # see runs/launch_rkd_qwen06b_v1_web1b.sh
    ... --teacher retro  --run-name rkd_retro_qwen06b_v1_web1b          # RETRO-layer teacher, dropped after training
    ... --teacher none   --run-name lora_qwen06b_v5_web1b               # control: same LoRA, data and order, CE only
    ... --resume

The deployed model is the base + the LoRA only. Per token the student loss is
    (1 - alpha) * CE + alpha * gate * KL(teacher || student)
with the teacher (cpt_trainer/retro/distill.py):
  context  the same decoder reading up to --ctx-k retrieved neighbours placed before every --ctx-window tokens;
           --teacher-weights self = with the current LoRA (default: the teacher keeps up with the student and the
           KL carries only what the neighbours add), base = the frozen base model (also anchors the student to it)
  retro    base + RETRO layers (configs/retro/*.yaml) trained in the same step on the teacher path (LoRA switched
           off) with their own CE; the student never sees them
  none     plain CE (control run; same data order, same evaluation)
--kd-gate teacher_better applies the KL only where the teacher gives the true token more probability than the
student (the tokens the neighbours helped with).

Evaluation (the student alone, exactly as the web5b run): at --eval-at, the plain LoRA adapter is saved to
epoch<x>/adapter and scored by scripts/eval_model.py (cpt_trainer.mmlu.run_eval: 5-shot MMLU incl. ai_large, the AI
MCQ; --final-eval-tasks at the last point); at every --val-every the validation losses of the student and of the
teacher (the gap = what the teacher still knows that the student does not) on the web5b validation sequences.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ai_qa_train.common import ROOT, load_model, load_tokenizer  # noqa: E402
from cpt_trainer import lora_plan  # noqa: E402
from cpt_trainer.longctx import chunked_lm_loss  # noqa: E402
from cpt_trainer.mmlu import DEFAULT_TASKS, history_table, run_eval  # noqa: E402
from cpt_trainer.retro.dataset import RetroCollator, RetroSplit  # noqa: E402
from cpt_trainer.retro.db import Database  # noqa: E402
from cpt_trainer.retro.distill import (ContextTeacher, context_teacher_hidden, export_student, kd_loss,  # noqa: E402
                                       lora_disabled, lora_layers)
from cpt_trainer.retro.model import (RetroConfig, attach_retro, attention_mask_for, param_report,  # noqa: E402
                                     retro_hidden, trainable_split, unwrap)
from cpt_trainer.train import Logger, epoch_label, interval_train_loss  # noqa: E402

DEFAULT_BASE = os.path.join(ROOT, "models", "Qwen3-0.6B-Base")


class DistillCollator:
    """RetroCollator (mode "on" for the RETRO teacher: per-chunk neighbour tokens; "off" otherwise) + the context
    teacher's sequence (t_* fields)."""

    def __init__(self, retro_col: RetroCollator, ctx: ContextTeacher | None):
        self.retro_col, self.ctx = retro_col, ctx

    @property
    def step(self):
        return self.retro_col.step

    @step.setter
    def step(self, v):
        self.retro_col.step = v

    @property
    def oracle_rate(self):
        return self.retro_col.oracle_rate

    @oracle_rate.setter
    def oracle_rate(self, v):
        self.retro_col.oracle_rate = v

    def __call__(self, batch: list[dict]) -> dict:
        out = self.retro_col(batch)
        if self.ctx is not None:
            out.update(self.ctx.build(batch[0], out))
        return out


def lr_lambda(total: int, warmup: int, scheduler: str):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        if scheduler == "constant":
            return 1.0
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    return f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--set", required=True, help="training set folder (cpt_trainer.retro.data + neighbours)")
    ap.add_argument("--db", default=None, help="retrieval database folder (cpt_trainer.retro.db)")
    ap.add_argument("--nbr-name", default=None, help="neighbour set under <set>/nbr/ (default: the database name)")
    ap.add_argument("--lora-config", default=os.path.join(ROOT, "configs", "lora", "cpt_qwen06b_rs_adaptive_v5_r16_128.yaml"))
    # teacher / distillation
    ap.add_argument("--teacher", default="context", choices=["context", "retro", "none"])
    ap.add_argument("--teacher-weights", default="self", choices=["self", "base"],
                    help="context teacher: current LoRA (self) or the frozen base model (base)")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of the KL term (1 - alpha on the CE)")
    ap.add_argument("--kd-gate", default="none", choices=["none", "teacher_better"])
    ap.add_argument("--kd-start-frac", type=float, default=0.0, help="alpha = 0 before this share of the steps")
    ap.add_argument("--kd-chunk", type=int, default=512, help="tokens per logits chunk of the KD loss")
    ap.add_argument("--ctx-k", type=int, default=4, help="context teacher: neighbours per window")
    ap.add_argument("--ctx-nbr-len", type=int, default=128, help="context teacher: tokens per neighbour")
    ap.add_argument("--ctx-window", type=int, default=1024, help="context teacher: window length (multiple of 64)")
    ap.add_argument("--ctx-max-overlap", type=float, default=0.5,
                    help="drop neighbours whose 8-grams lie in the window at least this much (copies)")
    ap.add_argument("--ctx-sep", default="\n\n", help="text appended to every neighbour in the teacher's context")
    ap.add_argument("--retro-config", default=os.path.join(ROOT, "configs", "retro", "retro_qwen06b_v1.yaml"))
    ap.add_argument("--retro-lr", type=float, default=2e-4)
    ap.add_argument("--retro-loss-weight", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=None, help="RETRO teacher: neighbours per chunk (default: retro config)")
    ap.add_argument("--oracle-rate", type=float, default=0.5, help="RETRO teacher: initial share of oracle neighbours")
    ap.add_argument("--oracle-until", type=float, default=0.05)
    # optimisation
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--max-seqs", type=int, default=None, help="smoke tests: cap training sequences")
    ap.add_argument("--lr", type=float, default=5e-5, help="LoRA learning rate")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--scheduler", default="cosine", choices=["cosine", "constant"])
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--attn", default="flex_attention", choices=["flex_attention", "sdpa"])
    ap.add_argument("--pad-multiple", type=int, default=2048)
    ap.add_argument("--loss-chunk", type=int, default=4096)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    # evaluation
    ap.add_argument("--eval-at", default="0.25,0.5,0.75,1", help="adapter export + eval_model.py")
    ap.add_argument("--val-every", type=float, default=0.1, help="validation losses (student, teacher) every this share")
    ap.add_argument("--max-val-seqs", type=int, default=32)
    ap.add_argument("--no-teacher-val", action="store_true", help="student validation loss only")
    ap.add_argument("--eval-tasks", default="mmlu,mcq")
    ap.add_argument("--final-eval-tasks", default=DEFAULT_TASKS)
    ap.add_argument("--mmlu-shots", type=int, default=5)
    ap.add_argument("--eval-bs", type=int, default=8)
    ap.add_argument("--eval-chat-model", default=os.path.join(ROOT, "models", "Qwen3-0.6B"),
                    help="open_qa runs on this model with the adapter (as the web5b run); empty = none")
    ap.add_argument("--base-eval-from", default=os.path.join(ROOT, "runs", "cpt_qwen06b_rs_adaptive_v4_web5b", "eval_history.json"))
    ap.add_argument("--no-eval", action="store_true", help="validation losses only")
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
    teacher = args.teacher
    rcfg = RetroConfig.from_file(args.retro_config) if teacher == "retro" else None
    k = args.k or (rcfg.neighbours if rcfg else 0)
    run = args.run_name or f"rkd_{teacher}_{time.strftime('%m%d_%H%M')}"
    out = os.path.join(args.output_root, run)
    os.makedirs(out, exist_ok=True)
    log = Logger(os.path.join(out, "train.log"))
    log(f"run {run} -> {out}")
    log("args: " + json.dumps(vars(args)))
    log(f"LoRA rank plan {plan.get('name')}, {plan['method']} alpha={plan['alpha']}:\n"
        + lora_plan.format_table(plan, shapes, plan["method"], plan["alpha"]))
    if rcfg:
        log(f"RETRO teacher {os.path.relpath(os.path.abspath(args.retro_config), ROOT)}: {json.dumps(rcfg.__dict__)} -> {param_report(rcfg)}")

    # ---------------- data ----------------
    tok = load_tokenizer(base)
    set_dir = os.path.abspath(args.set)
    with open(os.path.join(set_dir, "meta.json")) as f:
        set_meta = json.load(f)
    db = None
    if teacher != "none":
        if not args.db:
            sys.exit(f"--db is required with --teacher {teacher}")
        db = Database(os.path.abspath(args.db))
        if rcfg and db.chunk_len != rcfg.chunk_len:
            sys.exit(f"database chunk_len {db.chunk_len} != retro config {rcfg.chunk_len}")
    nbr_name = (args.nbr_name or db.meta.get("name")) if db else None
    train_ds = RetroSplit(set_dir, "train", nbr_name, max_seqs=args.max_seqs)
    val_names = sorted(c for c in set_meta["stats"]["val"])
    val_ds = {c: RetroSplit(set_dir, f"val_{c}", nbr_name, max_seqs=args.max_val_seqs) for c in val_names}
    train_tokens = int(train_ds.lengths().sum())
    log(f"  training set {os.path.relpath(set_dir, ROOT)}: {len(train_ds):,} sequences, {train_tokens / 1e6:.1f} M tokens; "
        f"val {', '.join(f'{c} {len(v)}' for c, v in val_ds.items())} sequences"
        + (f"; database {db.meta['name']}: {len(db):,} chunks, {db.meta['tokens'] / 1e9:.2f} B tokens" if db else ""))

    def make_ctx():
        if teacher != "context":
            return None
        sep = tok.encode(args.ctx_sep, add_special_tokens=False) if args.ctx_sep else []
        return ContextTeacher(db, k=args.ctx_k, nbr_len=args.ctx_nbr_len, window=args.ctx_window, sep_ids=sep,
                              max_overlap=args.ctx_max_overlap, pad_multiple=args.pad_multiple, pad_id=tok.pad_token_id)

    def make_collator(seed: int):
        rc = RetroCollator(db, k, rcfg.nbr_len if rcfg else 0, tok.pad_token_id, args.pad_multiple, seed=seed,
                           mode="on" if teacher == "retro" else "off")
        return DistillCollator(rc, make_ctx())

    collator = make_collator(args.seed)

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
    loras = lora_layers(model)
    n_retro, n_lora = sum(p.numel() for p in retro_p), sum(p.numel() for p in lora_p)
    log(f"  trainable: LoRA {n_lora / 1e6:.1f} M (deployed)" + (f" + RETRO teacher layers {n_retro / 1e6:.1f} M (dropped after training)"
                                                                if retro_p else "") + "; web5b adapter: 158.3 M")
    W = unwrap(model)[0].lm_head.weight

    steps_per_epoch = max(1, math.ceil(len(train_ds) / (args.bs * args.grad_accum)))
    total = args.max_steps if args.max_steps > 0 else int(math.ceil(steps_per_epoch * args.epochs))
    warm = max(1, int(total * args.warmup))
    kd_start = int(total * args.kd_start_frac)
    oracle_until = int(total * args.oracle_until) if rcfg else 0

    def points(fracs):
        res = {}
        for fr in sorted(fracs):
            res[min(total, max(1, int(round(fr * steps_per_epoch))))] = min(fr, total / steps_per_epoch)
        return res
    eval_points = points(float(x) for x in args.eval_at.split(",") if x.strip())
    val_points = points([args.val_every * i for i in range(1, int(1 / args.val_every + 1e-9) + 1)]) if args.val_every > 0 else {}
    val_points = {s: f for s, f in val_points.items() if s not in eval_points}
    final_step = max(eval_points) if eval_points else total
    log(f"  {total} optimizer steps ({args.bs} x {args.grad_accum} sequences, ~{args.grad_accum * train_tokens / max(1, len(train_ds)) / 1e3:.0f} k "
        f"tokens per step); warmup {warm}; teacher {teacher}"
        + (f" ({args.teacher_weights} weights, k={args.ctx_k} x {args.ctx_nbr_len} tokens per {args.ctx_window}-token window)" if teacher == "context" else "")
        + (f"; alpha {args.alpha} from step {kd_start}, gate {args.kd_gate}" if teacher != "none" else "")
        + f"; evals at steps {list(eval_points)}; validation-only points {list(val_points)}")

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
            history.append({**prev, "copied_from": os.path.relpath(os.path.abspath(args.base_eval_from), ROOT)})
            save_history()
            log(f"epoch 0: base-model scores copied from {args.base_eval_from} (same validation sequences)")

    # ---------------- losses ----------------
    @torch.no_grad()
    def teacher_forward(m, b: dict, mask):
        """Teacher hidden states [Lp, d] for validation (training: DistillTrainer.compute_loss)."""
        if teacher == "context":
            return context_teacher_hidden(m, b, args.attn, loras, use_lora=args.teacher_weights == "self")
        with lora_disabled(loras):
            return retro_hidden(m, b, mask, retrieval=True)

    @torch.no_grad()
    def val_losses(m, with_teacher: bool) -> dict:
        was = m.training
        m.eval()
        col = make_collator(args.seed + 1)
        res = {"student": {}, "teacher": {}}
        for c, ds in val_ds.items():
            n = 0.0
            tot = {"student": 0.0, "teacher": 0.0}
            for i in range(len(ds)):
                b = col([ds[i]])
                b = {kk: v.to(W.device) if torch.is_tensor(v) else v for kk, v in b.items()}
                y = b["labels"][0, 1:]
                cnt = float((y != -100).sum())
                mask = attention_mask_for(b["doc_ids"][0], args.attn)
                hs = retro_hidden(m, b, mask, retrieval=False)
                tot["student"] += float(chunked_lm_loss(hs[:-1], W, y, args.loss_chunk))
                if with_teacher:
                    ht = teacher_forward(m, b, mask)
                    tot["teacher"] += float(chunked_lm_loss(ht[:-1], W, y, args.loss_chunk))
                n += cnt
            for key in tot:
                if key == "student" or with_teacher:
                    res[key][c] = tot[key] / max(1.0, n)
        if was:
            m.train()
        return res

    # ---------------- trainer ----------------
    class DistillTrainer(Trainer):
        stats = None          # [ce, kl, teacher ce, gated, tokens] since the last log
        ctx_stats = None      # [windows, windows with neighbours, neighbours, neighbour tokens]
        retro_ce = None
        alpha = 0.0

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if not model.training:
                raise RuntimeError("evaluation goes through val_losses")
            if num_items_in_batch is None:
                raise RuntimeError("Trainer did not pass num_items_in_batch; the loss would be mis-scaled")
            dev = inputs["input_ids"].device
            if self.stats is None:
                self.stats = torch.zeros(5, device=dev)
                self.ctx_stats = torch.zeros(4, device=dev)
                self.retro_ce = torch.zeros(2, device=dev)
            y = inputs["labels"][0, 1:]
            mask = attention_mask_for(inputs["doc_ids"][0], args.attn)
            ht = None
            if teacher == "retro":
                # teacher path: base + RETRO layers, LoRA off; its own CE trains the RETRO layers. Backward now, inside
                # the LoRA-off context, so that recomputed checkpoints see the same weights.
                with lora_disabled(loras):
                    th = retro_hidden(model, inputs, mask, retrieval=True, grad_ckpt=grad_ckpt)
                    tl = chunked_lm_loss(th[:-1], W, y, args.loss_chunk)
                    self.accelerator.backward(tl * (args.retro_loss_weight / num_items_in_batch))
                ht = th.detach()
                self.retro_ce += torch.stack([tl.detach(), (y != -100).sum().float()])
            elif teacher == "context" and self.alpha > 0:
                ht = context_teacher_hidden(model, inputs, args.attn, loras, use_lora=args.teacher_weights == "self")
                self.ctx_stats += inputs["t_stats"].float()
            hs = retro_hidden(model, inputs, mask, retrieval=False)
            if ht is None or self.alpha <= 0:
                loss_sum = chunked_lm_loss(hs[:-1], W, y, args.loss_chunk)
                self.stats += torch.stack([loss_sum.detach(), loss_sum.new_zeros(()), loss_sum.new_zeros(()),
                                           loss_sum.new_zeros(()), (y != -100).sum().float()])
            else:
                loss_sum, st = kd_loss(hs[:-1], ht[:-1], W, y, self.alpha, args.kd_gate, args.kd_chunk)
                self.stats += st
            loss = loss_sum / num_items_in_batch
            return (loss, {}) if return_outputs else loss

        def create_optimizer(self, model=None):
            if self.optimizer is None:
                groups = [{"params": lora_p, "lr": args.lr, "weight_decay": 0.0}]
                if retro_p:
                    groups.append({"params": retro_p, "lr": args.retro_lr, "weight_decay": args.weight_decay})
                self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, fused=torch.cuda.is_available())
                # gradient clipping per parameter group (TrainingArguments.max_grad_norm is 0): the randomly initialised
                # RETRO teacher layers have large early gradients that would otherwise shrink the LoRA's update
                trainer_ref = self

                def clip(opt, *a, **kw):
                    mx = args.max_grad_norm if args.max_grad_norm > 0 else float("inf")
                    trainer_ref.group_norms = [
                        float(torch.nn.utils.clip_grad_norm_([p for p in g["params"] if p.grad is not None], mx))
                        if any(p.grad is not None for p in g["params"]) else 0.0 for g in opt.param_groups]
                self.optimizer.register_step_pre_hook(clip)
            return self.optimizer

        def create_scheduler(self, num_training_steps, optimizer=None):
            if self.lr_scheduler is None:
                opt = optimizer or self.optimizer
                f = lr_lambda(total, warm, args.scheduler)
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, [f] * len(opt.param_groups))
            return self.lr_scheduler

        def _save(self, output_dir=None, state_dict=None):
            super()._save(output_dir, state_dict)
            if rcfg:
                rcfg.save(output_dir or self.args.output_dir)

    kd_hist: dict = {"since": 0}

    def take_stats(tr) -> dict:
        """Mean distillation statistics since the last call (per target token / per window)."""
        res = {}
        if tr.stats is not None and float(tr.stats[4]) > 0:
            s = tr.stats.tolist()
            res = {"ce": s[0] / s[4], "kl": s[1] / s[4], "teacher_ce": s[2] / s[4] if s[2] else None, "gated": s[3] / s[4]}
            if tr.ctx_stats is not None and float(tr.ctx_stats[0]) > 0:
                c = tr.ctx_stats.tolist()
                res.update({"win_with_nbr": c[1] / c[0], "nbr_per_win": c[2] / c[0], "nbr_tok_per_win": c[3] / c[0]})
            if tr.retro_ce is not None and float(tr.retro_ce[1]) > 0:
                res["retro_teacher_ce"] = float(tr.retro_ce[0] / tr.retro_ce[1])
            tr.stats.zero_(); tr.ctx_stats.zero_(); tr.retro_ce.zero_()
        return res

    interval: dict = {"acc": None}

    def merge_interval(st: dict):
        """Token-weighted means over an evaluation interval are approximated by the mean of the logged means."""
        if not st:
            return
        a = interval["acc"] or {"n": 0}
        for kk, v in st.items():
            if v is not None:
                a[kk] = a.get(kk, 0.0) + v
        a["n"] += 1
        interval["acc"] = a

    def pop_interval() -> dict | None:
        a = interval["acc"]
        interval["acc"] = None
        if not a:
            return None
        n = a.pop("n")
        return {kk: round(v / n, 4) for kk, v in a.items()}

    class Callback(TrainerCallback):
        def on_step_begin(self, targs, state, control, **kw):
            st = state.global_step
            collator.step = st
            if rcfg:
                collator.oracle_rate = args.oracle_rate * max(0.0, 1 - st / oracle_until) if oracle_until > 0 else 0.0
            trainer.alpha = args.alpha if (teacher != "none" and st >= kd_start) else 0.0

        def on_log(self, targs, state, control, logs=None, **kw):
            if not logs:
                return
            parts = [f"step {state.global_step}/{state.max_steps}", f"epoch {logs.get('epoch', state.epoch or 0):.2f}"]
            if "loss" in logs:
                parts.append(f"loss {logs['loss']:.4g}")
            gn = getattr(trainer, "group_norms", None)
            if gn:
                parts.append("grad_norm " + "/".join(f"{x:.4g}" for x in gn))
            lrs = trainer.lr_scheduler.get_last_lr() if trainer.lr_scheduler else []
            if lrs:
                parts.append("lr " + "/".join(f"{x:.2e}" for x in lrs) + (" (lora/retro)" if len(lrs) == 2 else ""))
            st = take_stats(trainer)
            merge_interval(st)
            if st:
                parts.append(f"ce {st['ce']:.4f}")
                if teacher != "none":
                    parts.append(f"kl {st['kl']:.4f}")
                    if st.get("teacher_ce"):
                        parts.append(f"teacher ce {st['teacher_ce']:.4f}")
                    if args.kd_gate != "none":
                        parts.append(f"gated {st['gated']:.2f}")
                    if "nbr_per_win" in st:
                        parts.append(f"nbr/win {st['nbr_per_win']:.2f} ({st['nbr_tok_per_win']:.0f} tok)")
                    if "retro_teacher_ce" in st:
                        parts.append(f"retro ce {st['retro_teacher_ce']:.4f} oracle {collator.oracle_rate:.2f}")
            if "num_input_tokens_seen" in logs:
                parts.append(f"{logs['num_input_tokens_seen'] / 1e6:.1f} M tok")
            log("  " + " | ".join(parts))

        def on_step_end(self, targs, state, control, **kw):
            if state.global_step in eval_points or state.global_step in val_points:
                control.should_save = True

        def on_save(self, targs, state, control, model=None, **kw):
            st = state.global_step
            is_eval, is_val = st in eval_points, st in val_points
            if not (is_eval or is_val) or any(h.get("step") == st for h in history):
                return
            frac = eval_points.get(st, val_points.get(st))
            ep = epoch_label(frac)
            prev = max([h.get("step", 0) for h in history] + [0])
            t0 = time.time()
            vl = val_losses(model, with_teacher=teacher != "none" and not args.no_teacher_val)
            entry = {"epoch": ep, "step": st, "train_loss": interval_train_loss(state.log_history, prev, st),
                     "val_loss": vl["student"], "teacher_val_loss": vl["teacher"] or None, "kd": pop_interval()}
            gap = {c: round(vl["student"][c] - vl["teacher"][c], 4) for c in vl["teacher"]} if vl["teacher"] else {}
            log(f"epoch {ep}: val loss student {json.dumps({c: round(v, 4) for c, v in vl['student'].items()})}"
                + (f"; teacher {json.dumps({c: round(v, 4) for c, v in vl['teacher'].items()})}; gap {json.dumps(gap)}" if gap else "")
                + f" ({time.time() - t0:.0f}s)")
            if is_eval:
                adapter_dir = os.path.join(out, f"epoch{ep}", "adapter")
                export_student(model, adapter_dir, tok)
                entry["adapter"] = os.path.relpath(adapter_dir, ROOT)
                torch.cuda.empty_cache()
                if not args.no_eval and args.eval_tasks:
                    tasks = args.final_eval_tasks if st == final_step and args.final_eval_tasks else args.eval_tasks
                    try:
                        entry.update(run_eval(adapter_dir, f"{run}_epoch{ep}", tasks=tasks, shots=args.mmlu_shots,
                                              bs=args.eval_bs, fmt="base",
                                              chat_model=os.path.abspath(args.eval_chat_model) if args.eval_chat_model else None,
                                              log_path=os.path.join(out, "eval_logs", f"epoch{ep}.log"), log=log))
                    except Exception as e:  # noqa: BLE001 - keep training; the adapter is saved
                        entry["eval_error"] = str(e)
                        log(f"  ! evaluation failed: {e}")
            else:
                entry["val_only"] = True
            history.append(entry)
            history.sort(key=lambda h: h.get("step", 0))
            save_history()
            log("\n" + table() + "\n")

    def table() -> str:
        t = history_table(history, val_names)
        rows = [h for h in history if h.get("teacher_val_loss")]
        if rows:
            t += "\n\nteacher validation loss (gap = student - teacher):\n\n| epoch | step | " + " | ".join(val_names) + " |\n|" \
                 + "---:|" * (len(val_names) + 2) + "\n"
            for h in rows:
                t += f"| {h['epoch']} | {h['step']} | " + " | ".join(
                    f"{h['teacher_val_loss'][c]:.4f} ({h['val_loss'][c] - h['teacher_val_loss'][c]:+.4f})" for c in val_names) + " |\n"
        return t

    targs = TrainingArguments(
        output_dir=out, run_name=run, seed=args.seed, num_train_epochs=args.epochs, max_steps=total,
        per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, weight_decay=args.weight_decay, max_grad_norm=0.0,   # clipped per group (create_optimizer)
        bf16=torch.cuda.is_available(), logging_steps=args.logging_steps, report_to="none", disable_tqdm=True,
        eval_strategy="no", save_strategy="steps", save_steps=args.save_steps or total + 1, save_total_limit=2,
        remove_unused_columns=False, dataloader_num_workers=args.num_workers, include_num_input_tokens_seen=True,
        use_cpu=not torch.cuda.is_available())
    trainer = DistillTrainer(model=model, args=targs, train_dataset=train_ds, data_collator=collator, callbacks=[Callback()])
    trainer.loss_is_scaled_for_ga = True   # the losses divide by the token count of the accumulated batch
    trainer.remove_callback(PrinterCallback)
    if getattr(trainer.accelerator, "gradient_accumulation_steps", 1) != 1:
        sys.exit("accelerator scales backward by gradient accumulation: the RETRO teacher's extra backward would be mis-scaled")

    with open(os.path.join(out, "run_config.json"), "w") as f:
        json.dump({"args": vars(args), "base": base, "set_meta": set_meta,
                   "db_meta": {k_: v for k_, v in (db.meta if db else {}).items() if k_ != "contamination_examples"},
                   "retro": rcfg.__dict__ if rcfg else None,
                   "lora_plan": lora_plan.resolved_json(plan, plan["method"], plan["alpha"], plan.get("dropout", 0.0), shapes),
                   "trainable": {"lora": n_lora, "retro_teacher": n_retro}, "total_steps": total,
                   "eval_points": {str(k_): v for k_, v in eval_points.items()},
                   "val_points": {str(k_): v for k_, v in val_points.items()}}, f, indent=2, default=str)

    # teacher's head start before any training: base + neighbours vs base (context teacher only; the RETRO teacher
    # starts as the base model because its output projections are zero)
    if teacher == "context" and not args.no_teacher_val and not any(h.get("teacher_val_loss") for h in history if h.get("epoch") == 0):
        t0 = time.time()
        vl = val_losses(model, with_teacher=True)
        e0 = next((h for h in history if h.get("epoch") == 0), None)
        if e0 is None:
            e0 = {"epoch": 0, "step": 0}
            history.insert(0, e0)
        e0.setdefault("val_loss", vl["student"])
        e0["teacher_val_loss"] = vl["teacher"]
        save_history()
        log(f"epoch 0: base {json.dumps({c: round(v, 4) for c, v in vl['student'].items()})}; base + neighbours "
            f"{json.dumps({c: round(v, 4) for c, v in vl['teacher'].items()})} ({time.time() - t0:.0f}s)")

    resume = None
    if args.resume:
        cks = sorted([d for d in os.listdir(out) if d.startswith("checkpoint-")], key=lambda d: int(d.split("-")[1]))
        resume = os.path.join(out, cks[-1]) if cks else None
        log(f"resuming from {resume}" if resume else "no checkpoint to resume from; starting fresh")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    minutes = (time.time() - t0) / 60

    final = export_student(trainer.model, os.path.join(out, "adapter"), tok)
    if rcfg:
        trainer.model.save_pretrained(os.path.join(out, "teacher_retro"))
        rcfg.save(os.path.join(out, "teacher_retro"))
    with open(os.path.join(out, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    tdesc = {"context": f"context teacher ({args.teacher_weights} weights; {args.ctx_k} neighbours x {args.ctx_nbr_len} tokens "
                        f"per {args.ctx_window}-token window)",
             "retro": f"RETRO-layer teacher ({n_retro / 1e6:.1f} M, lr {args.retro_lr}; dropped)", "none": "none (control)"}[teacher]
    lines = [f"# Retrieval-distilled LoRA run `{run}`", "",
             f"- base `{os.path.relpath(base, ROOT)}`; deployed adapter: LoRA {plan.get('name')} ({n_lora / 1e6:.1f} M) -> "
             f"`{os.path.relpath(final, ROOT)}`",
             f"- teacher: {tdesc}" + (f"; alpha {args.alpha}, gate {args.kd_gate}" if teacher != "none" else ""),
             f"- lr {args.lr}, {args.scheduler}; {args.bs} x {args.grad_accum} sequences per step; {trainer.state.global_step} steps, "
             f"{minutes:.0f} min",
             f"- data: {os.path.relpath(set_dir, ROOT)} ({train_tokens / 1e6:.0f} M tokens)"
             + (f"; database {db.meta['name']} ({db.meta['tokens'] / 1e9:.2f} B tokens)" if db else ""),
             "", "## Scores (student, no retrieval)", "", table(), ""]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("\n".join(lines))


if __name__ == "__main__":
    main()
