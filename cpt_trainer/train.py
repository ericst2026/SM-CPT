#!/usr/bin/env python3
"""Continued pre-training (CPT) of Qwen3-0.6B-Base with a high-rank LoRA adapter on the collected
corpora, with an evaluation (5-shot MMLU, AI MCQ, open QA, DS-1000 code, SFT validation loss) after every epoch.

    python -m cpt_trainer.train                                          # corpus/ai + corpus/replay, r=64, 4 epochs
    python -m cpt_trainer.train --run-name cpt_r64 --epochs 4 --r 64 --alpha 128 --lr 1e-4
    python -m cpt_trainer.train --run-name cpt_r64 --resume              # continue after an interruption
    python -m cpt_trainer.train --run-name smoke --max-blocks 64 --epochs 1 --grad-accum 2 \
                                --mmlu-max-per-subject 2 --eval-limit 8  # end-to-end smoke test (minutes)
    python -m cpt_trainer.train --corpora cpt_data/corpus_full/ai cpt_data/corpus_full/replay \
                                --r 128 --alpha 256 --epochs 2 --run-name cpt_qwen06b_r128_full
    python -m cpt_trainer.train --corpora cpt_data/corpus_full/{ai,replay,stem,long} --varlen-min 16384 \
                                --varlen-max 24576 --method rslora --r 128 --alpha 16 --epochs 1 --eval-at 0.5,1 \
                                --run-name cpt_qwen06b_rs128_long              # long-context packing (see longctx.py)
    python -m cpt_trainer.train --lora-config configs/lora/cpt_qwen06b_rs_adaptive_v1.yaml --epochs 1 \
                                --run-name cpt_qwen06b_rs_adaptive_v1          # per-layer/module ranks (see lora_plan.py)

Data: every text/<source>/<id>.txt of the corpus folders (documents flagged low_quality in the manifest
are skipped), tokenised, EOS-terminated and packed into --seq-len blocks. 1 % of the (non-book) documents
of each corpus are held out for a validation loss per corpus, so the log shows what happens to AI text
and to general (replay) text separately.
With --varlen-min/--varlen-max, documents are instead packed into variable-length sequences (e.g. 16 k-24 k
tokens) with a per-document attention mask (flex attention) and RoPE positions restarting per document;
long documents fill whole sequences. --legacy-val-cache keeps the validation blocks of an earlier
fixed-block run for the corpora it covers, so val losses stay comparable. See cpt_trainer/longctx.py.

Evaluation: before training (epoch 0 = base model) and after every epoch (or at --eval-at epoch fractions,
e.g. 0.5,1) the adapter is saved and
scripts/eval_model.py runs --eval-tasks in a subprocess (default mmlu,mcq,open_qa,code,val_loss: MMLU from
the local copy in data/eval/mmlu, --mmlu-shots, answer letter by log-likelihood, base-model prompt format;
the AI MCQ set; open QA per learner level (ROUGE-L, level fit); DS-1000 code execution (pass@1); the
assistant-token loss on data/final/sft_val.jsonl). Scores go to eval_history.json / summary.md. A failed
epoch-end evaluation is logged and training continues (the adapter is saved first, so it can be re-run).

Output (runs/<run-name>/):
    epoch<N>/adapter/     LoRA adapter after epoch N (+ tokenizer)  -> eval_model.py / chat_feedback.py
    adapter/              final adapter (= last epoch)
    checkpoint-*/         Trainer checkpoints (for --resume)
    eval_logs/            stdout of each evaluation run
    eval_history.json     evaluation scores + losses per epoch
    log_history.json      every logged training / validation step
    run_config.json, train.log, summary.md
Evaluation details land in eval_results/<run-name>_epoch<N>/ (summary.json, summary.md, report.html, predictions).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The 152k-token vocabulary makes the fp32 logits of a block the largest tensor; let the allocator grow
# segments instead of fragmenting (set before torch is imported).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from ai_qa_train.common import LORA_METHODS, ROOT, build_lora_config, is_quantized, load_model, load_tokenizer  # noqa: E402
from cpt_trainer import lora_plan  # noqa: E402
from cpt_trainer.data import BlockDataset, build_blocks  # noqa: E402
from cpt_trainer.longctx import VarlenCollator, VarlenDataset, build_varlen, varlen_loss  # noqa: E402
from cpt_trainer.mmlu import DEFAULT_TASKS, history_table, run_eval  # noqa: E402

DEFAULT_BASE = os.path.join(ROOT, "models", "Qwen3-0.6B-Base")
DEFAULT_CORPORA = [os.path.join(ROOT, "corpus", "ai"), os.path.join(ROOT, "corpus", "replay")]


class Logger:
    """print + append to <run>/train.log."""

    def __init__(self, path: str):
        self.path = path

    def __call__(self, msg: str = "") -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else ""
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def interval_train_loss(log_history: list[dict], lo_step: int, hi_step: int) -> float | None:
    """Mean logged training loss over optimizer steps (lo_step, hi_step]."""
    vals = [h["loss"] for h in log_history if "loss" in h and lo_step < h.get("step", 0) <= hi_step]
    return sum(vals) / len(vals) if vals else None


def epoch_label(x: float):
    return int(x) if abs(x - round(x)) < 1e-9 else round(x, 3)


def latest_val_losses(log_history: list[dict], corpora: list[str]) -> dict:
    out = {}
    for h in reversed(log_history):
        for c in corpora:
            if c not in out and f"eval_{c}_loss" in h:
                out[c] = h[f"eval_{c}_loss"]
        if len(out) == len(corpora):
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # model / adapter
    ap.add_argument("--base", default=DEFAULT_BASE, help="base model dir (default models/Qwen3-0.6B-Base)")
    ap.add_argument("--method", default="lora", choices=list(LORA_METHODS),
                    help="lora | rslora | dora | pissa | olora | ... (see ai_qa_train.common.LORA_METHODS)")
    ap.add_argument("--r", type=int, default=64, help="LoRA rank")
    ap.add_argument("--alpha", type=int, default=128, help="LoRA alpha (scale = alpha / r)")
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--targets", default="all-linear", help="all-linear | attn | mlp | qv | comma list of module names")
    ap.add_argument("--lora-config", default=None, help="YAML rank plan with per-layer / per-module ranks, e.g. "
                    "configs/lora/cpt_qwen06b_rs_adaptive_v1.yaml; its method/alpha/dropout/targets/default_r replace "
                    "--method/--alpha/--dropout/--targets/--r (see cpt_trainer/lora_plan.py)")
    # data
    ap.add_argument("--corpora", nargs="+", default=DEFAULT_CORPORA, help="corpus folders (manifest.jsonl + text/)")
    ap.add_argument("--include-low-quality", action="store_true", help="also use documents flagged low_quality")
    ap.add_argument("--exclude-sources", default="", help="comma list of source names to leave out, e.g. wiki_simple")
    ap.add_argument("--seq-len", type=int, default=2048, help="tokens per packed block")
    ap.add_argument("--val-frac", type=float, default=0.01, help="share of documents per corpus held out for validation")
    ap.add_argument("--max-val-blocks", type=int, default=64, help="validation blocks per corpus")
    ap.add_argument("--max-blocks", type=int, default=None, help="cap training blocks (smoke tests)")
    ap.add_argument("--cache-dir", default=None, help="packed-block cache (default <output-root>/_cpt_cache)")
    ap.add_argument("--varlen-min", type=int, default=0, help="variable-length packing: minimum sequence length "
                    "(0 = fixed --seq-len blocks); e.g. 16384")
    ap.add_argument("--varlen-max", type=int, default=0, help="variable-length packing: maximum sequence length, e.g. 24576")
    ap.add_argument("--pad-multiple", type=int, default=2048, help="varlen: pad sequences to a multiple of this "
                    "(few distinct shapes for the compiled attention kernel)")
    ap.add_argument("--loss-chunk", type=int, default=4096, help="varlen: tokens per LM-head chunk in the loss")
    ap.add_argument("--exclude-docs", default=None, help="text file of '<corpus>/<doc id>' lines (# comments) kept out of "
                    "the training split, after the train/val split (e.g. configs/corpus/corpus_full_exclude_noisy_long.txt)")
    ap.add_argument("--max-val-seqs", type=int, default=32, help="varlen: validation sequences per corpus")
    ap.add_argument("--legacy-val-cache", default=None, help="varlen: fixed-block .npz of an earlier run whose "
                    "validation blocks are reused for --legacy-val-corpora (comparable val losses)")
    ap.add_argument("--legacy-val-corpora", default="ai,replay")
    # optimisation
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=2, help="blocks per device step (the fp32 logits of one 2048-token "
                    "block take ~1.2 GB; 2 fits next to other GPU users on a 24 GB card)")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--eval-blocks-bs", type=int, default=None, help="blocks per validation step (default --bs)")
    ap.add_argument("--warmup", type=float, default=0.03, help="warmup fraction of total steps")
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--grad-ckpt-frac", type=float, default=1.0, help="share of decoder layers that keep gradient "
                    "checkpointing (1.0 = all; e.g. 0.5 = every second layer stores its activations instead of recomputing)")
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=0, help="resume checkpoint every N steps (0 = only at eval points)")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    # evaluation
    ap.add_argument("--eval-tasks", default=DEFAULT_TASKS,
                    help="comma list of scripts/eval_model.py tasks run after every epoch: mmlu, mcq, open_qa, code, val_loss")
    ap.add_argument("--eval-limit", type=int, default=None, help="quick runs: max items per non-MMLU task")
    ap.add_argument("--eval-gen-bs", type=int, default=16, help="generation batch size of the open_qa / code tasks")
    ap.add_argument("--mmlu-shots", type=int, default=5)
    ap.add_argument("--mmlu-subjects", default="all", help="all | ai | STEM | comma list (see eval_model.py)")
    ap.add_argument("--mmlu-max-per-subject", type=int, default=None, help="quick runs: cap questions per subject")
    ap.add_argument("--eval-bs", type=int, default=8, help="batch size of the MMLU run (5-shot prompts are long: "
                    "~0.5 GB per prompt in the batch; halved automatically if the run fails)")
    ap.add_argument("--eval-at", default="", help="epoch fractions where the adapter is saved and evaluated, "
                    "e.g. 0.5,1 (default: the end of every epoch)")
    ap.add_argument("--eval-chat-model", default=None, help="run the chat-format tasks of --eval-chat-tasks on this "
                    "model (e.g. models/Qwen3-0.6B) with the adapter applied, instead of on the -Base model whose chat "
                    "special tokens are untrained; epoch 0 then scores the chat model alone on those tasks")
    ap.add_argument("--eval-chat-tasks", default="open_qa", help="comma list of tasks evaluated on --eval-chat-model")
    ap.add_argument("--no-eval-base", action="store_true", help="skip the epoch-0 (base model) evaluation")
    ap.add_argument("--base-eval-from", default=None, help="eval_history.json of an earlier run with the same base model "
                    "and eval settings: copy its epoch-0 scores instead of re-running them")
    ap.add_argument("--base-val-loss", action="store_true", help="compute the validation losses of the base model "
                    "(before training) and store them in the epoch-0 entry")
    ap.add_argument("--no-eval", "--no-mmlu", dest="no_eval", action="store_true", help="train only; no evaluation runs")
    # output
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--output-root", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--resume", action="store_true", help="resume from the last checkpoint in the run dir")
    ap.add_argument("--merge", action="store_true", help="also save a merged full bf16 model (not for 4-bit)")
    ap.add_argument("--prepare-only", action="store_true", help="tokenise + pack the data into the cache, then exit "
                    "(CPU only; run it with the same data flags so the real run starts from the cache)")
    args = ap.parse_args()

    from transformers import Trainer, TrainerCallback, TrainingArguments, default_data_collator, set_seed
    from transformers.trainer_callback import PrinterCallback

    set_seed(args.seed)
    base = os.path.abspath(args.base)
    if not os.path.isdir(base):
        sys.exit(f"base model not found: {base}")
    plan = None
    if args.lora_config:
        num_layers, shapes = lora_plan.model_shapes(base)
        try:
            plan = lora_plan.load_plan(args.lora_config, num_layers)
        except (OSError, ValueError) as e:
            sys.exit(f"--lora-config: {e}")
        for key, attr in (("method", "method"), ("alpha", "alpha"), ("dropout", "dropout"), ("default_r", "r")):
            if key in plan:
                setattr(args, attr, plan[key])
        args.targets = ",".join(plan["targets"])
    tag = f"{plan.get('name') or 'plan'}" if plan else f"r{args.r}"
    run = args.run_name or f"cpt_{os.path.basename(base).lower().replace('-base', '')}_{tag}_{time.strftime('%m%d_%H%M')}"
    out = os.path.join(args.output_root, run)
    os.makedirs(out, exist_ok=True)
    log = Logger(os.path.join(out, "train.log"))
    log(f"run {run} -> {out}")
    log("args: " + json.dumps(vars(args)))
    if plan:
        log(f"LoRA rank plan {plan.get('name')} ({os.path.relpath(plan['path'], ROOT)}), {args.method} alpha={args.alpha}:\n"
            + lora_plan.format_table(plan, shapes, args.method, args.alpha))
        with open(os.path.join(out, "lora_plan.json"), "w") as f:
            json.dump(lora_plan.resolved_json(plan, args.method, args.alpha, args.dropout, shapes), f, indent=2)

    # ---------------- data ----------------
    tok = load_tokenizer(base)
    corpora = [os.path.abspath(c) for c in args.corpora]
    exclude = {s.strip() for s in args.exclude_sources.split(",") if s.strip()}
    exclude_train = frozenset()
    if args.exclude_docs:
        with open(args.exclude_docs, encoding="utf-8") as f:
            exclude_train = frozenset(k for k in (line.split("#", 1)[0].strip() for line in f) if k)
        log(f"  --exclude-docs {args.exclude_docs}: {len(exclude_train):,} documents kept out of training")
    varlen = args.varlen_min > 0
    cache_dir = args.cache_dir or os.path.join(args.output_root, "_cpt_cache")
    if varlen:
        if not args.varlen_max >= args.varlen_min:
            sys.exit("--varlen-max must be >= --varlen-min")
        if args.bs != 1:
            log(f"  varlen packing: --bs {args.bs} -> 1 (one packed sequence per device step)")
            args.bs = 1
        legacy = None
        if args.legacy_val_cache:
            legacy = {"cache": os.path.abspath(args.legacy_val_cache), "split_seed": args.seed,
                      "corpora": [c.strip() for c in args.legacy_val_corpora.split(",") if c.strip()]}
        data = build_varlen(corpora, tok, args.varlen_min, args.varlen_max, args.val_frac, args.seed, cache_dir,
                            include_low_quality=args.include_low_quality, exclude_sources=exclude,
                            max_val_seqs=args.max_val_seqs, legacy_val=legacy, exclude_train=exclude_train, log=log)
        train_ds = VarlenDataset(data["train"], max_seqs=args.max_blocks)
        val_ds = {c: VarlenDataset(v) for c, v in data["val"].items()}
        st = data["stats"]
        for c, s in st["corpora"].items():
            v = st["val"].get(c, {})
            log(f"  {c:8s} {s['docs']:7,d} train docs  {s['tokens'] / 1e6:7.2f} M tokens  ({s['long_tokens'] / 1e6:.1f} M in "
                f"{s['long_docs']:,} docs >= {args.varlen_min} tokens) | val {v.get('seqs', 0)} "
                f"{'fixed blocks' if v.get('fixed_blocks') else 'sequences'}, {v.get('tokens', 0) / 1e6:.2f} M tokens")
        train_tokens = int(train_ds.lengths().sum())
        log(f"  training on {len(train_ds):,} sequences of {args.varlen_min}-{args.varlen_max} tokens = "
            f"{train_tokens / 1e6:.2f} M tokens per epoch (length quantiles {st['seq_len_quantiles']}, "
            f"{st['pieces_per_seq_mean']:.1f} documents per sequence)")
        mean_len = train_tokens / max(1, len(train_ds))
    else:
        data = build_blocks(corpora, tok, args.seq_len, args.val_frac, args.seed, cache_dir,
                            include_low_quality=args.include_low_quality, exclude_sources=exclude,
                            max_val_blocks=args.max_val_blocks, exclude_train=exclude_train, log=log)
        train_blocks = data["train"][: args.max_blocks] if args.max_blocks else data["train"]
        train_ds = BlockDataset(train_blocks)
        val_ds = {c: BlockDataset(b) for c, b in data["val"].items() if len(b)}
        for c, s in data["stats"]["corpora"].items():
            log(f"  {c:8s} {s['docs']:5d} train docs  {s['tokens'] / 1e6:6.2f} M tokens  {s['train_blocks']:6d} blocks | "
                f"val {s['val_docs']} docs -> {s['val_blocks']} blocks")
        train_tokens = len(train_ds) * args.seq_len
        log(f"  training on {len(train_ds):,} blocks x {args.seq_len} = {train_tokens / 1e6:.2f} M tokens per epoch")
        mean_len = args.seq_len
    val_names = sorted(val_ds)
    if args.prepare_only:
        log("--prepare-only: data cache is ready; exiting without training")
        return

    # ---------------- evaluation history (epoch 0 = base model) ----------------
    hist_path = os.path.join(out, "eval_history.json")
    history: list[dict] = []
    if args.resume:
        for cand in (hist_path, os.path.join(out, "mmlu_history.json")):  # older runs used the second name
            if os.path.exists(cand):
                with open(cand) as f:
                    history = json.load(f)
                break

    def save_history():
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

    eval_kw = dict(tasks=args.eval_tasks, shots=args.mmlu_shots, bs=args.eval_bs, gen_bs=args.eval_gen_bs, fmt="base",
                   subjects=args.mmlu_subjects, max_per_subject=args.mmlu_max_per_subject, limit=args.eval_limit,
                   chat_model=os.path.abspath(args.eval_chat_model) if args.eval_chat_model else None,
                   chat_tasks=args.eval_chat_tasks, log=log)
    chat_tasks = [t for t in args.eval_chat_tasks.split(",") if t in args.eval_tasks.split(",")] if args.eval_chat_model else []
    if args.base_eval_from and not any(h.get("epoch") == 0 for h in history):
        with open(args.base_eval_from) as f:
            prev = next((h for h in json.load(f) if h.get("epoch") == 0), None)
        if prev is None:
            sys.exit(f"no epoch-0 entry in {args.base_eval_from}")
        prev = {k: v for k, v in prev.items() if k != "val_loss"}
        history.append({**prev, "copied_from": os.path.relpath(os.path.abspath(args.base_eval_from), ROOT)})
        save_history()
        log(f"epoch 0: base-model scores copied from {args.base_eval_from} (MMLU {prev.get('accuracy')})")
    base_entry = next((h for h in history if h.get("epoch") == 0), None)
    if chat_tasks and not args.no_eval and base_entry is not None and base_entry.get("copied_from") \
            and base_entry.get("chat_model") is None:
        # the copied epoch-0 scores of the chat-format tasks come from the -Base model; replace them with the chat
        # model's own scores (no adapter), the reference the adapter-on-chat-model scores are compared with
        log(f"epoch 0: {','.join(chat_tasks)} of the chat model {args.eval_chat_model} (no adapter)")
        res = run_eval(base, f"{run}_epoch0_chat", log_path=os.path.join(out, "eval_logs", "epoch0_chat.log"),
                       **{**eval_kw, "tasks": ",".join(chat_tasks)})
        base_entry.update({k: v for k, v in res.items() if k not in ("tag", "model", "tasks", "wall_seconds")})
        save_history()
    if not args.no_eval and base_entry is not None and base_entry.get("copied_from") and "mmlu" in args.eval_tasks.split(",") \
            and base_entry.get("ai_large_accuracy") is None and args.mmlu_subjects == "all" \
            and os.path.isdir(os.path.join(ROOT, "data", "eval", "mmlu", "ai_large")):
        # the copied epoch-0 scores predate the ai_large subject: score the base model on it alone
        log("epoch 0: ai_large subject of the base model (the copied scores predate it)")
        res = run_eval(base, f"{run}_epoch0_ai_large", log_path=os.path.join(out, "eval_logs", "epoch0_ai_large.log"),
                       **{**eval_kw, "tasks": "mmlu", "subjects": "ai_large", "chat_model": None})
        base_entry.update({k: v for k, v in res.items() if k.startswith("ai_large_")})
        save_history()
    if not args.no_eval and not args.no_eval_base and not any(h.get("epoch") == 0 for h in history):
        log(f"epoch 0: evaluation of the base model (before training): {args.eval_tasks}")
        res = run_eval(base, f"{run}_epoch0", log_path=os.path.join(out, "eval_logs", "epoch0.log"), **eval_kw)
        history.append({"epoch": 0, "step": 0, **res})
        save_history()

    # ---------------- model ----------------
    q4 = is_quantized(args.method)
    if varlen:
        torch._dynamo.config.recompile_limit = 64  # one compiled attention kernel per padded sequence length
        torch._dynamo.config.cache_size_limit = 64
    model = load_model(base, quantize_4bit=q4, for_training=True, attn="flex_attention" if varlen else "sdpa")
    model.config.use_cache = False
    if q4:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=not args.no_grad_ckpt)
    from peft import get_peft_model
    if plan:
        model = get_peft_model(model, lora_plan.build_config(plan, args.method, args.alpha, args.dropout))
        lora_plan.check_model(model, plan)  # every module got the planned rank
    else:
        model = get_peft_model(model, build_lora_config(args.method, args.r, args.alpha, args.dropout, args.targets))
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        if args.grad_ckpt_frac < 1.0:  # partial: evenly spaced decoder layers store activations instead of recomputing
            from transformers.modeling_layers import GradientCheckpointingLayer
            layers = [m for m in model.modules() if isinstance(m, GradientCheckpointingLayer)]
            n_off = len(layers) - round(args.grad_ckpt_frac * len(layers))
            off = sorted({min(len(layers) - 1, int((j + 0.5) * len(layers) / n_off)) for j in range(n_off)}) if n_off else []
            for i in off:
                layers[i].gradient_checkpointing = False
            log(f"  gradient checkpointing on {len(layers) - len(off)} of {len(layers)} decoder layers (off: {off})")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    lora_desc = (f"{args.method} rank plan {plan.get('name')} (r {min(plan['rank_table'].values())}-"
                 f"{max(plan['rank_table'].values())}) alpha={args.alpha}" if plan else f"{args.method} r={args.r} alpha={args.alpha}")
    log(f"  {lora_desc} on {args.targets}: {trainable / 1e6:.1f} M trainable "
        f"of {total_params / 1e6:.0f} M ({100 * trainable / total_params:.2f} %)")

    steps_per_epoch = max(1, math.ceil(len(train_ds) / (args.bs * args.grad_accum)))
    total = args.max_steps if args.max_steps > 0 else int(math.ceil(steps_per_epoch * args.epochs))
    log(f"  {steps_per_epoch} optimizer steps per epoch ({args.bs} x {args.grad_accum} sequences = "
        f"~{args.bs * args.grad_accum * mean_len / 1e3:.0f} k tokens per step), {total} steps in total")

    # evaluation points: epoch fractions -> optimizer steps
    fracs = ([float(x) for x in args.eval_at.split(",") if x.strip()] if args.eval_at
             else [float(e) for e in range(1, int(math.ceil(args.epochs)) + 1)])
    eval_points: dict[int, float] = {}
    for fr in sorted(fracs):
        st_ = min(total, max(1, int(round(fr * steps_per_epoch))))
        eval_points[st_] = min(fr, total / steps_per_epoch)
    log(f"  evaluation at epochs {[epoch_label(v) for v in eval_points.values()]} = steps {list(eval_points)}")

    # ---------------- epoch-end callback: save adapter, run the evaluation ----------------
    class EpochEvalCallback(TrainerCallback):
        def on_log(self, targs, state, control, logs=None, **kw):
            if not logs:
                return
            parts = [f"step {state.global_step}/{state.max_steps}", f"epoch {logs.get('epoch', state.epoch or 0):.2f}"]
            for k in ("loss", "grad_norm", "learning_rate"):
                if k in logs:
                    parts.append(f"{k} {logs[k]:.4g}" if k != "learning_rate" else f"lr {logs[k]:.2e}")
            parts += [f"{k} {v:.4f}" for k, v in logs.items() if k.endswith("_loss") and k.startswith("eval_")]
            if "num_input_tokens_seen" in logs:
                parts.append(f"{logs['num_input_tokens_seen'] / 1e6:.1f} M tok")
            log("  " + " | ".join(parts))

        def on_step_end(self, targs, state, control, **kw):
            if state.global_step in eval_points:
                control.should_evaluate = bool(val_ds)
                control.should_save = True

        def on_save(self, targs, state, control, model=None, **kw):
            # Fires after the validation of an evaluation point, so its eval losses are in log_history.
            if state.global_step not in eval_points or any(h.get("step") == state.global_step for h in history):
                return
            epoch = epoch_label(eval_points[state.global_step])
            prev_step = max([h.get("step", 0) for h in history] + [0])
            adapter_dir = os.path.join(out, f"epoch{epoch}", "adapter")
            model.save_pretrained(adapter_dir)
            tok.save_pretrained(adapter_dir)
            entry = {"epoch": epoch, "step": state.global_step,
                     "train_loss": interval_train_loss(state.log_history, prev_step, state.global_step),
                     "val_loss": latest_val_losses(state.log_history, val_names),
                     "adapter": os.path.relpath(adapter_dir, ROOT)}
            log(f"epoch {epoch}: adapter saved to {entry['adapter']}; train loss {entry['train_loss']}; "
                f"val loss {entry['val_loss']}")
            if not args.no_eval:
                torch.cuda.empty_cache()  # free cached activations before the evaluation process starts
                try:
                    entry.update(run_eval(adapter_dir, f"{run}_epoch{epoch}",
                                          log_path=os.path.join(out, "eval_logs", f"epoch{epoch}.log"), **eval_kw))
                except Exception as e:  # noqa: BLE001 - a failed evaluation must not end a long training run
                    entry["eval_error"] = str(e)
                    log(f"  ! evaluation of epoch {epoch} failed: {e}\n    training continues; re-run it with: "
                        f"python scripts/eval_model.py --model {entry['adapter']} --tasks {args.eval_tasks} "
                        f"--mmlu-shots {args.mmlu_shots} --mmlu-format base --tag {run}_epoch{epoch} --no-ui")
            history.append(entry)
            save_history()
            log("\n" + history_table(history, val_names) + "\n")

    targs = TrainingArguments(
        output_dir=out, run_name=run, seed=args.seed,
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.bs, per_device_eval_batch_size=1 if varlen else (args.eval_blocks_bs or args.bs),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type=args.scheduler,
        warmup_steps=max(1, int(total * args.warmup)), weight_decay=args.weight_decay, max_grad_norm=1.0,
        bf16=True, optim="paged_adamw_8bit" if q4 else "adamw_torch_fused",
        logging_steps=args.logging_steps, report_to="none", disable_tqdm=True,
        # evaluation and the matching checkpoint are triggered by EpochEvalCallback at the eval points;
        # --save-steps adds resume checkpoints in between
        eval_strategy="no", save_strategy="steps", save_steps=args.save_steps or total + 1, save_total_limit=2,
        remove_unused_columns=False, dataloader_num_workers=args.num_workers,
        include_num_input_tokens_seen=True, prediction_loss_only=varlen,
    )

    class VarlenTrainer(Trainer):
        """Document-masked loss on packed variable-length sequences (cpt_trainer.longctx.varlen_loss)."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if model.training and num_items_in_batch is None:
                raise RuntimeError("Trainer did not pass num_items_in_batch; the loss would be mis-scaled")
            loss = varlen_loss(model, inputs, num_items_in_batch if model.training else None, args.loss_chunk)
            return (loss, {}) if return_outputs else loss

    trainer_cls = VarlenTrainer if varlen else Trainer
    collator = VarlenCollator(tok.pad_token_id, args.pad_multiple) if varlen else default_data_collator
    trainer = trainer_cls(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds or None,
                          data_collator=collator, callbacks=[EpochEvalCallback()])
    if varlen:
        trainer.loss_is_scaled_for_ga = True  # varlen_loss divides by the token count of the accumulated batch
    trainer.remove_callback(PrinterCallback)

    with open(os.path.join(out, "run_config.json"), "w") as f:
        json.dump({"args": vars(args), "base": base, "corpora": corpora, "data_stats": data["stats"],
                   "train_blocks_used": len(train_ds), "train_tokens_per_epoch": train_tokens, "trainable_params": trainable,
                   "steps_per_epoch": steps_per_epoch, "total_steps": total,
                   "eval_points": {str(k): v for k, v in eval_points.items()},
                   "lora_plan": lora_plan.resolved_json(plan, args.method, args.alpha, args.dropout, shapes) if plan else None},
                  f, indent=2)

    base_entry = next((h for h in history if h.get("epoch") == 0), None)
    if args.base_val_loss and val_ds and base_entry is not None and not base_entry.get("val_loss"):
        # LoRA B is initialised to zero, so the untrained adapter computes exactly the base model
        log("epoch 0: validation losses of the base model")
        metrics = trainer.evaluate()
        base_entry["val_loss"] = {c: metrics.get(f"eval_{c}_loss") for c in val_names}
        save_history()
        log(f"  base val loss {base_entry['val_loss']}")

    resume = None
    if args.resume:
        cks = sorted([d for d in os.listdir(out) if d.startswith("checkpoint-")], key=lambda d: int(d.split("-")[1]))
        resume = os.path.join(out, cks[-1]) if cks else None
        log(f"resuming from {resume}" if resume else "no checkpoint to resume from; starting fresh")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    train_seconds = time.time() - t0

    final = os.path.join(out, "adapter")
    trainer.model.save_pretrained(final)
    tok.save_pretrained(final)
    with open(os.path.join(out, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    if args.merge and not q4:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(os.path.join(out, "merged"))
        tok.save_pretrained(os.path.join(out, "merged"))

    stats = data["stats"]
    lines = [f"# CPT run `{run}`", "",
             f"- base `{os.path.relpath(base, ROOT)}`, {lora_desc} on {args.targets} "
             f"({trainable / 1e6:.1f} M trainable params), lr {args.lr} {args.scheduler}, "
             f"{args.bs} x {args.grad_accum} x {mean_len:,.0f} tokens per step, {args.epochs:g} epochs, "
             f"{trainer.state.global_step} steps, {train_seconds / 60:.0f} min of training",
             (f"- data: {len(train_ds):,} sequences of {args.varlen_min}-{args.varlen_max} tokens (document-masked "
              f"packing) = {train_tokens / 1e6:.2f} M tokens per epoch " if varlen else
              f"- data: {len(train_ds):,} blocks x {args.seq_len} = {train_tokens / 1e6:.2f} M tokens per epoch ")
             + f"from {', '.join(os.path.relpath(c, ROOT) for c in corpora)}",
             f"- evaluation at epochs {[epoch_label(v) for v in eval_points.values()]}: `{args.eval_tasks}` (MMLU {args.mmlu_shots}-shot, subjects "
             f"`{args.mmlu_subjects}`, base prompt format); epoch 0 is the base model", "",
             "## Scores per epoch", "", history_table(history, val_names), ""]
    if varlen:
        lines += ["## Training data", "", "| corpus | train docs | M tokens | long docs | M tokens in long docs | val |",
                  "|---|---:|---:|---:|---:|---|"]
        lines += [f"| {c} | {s['docs']:,} | {s['tokens'] / 1e6:.2f} | {s['long_docs']:,} | {s['long_tokens'] / 1e6:.2f} | "
                  f"{stats['val'].get(c, {}).get('seqs', 0)} {'fixed 2048 blocks' if stats['val'].get(c, {}).get('fixed_blocks') else 'sequences'} |"
                  for c, s in stats["corpora"].items()]
    else:
        lines += ["## Training data", "", "| corpus | train docs | M tokens | blocks | val docs | val blocks |",
                  "|---|---:|---:|---:|---:|---:|"]
        lines += [f"| {c} | {s['docs']:,} | {s['tokens'] / 1e6:.2f} | {s['train_blocks']:,} | {s['val_docs']} | {s['val_blocks']} |"
                  for c, s in stats["corpora"].items()]
    lines += ["", "| source | docs | M tokens |", "|---|---:|---:|"]
    lines += [f"| {k} | {v['docs']:,} | {v['tokens'] / 1e6:.2f} |" for k, v in sorted(stats["sources"].items())]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("\n" + "\n".join(lines[:12]))
    log(f"adapter saved to {final}\n"
        f"  evaluate:  python scripts/eval_model.py --model {final} --tasks {args.eval_tasks} --mmlu-shots {args.mmlu_shots}\n"
        f"  summary:   {os.path.join(out, 'summary.md')}")


if __name__ == "__main__":
    main()
