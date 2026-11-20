"""Run scripts/eval_model.py on a model or adapter in a subprocess and collect the scores; format the
per-epoch history as a Markdown table.

Tasks (--eval-tasks of cpt_trainer.train, all from scripts/eval_model.py):
    mmlu      k-shot MMLU from the local copy, answer letter by log-likelihood, base-model prompt format; also the
              ai_large subject (1,000 ML / AI questions), reported as AI_Large outside the MMLU accuracy
              (chat format at 0-shot is scored generatively instead: --mmlu-scoring of eval_model.py)
    mcq       the AI multiple-choice set (data/final/eval/mcq.jsonl), same letter scoring, 5-shot
    open_qa   open questions per learner level: ROUGE-L / token-F1 vs the reference, reading-level fit
    code      DS-1000: generated code executed against the official tests -> pass@1
    val_loss  assistant-token loss / perplexity on data/final/sft_val.jsonl
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_SCRIPT = os.path.join(ROOT, "scripts", "eval_model.py")
RESULTS_ROOT = os.path.join(ROOT, "eval_results")
CATEGORIES = ("STEM", "humanities", "social_sciences", "other")
DEFAULT_TASKS = "mmlu,mcq,open_qa,code,val_loss"

# history key -> (table header, summary.json path, decimals)
EXTRA_COLUMNS = [
    ("mcq_accuracy", "MCQ acc", 4),
    ("open_qa_rougeL", "openQA ROUGE-L", 4),
    ("open_qa_token_f1", "openQA tok-F1", 4),
    ("open_qa_level_fit", "level fit", 4),
    ("code_pass1", "DS-1000 pass@1", 4),
    ("sft_val_loss", "SFT val loss", 4),
]


def _collect(summary: dict) -> dict:
    """Flatten the scores of every task in eval_model.py's summary.json into one history entry."""
    t = summary["tasks"]
    res: dict = {}
    if "mmlu" in t:
        mm = t["mmlu"]
        res.update({"n": mm["n"], "shots": mm["shots"], "format": mm["format"], "scoring": mm.get("scoring", "loglik"),
                    "parse_rate": mm.get("parse_rate"), "accuracy": mm["accuracy"],
                    "macro_accuracy": mm["macro_accuracy"], "ai_subjects_accuracy": mm["ai_subjects_accuracy"],
                    "ai_related_accuracy": mm.get("ai_related_accuracy"), "ai_related_n": mm.get("ai_related_n"),
                    "ai_large_accuracy": mm.get("ai_large_accuracy"), "ai_large_n": mm.get("ai_large_n"),
                    "ai_large_by_topic": {k: v["acc"] for k, v in (mm.get("ai_large_by_topic") or {}).items()} or None,
                    "by_category": {c: v["acc"] for c, v in mm["by_category"].items()},
                    "letter_prob_mass": mm["letter_prob_mass"], "warning": mm.get("warning"),
                    "eval_seconds": mm["seconds"]})
    if "mcq" in t:
        m = t["mcq"]
        res.update({"mcq_accuracy": m["accuracy"], "mcq_n": m["n"], "mcq_shots": m.get("shots"),
                    "mcq_letter_prob_mass": m.get("letter_prob_mass"), "mcq_warning": m.get("warning")})
    if "open_qa" in t:
        o = t["open_qa"]
        res.update({"open_qa_rougeL": o["rougeL"], "open_qa_token_f1": o["token_f1"], "open_qa_level_fit": o["level_fit"],
                    "open_qa_n": o["n"], "open_qa_seconds": o.get("seconds"),
                    "open_qa_per_level": {lv: {"rougeL": v["rougeL"], "level_fit": v["level_fit"]} for lv, v in o["per_level"].items()}})
    if "code" in t:
        c = t["code"]
        res.update({"code_pass1": c["pass@1"], "code_pass1_valid": c["pass@1_valid"], "code_n": c["n"],
                    "code_n_valid": c["n_valid"], "code_by_library": {k: v["pass@1"] for k, v in c["by_library"].items()}})
    if "val_loss" in t:
        v = t["val_loss"]
        res.update({"sft_val_loss": v["loss"], "sft_val_ppl": v["ppl"], "sft_val_n": v["n"]})
    return res


def run_eval(model_path: str, tag: str, tasks: str = DEFAULT_TASKS, shots: int = 5, bs: int = 32, gen_bs: int = 16,
             fmt: str = "base", subjects: str = "all", max_per_subject: int | None = None, limit: int | None = None,
             log_path: str | None = None, chat_model: str | None = None, chat_tasks: str = "open_qa",
             scoring: str = "auto", log=print) -> dict:
    """Evaluate `model_path` (base model dir or LoRA adapter dir) on `tasks` and return the scores.
    With `chat_model`, the tasks in `chat_tasks` run on that model with the same adapter applied (eval_model.py
    --chat-model). Full results land in eval_results/<tag>/ (summary.json, summary.md, report.html, predictions)."""
    base_cmd = [sys.executable, "-u", EVAL_SCRIPT, "--model", model_path, "--tasks", tasks,
                "--mmlu-shots", str(shots), "--mmlu-format", fmt, "--mmlu-scoring", scoring, "--mmlu-subjects", subjects,
                "--no-ui", "--tag", tag]
    if chat_model:
        base_cmd += ["--chat-model", chat_model, "--chat-tasks", chat_tasks]
    if max_per_subject:
        base_cmd += ["--mmlu-max-per-subject", str(max_per_subject)]
    if limit:
        base_cmd += ["--limit", str(limit)]
    log(f"  eval [{tasks}] (MMLU {shots}-shot) on {os.path.relpath(model_path, ROOT)} -> eval_results/{tag}"
        + (f"  (log: {os.path.relpath(log_path, ROOT)})" if log_path else ""))
    t0 = time.time()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # The evaluation runs next to the training process (and whatever else uses the GPU); if it fails,
    # most likely out of memory, retry with smaller batches before giving up.
    for attempt, div in enumerate((1, 2, 4)):
        cmd = base_cmd + ["--bs", str(max(1, bs // div)), "--gen-bs", str(max(1, gen_bs // div))]
        if log_path:
            with open(log_path, "a" if attempt else "w", encoding="utf-8") as f:
                f.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
                f.flush()
                proc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
        else:
            proc = subprocess.run(cmd, cwd=ROOT, env=env)
        if proc.returncode == 0:
            break
        log(f"  ! eval_model.py exited with {proc.returncode} at --bs {max(1, bs // div)} --gen-bs {max(1, gen_bs // div)}"
            + (f" (see {os.path.relpath(log_path, ROOT)})" if log_path else "")
            + ("; retrying with smaller batches" if attempt < 2 else ""))
    else:
        raise RuntimeError(f"eval_model.py failed three times (last exit {proc.returncode})"
                           + (f"; see {log_path}" if log_path else ""))
    with open(os.path.join(RESULTS_ROOT, tag, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    res = {"tag": tag, "model": os.path.relpath(model_path, ROOT), "tasks": tasks, **_collect(summary),
           "wall_seconds": round(time.time() - t0, 1)}
    if summary.get("chat_model"):
        res["chat_model"] = os.path.relpath(summary["chat_model"], ROOT)
        res["chat_tasks"] = summary["chat_tasks"]
    parts = []
    if "accuracy" in res:
        rel = f", AI-rel. {res['ai_related_accuracy']:.4f}" if res.get("ai_related_accuracy") is not None else ""
        rel += f", AI_Large {res['ai_large_accuracy']:.4f}" if res.get("ai_large_accuracy") is not None else ""
        parts.append(f"MMLU {res['accuracy']:.4f} (macro {res['macro_accuracy']:.4f}, AI subj. {res['ai_subjects_accuracy']:.4f}{rel}, n={res['n']})")
    for key, head, nd in EXTRA_COLUMNS:
        if res.get(key) is not None:
            parts.append(f"{head} {res[key]:.{nd}f}")
    log(f"  eval {tag}: " + " | ".join(parts) + f" | {res['wall_seconds']:.0f}s")
    for w in (res.get("warning"), res.get("mcq_warning")):
        if w:
            log(f"  ! {w}")
    return res


def run_mmlu(model_path: str, tag: str, **kw) -> dict:
    """MMLU only (the original epoch-end evaluation)."""
    return run_eval(model_path, tag, tasks="mmlu", **kw)


def history_table(history: list[dict], val_corpora: list[str] = ()) -> str:
    """Markdown table: one row per epoch (epoch 0 = base model before training). Columns of tasks that
    never ran are left out."""
    base = next((h for h in history if h.get("epoch") == 0 and "accuracy" in h), None)
    extra = [(k, head, nd) for k, head, nd in EXTRA_COLUMNS if any(h.get(k) is not None for h in history)]
    large = any(h.get("ai_large_accuracy") is not None for h in history)  # the ai_large subject (not in MMLU acc)
    head = ["epoch", "step", "train loss"] + [f"val loss {c}" for c in val_corpora] + \
           ["MMLU acc", "Δ vs base", "macro", "AI subj.", "AI-rel."] + (["AI_Large"] if large else []) + \
           list(CATEGORIES) + ["letter mass"] + [h for _, h, _ in extra]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---:" for _ in head) + "|"]

    def f(x, nd=4):
        return "" if x is None else f"{x:.{nd}f}"

    for h in sorted(history, key=lambda h: h.get("epoch", 0)):
        acc = h.get("accuracy")
        delta = f(acc - base["accuracy"], 4) if base and acc is not None and h is not base else ""
        if delta and not delta.startswith("-"):
            delta = "+" + delta
        row = [str(h.get("epoch", "")), str(h.get("step", "")), f(h.get("train_loss"))]
        row += [f((h.get("val_loss") or {}).get(c)) for c in val_corpora]
        row += [f(acc), delta, f(h.get("macro_accuracy")), f(h.get("ai_subjects_accuracy")), f(h.get("ai_related_accuracy"))]
        row += [f(h.get("ai_large_accuracy"))] if large else []
        row += [f((h.get("by_category") or {}).get(c)) for c in CATEGORIES]
        row += [f(h.get("letter_prob_mass"), 3)]
        row += [f(h.get(k), nd) for k, _, nd in extra]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
