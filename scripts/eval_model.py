#!/usr/bin/env python3
"""Evaluate the base model or a LoRA adapter on the held-out AI eval sets.

    python scripts/eval_model.py                                   # base Qwen3-1.7B, all tasks
    python scripts/eval_model.py --model runs/lora_r16/adapter     # trained adapter
    python scripts/eval_model.py --model runs/lora_r16 --tasks mcq,open_qa --limit 200
    python scripts/eval_model.py --model runs/lora_r16 --compare eval_results/base/summary.json
    python scripts/eval_model.py --tasks math --math-file data/final/rl_math.jsonl --limit 200
    python scripts/eval_model.py --tasks mmlu                                  # full local MMLU (57 subjects)
    python scripts/eval_model.py --tasks mmlu --mmlu-shots 5                   # standard 5-shot (log-likelihood)
    python scripts/eval_model.py --model runs/sft --tasks mmlu --mmlu-max-per-subject 50   # chat 0-shot = generative
    python scripts/eval_model.py --tasks mmlu --mmlu-subjects ai               # AI-related subjects only
    python scripts/eval_model.py --tasks mmlu --mmlu-subjects ai_related       # zero-shot-selected AI questions
    python scripts/eval_model.py --tasks mmlu --mmlu-subjects ai_large         # the 1,000-question ML / AI subject only
    python scripts/eval_model.py --tasks mmlu --mmlu-subjects machine_learning,college_computer_science

Tasks
  mmlu      the local MMLU copy in data/eval/mmlu (<subject>/{dev,validation,test}-*.parquet).
            0- or k-shot (dev split examples), answer letter by log-likelihood -> micro accuracy; for a chat model
            at 0-shot (--mmlu-scoring auto/generate) the model instead replies in non-thinking mode, ends with
            "Answer: X", and the letter is parsed from the text (parse_rate reported; thinking mode:
            scripts/eval_mmlu_thinking.py),
            macro accuracy over subjects, AI-related-subjects accuracy, ai_related accuracy (machine_learning plus
            math / CS / formal_logic questions picked by scripts/build_mmlu_ai_related.py, once built), per category
            (STEM / humanities / social sciences / other) and per subject. Extra subjects in the same folder that are
            not MMLU (ai_large: 1,000 ML / AI questions, scripts/build_mmlu_ai_large.py) run with "all" but are
            reported on their own (ai_large_accuracy, per topic), outside the MMLU accuracy / macro / categories.
  mcq       MMLU / MMLU-Pro AI items selected by the data pipeline. Scored by next-token log-likelihood of the answer letter
            (non-thinking mode) -> accuracy, overall / per source / per subject.
  open_qa   held-out open questions per learner level. Generates answers and reports ROUGE-L F1,
            token F1 vs. the reference, and *level fit* = share of answers whose Flesch-Kincaid grade
            is inside the target range of their level (does the model talk to children like children?).
            --level-judge additionally classifies each answer's audience with the zero-shot model.
  code      DS-1000 (numpy/pandas/sklearn/torch/...): generated code is executed against the official
            test harness in a subprocess -> pass@1, per library. (Not a security sandbox: only run on
            a machine where executing model-written code is acceptable.)
  val_loss  assistant-token loss / perplexity on data/final/sft_val.jsonl.
  math      optional: \\boxed{} answer accuracy with math-verify on a prompt/ground_truth jsonl.

Results go to eval_results/<tag>/: summary.json, summary.md, report.html and per-task *_predictions.jsonl
(open_qa predictions can be reviewed and rated in chat_feedback.py, "Review" tab).

Web report: when the run finishes, report.html is served at http://127.0.0.1:7870/<tag>/report.html and
opened in the browser (Ctrl-C stops the server). It shows score tiles, bar charts for every score
(overall, MMLU per category / subject, MCQ per subject, open-QA per level, reading grade vs target,
DS-1000 per library, val loss per level; side by side with --compare) and browsable examples per area
with filters (right / wrong, subject / level / library, text search). eval_results/index.html lists all runs.
    python scripts/eval_model.py --report-only base                       # reopen an earlier run
    python scripts/eval_model.py --report-only lora_r16 --compare base    # ... compared with another run
    python scripts/eval_model.py --tasks mmlu --no-ui                     # headless / batch jobs
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from ai_qa_train.common import (ROOT, encode_chat, generate, load_model, load_tokenizer, read_jsonl,  # noqa: E402
                                render, resolve_adapter, split_thinking)
from ai_qa_data.levels import fk_grade  # noqa: E402
from ai_qa_data.taxonomy import LEVELS  # noqa: E402

EVAL_DIR = os.path.join(ROOT, "data", "final", "eval")
RESULTS_ROOT = os.path.join(ROOT, "eval_results")
LETTERS = "ABCDEFGHIJ"
WORD = re.compile(r"\w+")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def rouge_l(pred: str, ref: str) -> float:
    a, b = WORD.findall(pred.lower())[:600], WORD.findall(ref.lower())[:600]
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    p, r = lcs / len(a), lcs / len(b)
    return 0.0 if lcs == 0 else 2 * p * r / (p + r)


def token_f1(pred: str, ref: str) -> float:
    a, b = Counter(WORD.findall(pred.lower())), Counter(WORD.findall(ref.lower()))
    common = sum((a & b).values())
    if not common:
        return 0.0
    p, r = common / sum(a.values()), common / sum(b.values())
    return 2 * p * r / (p + r)


def mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


# --------------------------------------------------------------------------- #
# tasks
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def score_letters(model, tok, prompts: list, n_opts: list[int], bs: int, system: str | None = None,
                  raw: bool = False, return_mass: bool = False):
    """Pick the answer letter by next-token log-likelihood. Only the last position's logits are computed.
    chat (raw=False): each prompt is a user string or a full message list; rendered with the chat template and
                      scored right after the assistant prefix in non-thinking mode; letters "A","B",...
    raw  (raw=True):  plain completion strings ending in "Answer:" (how base models are evaluated in the
                      Qwen3 report / lm-eval-harness); letters " A"," B",... (space-prefixed tokens).
    With return_mass=True also returns, per item, the probability mass the model puts on the option letters
    (~1.0 when the prompt format suits the model; ~0.0 means the scores are meaningless)."""
    letter_ids = [tok.encode((" " if raw else "") + L, add_special_tokens=False)[0] for L in LETTERS]
    texts = []
    for p in prompts:
        if raw:
            texts.append(p)
            continue
        msgs = p if isinstance(p, list) else [{"role": "user", "content": p}]
        if system and not any(m["role"] == "system" for m in msgs):
            msgs = [{"role": "system", "content": system}] + msgs
        texts.append(render(tok, msgs, add_generation_prompt=True, enable_thinking=False))
    preds, mass = [""] * len(texts), [0.0] * len(texts)
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))  # length-bucketed batches
    tok.padding_side = "left"
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        probs = logits.softmax(-1)
        for i, lg, pr in zip(idx, logits, probs):
            ids = letter_ids[:n_opts[i]]
            preds[i] = LETTERS[int(lg[ids].argmax())]
            mass[i] = float(pr[ids].sum())
    tok.padding_side = "right"
    return (preds, mass) if return_mass else preds


def collapse_warning(preds: list[str], mass: list[float], fmt: str, task: str = "mmlu", shots: int = 0,
                     gold: list[str] | None = None, scoring: str = "loglik") -> str | None:
    """Detect degenerate letter scoring: little probability on the letters (loglik) or few readable answers
    (generate), one letter for (almost) everything, or a letter predicted far more often than it is correct
    (position bias: what first-token scoring of a chat model at 0-shot produces, where the first token is a word)."""
    if not preds:
        return None
    n = len(preds)
    top, cnt = Counter(preds).most_common(1)[0]
    avg_mass = sum(mass) / n
    problems = []
    if avg_mass < 0.8:
        problems.append(f"only {avg_mass:.1%} of the replies end in a readable answer letter" if scoring == "generate"
                        else f"the model puts only {avg_mass:.1%} probability on the answer letters")
    if top and cnt / n > 0.7:
        problems.append(f"{cnt / n:.0%} of predictions are '{top}'")
    if gold:
        g = Counter(gold)
        problems += [f"'{L}' is predicted {c / n:.0%} of the time but correct {g[L] / n:.0%}"
                     for L, c in Counter(preds).items() if L and c / n > 0.35 and c / n > 1.6 * g[L] / n]
    if not problems:
        return None
    if scoring == "generate" and avg_mass >= 0.8:  # letters read from the replies: a skew is the model's own answers
        return (f"POSITION BIAS in {task}: " + "; ".join(problems) + ". The letters are parsed from the replies "
                f"({avg_mass:.1%} readable), so the accuracy is valid; the model itself favours these options.")
    if scoring == "generate":
        hints = ["--mmlu-max-new-tokens 1024 (replies may be cut off before the answer line)",
                 "--mmlu-shots 5 --mmlu-scoring loglik"]
    else:
        other = "base" if fmt == "chat" else "chat"
        hints = [f"--mmlu-format {other} (this model may not suit the '{fmt}' format)"]
        if shots == 0:
            hints.insert(0, f"--{task}-shots 5 (at 0-shot chat models often want to start with a word such as 'The ...')")
            if task == "mmlu" and fmt == "chat":
                hints.insert(0, "--mmlu-scoring generate")
    return (f"DEGENERATE SCORING in {task}: " + "; ".join(problems) + ", so this accuracy is not meaningful. "
            "Try " + " or ".join(hints) + ".")


@torch.inference_mode()
def mcq_prompt(r: dict, shots: list[dict], fmt: str):
    """Same scheme as MMLU: base = one completion string; chat = few-shot examples as earlier chat turns.
    MCQ questions already contain their lettered options."""
    head = "The following are multiple choice questions (with answers) about artificial intelligence and computing.\n\n"
    if fmt == "base":
        return head + "".join(mmlu_block(x, True) for x in shots) + r["question"].strip() + "\nAnswer:"

    def q(x):
        return x["question"].strip() + "\n" + "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(x["choices"]))

    msgs = []
    for k, ex in enumerate(shots):
        msgs += [{"role": "user", "content": (head if k == 0 else "") + q(ex) + "\n\n" + CHAT_INSTRUCTION},
                 {"role": "assistant", "content": LETTERS[int(ex["answer"])]}]
    msgs.append({"role": "user", "content": ("" if shots else head) + r["question"].strip() + "\n\n" + CHAT_INSTRUCTION})
    return msgs


def eval_mcq(model, tok, rows, out_dir, args):
    """AI multiple choice (MMLU machine-learning + filtered MMLU-Pro items from the data pipeline).
    Few-shot examples come from the local MMLU machine_learning *dev* split (never from the test items)."""
    preds, by_src, by_subj = [], defaultdict(list), defaultdict(list)
    shots = []
    if args.mcq_shots:
        if os.path.isdir(os.path.join(args.mmlu_dir, "machine_learning")):
            shots = load_mmlu_split(args.mmlu_dir, "machine_learning", "dev")[:args.mcq_shots]
        else:
            print(f"! --mcq-shots needs {args.mmlu_dir}/machine_learning/dev-*.parquet for examples; running 0-shot")
    n_shots = len(shots)
    prompts = [mcq_prompt(r, shots, args.mmlu_format) for r in rows]
    n_opts = [len(re.findall(r"^[A-J]\. ", r["question"], re.MULTILINE)) or 4 for r in rows]
    letters, mass = score_letters(model, tok, prompts, n_opts, args.bs, raw=args.mmlu_format == "base",
                                  return_mass=True)
    warning = collapse_warning(letters, mass, args.mmlu_format, task="mcq", shots=n_shots,
                               gold=[r.get("ground_truth") for r in rows])
    if warning:
        print("\n" + "!" * 100 + "\n" + warning + "\n" + "!" * 100 + "\n", flush=True)
    for r, pred in zip(rows, letters):
        ok = pred == r.get("ground_truth")
        subj = (r.get("meta") or {}).get("subject") or (r.get("meta") or {}).get("category") or "?"
        preds.append({"id": r["id"], "source": r["source"], "subject": subj, "question": r["question"],
                      "answer": r.get("answer"), "pred": pred, "gold": r.get("ground_truth"), "correct": ok})
        by_src[r["source"]].append(ok)
        subj = (r.get("meta") or {}).get("subject") or (r.get("meta") or {}).get("category") or "?"
        by_subj[subj].append(ok)
    _dump(out_dir, "mcq_predictions.jsonl", preds)
    return {"n": len(preds), "accuracy": mean(p["correct"] for p in preds), "shots": n_shots,
            "format": args.mmlu_format, "pred_distribution": dict(Counter(letters)),
            "letter_prob_mass": mean(mass), "warning": warning,
            "by_source": {k: mean(v) for k, v in by_src.items()},
            "by_subject": {k: {"acc": mean(v), "n": len(v)} for k, v in sorted(by_subj.items())}}


# --------------------------------------------------------------------------- #
# MMLU from a local copy (Hugging Face `cais/mmlu` parquet layout):
#   <mmlu-dir>/<subject>/{dev,validation,test}-00000-of-00001.parquet
#   columns: question, subject, choices (list[str]), answer (int index)
# --------------------------------------------------------------------------- #
MMLU_DIR = os.path.join(ROOT, "data", "eval", "mmlu")

# Official subject -> sub-category -> category mapping (hendrycks/test, categories.py)
MMLU_SUBCATEGORY = {
    "abstract_algebra": "math", "anatomy": "health", "astronomy": "physics", "business_ethics": "business",
    "clinical_knowledge": "health", "college_biology": "biology", "college_chemistry": "chemistry",
    "college_computer_science": "computer science", "college_mathematics": "math", "college_medicine": "health",
    "college_physics": "physics", "computer_security": "computer science", "conceptual_physics": "physics",
    "econometrics": "economics", "electrical_engineering": "engineering", "elementary_mathematics": "math",
    "formal_logic": "philosophy", "global_facts": "other", "high_school_biology": "biology",
    "high_school_chemistry": "chemistry", "high_school_computer_science": "computer science",
    "high_school_european_history": "history", "high_school_geography": "geography",
    "high_school_government_and_politics": "politics", "high_school_macroeconomics": "economics",
    "high_school_mathematics": "math", "high_school_microeconomics": "economics", "high_school_physics": "physics",
    "high_school_psychology": "psychology", "high_school_statistics": "math", "high_school_us_history": "history",
    "high_school_world_history": "history", "human_aging": "health", "human_sexuality": "culture",
    "international_law": "law", "jurisprudence": "law", "logical_fallacies": "philosophy",
    "machine_learning": "computer science", "management": "business", "marketing": "business",
    "medical_genetics": "health", "miscellaneous": "other", "moral_disputes": "philosophy",
    "moral_scenarios": "philosophy", "nutrition": "health", "philosophy": "philosophy", "prehistory": "history",
    "professional_accounting": "other", "professional_law": "law", "professional_medicine": "health",
    "professional_psychology": "psychology", "public_relations": "politics", "security_studies": "politics",
    "sociology": "culture", "us_foreign_policy": "politics", "virology": "health", "world_religions": "philosophy",
}
MMLU_CATEGORY = {
    "STEM": {"physics", "chemistry", "biology", "computer science", "math", "engineering"},
    "humanities": {"history", "philosophy", "law"},
    "social_sciences": {"politics", "culture", "economics", "geography", "psychology"},
    "other": {"other", "business", "health"},
}
# Subjects most relevant to this AI tutor (used by --mmlu-subjects ai)
MMLU_AI_SUBJECTS = ["machine_learning", "college_computer_science", "high_school_computer_science",
                    "computer_security", "high_school_statistics", "college_mathematics", "high_school_mathematics",
                    "elementary_mathematics", "econometrics", "electrical_engineering", "formal_logic"]
# Extra subjects in the same layout that are not part of MMLU (written for this project), subject -> sub-category.
# "all" evaluates them too, but they stay out of accuracy / macro / by_category / ai_subjects so those remain
# comparable with runs made before the subject existed; each gets its own <subject>_accuracy in the summary.
#   ai_large: 1,000 ML / AI questions in 8 topics (data/eval/mmlu_ai_large, scripts/build_mmlu_ai_large.py)
MMLU_EXTRA_SUBJECTS = {"ai_large": "computer science"}
MMLU_SUBJECT_TITLE = {"ai_large": "machine learning and artificial intelligence"}  # prompt header name
# The "ai_related" subject: all of machine_learning plus the math / computer-science / formal_logic questions the
# zero-shot model judged AI-related, built by scripts/build_mmlu_ai_related.py (used by --mmlu-subjects ai_related
# and the ai_related_accuracy score)
MMLU_AI_RELATED = os.path.join(ROOT, "data", "eval", "mmlu_ai_related", "selection.json")


def load_ai_related(path: str | None) -> set[str]:
    """Question ids (mmlu/<subject>/<split>/<index>) of the ai_related subject; empty if it was not built."""
    if not path or not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return set(json.load(f)["ids"])


def mmlu_category(subject: str) -> str:
    sub = MMLU_SUBCATEGORY.get(subject) or MMLU_EXTRA_SUBJECTS.get(subject, "other")
    return next((c for c, subs in MMLU_CATEGORY.items() if sub in subs), "other")


def subject_title(subject: str) -> str:
    return MMLU_SUBJECT_TITLE.get(subject, subject.replace("_", " "))


def list_mmlu_subjects(mmlu_dir: str, extra: bool = False) -> list[str]:
    """MMLU's own subjects found in `mmlu_dir`; with extra=True also the MMLU_EXTRA_SUBJECTS present there."""
    import glob

    subs = []
    for d in sorted(os.listdir(mmlu_dir)):
        if d.startswith(".") or d in ("all", "auxiliary_train") or (d in MMLU_EXTRA_SUBJECTS and not extra):
            continue
        if glob.glob(os.path.join(mmlu_dir, d, "test-*.parquet")):
            subs.append(d)
    return subs


def load_mmlu_split(mmlu_dir: str, subject: str, split: str) -> list[dict]:
    import glob

    import pyarrow.parquet as pq

    rows = []
    for f in sorted(glob.glob(os.path.join(mmlu_dir, subject, f"{split}-*.parquet"))):
        names = pq.read_schema(f).names  # extra subjects also carry a "topic" column
        rows += pq.read_table(f, columns=[c for c in ("question", "subject", "choices", "answer", "topic")
                                          if c in names]).to_pylist()
    return rows


def mmlu_block(r: dict, with_answer: bool) -> str:
    s = r["question"].strip() + "\n" + "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(r["choices"]))
    return s + (f"\nAnswer: {LETTERS[int(r['answer'])]}\n\n" if with_answer else "\nAnswer:")


CHAT_INSTRUCTION = "Reply with only the letter of the correct option."


def mmlu_prompt(r: dict, subject: str, shots: list[dict], fmt: str = "chat"):
    """base: one completion string (standard MMLU few-shot prompt, ends in 'Answer:').
    chat: a message list; few-shot examples are earlier user/assistant turns (multi-turn few-shot), so a chat
          model sees exactly the interaction it is asked to continue."""
    head = f"The following are multiple choice questions (with answers) about {subject_title(subject)}.\n\n"
    if fmt == "base":
        return head + "".join(mmlu_block(s, True) for s in shots) + mmlu_block(r, False)

    def q(x):
        return x["question"].strip() + "\n" + "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(x["choices"]))

    msgs = []
    for k, ex in enumerate(shots):
        msgs += [{"role": "user", "content": (head if k == 0 else "") + q(ex) + "\n\n" + CHAT_INSTRUCTION},
                 {"role": "assistant", "content": LETTERS[int(ex["answer"])]}]
    msgs.append({"role": "user", "content": ("" if shots else head) + q(r) + "\n\n" + CHAT_INSTRUCTION})
    return msgs


# Zero-shot generative MMLU for chat models (--mmlu-scoring generate; simple-evals style). At 0-shot a chat model
# rarely starts its reply with the bare letter ("The correct answer is: **C. ...**", "We are given ..."), so the
# first-token letter logits are noise. Instead the model replies freely in non-thinking mode, is told to end with
# "Answer: X", and the letter is read from the text.
GEN_INSTRUCTION = ("Reason briefly if needed, then give your final answer on the last line in exactly this format: "
                   "\"Answer: X\", where X is {letters}.")
_LETTER = r"[\*_\s]*[\(\[]?\s*([A-J])(?![A-Za-z0-9])"  # optional bold / bracket, then a stand-alone capital letter
_ANSWER_PATTERNS = [  # (name, pattern, which match counts), most explicit first
    ("answer_line", re.compile(r"(?i:\banswer)\s*[:：]" + _LETTER), "last"),  # Answer: C / **Final Answer: (C)**
    ("boxed", re.compile(r"\\boxed\{\s*(?:\\text\{)?\s*\(?([A-J])(?![A-Za-z0-9])"), "last"),  # \boxed{C} (math style)
    ("answer_phrase", re.compile(r"(?i:\b(?:answer|option|choice)\s+(?:is|would be)\s*:?)" + _LETTER), "last"),
    ("leading_letter", re.compile(r"\A\s*[\*_\(\[]*([A-J])(?:[\.\):\]\*]|\s*$)", re.MULTILINE), "first"),  # reply "C. ..."
    ("bold_option", re.compile(r"\*\*\s*[\(\[]?([A-J])[\.\):\]]"), "unique"),  # one option in bold: "**C. text**"
]


def mmlu_gen_prompt(r: dict, subject: str) -> list[dict]:
    """One user turn: the question, its options and the answer-format instruction (last, nearest the reply)."""
    letters = [LETTERS[i] for i in range(len(r["choices"]))]
    opts = "\n".join(f"{L}. {c}" for L, c in zip(letters, r["choices"]))
    head = f"The following is a multiple choice question about {subject_title(subject)}.\n\n"
    ins = GEN_INSTRUCTION.format(letters=", ".join(letters[:-1]) + " or " + letters[-1])
    return [{"role": "user", "content": head + r["question"].strip() + "\n\n" + opts + "\n\n" + ins}]


def extract_choice(text: str, n_opts: int) -> tuple[str | None, str | None]:
    """(letter, pattern name) of the answer stated in a generated reply, or (None, None). The final "Answer: X" line
    wins over letters mentioned while reasoning; capitals only, so the article "a" never counts."""
    valid = set(LETTERS[:n_opts])
    for name, pat, pick in _ANSWER_PATTERNS:
        found = [m.group(1) for m in pat.finditer(text) if m.group(1) in valid]
        if pick == "unique":
            found = found if len(set(found)) == 1 else []
        if found:
            return (found[-1] if pick == "last" else found[0]), name
    return None, None


def resolve_mmlu_scoring(scoring: str, fmt: str, shots: int) -> str:
    """auto = generate for a chat-format 0-shot run (the case first-token scoring gets wrong), loglik otherwise
    (k-shot chat turns and the base-format prompt make the next token the letter, so loglik is valid and cheap)."""
    if scoring == "auto":
        scoring = "generate" if fmt == "chat" and shots == 0 else "loglik"
    if scoring == "generate" and (fmt != "chat" or shots):
        raise SystemExit("--mmlu-scoring generate is zero-shot with the chat template: use --mmlu-format chat "
                         "--mmlu-shots 0 (or --mmlu-scoring loglik)")
    return scoring


def is_base_checkpoint(path: str | None) -> bool:
    """True for pre-trained (not chat-tuned) checkpoints such as Qwen3-1.7B-Base. Uses the directory / repo
    name, which is how Qwen (and most model families) mark base models."""
    return bool(path) and re.search(r"(^|[-_/])base($|[-_/])", os.path.basename(os.path.normpath(path)).lower()) is not None


def resolve_mmlu_format(fmt: str, base_path: str) -> str:
    if fmt != "auto":
        return fmt
    return "base" if is_base_checkpoint(base_path) else "chat"


def resolve_mmlu_subjects(spec: str, available: list[str], related: set[str] = frozenset()) -> list[str]:
    if spec == "all":
        return available
    if spec == "ai":
        return [s for s in MMLU_AI_SUBJECTS if s in available]
    if spec == "ai_related":
        if not related:
            raise SystemExit(f"no ai_related subject at {MMLU_AI_RELATED}: run scripts/build_mmlu_ai_related.py first")
        subs = {i.split("/")[1] for i in related}
        return [s for s in available if s in subs]
    if spec in MMLU_CATEGORY:
        return [s for s in available if mmlu_category(s) == spec and s not in MMLU_EXTRA_SUBJECTS]
    want = [s.strip() for s in spec.split(",") if s.strip()]
    missing = [s for s in want if s not in available]
    if missing:
        raise SystemExit(f"unknown MMLU subjects {missing}; available: {available}")
    return want


@torch.inference_mode()
def eval_mmlu(model, tok, out_dir, args):
    mdir = args.mmlu_dir
    if not os.path.isdir(mdir):
        z = mdir.rstrip("/") + ".zip"
        hint = f" (found {z}: unzip it first: unzip {z} -d {os.path.dirname(mdir)})" if os.path.exists(z) else ""
        raise SystemExit(f"MMLU directory not found: {mdir}{hint}")
    related = load_ai_related(args.mmlu_ai_related)
    only_related = args.mmlu_subjects == "ai_related"
    subjects = resolve_mmlu_subjects(args.mmlu_subjects, list_mmlu_subjects(mdir, extra=True), related)
    generative = args.mmlu_scoring == "generate"
    texts, n_opts, meta = [], [], []
    for subj in subjects:
        rows = load_mmlu_split(mdir, subj, args.mmlu_split)
        if args.mmlu_max_per_subject:
            rows = rows[:args.mmlu_max_per_subject]
        shots = load_mmlu_split(mdir, subj, "dev")[:args.mmlu_shots] if args.mmlu_shots else []
        for i, r in enumerate(rows):
            qid = f"mmlu/{subj}/{args.mmlu_split}/{i}"
            if only_related and qid not in related:
                continue
            texts.append(mmlu_gen_prompt(r, subj) if generative else mmlu_prompt(r, subj, shots, args.mmlu_format))
            n_opts.append(len(r["choices"]))
            meta.append({"id": qid, "subject": subj, "category": mmlu_category(subj),
                         "question": r["question"], "choices": list(r["choices"]),
                         "gold": LETTERS[int(r["answer"])], **({"topic": r["topic"]} if r.get("topic") else {})})
    t0 = time.time()
    if generative:  # non-thinking, greedy; mass = 1 where a letter could be read from the reply, else 0
        gens = generate(model, tok, texts, max_new_tokens=args.mmlu_max_new_tokens, enable_thinking=False,
                        temperature=0.0, batch_size=args.gen_bs)
        replies = [split_thinking(g)[1] or g for g in gens]
        parsed = [extract_choice(t, n) for t, n in zip(replies, n_opts)]
        letters = [L or "" for L, _ in parsed]
        mass = [1.0 if L else 0.0 for L in letters]
    else:
        letters, mass = score_letters(model, tok, texts, n_opts, args.bs, raw=args.mmlu_format == "base",
                                      return_mass=True)
    preds, by_subj = [], defaultdict(list)
    for k, (m, pred, pm) in enumerate(zip(meta, letters, mass)):
        ok = pred == m["gold"]
        if generative:
            preds.append({**m, "pred": pred or None, "correct": ok, "parse": parsed[k][1], "output": replies[k][-2000:]})
        else:
            preds.append({**m, "pred": pred, "correct": ok, "letter_prob": round(pm, 4)})
        by_subj[m["subject"]].append(ok)
    warning = collapse_warning(letters, mass, args.mmlu_format, task="mmlu", shots=args.mmlu_shots,
                               gold=[m["gold"] for m in meta], scoring=args.mmlu_scoring)
    if warning:
        print("\n" + "!" * 100 + "\n" + warning + "\n" + "!" * 100 + "\n", flush=True)
    _dump(out_dir, "mmlu_predictions.jsonl", preds)
    core = [p for p in preds if p["subject"] not in MMLU_EXTRA_SUBJECTS] or preds  # headline scores: MMLU itself
    by_cat = defaultdict(list)
    for p in core:
        by_cat[mmlu_category(p["subject"])].append(p["correct"])
    subj_acc = {s: {"acc": mean(v), "n": len(v), "category": mmlu_category(s),
                    **({"extra": True} if s in MMLU_EXTRA_SUBJECTS else {})} for s, v in sorted(by_subj.items())}
    core_subj = [v["acc"] for s, v in subj_acc.items() if s not in MMLU_EXTRA_SUBJECTS] or [v["acc"] for v in subj_acc.values()]
    ai = [p["correct"] for p in preds if p["subject"] in MMLU_AI_SUBJECTS]
    rel = [p["correct"] for p in preds if p["id"] in related]
    extra = {}
    for s in MMLU_EXTRA_SUBJECTS:
        ps = [p for p in preds if p["subject"] == s]
        if ps:
            topics = defaultdict(list)
            for p in ps:
                topics[p.get("topic") or "?"].append(p["correct"])
            extra.update({f"{s}_accuracy": mean(p["correct"] for p in ps), f"{s}_n": len(ps),
                          f"{s}_by_topic": {k: {"acc": mean(v), "n": len(v)} for k, v in sorted(topics.items())}})
    return {"n": len(core), "subjects": len(core_subj), "shots": args.mmlu_shots, "split": args.mmlu_split,
            "format": args.mmlu_format,
            "scoring": args.mmlu_scoring,
            "letter_prob_mass": None if generative else mean(mass),
            **({"thinking": False, "max_new_tokens": args.mmlu_max_new_tokens,
                "parse_rate": mean(mass),                                       # replies with a readable letter
                "parse_methods": dict(Counter(p["parse"] or "none" for p in preds))} if generative else {}),
            "pred_distribution": dict(Counter(L or "none" for L in letters)),
            "warning": warning,
            "seconds": round(time.time() - t0, 1),
            "accuracy": mean(p["correct"] for p in core),                        # micro (per question)
            "macro_accuracy": mean(core_subj),                                   # mean over subjects
            "ai_subjects_accuracy": mean(ai),
            "ai_related_accuracy": mean(rel),                                    # None until the subject is built
            "ai_related_n": len(rel),
            **extra,                                                             # ai_large_accuracy / _n / _by_topic
            "by_category": {c: {"acc": mean(v), "n": len(v)} for c, v in sorted(by_cat.items())},
            "by_subject": subj_acc}


def eval_open_qa(model, tok, rows, out_dir, args):
    convs = [[{"role": "system", "content": r["system"]}, {"role": "user", "content": r["question"]}] for r in rows]
    t0 = time.time()
    gens = generate(model, tok, convs, max_new_tokens=args.max_new_tokens, enable_thinking=args.thinking,
                    temperature=args.temperature, batch_size=args.gen_bs)
    preds = []
    for r, g in zip(rows, gens):
        think, ans = split_thinking(g)
        fk = fk_grade(ans)
        lo, hi = LEVELS[r["level"]]["fk"]
        preds.append({"id": r["id"], "level": r["level"], "topics": r.get("topics"), "system": r["system"],
                      "question": r["question"], "reference": r["reference"], "thinking": think, "answer": ans,
                      "rougeL": round(rouge_l(ans, r["reference"]), 4), "token_f1": round(token_f1(ans, r["reference"]), 4),
                      "fk_grade": round(fk, 2) if fk is not None else None,
                      "level_fit": (lo - 1 <= fk <= hi + 1) if fk is not None else None,
                      "answer_words": len(ans.split())})
    if args.level_judge:
        from ai_qa_data.zeroshot import ZeroShotClassifier
        zs = ZeroShotClassifier(batch_size=64, max_length=256)
        probs = zs.single_label([p["answer"][:1500] or " " for p in preds], {k: v["zs"] for k, v in LEVELS.items()})
        for p, pr in zip(preds, probs):
            p["judged_level"] = max(pr, key=pr.get)
            p["judged_match"] = p["judged_level"] == p["level"]
        del zs
    _dump(out_dir, "open_qa_predictions.jsonl", preds)
    per_level = {}
    for lv in LEVELS:
        ps = [p for p in preds if p["level"] == lv]
        if ps:
            per_level[lv] = {"n": len(ps), "rougeL": mean(p["rougeL"] for p in ps),
                             "token_f1": mean(p["token_f1"] for p in ps),
                             "fk_grade": mean(p["fk_grade"] for p in ps if p["fk_grade"] is not None),
                             "level_fit": mean(p["level_fit"] for p in ps if p["level_fit"] is not None),
                             "judged_match": mean(p["judged_match"] for p in ps) if args.level_judge else None,
                             "answer_words": mean(p["answer_words"] for p in ps)}
    return {"n": len(preds), "seconds": round(time.time() - t0, 1),
            "rougeL": mean(p["rougeL"] for p in preds), "token_f1": mean(p["token_f1"] for p in preds),
            "level_fit": mean(p["level_fit"] for p in preds if p["level_fit"] is not None),
            "judged_match": mean(p["judged_match"] for p in preds) if args.level_judge else None,
            "per_level": per_level}


_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
_CODE_TAG = re.compile(r"<code>\s*(.*?)\s*</code>", re.S)


def extract_code(text: str, prompt: str = "") -> str:
    m = _CODE_BLOCK.findall(text) or _CODE_TAG.findall(text)
    if m:
        code = m[-1]
    elif "```" in text:  # unclosed fence (generation hit max_new_tokens)
        code = re.sub(r"^(?:python|py)?\s*\n", "", text.split("```", 1)[1])
    else:
        code = text
    code = code.replace("END SOLUTION", "").replace("BEGIN SOLUTION", "").strip()
    # Models often re-type the problem's setup code (df = ..., List = np.random...), which would overwrite
    # the harness' test inputs. Drop solution lines that literally repeat a non-import setup line.
    setup = _CODE_TAG.findall(prompt)
    if setup:
        seen = {ln.strip() for ln in setup[0].splitlines() if ln.strip() and not ln.strip().startswith(("import", "from"))}
        code = "\n".join(ln for ln in code.splitlines() if ln.strip() not in seen)
    return code.strip()


_BODY_SLOT = re.compile(r"def (\w+)\(.*?\):\s*\n\s*\[insert\]")


def fit_to_slot(ctx: str, solution: str) -> str:
    """Some DS-1000 problems insert the solution as the *body* of `def f(...)`. Models usually write the
    whole function: keep only its body; or indent a bare snippet."""
    m = _BODY_SLOT.search(ctx)
    if not m:
        return solution
    lines = solution.splitlines()
    for i, ln in enumerate(lines):
        if re.match(rf"\s*def {m.group(1)}\(", ln):
            indent = len(ln) - len(ln.lstrip())
            body = []
            for b in lines[i + 1:]:
                if b.strip() and len(b) - len(b.lstrip()) <= indent:
                    break
                body.append(b)
            return "\n".join(body)
    return "\n".join(("    " + ln if ln.strip() and not ln.startswith((" ", "\t")) else ln) for ln in lines)


def run_ds1000(ctx: str, solution: str, timeout: int) -> tuple[bool, str]:
    solution = fit_to_slot(ctx, solution)
    prog = ctx + f"\n\nsolution = {solution!r}\ntest_execution(solution)\n"
    if "def test_string" in ctx:
        prog += "test_string(solution)\n"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(prog)
        try:
            r = subprocess.run([sys.executable, path], cwd=d, capture_output=True, text=True, timeout=timeout,
                               env={**os.environ, "MPLBACKEND": "Agg", "CUDA_VISIBLE_DEVICES": ""})
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if r.returncode == 0:
            return True, ""
        lines = r.stderr.strip().splitlines() or ["error"]
        # the exception line, not trailing warnings / numpy assertion dumps
        exc = [ln for ln in lines if re.match(r"^[\w.]*(Error|Exception|Interrupt|Exit)\b", ln)]
        err = exc[-1] if exc else lines[-1]
        return False, ("missing_lib: " + err) if "ModuleNotFoundError" in err else err[:300]


def eval_code(model, tok, rows, out_dir, args):
    convs = []
    for r in rows:
        q = r["question"] + ("\n\nComplete the solution. Reply with only the code that replaces "
                             "BEGIN SOLUTION, inside a single ```python``` block.")
        convs.append([{"role": "system", "content": LEVELS["ai_student"]["system"]}, {"role": "user", "content": q}])
    gens = generate(model, tok, convs, max_new_tokens=min(args.max_new_tokens, 768), enable_thinking=args.thinking,
                    temperature=0.0, batch_size=args.gen_bs)
    sols = [extract_code(split_thinking(g)[1] or g, r["question"]) for r, g in zip(rows, gens)]
    with ThreadPoolExecutor(args.exec_workers) as ex:
        results = list(ex.map(lambda x: run_ds1000(x[0]["tests"], x[1], args.exec_timeout), zip(rows, sols)))
        # environment ceiling: DS-1000 was written for older library versions (pandas 1.x, ...);
        # problems whose *reference* solution fails here are reported separately.
        refs = list(ex.map(lambda r: run_ds1000(r["tests"], r.get("answer") or "", args.exec_timeout), rows)) \
            if not args.no_ref_check else [(True, "")] * len(rows)
    preds, by_lib = [], defaultdict(list)
    for r, s, (ok, err), (ref_ok, _) in zip(rows, sols, results, refs):
        lib = (r.get("meta") or {}).get("library", "?")
        by_lib[lib].append((ok, ref_ok))
        preds.append({"id": r["id"], "library": lib, "passed": ok, "reference_passes": ref_ok, "error": err,
                      "solution": s, "question": r["question"][:4000], "reference": r.get("answer")})
    _dump(out_dir, "code_predictions.jsonl", preds)
    valid = [p for p in preds if p["reference_passes"]]
    return {"n": len(preds), "pass@1": mean(p["passed"] for p in preds),
            "reference_pass_rate": mean(p["reference_passes"] for p in preds),
            "pass@1_valid": mean(p["passed"] for p in valid), "n_valid": len(valid),
            "by_library": {k: {"pass@1": mean(o for o, _ in v), "reference_pass": mean(rf for _, rf in v), "n": len(v)}
                           for k, v in sorted(by_lib.items())},
            "env_missing_lib": sum(1 for p in preds if p["error"].startswith("missing_lib"))}


@torch.inference_mode()
def eval_val_loss(model, tok, path, limit, max_len):
    """Assistant-token loss. One sequence at a time; the LM head is applied only at supervised
    positions (full-vocab logits for a 2k-token batch would need tens of GB)."""
    rows = [r for _, r in zip(range(limit), read_jsonl(path))]
    tot_loss, tot_tok = 0.0, 0
    per_level = defaultdict(lambda: [0.0, 0])
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    for r in rows:
        e = encode_chat(tok, r["messages"], max_len)
        if not e:
            continue
        ids = torch.tensor([e["input_ids"]], device=model.device)
        lab = torch.tensor(e["labels"][1:], device=model.device)
        hid = base.model(input_ids=ids).last_hidden_state[0, :-1]
        m = lab != -100
        logits = base.lm_head(hid[m]).float()
        l = float(torch.nn.functional.cross_entropy(logits, lab[m], reduction="sum"))
        n = int(m.sum())
        tot_loss, tot_tok = tot_loss + l, tot_tok + n
        pl = per_level[r.get("level", "?")]
        pl[0] += l
        pl[1] += n
    avg = tot_loss / max(1, tot_tok)
    return {"n": len(rows), "loss": round(avg, 4), "ppl": round(math.exp(avg), 3),
            "per_level_loss": {k: round(v[0] / max(1, v[1]), 4) for k, v in per_level.items()}}


def eval_math(model, tok, path, limit, args):
    from math_verify import parse, verify

    rows = [r for _, r in zip(range(limit), read_jsonl(path))]
    gens = generate(model, tok, [r["prompt"] for r in rows], max_new_tokens=args.max_new_tokens,
                    enable_thinking=args.thinking, temperature=0.0, batch_size=args.gen_bs)
    preds = []
    for r, g in zip(rows, gens):
        ans = split_thinking(g)[1] or g
        try:
            ok = bool(verify(parse("$" + r["ground_truth"] + "$"), parse(ans)))
        except Exception:  # noqa: BLE001
            ok = False
        q = next((m["content"] for m in r["prompt"] if m["role"] == "user"), "")
        preds.append({"id": r["id"], "correct": ok, "gold": r["ground_truth"], "question": q, "output": ans[-1500:]})
    _dump(args._out, "math_predictions.jsonl", preds)
    return {"n": len(preds), "accuracy": mean(p["correct"] for p in preds),
            "note": "only meaningful on problems that were NOT used for training"}


def _dump(out_dir, name, rows):
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
def to_markdown(summary: dict, compare: dict | None) -> str:
    def cmp(path, val):
        if compare is None or val is None:
            return ""
        ref = compare
        for k in path:
            ref = (ref or {}).get(k) if isinstance(ref, dict) else None
        return f" ({val - ref:+.4f})" if isinstance(ref, (int, float)) else ""

    L = [f"# Eval: {summary['tag']}", "", f"model: `{summary['model']}`  thinking={summary['thinking']}", ""]
    t = summary["tasks"]
    if "mmlu" in t:
        m = t["mmlu"]
        gen = m.get("scoring") == "generate"
        how = ("0-shot generative, non-thinking" if gen else f"{m['shots']}-shot, {m.get('format', 'chat')} format")
        L += [f"- **MMLU** ({m['split']}, {how}, {m['subjects']} subjects, n={m['n']}): "
              f"accuracy {m['accuracy']}{cmp(['tasks', 'mmlu', 'accuracy'], m['accuracy'])}, "
              f"macro {m['macro_accuracy']}{cmp(['tasks', 'mmlu', 'macro_accuracy'], m['macro_accuracy'])}, "
              f"AI-related subjects {m['ai_subjects_accuracy']}"
              f"{cmp(['tasks', 'mmlu', 'ai_subjects_accuracy'], m['ai_subjects_accuracy'])}"
              + (f", ai_related {m['ai_related_accuracy']}{cmp(['tasks', 'mmlu', 'ai_related_accuracy'], m['ai_related_accuracy'])}"
                 f" (n={m['ai_related_n']})" if m.get("ai_related_accuracy") is not None else "")
              + (f", ai_large {m['ai_large_accuracy']}{cmp(['tasks', 'mmlu', 'ai_large_accuracy'], m['ai_large_accuracy'])}"
                 f" (n={m['ai_large_n']}, not in the MMLU accuracy)" if m.get("ai_large_accuracy") is not None else "")
              + (f"; answer letter parsed from {m.get('parse_rate')} of replies ({m.get('parse_methods')})" if gen
                 else f"; probability on answer letters {m.get('letter_prob_mass')}")]
        if m.get("warning"):
            L += [f"- ⚠️ **{m['warning']}**"]
    if "mcq" in t:
        L += [f"- **MCQ accuracy**: {t['mcq']['accuracy']}{cmp(['tasks', 'mcq', 'accuracy'], t['mcq']['accuracy'])} "
              f"(n={t['mcq']['n']}, {t['mcq'].get('shots', 0)}-shot, {t['mcq'].get('format', 'chat')} format, "
              f"probability on answer letters {t['mcq'].get('letter_prob_mass')})"]
        if t["mcq"].get("warning"):
            L += [f"- ⚠️ **{t['mcq']['warning']}**"]
    if "code" in t:
        c = t["code"]
        L += [f"- **DS-1000 pass@1**: {c['pass@1']}{cmp(['tasks', 'code', 'pass@1'], c['pass@1'])} (n={c['n']}); "
              f"on env-valid problems {c['pass@1_valid']} (n={c['n_valid']}, reference pass rate "
              f"{c['reference_pass_rate']}, missing-lib errors {c['env_missing_lib']})"]
    if "val_loss" in t:
        L += [f"- **Val loss**: {t['val_loss']['loss']}{cmp(['tasks', 'val_loss', 'loss'], t['val_loss']['loss'])} "
              f"(ppl {t['val_loss']['ppl']})"]
    if "math" in t:
        L += [f"- **Math accuracy**: {t['math']['accuracy']} (n={t['math']['n']})"]
    if "open_qa" in t:
        o = t["open_qa"]
        L += [f"- **Open QA**: ROUGE-L {o['rougeL']}{cmp(['tasks', 'open_qa', 'rougeL'], o['rougeL'])}, "
              f"token-F1 {o['token_f1']}, level-fit {o['level_fit']}{cmp(['tasks', 'open_qa', 'level_fit'], o['level_fit'])}"
              + (f", judged-level match {o['judged_match']}" if o.get("judged_match") is not None else ""), "",
              "| level | n | ROUGE-L | token-F1 | FK grade | level fit | judged match | words |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for lv, m in o["per_level"].items():
            L.append(f"| {lv} | {m['n']} | {m['rougeL']} | {m['token_f1']} | {m['fk_grade']} | {m['level_fit']} | "
                     f"{m['judged_match'] if m['judged_match'] is not None else '-'} | {m['answer_words']} |")
    if "mmlu" in t:
        m = t["mmlu"]
        L += ["", "## MMLU by category", "", "| category | n | accuracy |", "|---|---:|---:|"]
        L += [f"| {c} | {v['n']} | {v['acc']}{cmp(['tasks', 'mmlu', 'by_category', c, 'acc'], v['acc'])} |"
              for c, v in m["by_category"].items()]
        L += ["", "## MMLU by subject", "", "| subject | category | n | accuracy |", "|---|---|---:|---:|"]
        L += [f"| {s} | {v['category']} | {v['n']} | {v['acc']}{cmp(['tasks', 'mmlu', 'by_subject', s, 'acc'], v['acc'])} |"
              for s, v in m["by_subject"].items()]
        for s in MMLU_EXTRA_SUBJECTS:
            if m.get(f"{s}_by_topic"):
                L += ["", f"## {s} by topic (not part of the MMLU accuracy)", "", "| topic | n | accuracy |", "|---|---:|---:|"]
                L += [f"| {k} | {v['n']} | {v['acc']}{cmp(['tasks', 'mmlu', f'{s}_by_topic', k, 'acc'], v['acc'])} |"
                      for k, v in m[f"{s}_by_topic"].items()]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="adapter dir, run dir, merged model dir, or empty = base model")
    ap.add_argument("--tasks", default="mmlu,mcq,open_qa,code,val_loss",
                    help="comma list of: mmlu, mcq, open_qa, code, val_loss, math")
    ap.add_argument("--eval-dir", default=EVAL_DIR)
    ap.add_argument("--val-file", default=os.path.join(ROOT, "data", "final", "sft_val.jsonl"))
    ap.add_argument("--math-file", default=None)
    ap.add_argument("--mmlu-dir", default=MMLU_DIR, help="local MMLU (HF cais/mmlu parquet layout)")
    ap.add_argument("--mmlu-subjects", default="all",
                    help="all | ai | ai_related | STEM | humanities | social_sciences | other | comma list of subjects")
    ap.add_argument("--mmlu-ai-related", default=MMLU_AI_RELATED,
                    help="selection.json of the ai_related subject (scripts/build_mmlu_ai_related.py, or the LLM-judged one from scripts/build_mmlu_ai_related_llm.py)")
    ap.add_argument("--mmlu-split", default="test", choices=["test", "validation", "dev"])
    ap.add_argument("--mmlu-shots", type=int, default=0, help="k-shot examples from the dev split (max 5)")
    ap.add_argument("--mmlu-format", choices=["auto", "chat", "base"], default="auto",
                    help="auto (default) = 'base' for pre-trained *-Base checkpoints (and adapters trained on one), "
                         "'chat' otherwise. chat = chat template, non-thinking, few-shot as earlier chat turns, "
                         "letter after the assistant prefix (how the tutor is used). base = plain completion prompt "
                         "ending in 'Answer:' (Qwen3 report / lm-eval-harness style)")
    ap.add_argument("--mmlu-scoring", choices=["auto", "loglik", "generate"], default="auto",
                    help="loglik = answer letter by next-token log-likelihood (one forward pass). generate = 0-shot chat "
                         "only: the model replies in non-thinking mode (greedy), told to end with 'Answer: X', and the "
                         "letter is parsed from the text; unparsed replies count as wrong (parse_rate is reported). "
                         "auto (default) = generate for chat format at 0-shot, where a chat model's first token is "
                         "usually a word and loglik turns into position bias; loglik otherwise")
    ap.add_argument("--mmlu-max-new-tokens", type=int, default=512,
                    help="--mmlu-scoring generate: reply budget (small models reason briefly even in non-thinking mode)")
    ap.add_argument("--mcq-shots", type=int, default=5,
                    help="few-shot examples for the AI MCQ task (from MMLU machine_learning dev; 0 = zero-shot)")
    ap.add_argument("--mmlu-max-per-subject", type=int, default=None, help="quick runs: cap questions per subject")
    ap.add_argument("--limit", type=int, default=None, help="max items per task")
    ap.add_argument("--thinking", action="store_true", help="generate in Qwen3 thinking mode")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--gen-bs", type=int, default=16)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--level-judge", action="store_true", help="zero-shot audience classification of answers")
    ap.add_argument("--exec-workers", type=int, default=8)
    ap.add_argument("--exec-timeout", type=int, default=60)
    ap.add_argument("--no-ref-check", action="store_true", help="skip running DS-1000 reference solutions")
    ap.add_argument("--chat-model", default=None,
                    help="model dir for the chat-format generation tasks in --chat-tasks: the adapter (if any) is applied "
                         "to this model instead of its own base. Use the chat model (models/Qwen3-0.6B) for CPT adapters "
                         "trained on a -Base checkpoint, whose chat special tokens are untrained")
    ap.add_argument("--chat-tasks", default="open_qa", help="comma list of tasks evaluated on --chat-model")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--base", default=None,
                    help="load the adapter on this model instead of the base recorded in its adapter_config.json, e.g. "
                         "the chat model models/Qwen3-0.6B to test a CPT adapter on the SFT model (all tasks)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--compare", default=None, help="previous run to diff against: summary.json, its folder, or a tag")
    ap.add_argument("--no-ui", action="store_true", help="do not start the web report server at the end")
    ap.add_argument("--no-browser", action="store_true", help="serve the report but do not open a browser")
    ap.add_argument("--ui-host", default="127.0.0.1")
    ap.add_argument("--ui-port", type=int, default=7870)
    ap.add_argument("--report-examples", type=int, default=300, help="examples embedded per area in the report")
    ap.add_argument("--report-only", default=None, metavar="TAG",
                    help="skip evaluation; rebuild and open the report of an existing run (tag or folder)")
    args = ap.parse_args()

    if args.report_only:
        out = args.report_only if os.path.isdir(args.report_only) else os.path.join(RESULTS_ROOT, args.report_only)
        with open(os.path.join(out, "summary.json")) as f:
            summary = json.load(f)
        open_report(out, summary, load_compare(args.compare), args)
        return

    base, adapter = resolve_adapter(args.model)
    if args.base:
        if not adapter:
            raise SystemExit("--base only makes sense with an adapter as --model")
        base = os.path.abspath(args.base)
    tag = args.tag or (os.path.basename(os.path.dirname(adapter.rstrip("/"))) if adapter else "base")
    out = os.path.join(RESULTS_ROOT, tag)
    os.makedirs(out, exist_ok=True)
    args._out = out
    args.mmlu_format = resolve_mmlu_format(args.mmlu_format, base)
    args.mmlu_scoring = resolve_mmlu_scoring(args.mmlu_scoring, args.mmlu_format, args.mmlu_shots)
    print(f"base={base} adapter={adapter} -> {out}  (MMLU prompt format: {args.mmlu_format}, scoring: {args.mmlu_scoring})")
    tok = load_tokenizer(base)
    model = load_model(base, adapter=adapter, quantize_4bit=args.load_4bit, merge=True)

    tasks = args.tasks.split(",")
    chat_tasks = [t for t in args.chat_tasks.split(",") if t in tasks] if args.chat_model else []
    lim = args.limit
    summary = {"tag": tag, "model": adapter or base, "base": base, "thinking": args.thinking,
               "time": time.strftime("%F %T"), "tasks": {}}
    prev = load_previous_summary(out, adapter or base)
    if prev:  # a partial run keeps the earlier results of the tasks it does not repeat
        kept = {t: v for t, v in prev.get("tasks", {}).items() if t not in tasks}
        if kept:
            summary["tasks"].update(kept)
            summary["previous_time"] = prev.get("time")
            for k in ("chat_model", "chat_tasks"):
                if k in prev and not args.chat_model:
                    summary[k] = prev[k]
            print(f"keeping earlier results of {sorted(kept)} from {prev.get('time')} (not run this time)")
    if chat_tasks:
        summary["chat_model"] = args.chat_model
        summary["chat_tasks"] = chat_tasks
        print(f"chat-format tasks {chat_tasks} on {args.chat_model}" + (f" + adapter {adapter}" if adapter else ""))
    if "mmlu" in tasks:
        summary["tasks"]["mmlu"] = eval_mmlu(model, tok, out, args)
        mm = summary["tasks"]["mmlu"]
        print("mmlu", {k: mm[k] for k in ("n", "accuracy", "macro_accuracy", "ai_subjects_accuracy", "ai_related_accuracy")})
    if "mcq" in tasks and os.path.exists(os.path.join(args.eval_dir, "mcq.jsonl")):
        rows = list(read_jsonl(os.path.join(args.eval_dir, "mcq.jsonl")))[:lim]
        rows = [r for r in rows if r.get("ground_truth")]
        summary["tasks"]["mcq"] = eval_mcq(model, tok, rows, out, args)
        print("mcq", summary["tasks"]["mcq"]["accuracy"])
    def generation_tasks(m, t, names):
        if "open_qa" in names and os.path.exists(os.path.join(args.eval_dir, "open_qa.jsonl")):
            rows = list(read_jsonl(os.path.join(args.eval_dir, "open_qa.jsonl")))[:lim]
            summary["tasks"]["open_qa"] = eval_open_qa(m, t, rows, out, args)
            print("open_qa", {k: summary["tasks"]["open_qa"][k] for k in ("rougeL", "token_f1", "level_fit")})
        if "code" in names and os.path.exists(os.path.join(args.eval_dir, "code.jsonl")):
            rows = [r for r in read_jsonl(os.path.join(args.eval_dir, "code.jsonl")) if r.get("tests")][:lim]
            summary["tasks"]["code"] = eval_code(m, t, rows, out, args)
            print("code", summary["tasks"]["code"]["pass@1"])

    generation_tasks(model, tok, [t for t in tasks if t not in chat_tasks])
    if "val_loss" in tasks and os.path.exists(args.val_file):
        summary["tasks"]["val_loss"] = eval_val_loss(model, tok, args.val_file, lim or 1000, 2048)
        print("val_loss", summary["tasks"]["val_loss"]["loss"])
    if "math" in tasks and args.math_file:
        summary["tasks"]["math"] = eval_math(model, tok, args.math_file, lim or 200, args)
    if chat_tasks:
        del model
        torch.cuda.empty_cache()
        tok = load_tokenizer(args.chat_model)
        model = load_model(args.chat_model, adapter=adapter, quantize_4bit=args.load_4bit, merge=True)
        generation_tasks(model, tok, chat_tasks)
        for t in chat_tasks:
            if t in summary["tasks"]:
                summary["tasks"][t]["model"] = args.chat_model + (f" + {adapter}" if adapter else "")

    compare = load_compare(args.compare)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    md = to_markdown(summary, compare)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(md)
    print("\n" + md)
    del model
    torch.cuda.empty_cache()
    open_report(out, summary, compare, args)


def load_previous_summary(out: str, model_id: str) -> dict | None:
    """summary.json already in this tag's folder, if it evaluated the same model (so a run of some tasks can keep
    the others' results); None when absent or made with another model."""
    path = os.path.join(out, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        prev = json.load(f)

    def norm(p: str) -> str:
        return os.path.normpath(p if os.path.isabs(p) else os.path.join(ROOT, p))

    if norm(prev.get("model") or "") != norm(model_id):
        print(f"! {path} was made with {prev.get('model')}, not {model_id}: its tasks are not kept")
        return None
    return prev


def load_compare(path: str | None) -> dict | None:
    if not path:
        return None
    if os.path.isdir(path):  # accept a run dir / tag dir as well as summary.json
        path = os.path.join(path, "summary.json")
    elif not os.path.exists(path) and os.path.exists(os.path.join(RESULTS_ROOT, path, "summary.json")):
        path = os.path.join(RESULTS_ROOT, path, "summary.json")
    with open(path) as f:
        return json.load(f)


def open_report(out: str, summary: dict, compare: dict | None, args):
    """Write report.html (+ the all-runs index) and serve it as a local web page."""
    from ai_qa_train.report import build_index, build_report, serve

    page = build_report(out, summary, compare, per_area=args.report_examples)
    build_index(RESULTS_ROOT)
    print(f"report written: {page}")
    if not args.no_ui:
        serve(RESULTS_ROOT, f"{os.path.basename(out)}/report.html", args.ui_host, args.ui_port,
              open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
