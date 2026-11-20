"""General-reasoning filter: is a record a good *reasoning* SFT example, whatever its topic?

The AI filter (keywords.py + zero-shot) answers "is this about AI?". This module answers a different
question for the `reasoning_sft` bucket - math derivations, Python problem solving, logic puzzles and
science reasoning that teach the model *how to think*, independent of the AI topic:

  * complete      the <think> trace is closed and followed by a real answer (no truncated R1 dumps)
  * fits          question + trace + answer <= max_chars (the training context, ~2048 tokens ~= 7k chars)
  * not degenerate  no repetition loops, no refusals
  * correct       when the source ships a ground truth, the final \\boxed{} / option letter must match it
  * worked        records without a trace qualify only as worked solutions: step-by-step math ending in
                  \\boxed{}, or a Python code block plus explanation

Used by collect_datasets.py (to keep reasoning rows that the AI pre-filter rejects) and by
split_ai_data.py (to route non-AI reasoning rows to data/split/reasoning_sft).
"""
from __future__ import annotations

import re

from .sources import last_boxed, split_think

DEFAULT_MAX_CHARS = 12000   # collection budget; the mix recipe applies its own (tighter) max_reasoning_chars
MIN_TRACE_CHARS = 200
CATEGORIES = ("math", "code", "logic", "science", "general")

_REFUSAL = re.compile(r"^\s*(?:I'?m sorry|I am sorry|I apologi[sz]e|As an AI|I (?:can(?:no|')t|am unable to) (?:help|assist))", re.I)
_CODE = re.compile(r"```(?:python|py)?\s*\n|^\s*(?:def|class) \w+.*:\s*$|^\s*(?:import \w+|from \w+(?:\.\w+)* import )", re.M)
_MATH = re.compile(r"\\boxed|\\frac|\\sqrt|\\sum|\\int|\\cdot|\$[^$\n]{1,80}\$|\\\(|\\\[|\b\d+\s*[-+*/^=]\s*\d+")
_LOGIC = re.compile(r"\b(?:knights?\s+and\s+knaves|truth[\s-]?tellers?|liars?\b|logic(?:al)?\s+puzzles?|riddle|sudoku|"
                    r"syllogism|premises?|valid\s+argument|deduc(?:e|tion)|if\s+and\s+only\s+if|seating\s+arrangement|"
                    r"who\s+(?:is|owns|lives)|zebra\s+puzzle|grid\s+puzzle)\b", re.I)
_SCIENCE = re.compile(r"\b(?:physics|chemistry|biology|molecules?|atoms?|electrons?|reaction|enzyme|protein|cells?|"
                      r"velocity|acceleration|momentum|energy|quantum|thermodynamic\w*|wavelength|photons?|genes?|"
                      r"dna|organism|ecosystem|circuit|voltage|newton|joule|kelvin|moles?\b|ph\b)", re.I)
_LETTER_ANS = re.compile(r"(?:\\boxed\{\s*(?:\\text\{)?\s*\(?([A-J])\)?\s*\}?\s*\}|answer\s*(?:is|:)\s*\**\s*\(?([A-J])\)?\b|"
                         r"^\s*\**\(?([A-J])\)?\**\s*$)", re.I | re.M)
_WS = re.compile(r"\s+")
_OPTION_PREFIX = re.compile(r"^\(?[a-e]\)\s*|(?<=.)\([a-e]\)$")
_ASSIGN_PREFIX = re.compile(r"^[a-z](?:_\{?\w+\}?)?=(?=.)")
_LETTER_ONLY = re.compile(r"\(?[a-j]\)?")


def _norm_ans(s: str) -> str:
    s = str(s).strip().strip("$").strip()
    for a, b in (("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\dfrac", "\\frac"),
                 ("\\tfrac", "\\frac"), ("^{\\circ}", ""), ("^\\circ", ""), ("\\%", ""), ("%", ""),
                 ("{,}", ""), ("~", "")):
        s = s.replace(a, b)
    for a in ("\\ ", "\\:", "\\;", "\\quad"):
        s = s.replace(a, " ")
    s = re.sub(r"\\(?:text|textbf|mathrm|mathbf)\{\s*([^}]*)\}", r"\1", s)
    s = _WS.sub("", s).rstrip(".").lower()
    if _LETTER_ONLY.fullmatch(s):
        return s.strip("()")                                               # "(c)" -> "c"
    s = _OPTION_PREFIX.sub("", s)                                          # "(a)26", "40(b)" -> number
    s = _ASSIGN_PREFIX.sub("", s)                                          # "n=4" -> "4"
    return s


def trace_and_answer(rec: dict) -> tuple[str | None, str]:
    """(trace, final answer). Handles R1 dumps stored in `reasoning` with no separate answer (DeepMath)."""
    reasoning, answer = rec.get("reasoning") or None, (rec.get("answer") or "").strip()
    if reasoning and not answer and "</think>" in reasoning:
        reasoning, answer = split_think(reasoning)
    elif reasoning and "</think>" in reasoning:
        reasoning = split_think(reasoning)[0]
    return reasoning, answer


def category(rec: dict, answer: str) -> str:
    q = rec.get("question") or ""
    task, dom = rec.get("task"), rec.get("domain")
    meta = rec.get("meta") or {}
    hint = str(meta.get("domain") or meta.get("category") or meta.get("problem_type") or "").lower()
    if task in ("code", "rl_code") or dom == "ai_code" or "code" in hint or _CODE.search(answer):
        return "code"
    if "logic" in hint or "puzzle" in hint or _LOGIC.search(q):
        return "logic"
    if task == "rl_math" or dom == "ai_math" or "math" in hint or _MATH.search(q):
        return "math"
    if hint in ("science", "stem", "biology", "chemistry", "physics") or _SCIENCE.search(q):
        return "science"
    return "general"


def _repetitive(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 20]
    return len(lines) >= 12 and len(set(lines)) / len(lines) < 0.5


def _check_answer(rec: dict, answer: str) -> bool | None:
    """True = matches the ground truth, False = contradicts it, None = cannot tell."""
    gt = rec.get("ground_truth")
    if not gt or len(str(gt)) > 60:          # long "ground truths" are full solutions, not checkable answers
        return None
    gt = str(gt).strip()
    if re.fullmatch(r"\(?[A-J]\)?", gt):      # multiple choice letter
        letters = [next(x for x in m.groups() if x) for m in _LETTER_ANS.finditer(answer[-1500:])]
        if not letters:
            return None
        return letters[-1].upper() == gt.strip("()").upper()
    boxed = last_boxed(answer)
    g = _norm_ans(gt)
    if boxed is None:                         # unboxed solutions: the answer must be stated at the end
        return True if g and g in _norm_ans(answer[-300:]) else None
    b = _norm_ans(boxed)
    if _LETTER_ONLY.fullmatch(b) and not _LETTER_ONLY.fullmatch(g):
        return None                           # boxed option letter vs. value ground truth: cannot tell
    return b == g


def reasoning_check(rec: dict, max_chars: int = DEFAULT_MAX_CHARS) -> dict:
    """{"ok": bool, "why": str, "category": str, "chars": int, "has_trace": bool, "verified": bool | None}"""
    q = (rec.get("question") or "").strip()
    trace, answer = trace_and_answer(rec)
    cat = category(rec, answer)
    out = {"ok": False, "why": "", "category": cat, "chars": len(q) + len(trace or "") + len(answer),
           "has_trace": bool(trace), "verified": None}

    def no(why):
        out["why"] = why
        return out

    if len(q) < 15:
        return no("short_question")
    if "<think>" in answer or "</think>" in answer:
        return no("unclosed_think")             # truncated trace: the answer never came
    if len(answer) < 20:
        return no("no_answer")
    if out["chars"] > max_chars:
        return no("too_long")
    if _REFUSAL.search(answer):
        return no("refusal")
    if trace:
        if len(trace) < MIN_TRACE_CHARS:
            return no("short_trace")
        if _repetitive(trace):
            return no("repetitive")
    else:                                     # worked solutions only
        if cat == "code":
            prose = _CODE.sub("", re.sub(r"```.*?```", "", answer, flags=re.S)).strip()
            if not _CODE.search(answer) or len(prose) < 80:
                return no("code_without_explanation")
        elif cat == "math":
            ends_with_answer = last_boxed(answer) is not None or _check_answer(rec, answer) is True
            if not ends_with_answer or answer.count("\n") < 2 or len(answer) < 200:
                return no("not_worked_solution")
        else:
            return no("no_trace")
    v = _check_answer(rec, answer)
    out["verified"] = v
    if v is False:
        return no("wrong_answer")
    out["ok"], out["why"] = True, "ok"
    return out


def to_reasoning_record(rec: dict, chk: dict) -> dict:
    """Normalise a record for the reasoning_sft bucket (trace/answer split, no learner level)."""
    trace, answer = trace_and_answer(rec)
    return {**rec, "reasoning": trace, "answer": answer, "bucket": "reasoning_sft", "level": "none",
            "topics": [f"reasoning_{chk['category']}"], "task": "chat",
            "reasoning_check": {k: chk[k] for k in ("category", "chars", "has_trace", "verified")}}
