"""Learner-level inference.

level score = 3.0 * metadata prior          (cosmopedia `audience`, school grade, source default)
            + 1.0 * readability fit          (Flesch-Kincaid grade of the answer vs level range)
            + 1.0 * technicality fit         (LaTeX / code / jargon density)
            + 1.0 * question-phrasing cues   ("explain like I'm 5", "for my biology thesis", ...)
            + 1.5 * zero-shot probability    (optional, single-label over LEVELS[*]["zs"])

The weights are deliberately simple; metadata dominates when it exists, the zero-shot
model breaks ties when it does not.
"""
from __future__ import annotations

import re

from .taxonomy import LEVELS, LEVEL_NAMES

_VOWELS = re.compile(r"[aeiouy]+")
_SENT_SPLIT = re.compile(r"[.!?]+(?:\s|$)")
_WORD = re.compile(r"[A-Za-z]+")
_LATEX = re.compile(r"\\(?:frac|sum|int|partial|nabla|mathbb|mathbf|theta|sigma|lambda|alpha|beta|log|exp|cdot|top|left|right)|\$[^$]{3,}\$")
_CODE = re.compile(r"```|^\s{4}\S|\bdef \w+\(|\bimport \w+|\w+\.\w+\(")
_JARGON = re.compile(
    r"\b(?:gradient|backprop\w*|stochastic|convex|eigen\w*|regulari[sz]\w*|hyperparameter\w*|"
    r"variational|posterior|likelihood|entropy|tensor\w*|embedding\w*|attention|transformer\w*|"
    r"convolution\w*|logits?|softmax|optimi[sz]er|ablation|architecture|latent|inference)\b", re.I)

QUESTION_CUES: dict[str, list[str]] = {
    "primary_school": [r"explain (?:it )?like i'?m (?:5|five)", r"\beli5\b", r"for (?:a |my )?(?:kid|child|children|\d-year-old)s?\b", r"\bkids?\b"],
    "middle_school": [r"middle school", r"\b(?:6|7|8)th grade", r"\bteen(?:ager)?s?\b"],
    "high_school": [r"high school", r"\b(?:9|10|11|12)th grade", r"\bAP (?:computer science|statistics|calculus)\b", r"homework"],
    "undergrad_non_ai": [r"\b(?:my|our) (?:course|class|professor|lecture)\b", r"\bundergrad\w*", r"\bintro(?:ductory)? (?:course|class)\b"],
    "ai_student": [r"\b(?:cs|computer science|ml|machine learning) (?:course|class|student|major)\b", r"\bderive\b", r"\bimplement\w* from scratch\b", r"\binterview\b"],
    "general_public": [r"\bwill (?:ai|robots?) (?:take|replace|steal)\b", r"\bis (?:ai|chatgpt) (?:dangerous|safe|conscious|alive)\b", r"\bin (?:simple|plain|layman'?s?) (?:terms|words|english)\b", r"\bmy (?:job|kids|parents|phone)\b"],
    "researcher_non_ai": [r"\bin my (?:biology|chemistry|physics|medical|clinical|economics|social science|psychology|geology|ecology|astronomy|lab|research|thesis|phd)\b", r"\b(?:my|our) (?:experiment|samples|patients|survey|field data)\b", r"\bapply(?:ing)? (?:machine learning|ml|ai|deep learning) to\b"],
    "ai_researcher": [r"\bstate[- ]of[- ]the[- ]art\b", r"\bablation\b", r"\bconvergence (?:rate|proof|guarantee)\b", r"\bpaper\b.*\b(?:propose|method)\b", r"\bneurips|icml|iclr|arxiv\b"],
}
_CUE_RE = {k: [re.compile(p, re.I) for p in v] for k, v in QUESTION_CUES.items()}

# Map cosmopedia `audience` values (typos included, they are in the data) -> level prior.
COSMOPEDIA_AUDIENCE = {
    "young_children": {"primary_school": 1.0},
    "children": {"primary_school": 1.0},
    "grade_school_students": {"primary_school": 0.7, "middle_school": 0.3},
    "middle_school_students": {"middle_school": 1.0},
    "high_school_studnets": {"high_school": 1.0},
    "high_school_students": {"high_school": 1.0},
    "college_students": {"undergrad_non_ai": 0.5, "ai_student": 0.5},
    "college_studnets": {"undergrad_non_ai": 0.5, "ai_student": 0.5},
    "researchers": {"ai_researcher": 0.6, "researcher_non_ai": 0.4},
    "general": {"general_public": 1.0},
    "requires_details": {"ai_student": 0.5, "general_public": 0.5},
}


def _syllables(word: str) -> int:
    w = word.lower()
    n = len(_VOWELS.findall(w))
    if w.endswith("e") and n > 1 and not w.endswith("le"):
        n -= 1
    return max(1, n)


def fk_grade(text: str) -> float | None:
    """Flesch-Kincaid grade level (None if text too short)."""
    words = _WORD.findall(text[:6000])
    if len(words) < 30:
        return None
    sents = max(1, len(_SENT_SPLIT.findall(text[:6000])))
    syl = sum(_syllables(w) for w in words)
    return 0.39 * (len(words) / sents) + 11.8 * (syl / len(words)) - 15.59


def technicality(text: str) -> dict[str, float]:
    """Densities in 0..1: math (LaTeX/formulas), code, jargon."""
    if not text:
        return {"math": 0.0, "code": 0.0, "jargon": 0.0}
    n = max(50, len(text.split()))
    return {
        "math": min(1.0, len(_LATEX.findall(text)) / n * 20),
        "code": min(1.0, len(_CODE.findall(text)) / n * 10),
        "jargon": min(1.0, len(_JARGON.findall(text)) / n * 15),
    }


def _range_fit(x: float, lo: float, hi: float, slack: float = 2.0) -> float:
    if lo <= x <= hi:
        return 1.0
    d = lo - x if x < lo else x - hi
    return max(0.0, 1.0 - d / slack)


# How much each level likes math / code / jargon density (negative = prefers plain prose).
# Code-heavy debugging questions are typical for AI students/practitioners, formula-heavy text for researchers.
_TECH_PREF = {
    #                    math   code  jargon
    "primary_school":    (-1.0, -0.8, -1.0),
    "middle_school":     (-0.7, -0.4, -0.7),
    "high_school":       (-0.2,  0.0, -0.3),
    "undergrad_non_ai":  ( 0.0, -0.2,  0.0),
    "ai_student":        ( 0.5,  0.9,  0.5),
    "general_public":    (-0.8, -0.8, -0.8),
    "researcher_non_ai": ( 0.3,  0.3,  0.2),
    "ai_researcher":     ( 1.0,  0.2,  0.8),
}


def infer_level(question: str, answer: str | None, prior: dict[str, float] | None = None,
                zs_probs: dict[str, float] | None = None) -> tuple[str, dict[str, float]]:
    scores = {k: 0.0 for k in LEVEL_NAMES}
    if prior:
        for k, v in prior.items():
            if k in scores:
                scores[k] += 3.0 * v

    body = answer or question
    fk = fk_grade(body)
    if fk is not None:
        for k in LEVEL_NAMES:
            lo, hi = LEVELS[k]["fk"]
            scores[k] += 1.0 * _range_fit(fk, lo, hi)

    tech = technicality((question or "") + "\n" + (answer or ""))
    plain = 1.0 - max(tech.values())
    for k in LEVEL_NAMES:
        for dim, pref in zip(("math", "code", "jargon"), _TECH_PREF[k]):
            scores[k] += pref * tech[dim]
        if min(_TECH_PREF[k]) < 0 and max(_TECH_PREF[k]) <= 0:  # plain-language levels get a bonus for plain text
            scores[k] += 0.5 * plain

    for k, rxs in _CUE_RE.items():
        if any(r.search(question or "") for r in rxs):
            scores[k] += 1.0

    if zs_probs:
        for k, v in zs_probs.items():
            if k in scores:
                scores[k] += 1.5 * v

    best = max(scores, key=scores.get)
    return best, {k: round(v, 3) for k, v in scores.items()}
