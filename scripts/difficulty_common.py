"""Shared definitions for the 5-level reading-difficulty classifiers.

Levels (index 1..5):
  1 very_easy       primary school
  2 easy            middle school
  3 medium          high school
  4 difficult       university student
  5 very_difficult  researcher / expert

Two cheap measures live here: zero-shot NLI hypotheses (one set of 5 per template) and length features
(sentence length, word length, Flesch-Kincaid grade, document length) with fixed thresholds -> level.
The rubric string is the one shown to the LLM judges (Qwen3-1.7B thinking, Claude) so all judges use the same scale.
"""
from __future__ import annotations

import re

LEVELS = [
    (1, "very_easy", "Very Easy", "primary school (ages 6-11)"),
    (2, "easy", "Easy", "middle school (ages 11-14)"),
    (3, "medium", "Medium", "high school (ages 14-18)"),
    (4, "difficult", "Difficult", "university student"),
    (5, "very_difficult", "Very Difficult", "researcher or domain expert"),
]
LEVEL_KEYS = [k for _, k, _, _ in LEVELS]
LEVEL_TITLE = {i: t for i, _, t, _ in LEVELS}
LEVEL_INDEX = {k: i for i, k, _, _ in LEVELS}

RUBRIC = """Rate how difficult the text is to understand, i.e. the education level a reader needs to follow it comfortably
(vocabulary, sentence complexity, and the background knowledge the text assumes). Do not rate quality or usefulness.

1 = Very Easy (primary school, ages 6-11): simple everyday words, short sentences, no technical terms; a 10-year-old
    could follow it (children's stories, simple product blurbs, basic how-to in plain words).
2 = Easy (middle school, ages 11-14): ordinary general-audience writing: news, lifestyle, sports, shopping, forum chat,
    simple consumer-tech or health tips, basic explanations that define everything they use.
3 = Medium (high school, ages 14-18): assumes a general education; longer arguments, some specialised vocabulary that is
    explained or widely known (popular science, business and finance news, beginner programming tutorials, legal or
    medical information written for laypeople, opinion essays).
4 = Difficult (university student): assumes coursework in a field; technical terms used without explanation, formulas,
    non-trivial code, academic prose, professional documentation (API references, engineering specs, practitioner
    write-ups, textbook-level treatments).
5 = Very Difficult (researcher or expert): research papers and abstracts, dense mathematical derivations or proofs,
    advanced specialist discussion that assumes graduate-level background in the field.

Rules: judge the main body and ignore navigation menus, cookie notices and other boilerplate. A page that is only links,
keywords, listings or spam has no real content: rate its vocabulary (usually 2). Code: beginner tutorial = 3, library or
production code and non-trivial algorithms = 4, research-level implementations = 5. Question-answer pairs: rate the level
needed to understand the answer. Non-English text: rate the apparent content level if you can read it, otherwise 2."""

# Zero-shot NLI hypotheses, one per level, several phrasings to compare.
ZS_TEMPLATES: dict[str, list[str]] = {
    "audience": [
        "This text is written in very simple words for young children in primary school.",
        "This text is written for middle school students aged 11 to 14.",
        "This text is written for high school students.",
        "This text is written for university students and assumes specialised knowledge.",
        "This text is written for expert researchers and assumes advanced technical knowledge.",
    ],
    "understand": [
        "A primary school child could easily understand this text.",
        "A middle school student could understand this text.",
        "A high school student could understand this text.",
        "Only a university student in the field could understand this text.",
        "Only an expert researcher could understand this text.",
    ],
    "level": [
        "This text is very easy to read, at an elementary school level.",
        "This text is easy to read, at a middle school level.",
        "This text is moderately difficult to read, at a high school level.",
        "This text is difficult to read, at a university level.",
        "This text is very difficult to read, at an expert or research level.",
    ],
}
DEFAULT_ZS_TEMPLATE = "audience"   # best template on the 1k comparison (compare_difficulty.py)
ZS_TEMPERATURE = 2.0               # softmax(log-odds / T) over the 5 hypotheses before taking the expected level

# Fitted on the 1k Claude-labelled sample (compare_difficulty.py, 2026-11-02). Quantile-matched thresholds on the
# expected level of the default template (level = 1 + number of thresholds reached) and a ridge combo of the five
# entailment log-odds with length features (n_tokens enters as log1p). CV against Claude: zero-shot calibrated
# QWK 0.55 / exact 0.54 / within-one 0.97; combo QWK 0.58 / exact 0.56 / within-one 0.97.
ZS_CAL_THRESHOLDS = {"audience": (1.738, 3.369, 4.614, 4.879)}
COMBO = {"features": ["audience_lo1", "audience_lo2", "audience_lo3", "audience_lo4", "audience_lo5", "fk_grade", "long_word_ratio", "avg_word_len", "avg_sent_len", "n_tokens"], "coef": [-0.010324, -0.014545, 0.004726, -0.014462, 0.150551, 0.068718, 0.696315, -0.135243, -0.020788, 0.067264], "intercept": 2.583003, "thresholds": [1.7827, 2.6075, 3.3854, 3.7121], "log1p": ["n_tokens"]}
TOKENS_PER_WORD = 1.261            # median Qwen3 tokens per regex word on the sample; n_tokens fallback when no token column exists


def combo_score(row: dict) -> float:
    """difficulty_score of COMBO for a row (dict of feature -> value); missing features count as 0."""
    import math
    s = COMBO["intercept"]
    for f, c in zip(COMBO["features"], COMBO["coef"]):
        v = float(row.get(f) or 0.0)
        if f in COMBO["log1p"]:
            v = math.log1p(max(v, 0.0))
        s += c * v
    return s

# ----------------------------------------------------------------------------------------------------------------
# Length features
# ----------------------------------------------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_VOWELS = re.compile(r"[aeiouy]+")

LENGTH_FEATURES = ["n_chars", "n_words", "n_sents", "avg_sent_len", "max_sent_len", "p90_sent_len",
                   "avg_word_len", "long_word_ratio", "fk_grade"]


def _syllables(word: str) -> int:
    w = word.lower()
    n = len(_VOWELS.findall(w))
    if w.endswith("e") and n > 1 and not w.endswith("le"):
        n -= 1
    return max(1, n)


def length_features(text: str, max_chars: int | None = None) -> dict[str, float]:
    """Sentence-length / word-length / readability statistics of a document (regex only, no model)."""
    t = text or ""
    if max_chars:
        t = t[:max_chars]
    sents = [s for s in _SENT_SPLIT.split(t) if s and _WORD.search(s)]
    wps = sorted(len(_WORD.findall(s)) for s in sents)
    words = _WORD.findall(t)
    nw, ns = len(words), max(1, len(wps))
    if nw == 0:
        return {k: 0.0 for k in LENGTH_FEATURES} | {"n_chars": float(len(t))}
    syl = [_syllables(w) for w in words]
    return {
        "n_chars": float(len(t)),
        "n_words": float(nw),
        "n_sents": float(len(wps)),
        "avg_sent_len": nw / ns,
        "max_sent_len": float(wps[-1]),
        "p90_sent_len": float(wps[int(0.9 * (len(wps) - 1))]),
        "avg_word_len": sum(len(w) for w in words) / nw,
        "long_word_ratio": sum(s >= 3 for s in syl) / nw,
        "fk_grade": 0.39 * (nw / ns) + 11.8 * (sum(syl) / nw) - 15.59,
    }


# Fixed thresholds -> level 1..5 (value >= threshold moves one level up). fk_grade follows the usual US-grade reading
# of Flesch-Kincaid (<6 primary, 6-9 middle, 9-13 high school, 13-17 university, >=17 graduate/expert); the sentence
# and document-length thresholds are plain quantile guesses that compare_difficulty.py can re-fit on labelled data.
LENGTH_THRESHOLDS: dict[str, tuple[float, float, float, float]] = {
    "fk_grade": (6.0, 9.0, 13.0, 17.0),
    "avg_sent_len": (11.0, 15.0, 20.0, 26.0),
    "max_sent_len": (25.0, 40.0, 60.0, 90.0),
    "n_tokens": (150.0, 400.0, 1000.0, 2500.0),
}


def level_from_thresholds(x: float, thresholds) -> int:
    return 1 + sum(x >= t for t in thresholds)


def head(text: str, n_chars: int) -> str:
    """Whitespace-collapsed first n_chars of a document: the common input of all model judges."""
    return " ".join((text or "").split())[:n_chars]


def parse_level(answer: str) -> int | None:
    """Level from a judge's answer: 'Difficulty: N', else the last standalone digit 1-5, else a level name."""
    m = list(re.finditer(r"[Dd]ifficulty\s*[:=]?\s*\**\s*\(?([1-5])\b", answer))
    if m:
        return int(m[-1].group(1))
    m = list(re.finditer(r"(?<![\d.])([1-5])(?![\d.])", answer))
    if m:
        return int(m[-1].group(1))
    low = answer.lower()
    for i in (5, 1, 4, 3, 2):                     # 'very difficult' / 'very easy' before 'difficult' / 'easy'
        if LEVEL_TITLE[i].lower() in low:
            return i
    return None
