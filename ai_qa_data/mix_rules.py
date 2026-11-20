"""Rules used by build_training_mix.py (v0.2+): level-fit relabelling, reasoning-trace handling, RL routing.

Motivation (v0.1 evaluation, see docs/AI_QA_DATASET_GUIDE.md section 5):
  * reference answers labelled primary_school had Flesch-Kincaid grade ~10.9 (target <= 5.5) and 450 words;
    SFT on them made the model ignore the level system prompt (open-QA level fit 0.60 -> 0.47).
  * only ~600 AI examples carried reasoning traces.
"""
from __future__ import annotations

import re

from .levels import fk_grade, technicality
from .sources import split_think
from .taxonomy import LEVELS

# Increasing complexity. general_public is a plain-language branch next to high_school.
PLAIN_LADDER = ["primary_school", "middle_school", "high_school", "undergrad_non_ai", "ai_student"]
PUBLIC_LADDER = ["general_public", "undergrad_non_ai", "ai_student"]
PLAIN_LEVELS = {"primary_school", "middle_school", "high_school", "general_public"}
TECH_LEVELS = {"undergrad_non_ai", "ai_student", "researcher_non_ai", "ai_researcher"}

# Answers longer than this are not appropriate for the level (words)
MAX_WORDS = {"primary_school": 220, "middle_school": 350, "high_school": 500, "general_public": 450}
FK_SLACK = 1.5

AUDIENCE_PHRASE = {
    "primary_school": "young children", "middle_school": "middle school students",
    "high_school": "high school students", "undergrad_non_ai": "college students",
    "ai_student": "computer science students", "ai_researcher": "researchers",
    "general_public": "a general audience", "researcher_non_ai": "scientists from other fields",
}
_AUD_RE = re.compile(r" for (?:young children|children|middle school students|high school students|college students|"
                     r"researchers|a general audience|computer science students|scientists from other fields)\.\s*$")


def fits(level: str, answer: str) -> tuple[bool, float | None, int]:
    """Does the answer's reading grade / length suit the level?"""
    words = len(answer.split())
    fk = fk_grade(answer)
    lo, hi = LEVELS[level]["fk"]
    if words > MAX_WORDS.get(level, 10 ** 9):
        return False, fk, words
    if fk is None:                       # very short answers: length rule only
        return True, fk, words
    return (lo - FK_SLACK) <= fk <= (hi + FK_SLACK), fk, words


def fix_level(r: dict, mode: str = "relabel") -> tuple[str | None, str]:
    """Return (new_level or None to drop, reason). Relabels upward on the ladder for plain levels whose
    answer is too hard/long; moves 'ai_researcher' answers that read like plain prose down to ai_student."""
    level, answer = r["level"], r.get("answer") or ""
    ok, fk, words = fits(level, answer)
    if ok:
        return level, "fits"
    if level in PLAIN_LEVELS:
        if mode == "drop":
            return None, "drop_misfit"
        ladder = PUBLIC_LADDER if level == "general_public" else PLAIN_LADDER
        for cand in ladder[ladder.index(level) + 1:]:
            if fits(cand, answer)[0]:
                return cand, f"relabel_up:{level}->{cand}"
        tech = technicality(answer)
        return ("ai_student" if max(tech["code"], tech["math"]) > 0.15 else "undergrad_non_ai"), f"relabel_up:{level}->fallback"
    if level == "ai_researcher" and fk is not None and fk < 12 and max(technicality(answer).values()) < 0.15:
        return "ai_student", "relabel_down:ai_researcher->ai_student"
    return level, "kept_tech_level"          # technical levels have wide ranges; keep


def rewrite_audience(question: str, level: str) -> str:
    """Cosmopedia-derived questions name their audience ('... for young children.'); keep it consistent."""
    if _AUD_RE.search(question):
        return _AUD_RE.sub(f" for {AUDIENCE_PHRASE[level]}.", question)
    return question


def reasoning_ok(r: dict, max_chars: int) -> bool:
    """Reasoning examples must fit the training context (~2048 tokens ~= 6-7k characters incl. answer)."""
    return len(r.get("reasoning") or "") + len(r.get("answer") or "") + len(r.get("question") or "") <= max_chars


def rl_math_to_reasoning_sft(r: dict) -> dict | None:
    """DeepMath-style records keep the full R1 output in `reasoning` ('...thinking</think> final solution')."""
    raw = r.get("reasoning") or ""
    think, final = split_think(raw)
    if not think or not final:
        return None
    return {**r, "reasoning": think, "answer": final, "task": "chat", "bucket": "ai_sft",
            "source": f"{r['source']}_r1", "id": r["id"] + "#r1"}


_LETTER_ONLY = re.compile(r"^[A-J]$")


def is_mcq(r: dict) -> bool:
    """MCQ with a known answer letter (RL-MCQ needs the letter as its reward target)."""
    return bool(_LETTER_ONLY.match(str(r.get("ground_truth") or "").strip()))
