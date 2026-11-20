"""Task relevance scoring for candidate documents (title + abstract, or the start of the text).

Scorers:
  * builtin:ai       - the weighted AI lexicon of the data pipeline (ai_qa_data/keywords.py):
                       core + code + 0.5 * math
  * builtin:ai_core  - the same lexicon without the math vocabulary (core + code); for excluding AI
                       documents from a general corpus without also dropping ordinary math pages
  * terms            - any task: a list of {pattern, weight} regexes in the spec (case-insensitive)

Modes:
  * include (default) - a document passes when score >= threshold (build a corpus ABOUT the task)
  * exclude           - a document passes when score <  threshold (build a corpus of everything BUT the
                        task, e.g. replay text for continued pre-training)

Curated items (books) skip the check.
"""
from __future__ import annotations

import os
import re
import sys


class Relevance:
    def __init__(self, cfg: dict | None):
        cfg = cfg or {}
        self.scorer = cfg.get("scorer", "terms")
        self.mode = cfg.get("mode", "include")
        if self.mode not in ("include", "exclude"):
            raise ValueError(f"relevance.mode must be include or exclude, not {self.mode!r}")
        self.threshold = float(cfg.get("threshold", 3.0))
        self.terms = [(re.compile(t["pattern"], re.I), float(t.get("weight", 1.0))) for t in cfg.get("terms", [])]
        if self.scorer in ("builtin:ai", "builtin:ai_core"):
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from ai_qa_data.keywords import score_text
            self._score_ai = score_text
        elif self.scorer != "terms":
            raise ValueError(f"unknown relevance scorer {self.scorer!r}")

    def score(self, text: str) -> float:
        text = (text or "")[:6000]
        if self.scorer == "builtin:ai":
            kw = self._score_ai(text)
            return round(kw.core + kw.code + 0.5 * kw.math, 2)
        if self.scorer == "builtin:ai_core":
            kw = self._score_ai(text)
            return round(kw.core + kw.code, 2)
        s = 0.0
        for rx, w in self.terms:
            n = len(rx.findall(text))
            if n:
                s += w * (1 + 0.25 * min(n - 1, 4))
        return round(s, 2)

    def ok(self, text: str, threshold: float | None = None) -> tuple[bool, float]:
        s = self.score(text)
        hit = s >= (self.threshold if threshold is None else threshold)
        return (not hit if self.mode == "exclude" else hit), s
