"""Human feedback store (append-only JSONL) + export to training data.

Each feedback record:
{
  "id": uuid, "time": iso, "model": adapter-or-base path, "origin": "chat" | "eval_review",
  "level": "middle_school", "system": "...", "enable_thinking": false, "temperature": 0.7,
  "messages": [... conversation up to and including the user turn ...],
  "response": "model answer (without thinking)", "thinking": "...",
  "rating": 1..5, "verdict": "good" | "bad" | "neutral", "tags": ["wrong_level", ...],
  "comment": "...", "correction": "improved answer written by the reviewer" | null,
  "reference": "dataset reference answer (eval_review only)" | null
}

export():
  feedback_sft.jsonl  - chat records for SFT: the correction if given, else the response if rating >= 4 / good
  feedback_dpo.jsonl  - preference pairs {prompt, chosen, rejected}: correction (chosen) vs response (rejected)
                        (usable with TRL DPOTrainer / ORPO later)
"""
from __future__ import annotations

import json
import os
import time
import uuid
from collections import Counter

from .common import ROOT

FEEDBACK_DIR = os.path.join(ROOT, "data", "feedback")
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "feedback.jsonl")
TAGS = ["correct", "factually_wrong", "wrong_level", "too_long", "too_short", "unclear", "code_bug",
        "math_error", "off_topic", "unsafe", "great_example"]


def save_feedback(rec: dict, path: str = FEEDBACK_FILE) -> dict:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = {"id": uuid.uuid4().hex[:12], "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def load_feedback(path: str = FEEDBACK_FILE) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def stats(path: str = FEEDBACK_FILE) -> dict:
    fb = load_feedback(path)
    return {
        "total": len(fb),
        "verdicts": dict(Counter(r.get("verdict") for r in fb)),
        "mean_rating": round(sum(r.get("rating") or 0 for r in fb) / len(fb), 2) if fb else None,
        "with_correction": sum(1 for r in fb if (r.get("correction") or "").strip()),
        "levels": dict(Counter(r.get("level") for r in fb)),
        "tags": dict(Counter(t for r in fb for t in r.get("tags") or [])),
        "models": dict(Counter(r.get("model") for r in fb)),
    }


def export(path: str = FEEDBACK_FILE, out_dir: str = FEEDBACK_DIR, min_rating: int = 4) -> dict:
    fb = load_feedback(path)
    sft, dpo = [], []
    for r in fb:
        prompt = [m for m in r["messages"]]
        if r.get("system") and not any(m["role"] == "system" for m in prompt):
            prompt = [{"role": "system", "content": r["system"]}] + prompt
        corr = (r.get("correction") or "").strip()
        resp = (r.get("response") or "").strip()
        good = r.get("verdict") == "good" or (r.get("rating") or 0) >= min_rating
        target = corr or (resp if good else "")
        if target:
            sft.append({"id": f"feedback/{r['id']}", "messages": prompt + [{"role": "assistant", "content": target}],
                        "level": r.get("level"), "source": "feedback", "topics": [], "has_reasoning": False})
        if corr and resp and corr != resp:
            dpo.append({"id": f"feedback/{r['id']}", "prompt": prompt,
                        "chosen": [{"role": "assistant", "content": corr}],
                        "rejected": [{"role": "assistant", "content": resp}],
                        "level": r.get("level"), "tags": r.get("tags")})
    os.makedirs(out_dir, exist_ok=True)
    for name, rows in (("feedback_sft.jsonl", sft), ("feedback_dpo.jsonl", dpo)):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"feedback": len(fb), "sft": len(sft), "dpo_pairs": len(dpo),
            "sft_file": os.path.join(out_dir, "feedback_sft.jsonl"),
            "dpo_file": os.path.join(out_dir, "feedback_dpo.jsonl")}
