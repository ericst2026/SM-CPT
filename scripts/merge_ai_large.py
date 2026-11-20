#!/usr/bin/env python3
"""Copy ai_large scores from separate evaluations (--mmlu-subjects ai_large) into the eval results of runs that were
evaluated before the subject existed, so their reports and history tables show AI_Large next to the MMLU scores.

    python scripts/merge_ai_large.py            # the pairs in PAIRS whose source has finished
    python scripts/merge_ai_large.py --dry-run

Per pair (source tag -> target tag in eval_results/):
    summary.json            tasks.mmlu gets ai_large_accuracy / ai_large_n / ai_large_by_topic (+ ai_large_source)
    mmlu_predictions.jsonl  the source's ai_large predictions are appended (earlier ai_large rows replaced)
    summary.md              an "ai_large" section is appended (replaced on a re-run)
    report.html             rebuilt, plus the eval_results index
Run histories (runs/<run>/eval_history.json or mmlu_history.json): the entry with the target tag gets the ai_large
keys, epoch 0 gets the base model's; runs/<run>/summary.md's score table is redrawn.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from cpt_trainer.mmlu import history_table  # noqa: E402

RESULTS = os.path.join(ROOT, "eval_results")
BASE_5SHOT = "base06_ai_large_5shot"
PAIRS = [  # (source with the ai_large scores, target eval_results tag)
    (BASE_5SHOT, "cpt_qwen06b_r64_300mb_epoch0"),
    ("cpt_qwen06b_r64_300mb_epoch1_ai_large_5shot", "cpt_qwen06b_r64_300mb_epoch1"),
    ("cpt_qwen06b_rs_adaptive_q4_100m_epoch0.5_ai_large_5shot", "cpt_qwen06b_rs_adaptive_q4_100m_epoch0.5"),
    ("cpt_qwen06b_rs_adaptive_q4_100m_epoch1_ai_large_5shot", "cpt_qwen06b_rs_adaptive_q4_100m_epoch1"),
    ("chat06b_ai_large_gen", "chat06b_mmlu_gen"),
    ("cpt_qwen06b_r64_300mb_epoch1_onchat_ai_large_gen", "cpt_qwen06b_r64_300mb_epoch1_onchat_gen"),
    ("cpt_qwen06b_rs_adaptive_q4_100m_epoch1_onchat_ai_large_gen", "cpt_qwen06b_rs_adaptive_q4_100m_epoch1_onchat_gen"),
    ("cpt_qwen06b_rs_adaptive_long_epoch0.5_ai_large_5shot", "cpt_qwen06b_rs_adaptive_long_epoch0.5"),
    ("cpt_qwen06b_rs_adaptive_long_epoch1_ai_large_5shot", "cpt_qwen06b_rs_adaptive_long_epoch1"),
    ("cpt_qwen06b_rs_adaptive_long_epoch1_onchat_ai_large_gen", "cpt_qwen06b_rs_adaptive_long_epoch1_onchat_gen"),
    *[(f"cpt_qwen06b_r64_epoch{e}_ai_large_5shot", f"cpt_qwen06b_r64_epoch{e}") for e in (1, 2, 3, 4)],
    *[(f"cpt_qwen06b_r128_full_epoch{e}_ai_large_5shot", f"cpt_qwen06b_r128_full_epoch{e}") for e in (1, 2)],
    ("cpt_qwen06b_rs128_long_epoch0.5_ai_large_5shot", "cpt_qwen06b_rs128_long_epoch0.5"),
    ("cpt_qwen06b_ml_rs_adaptive_v1_epoch0.5_ai_large_5shot", "cpt_qwen06b_ml_rs_adaptive_v1_epoch0.5"),
    ("base17_ai_large_5shot", "mmlu_17base_5shot"),
]
RUNS = {  # run dir -> history file
    "cpt_qwen06b_r64_300mb": "mmlu_history.json",
    "cpt_qwen06b_rs_adaptive_q4_100m": "eval_history.json",
    "cpt_qwen06b_rs_adaptive_long": "eval_history.json",
    "cpt_qwen06b_r64": "mmlu_history.json",
    "cpt_qwen06b_r128_full": "eval_history.json",
    "cpt_qwen06b_rs128_long": "eval_history.json",
    "cpt_qwen06b_ml_rs_adaptive_v1": "eval_history.json",
}
KEYS = ("ai_large_accuracy", "ai_large_n", "ai_large_by_topic")
MD_MARK = "\n## ai_large (merged from "


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def source_scores(tag: str) -> dict | None:
    path = os.path.join(RESULTS, tag, "summary.json")
    if not os.path.exists(path):
        return None
    mm = load(path)["tasks"]["mmlu"]
    return {k: mm[k] for k in KEYS if k in mm} or None


def merge_result(src: str, dst: str, scores: dict, dry: bool) -> bool:
    out = os.path.join(RESULTS, dst)
    if not os.path.exists(os.path.join(out, "summary.json")):
        print(f"  ! {dst}: no summary.json, skipped")
        return False
    summary = load(os.path.join(out, "summary.json"))
    src_mm = load(os.path.join(RESULTS, src, "summary.json"))["tasks"]["mmlu"]
    dst_mm = summary["tasks"]["mmlu"]
    same = (src_mm.get("scoring"), src_mm.get("shots"), src_mm.get("format")) == \
           (dst_mm.get("scoring", "loglik"), dst_mm.get("shots"), dst_mm.get("format"))
    if not same:
        print(f"  ! {dst}: scoring differs from {src} ({dst_mm.get('scoring')}/{dst_mm.get('shots')}-shot/"
              f"{dst_mm.get('format')} vs {src_mm.get('scoring')}/{src_mm.get('shots')}-shot/{src_mm.get('format')}), skipped")
        return False
    print(f"  {dst}: ai_large {scores['ai_large_accuracy']:.4f} (n={scores['ai_large_n']}) from {src}")
    if dry:
        return True
    dst_mm.update(scores, ai_large_source=src)
    save(os.path.join(out, "summary.json"), summary)

    pred_path = os.path.join(out, "mmlu_predictions.jsonl")
    if os.path.exists(pred_path):
        with open(pred_path, encoding="utf-8") as f:
            rows = [l for l in f if l.strip() and json.loads(l).get("subject") != "ai_large"]
        with open(os.path.join(RESULTS, src, "mmlu_predictions.jsonl"), encoding="utf-8") as f:
            rows += [l for l in f if l.strip() and json.loads(l).get("subject") == "ai_large"]
        with open(pred_path, "w", encoding="utf-8") as f:
            f.writelines(l if l.endswith("\n") else l + "\n" for l in rows)

    md_path = os.path.join(out, "summary.md")
    if os.path.exists(md_path):
        with open(md_path, encoding="utf-8") as f:
            md = f.read().split(MD_MARK)[0].rstrip("\n") + "\n"
        md += (f"{MD_MARK}{src}, not part of the MMLU accuracy)\n\n"
               f"accuracy {scores['ai_large_accuracy']:.4f} (n={scores['ai_large_n']})\n\n| topic | n | accuracy |\n|---|---:|---:|\n"
               + "".join(f"| {k} | {v['n']} | {v['acc']:.4f} |\n" for k, v in scores["ai_large_by_topic"].items()))
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)

    from ai_qa_train.report import build_report
    build_report(out, summary, None)
    return True


def merge_histories(done: dict[str, dict], dry: bool):
    """done: target tag -> ai_large scores (as eval_model's summary has them)."""
    base = done.get("_base")
    for run, fname in RUNS.items():
        path = os.path.join(ROOT, "runs", run, fname)
        if not os.path.exists(path):
            continue
        history = load(path)
        changed = []
        for h in history:
            sc = base if h.get("epoch") == 0 else done.get(h.get("tag"))
            if sc:
                h.update({"ai_large_accuracy": sc["ai_large_accuracy"], "ai_large_n": sc["ai_large_n"],
                          "ai_large_by_topic": {k: v["acc"] for k, v in sc["ai_large_by_topic"].items()}})
                changed.append(h.get("epoch"))
        print(f"  runs/{run}/{fname}: ai_large added at epochs {changed}")
        if dry or not changed:
            continue
        save(path, history)
        md_path = os.path.join(ROOT, "runs", run, "summary.md")
        if os.path.exists(md_path):  # redraw the score table (val-loss columns from the history itself)
            val = sorted({c for h in history for c in (h.get("val_loss") or {})})
            with open(md_path, encoding="utf-8") as f:
                md = f.read()
            md = re.sub(r"(## Scores per epoch\n\n)(\|.*\|\n)+", lambda m: m.group(1) + history_table(history, val) + "\n", md)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    done = {}
    for src, dst in PAIRS:
        scores = source_scores(src)
        if scores is None:
            print(f"  - {src}: not evaluated yet")
            continue
        if merge_result(src, dst, scores, args.dry_run):
            done[dst] = scores
        if src == BASE_5SHOT:
            done["_base"] = scores
    merge_histories(done, args.dry_run)
    if not args.dry_run:
        from ai_qa_train.report import build_index
        build_index(RESULTS)


if __name__ == "__main__":
    main()
