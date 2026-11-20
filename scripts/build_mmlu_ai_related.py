#!/usr/bin/env python3
"""Build the MMLU "ai_related" subject: machine_learning in full, plus the questions of the math, computer and
formal-logic subjects that the zero-shot NLI model judges to be about AI's neighbouring areas.

    python scripts/build_mmlu_ai_related.py                    # classify the test split (GPU, under a minute)
    python scripts/build_mmlu_ai_related.py --threshold 0.5    # re-select from the cached scores (no model)
    python scripts/build_mmlu_ai_related.py --reclassify       # score all questions again
    python scripts/build_mmlu_ai_related.py --limit-per-subject 5 --out /tmp/x   # smoke test

Subject groups (GROUPS); a question is taken when the best score among its group's hypotheses >= --threshold:
    ml        machine_learning                                            every question, no classifier
    math      abstract_algebra, *_mathematics, high_school_statistics     hypothesis `math`
    computer  college/high_school_computer_science, computer_security     best of `code`, `cs`
    logic     formal_logic                                                hypothesis `overall`
Questions of any other subject are never selected.

Stage 1  every question of these subjects is scored against each of HYPOTHESES with
         models/deberta-v3-base-zeroshot-v2.0 (ai_qa_data/zeroshot.py, multi-label: independent P(entailment)).
         The premise is the question text followed by the correct answer text ("Answer: ..."); the wrong
         options and the subject name are not shown to the classifier.
         Scores are cached in scores.jsonl and reused while model, hypotheses, groups, premise and split are
         unchanged.
Stage 2  selection by group as above.

Output (--out, default data/eval/mmlu_ai_related/)
    scores.jsonl     every scored question: id, subject, group, per-hypothesis scores, and `score` / `sel_hyp`
                     (the group's best hypothesis, used for selection; null for machine_learning)
    selection.json   ids of the new subject + settings; scripts/eval_model.py reads it
    report.md        size per subject at several thresholds, borderline examples to check by hand,
                     and the ai_related accuracy of every comparable earlier run in eval_results/ (5-shot, base
                     prompt format, *-Base checkpoint alone or with an adapter; from its saved
                     mmlu_predictions.jsonl, no model needed)

In scripts/eval_model.py (once selection.json exists):
    every MMLU run also reports ai_related_accuracy / ai_related_n
    --mmlu-subjects ai_related   evaluates only these questions; each keeps its own subject's prompt header and
                                 dev shots, so the score equals the one inside a full MMLU run
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_model import (MMLU_AI_SUBJECTS, MMLU_DIR, RESULTS_ROOT, ROOT,  # noqa: E402
                        list_mmlu_subjects, load_mmlu_split)

OUT_DIR = os.path.join(ROOT, "data", "eval", "mmlu_ai_related")
ZS_MODEL = os.path.join(ROOT, "models", "deberta-v3-base-zeroshot-v2.0")
# One hypothesis per area plus the overall description.
HYPOTHESES = {
    "ai": "This question is about artificial intelligence, machine learning, deep learning, neural networks "
          "or data science.",
    "math": "This question is about mathematics used in machine learning, such as derivatives, gradients, "
            "linear algebra, matrices, probability, statistics or optimization.",
    "code": "This question is about programming code of the kind used for AI, such as Python, PyTorch or NumPy.",
    "cs": "This question is about basic computer science related to AI, such as algorithms, data structures, "
          "computational complexity, search or logic.",
    "overall": "This question is related to AI and its nearby areas, such as derivations in math, PyTorch in "
               "code, or basic computer science related to AI.",
}
# group -> {subjects, hypotheses used for selection (None = take every question)}
GROUPS = {
    "ml": {"subjects": ["machine_learning"], "hypotheses": None},
    "math": {"subjects": ["abstract_algebra", "college_mathematics", "elementary_mathematics",
                          "high_school_mathematics", "high_school_statistics"],
             "hypotheses": ["math"]},
    "computer": {"subjects": ["college_computer_science", "high_school_computer_science", "computer_security"],
                 "hypotheses": ["code", "cs"]},
    "logic": {"subjects": ["formal_logic"], "hypotheses": ["overall"]},
}
SUBJECT_GROUP = {s: g for g, info in GROUPS.items() for s in info["subjects"]}
USED_HYPOTHESES = {h for info in GROUPS.values() for h in (info["hypotheses"] or [])}
REPORT_THRESHOLDS = (0.3, 0.35, 0.4, 0.5, 0.7, 0.85)


def correct_answer(r: dict) -> str:
    return str(r["choices"][int(r["answer"])]).strip()


def premise(r: dict) -> str:
    return r["question"].strip() + "\nAnswer: " + correct_answer(r)


def load_questions(mmlu_dir: str, split: str, limit: int | None) -> list[dict]:
    qs = []
    for subj in list_mmlu_subjects(mmlu_dir):
        if subj not in SUBJECT_GROUP:
            continue
        rows = load_mmlu_split(mmlu_dir, subj, split)
        for i, r in enumerate(rows[:limit] if limit else rows):
            qs.append({"id": f"mmlu/{subj}/{split}/{i}", "subject": subj, "group": SUBJECT_GROUP[subj],
                       "text": premise(r), "question": r["question"].strip(), "answer": correct_answer(r)})
    missing = sorted(set(SUBJECT_GROUP) - {q["subject"] for q in qs})
    if missing:
        print(f"warning: subjects not found in {mmlu_dir}: {', '.join(missing)}")
    return qs


def group_score(group: str, scores: dict[str, float]) -> tuple[float | None, str | None]:
    """(score used for selection, hypothesis that gave it); (None, None) for groups taken in full."""
    hyps = GROUPS[group]["hypotheses"]
    if hyps is None:
        return None, None
    best = max(hyps, key=lambda h: scores[h])
    return scores[best], best


def is_selected(r: dict, threshold: float) -> bool:
    return r["score"] is None or r["score"] >= threshold


def classify(qs: list[dict], args, hyps: dict) -> list[dict]:
    from ai_qa_data.zeroshot import ZeroShotClassifier

    zs = ZeroShotClassifier(args.model, device=args.device, batch_size=args.batch_size, max_length=args.max_len)
    print(f"zero-shot: {len(qs):,} questions x {len(hyps)} hypotheses on {zs.device}", flush=True)
    out, t0 = [], time.time()
    for s in range(0, len(qs), args.chunk):
        part = qs[s:s + args.chunk]
        for q, p in zip(part, zs.multi_label([q["text"] for q in part], hyps)):
            score, sel_hyp = group_score(q["group"], p)
            out.append({"id": q["id"], "subject": q["subject"], "group": q["group"], "scores": p,
                        "max": max(p.values()), "best": max(p, key=p.get), "score": score, "sel_hyp": sel_hyp,
                        "question": q["question"][:300], "answer": q["answer"][:200]})
        print(f"  {len(out):,}/{len(qs):,}  {time.time() - t0:.0f}s", flush=True)
    return out


def ci95(acc: float, n: int) -> float:
    return 1.96 * math.sqrt(acc * (1 - acc) / n) if n else float("nan")


def fnum(x: float | None) -> str:
    return "" if x is None else f"{x:.4f}"


def run_setup(run_dir: str) -> dict:
    """How an eval run was made, from its summary.json: shots, prompt format, base checkpoint, adapter."""
    info = {"shots": None, "format": None, "base": None, "adapter": None, "model": None, "has_summary": False}
    path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(path):
        return info
    info["has_summary"] = True
    with open(path, encoding="utf-8") as f:
        summ = json.load(f)
    m = summ.get("tasks", {}).get("mmlu", {})
    info.update(shots=m.get("shots"), format=m.get("format", "chat"), model=summ.get("model"))
    model = summ.get("model") or ""
    model = model if os.path.isabs(model) else os.path.join(ROOT, model)
    cfg = os.path.join(model, "adapter_config.json")
    if os.path.exists(cfg):
        with open(cfg, encoding="utf-8") as f:
            info["base"] = json.load(f).get("base_model_name_or_path")
        info["adapter"] = model
    else:
        info["base"] = model
    if summ.get("base"):  # recorded by eval_model.py; differs from adapter_config when --base overrode it
        info["base"] = summ["base"]
    return info


def run_exclusion(info: dict) -> str | None:
    """Why a run is left out of the earlier-runs table (None = kept): only 5-shot, base-prompt-format runs of a
    pre-trained *-Base checkpoint (alone or with an adapter trained on it) are comparable."""
    if not info["has_summary"]:
        return "no summary.json"
    if info["shots"] is None:
        return "no MMLU result in summary.json"
    if info["shots"] != 5:
        return f"{info['shots']}-shot"
    if info["format"] != "base":
        return f"{info['format']} prompt format"
    if not os.path.basename((info["base"] or "").rstrip("/")).lower().endswith("-base"):
        return f"{os.path.basename((info['base'] or '?').rstrip('/'))} is not a base checkpoint"
    return None


def short_adapter(path: str | None) -> str:
    if not path:
        return "-"
    rel = os.path.relpath(path, ROOT)
    rel = rel[len("runs/"):] if rel.startswith("runs/") else rel
    return rel[:-len("/adapter")] if rel.endswith("/adapter") else rel


def rescore_runs(results_root: str, idsets: dict[str, set[str]], split: str) -> tuple[list[dict], list[dict]]:
    """Accuracy of every comparable earlier eval run on each id set, from its saved MMLU predictions.
    Returns (kept runs, excluded runs with a reason)."""
    runs, excluded = [], []
    for path in sorted(glob.glob(os.path.join(results_root, "*", "mmlu_predictions.jsonl"))):
        run_dir = os.path.dirname(path)
        name = os.path.basename(run_dir)
        info = run_setup(run_dir)
        why = run_exclusion(info)
        if why:
            excluded.append({"run": name, "why": why})
            continue
        with open(path, encoding="utf-8") as f:
            preds = [json.loads(line) for line in f if line.strip()]
        preds = [p for p in preds if p["id"].split("/")[2] == split]
        if not preds:
            excluded.append({"run": name, "why": f"no {split} predictions"})
            continue
        ai = [p["correct"] for p in preds if p["subject"] in MMLU_AI_SUBJECTS]
        sets = {}
        for nm, ids in idsets.items():
            rel = [p["correct"] for p in preds if p["id"] in ids]
            sets[nm] = {"n": len(rel), "acc": sum(rel) / len(rel) if rel else None}
        runs.append({"run": name, "base": os.path.basename(info["base"].rstrip("/")),
                     "adapter": short_adapter(info["adapter"]), "n_mmlu": len(preds),
                     "mmlu": sum(p["correct"] for p in preds) / len(preds),
                     "ai_subjects": sum(ai) / len(ai) if ai else None, "sets": sets})
    return runs, excluded


def effective_threshold(rows: list[dict], threshold: float, min_n: int | None) -> float:
    """The threshold itself, or a lower one (the score of the min_n-th question) when it selects fewer than min_n."""
    if not min_n or sum(is_selected(r, threshold) for r in rows) >= min_n:
        return threshold
    k = min_n - sum(r["score"] is None for r in rows)
    scores = sorted((r["score"] for r in rows if r["score"] is not None), reverse=True)
    if k <= 0:
        return threshold
    return scores[k - 1] if k <= len(scores) else 0.0


def print_runs(runs: list[dict], excluded: list[dict], set_names: list[str]) -> None:
    if runs:
        print(f"\n5-shot base-format runs on a *-Base checkpoint ({len(excluded)} other runs excluded):")
        print(f"{'run':44s} {'base':16s} {'MMLU':>7s} {'AI subj':>8s}"
              + "".join(f" {nm[:12]:>12s} {'±':>6s}" for nm in set_names) + "  n")
        for r in runs:
            cells = ""
            for nm in set_names:
                st = r["sets"][nm]
                ci = None if st["acc"] is None else ci95(st["acc"], st["n"])
                cells += f" {fnum(st['acc']):>12s} {fnum(ci):>6s}"
            ns = "/".join(str(r["sets"][nm]["n"]) for nm in set_names)
            print(f"{r['run']:44s} {r['base']:16s} {r['mmlu']:7.4f} {fnum(r['ai_subjects']):>8s}{cells}  {ns}")
    for e in excluded:
        print(f"  excluded {e['run']}: {e['why']}")


def cell(s: str, n: int) -> str:
    return " ".join(s.split())[:n].replace("|", "\\|")


def write_report(path: str, rows: list[dict], sel: dict, runs: list[dict], excluded: list[dict],
                 set_sizes: dict[str, int] | None = None, extra: list[str] | None = None, seed: int = 0) -> None:
    """report.md. `set_sizes` names the id sets rescored in `runs` (first = this subject) with their sizes;
    `extra` = markdown lines inserted before the earlier-runs section."""
    thr = sel["threshold"]
    set_sizes = set_sizes or {"ai_related": sel["n"]}
    names = list(set_sizes)
    filtered = [r for r in rows if r["score"] is not None]
    req = sel.get("threshold_requested", thr)
    thr_note = (f"**{thr}**" if req == thr
                else f"**{thr}** (lowered from {req} to reach at least {sel['min_n']} questions)")
    L = [f"# {sel.get('title', 'MMLU ai_related subject')}", "",
         f"- split `{sel['split']}`, judge: {sel['judge_note']}, threshold {thr_note} on the best score among "
         "the group's hypotheses",
         f"- {sel['premise_note']}",
         f"- **{sel['n']:,}** of {len(rows):,} scored questions selected: "
         + ", ".join(f"{g} {n}" for g, n in sel["by_group"].items()),
         "- groups:"]
    for g, info in sel["groups"].items():
        rule = ("every question, no classifier" if info["hypotheses"] is None
                else "hypothesis " + " / ".join(f"`{h}`" for h in info["hypotheses"]))
        L.append(f"  - `{g}` ({', '.join(info['subjects'])}): {rule}")
    L.append("- hypotheses:")
    L += [f"  - `{k}`: {v}" + ("" if k in USED_HYPOTHESES else " *(scored, not used for selection)*")
          for k, v in sel["hypotheses"].items()]

    # size per subject at several thresholds
    by_subj = defaultdict(list)
    for r in rows:
        by_subj[r["subject"]].append(r)
    ths = sorted(set(REPORT_THRESHOLDS) | {thr})
    L += ["", "## Questions selected per subject", "",
          "| group | subject | total | " + " | ".join(f"≥ {t}" for t in ths) + " |",
          "|---|---|---:|" + "---:|" * len(ths)]
    for g, info in sel["groups"].items():
        for s in info["subjects"]:
            xs = by_subj.get(s)
            if not xs:
                continue
            L.append(f"| {g} | {s} | {len(xs)} | "
                     + " | ".join(str(sum(is_selected(r, t) for r in xs)) for t in ths) + " |")
    L.append(f"| | **total** | {len(rows)} | "
             + " | ".join(f"**{sum(is_selected(r, t) for r in rows)}**" for t in ths) + " |")

    L += ["", "## Which hypothesis selected them", "",
          "| group | hypotheses | scored | selected | best hypothesis among selected |", "|---|---|---:|---:|---|"]
    for g, info in sel["groups"].items():
        xs = [r for r in rows if r["group"] == g]
        picked = [r for r in xs if is_selected(r, thr)]
        if info["hypotheses"] is None:
            L.append(f"| {g} | (all) | {len(xs)} | {len(picked)} | |")
        else:
            best = Counter(r["sel_hyp"] for r in picked)
            L.append(f"| {g} | {', '.join(info['hypotheses'])} | {len(xs)} | {len(picked)} | "
                     + ", ".join(f"{h} {n}" for h, n in best.most_common()) + " |")

    # borderline examples to check the threshold by hand (filtered groups only)
    rng = random.Random(seed)

    def sample(lo_, hi_, k=20):
        xs = [r for r in filtered if lo_ <= r["score"] < hi_]
        return sorted(rng.sample(xs, min(k, len(xs))), key=lambda r: -r["score"])

    def fmt(r):
        return f"| {r['score']:.3f} | {r['sel_hyp']} | {r['subject']} | {cell(r['question'], 140)} | {cell(r['answer'], 60)} |"

    for title, xs in ((f"Just selected (score {thr} – {min(1.0, thr + 0.05):.2f})", sample(thr, thr + 0.05)),
                      (f"Just rejected (score {max(0.0, thr - 0.1):.2f} – {thr})", sample(thr - 0.1, thr))):
        L += ["", f"## {title}", "", "| score | hypothesis | subject | question | answer |", "|---:|---|---|---|---|"]
        L += [fmt(r) for r in xs] or ["| | | | none | |"]

    if extra:
        L += [""] + list(extra)

    L += ["", "## Earlier runs (from eval_results/*/mmlu_predictions.jsonl)", "",
          "Only runs evaluated **5-shot** with the **base prompt format** on a pre-trained `*-Base` checkpoint, alone "
          "or with a LoRA adapter trained on it; 0-shot and chat-format / chat-model runs are not comparable and are "
          "left out. Scores are recomputed from each run's saved predictions, no model needed."]
    if runs:
        note = ("± = 95 % interval half-width of the subject score; differences between two runs on the same "
                "questions are tighter than that (paired test).")
        if len(names) == 2:
            note += f" Δ = {names[0]} acc − {names[1]} acc of the same run."
        L += ["", note, "",
              "| run | base model | adapter | MMLU acc | AI subj. (11) |"
              + "".join(f" {nm} n | {nm} acc | ± |" for nm in names) + (" Δ |" if len(names) == 2 else ""),
              "|---|---|---|---:|---:|" + "---:|---:|---:|" * len(names) + ("---:|" if len(names) == 2 else "")]
        for r in runs:
            cells = [r["run"], r["base"], r["adapter"], f"{r['mmlu']:.4f}", fnum(r["ai_subjects"])]
            accs = []
            for nm in names:
                st = r["sets"][nm]
                ci = fnum(None if st["acc"] is None else ci95(st["acc"], st["n"]))
                part = "" if st["n"] == set_sizes[nm] else " (partial)"
                cells += [f"{st['n']}{part}", fnum(st["acc"]), ci]
                accs.append(st["acc"])
            if len(names) == 2:
                cells.append("" if None in accs else f"{accs[0] - accs[1]:+.4f}")
            L.append("| " + " | ".join(cells) + " |")
    else:
        L += ["", "No comparable earlier MMLU predictions found."]
    if excluded:
        L += ["", "Excluded: " + ", ".join(f"`{e['run']}` ({e['why']})" for e in excluded) + "."]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mmlu-dir", default=MMLU_DIR)
    ap.add_argument("--split", default="test", choices=["test", "validation", "dev"])
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--model", default=ZS_MODEL, help="zero-shot NLI model")
    ap.add_argument("--hypotheses", default=None, help="JSON file {name: hypothesis} replacing the built-in ones")
    ap.add_argument("--threshold", type=float, default=0.4,
                    help="P(entailment) of the group's best hypothesis (0.4 gives ~300 questions on the test split)")
    ap.add_argument("--min-n", type=int, default=None,
                    help="lower the threshold until at least this many questions are selected")
    ap.add_argument("--reclassify", action="store_true", help="ignore cached scores.jsonl")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=256, help="premise tokens (question + correct answer)")
    ap.add_argument("--chunk", type=int, default=2048, help="questions per progress line")
    ap.add_argument("--limit-per-subject", type=int, default=None, help="smoke tests only")
    ap.add_argument("--results-root", default=RESULTS_ROOT, help="earlier eval runs to rescore")
    args = ap.parse_args()

    hyps = HYPOTHESES
    if args.hypotheses:
        with open(args.hypotheses, encoding="utf-8") as f:
            hyps = json.load(f)
        missing = sorted(USED_HYPOTHESES - set(hyps))
        if missing:
            raise SystemExit(f"--hypotheses must define the ones GROUPS use; missing: {', '.join(missing)}")
    os.makedirs(args.out, exist_ok=True)
    scores_path, meta_path = os.path.join(args.out, "scores.jsonl"), os.path.join(args.out, "scores.meta.json")
    meta = {"model": os.path.abspath(args.model), "hypotheses": hyps, "groups": GROUPS, "premise": "question+answer",
            "split": args.split, "max_len": args.max_len, "limit_per_subject": args.limit_per_subject}

    cached = None
    if not args.reclassify and os.path.exists(scores_path) and os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            if json.load(f) == meta:
                with open(scores_path, encoding="utf-8") as g:
                    cached = [json.loads(line) for line in g if line.strip()]
                print(f"reusing {len(cached):,} cached scores from {scores_path}")
            else:
                print("cached scores were made with other settings: classifying again")
    if cached is None:
        qs = load_questions(args.mmlu_dir, args.split, args.limit_per_subject)
        cached = classify(qs, args, hyps)
        with open(scores_path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in cached)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
    rows = cached

    thr = effective_threshold(rows, args.threshold, args.min_n)
    ids = sorted((r for r in rows if is_selected(r, thr)), key=lambda r: r["id"])
    sel = {"split": args.split, "threshold": thr, "threshold_requested": args.threshold, "min_n": args.min_n,
           "judge": "nli", "model": meta["model"], "title": "MMLU ai_related subject (zero-shot NLI judge)",
           "judge_note": f"zero-shot NLI model `{os.path.relpath(meta['model'], ROOT)}`, score = P(entailment)",
           "premise_note": "premise shown to the classifier: question text + `Answer: <correct answer text>` "
                           "(no wrong options, no subject name)",
           "premise": meta["premise"], "groups": GROUPS, "hypotheses": hyps, "created": time.strftime("%F %T"),
           "n": len(ids), "by_group": {g: sum(r["group"] == g for r in ids) for g in GROUPS},
           "by_subject": dict(Counter(r["subject"] for r in ids).most_common()), "ids": [r["id"] for r in ids]}
    with open(os.path.join(args.out, "selection.json"), "w", encoding="utf-8") as f:
        json.dump(sel, f, indent=1)

    runs, excluded = rescore_runs(args.results_root, {"ai_related": set(sel["ids"])}, args.split)
    write_report(os.path.join(args.out, "report.md"), rows, sel, runs, excluded, {"ai_related": sel["n"]})

    print(f"\nai_related: {sel['n']:,} of {len(rows):,} scored questions at threshold {thr}")
    for g, info in GROUPS.items():
        print(f"  [{g}] {'all' if info['hypotheses'] is None else '/'.join(info['hypotheses'])}: {sel['by_group'][g]}")
        for s in info["subjects"]:
            n_tot = sum(r["subject"] == s for r in rows)
            if n_tot:
                print(f"      {s:36s} {sel['by_subject'].get(s, 0):4d} / {n_tot}")
    print_runs(runs, excluded, ["ai_related"])
    print(f"\nwrote {args.out}/selection.json, scores.jsonl, report.md")


if __name__ == "__main__":
    main()
