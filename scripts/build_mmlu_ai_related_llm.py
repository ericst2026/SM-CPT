#!/usr/bin/env python3
"""Build the MMLU "ai_related" subject a second way: the Qwen3-1.7B chat model is asked, for every question of the
same subject groups, whether the question (shown with its correct answer) is about each hypothesis; P(yes) is the
score. Groups, hypotheses, premise and selection rule are those of scripts/build_mmlu_ai_related.py (zero-shot NLI
judge), so the two selections can be compared: the report does that and rescores the earlier 5-shot runs on both.

    python scripts/build_mmlu_ai_related_llm.py                   # judge with models/Qwen3-1.7b (GPU, a few minutes)
    python scripts/build_mmlu_ai_related_llm.py --threshold 0.9   # re-select from the cached scores (no model)
    python scripts/build_mmlu_ai_related_llm.py --min-n 0         # keep --threshold even if it gives < 300 questions
    python scripts/build_mmlu_ai_related_llm.py --limit-per-subject 3 --out /tmp/x   # smoke test

Judge  chat template with thinking off; system prompt SYSTEM, user prompt USER = question + correct answer + the
       hypothesis rephrased as a yes/no question ("This question is about X." -> "Is this question about X?").
       Score = P(yes) / (P(yes) + P(no)) over the first answer token (yes/Yes/YES/... variants summed); nothing is
       generated. `yes_no_mass` in scores.jsonl is P(yes) + P(no), i.e. how surely the model answered the question.
Selection  as in the NLI script: machine_learning in full; math subjects by `math`; computer subjects by the best of
       `code` / `cs`; formal_logic by `overall`; best score >= --threshold, lowered if needed to reach --min-n.

Output (--out, default data/eval/mmlu_ai_related_llm/): scores.jsonl, selection.json, report.md (same layout as the
NLI report plus a "Comparison with the zero-shot NLI selection" section and both subjects' scores for every earlier
5-shot run). Evaluate a model on it with
    python scripts/eval_model.py --tasks mmlu --mmlu-shots 5 --mmlu-subjects ai_related \\
        --mmlu-ai-related data/eval/mmlu_ai_related_llm/selection.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_mmlu_ai_related as nli  # noqa: E402
from build_mmlu_ai_related import (GROUPS, HYPOTHESES, ROOT, USED_HYPOTHESES, cell, effective_threshold,  # noqa: E402
                                   group_score, is_selected, load_questions, print_runs, rescore_runs,
                                   write_report)
from eval_model import MMLU_DIR, RESULTS_ROOT  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "eval", "mmlu_ai_related_llm")
JUDGE_MODEL = os.path.join(ROOT, "models", "Qwen3-1.7b")
SYSTEM = "You classify exam questions by topic. Reply with a single word: yes or no."
USER = ("Here is a multiple-choice exam question together with its correct answer.\n\n"
        "Question: {question}\nCorrect answer: {answer}\n\n{hypothesis} Answer yes or no.")
YES_WORDS, NO_WORDS = ["yes", "Yes", "YES", " yes", " Yes"], ["no", "No", "NO", " no", " No"]


def as_question(h: str) -> str:
    """'This question is about X.' -> 'Is this question about X?'"""
    h = h.strip().rstrip(".")
    if h.startswith("This question is "):
        h = "Is this question " + h[len("This question is "):]
    return h + "?"


class YesNoJudge:
    """P(yes) of a chat model's first answer token, batched with left padding; nothing is generated."""

    def __init__(self, model_path: str, batch_size: int = 16, max_chars: int = 3000):
        import torch
        from ai_qa_train.common import load_model, load_tokenizer, render

        self.torch, self.render = torch, render
        self.tok = load_tokenizer(model_path)
        self.tok.padding_side = "left"
        self.model = load_model(model_path)
        self.device = next(self.model.parameters()).device
        self.batch_size, self.max_chars = batch_size, max_chars
        self.yes_ids, self.no_ids = self._single_tokens(YES_WORDS), self._single_tokens(NO_WORDS)

    def _single_tokens(self, words: list[str]) -> list[int]:
        ids = []
        for w in words:
            t = self.tok.encode(w, add_special_tokens=False)
            if len(t) == 1 and t[0] not in ids:
                ids.append(t[0])
        return ids

    def prompt(self, question: str, answer: str, hypothesis_q: str) -> str:
        q = question if len(question) <= self.max_chars else question[:self.max_chars] + " ..."
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(question=q, answer=answer, hypothesis=hypothesis_q)}]
        return self.render(self.tok, msgs, add_generation_prompt=True, enable_thinking=False)

    def p_yes(self, prompts: list[str]) -> list[tuple[float, float]]:
        """(P(yes) / (P(yes) + P(no)), P(yes) + P(no)) per prompt."""
        torch = self.torch
        out: list[tuple[float, float] | None] = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # length bucketing
        with torch.inference_mode():
            for s in range(0, len(order), self.batch_size):
                idx = order[s:s + self.batch_size]
                enc = self.tok([prompts[i] for i in idx], padding=True, return_tensors="pt",
                               add_special_tokens=False).to(self.device)
                pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
                probs = self.model(**enc, position_ids=pos).logits[:, -1, :].float().softmax(-1)
                yes, no = probs[:, self.yes_ids].sum(-1), probs[:, self.no_ids].sum(-1)
                p = (yes / (yes + no).clamp(min=1e-9)).tolist()
                mass = (yes + no).tolist()
                for i, a, b in zip(idx, p, mass):
                    out[i] = (a, b)
        return out  # type: ignore[return-value]


def classify(qs: list[dict], args, hyps_q: dict[str, str]) -> list[dict]:
    judge = YesNoJudge(args.model, batch_size=args.batch_size, max_chars=args.max_chars)
    names = list(hyps_q)
    print(f"LLM judge {os.path.relpath(args.model, ROOT)}: {len(qs):,} questions x {len(names)} hypotheses "
          f"on {judge.device}", flush=True)
    out, t0 = [], time.time()
    for s in range(0, len(qs), args.chunk):
        part = qs[s:s + args.chunk]
        res = judge.p_yes([judge.prompt(q["question"], q["answer"], hyps_q[n]) for q in part for n in names])
        for j, q in enumerate(part):
            p = {n: round(res[j * len(names) + k][0], 4) for k, n in enumerate(names)}
            mass = min(res[j * len(names) + k][1] for k in range(len(names)))
            score, sel_hyp = group_score(q["group"], p)
            out.append({"id": q["id"], "subject": q["subject"], "group": q["group"], "scores": p,
                        "max": max(p.values()), "best": max(p, key=p.get), "score": score, "sel_hyp": sel_hyp,
                        "yes_no_mass": round(mass, 4), "question": q["question"][:300], "answer": q["answer"][:200]})
        print(f"  {len(out):,}/{len(qs):,}  {time.time() - t0:.0f}s", flush=True)
    return out


# --------------------------------------------------------------------------- comparison with the NLI selection
def load_nli(nli_dir: str) -> dict | None:
    sp, scp = os.path.join(nli_dir, "selection.json"), os.path.join(nli_dir, "scores.jsonl")
    if not (os.path.exists(sp) and os.path.exists(scp)):
        return None
    with open(sp, encoding="utf-8") as f:
        sel = json.load(f)
    with open(scp, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {"dir": nli_dir, "sel": sel, "rows": {r["id"]: r for r in rows}, "ids": set(sel["ids"])}


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x, y).correlation)
    except Exception:
        import numpy as np
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))  # ties broken by order
        return float(np.corrcoef(rx, ry)[0, 1])


def compare_section(rows: list[dict], sel: dict, other: dict) -> list[str]:
    llm_ids, nli_ids, nli_rows = set(sel["ids"]), other["ids"], other["rows"]
    L = ["## Comparison with the zero-shot NLI selection", "",
         f"NLI subject: `{os.path.relpath(os.path.join(other['dir'], 'selection.json'), ROOT)}` "
         f"(threshold {other['sel']['threshold']}, {len(nli_ids):,} questions). LLM subject: threshold "
         f"{sel['threshold']}, {len(llm_ids):,} questions. Both draw from the same {len(rows):,} questions (same "
         "groups, split and premise) and both hold all of machine_learning, so the differences are in the filtered "
         "groups.", "",
         "| group | subject | scored | NLI | LLM | both | NLI only | LLM only | Jaccard |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]

    def line(g, s, xs):
        ids = {r["id"] for r in xs}
        a, b = ids & nli_ids, ids & llm_ids
        both, union = a & b, a | b
        jac = f"{len(both) / len(union):.2f}" if union else ""
        return f"| {g} | {s} | {len(xs)} | {len(a)} | {len(b)} | {len(both)} | {len(a - b)} | {len(b - a)} | {jac} |"

    for g, info in GROUPS.items():
        for s in info["subjects"]:
            xs = [r for r in rows if r["subject"] == s]
            if xs:
                L.append(line(g, s, xs))
        if len(info["subjects"]) > 1:
            L.append(line(f"**{g}**", "*(group)*", [r for r in rows if r["group"] == g]))
    L.append(line("", "**total**", rows))

    filtered = [r for r in rows if r["score"] is not None]
    L += ["", "### Score agreement on the filtered groups", "",
          "Spearman ρ between the NLI score and the LLM score of the same question (each the best of the group's "
          "hypotheses); mean score per judge; and how often the LLM put at least 90 % of its first-token probability "
          "on yes/no (`yes_no_mass`).", "",
          "| group | n | Spearman ρ | NLI mean | LLM mean | LLM ≥ 0.5 | LLM ≥ 0.9 | yes/no mass ≥ 0.9 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g, info in GROUPS.items():
        if info["hypotheses"] is None:
            continue
        xs = [r for r in filtered if r["group"] == g and r["id"] in nli_rows and nli_rows[r["id"]]["score"] is not None]
        if not xs:
            continue
        x, y = [nli_rows[r["id"]]["score"] for r in xs], [r["score"] for r in xs]
        sure = sum(r.get("yes_no_mass", 1.0) >= 0.9 for r in xs)
        L.append(f"| {g} | {len(xs)} | {spearman(x, y):.2f} | {mean(x):.3f} | {mean(y):.3f} | "
                 f"{sum(v >= 0.5 for v in y)} | {sum(v >= 0.9 for v in y)} | {sure} ({100 * sure / len(xs):.0f} %) |")

    bins = [(i / 10, (i + 1) / 10) for i in range(10)]
    L += ["", "LLM score histogram over the filtered groups: "
          + ", ".join(f"{lo:.1f}–{hi:.1f}: {sum(lo <= r['score'] < (hi if hi < 1 else 1.01) for r in filtered)}"
                      for lo, hi in bins)]

    def fmt(r):
        o = nli_rows.get(r["id"], {})
        ns = "" if o.get("score") is None else f"{o['score']:.3f}"
        return (f"| {r['score']:.3f} | {ns} | {r['sel_hyp']} | {r['subject']} | {cell(r['question'], 140)} | "
                f"{cell(r['answer'], 60)} |")

    only_llm = sorted((r for r in filtered if r["id"] in llm_ids - nli_ids), key=lambda r: -r["score"])[:15]
    only_nli = sorted((r for r in filtered if r["id"] in nli_ids - llm_ids),
                      key=lambda r: -(nli_rows.get(r["id"], {}).get("score") or 0))[:15]
    for title, xs in (("Selected only by the LLM judge (highest LLM score first)", only_llm),
                      ("Selected only by the NLI model (highest NLI score first)", only_nli)):
        L += ["", f"### {title}", "", "| LLM | NLI | hypothesis | subject | question | answer |",
              "|---:|---:|---|---|---|---|"]
        L += [fmt(r) for r in xs] or ["| | | | | none | |"]
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mmlu-dir", default=MMLU_DIR)
    ap.add_argument("--split", default="test", choices=["test", "validation", "dev"])
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--model", default=JUDGE_MODEL, help="chat model used as the judge")
    ap.add_argument("--hypotheses", default=None,
                    help="JSON file {name: statement} replacing the built-in ones (rephrased to questions here)")
    ap.add_argument("--threshold", type=float, default=0.5, help="P(yes) of the group's best hypothesis")
    ap.add_argument("--min-n", type=int, default=300,
                    help="lower the threshold until at least this many questions are selected (0 = never)")
    ap.add_argument("--reclassify", action="store_true", help="ignore cached scores.jsonl")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-chars", type=int, default=3000, help="question text is cut after this many characters")
    ap.add_argument("--chunk", type=int, default=256, help="questions per progress line")
    ap.add_argument("--limit-per-subject", type=int, default=None, help="smoke tests only")
    ap.add_argument("--results-root", default=RESULTS_ROOT, help="earlier eval runs to rescore")
    ap.add_argument("--compare", default=nli.OUT_DIR, help="directory of the NLI selection to compare with ('' = skip)")
    args = ap.parse_args()

    hyps = HYPOTHESES
    if args.hypotheses:
        with open(args.hypotheses, encoding="utf-8") as f:
            hyps = json.load(f)
        missing = sorted(USED_HYPOTHESES - set(hyps))
        if missing:
            raise SystemExit(f"--hypotheses must define the ones GROUPS use; missing: {', '.join(missing)}")
    hyps_q = {k: as_question(v) for k, v in hyps.items()}
    os.makedirs(args.out, exist_ok=True)
    scores_path, meta_path = os.path.join(args.out, "scores.jsonl"), os.path.join(args.out, "scores.meta.json")
    meta = {"judge": "llm-yes-no", "model": os.path.abspath(args.model), "system": SYSTEM, "user": USER,
            "hypotheses": hyps_q, "groups": GROUPS, "premise": "question+answer", "split": args.split,
            "max_chars": args.max_chars, "limit_per_subject": args.limit_per_subject}

    cached = None
    if not args.reclassify and os.path.exists(scores_path) and os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            if json.load(f) == meta:
                with open(scores_path, encoding="utf-8") as g:
                    cached = [json.loads(line) for line in g if line.strip()]
                print(f"reusing {len(cached):,} cached scores from {scores_path}")
            else:
                print("cached scores were made with other settings: judging again")
    if cached is None:
        qs = load_questions(args.mmlu_dir, args.split, args.limit_per_subject)
        cached = classify(qs, args, hyps_q)
        with open(scores_path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in cached)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
    rows = cached

    thr = effective_threshold(rows, args.threshold, args.min_n)
    ids = sorted((r for r in rows if is_selected(r, thr)), key=lambda r: r["id"])
    sel = {"split": args.split, "threshold": thr, "threshold_requested": args.threshold, "min_n": args.min_n,
           "judge": "llm", "model": meta["model"],
           "title": f"MMLU ai_related subject (LLM judge, {os.path.basename(meta['model'])})",
           "judge_note": f"`{os.path.relpath(meta['model'], ROOT)}` asked one yes/no question per hypothesis (chat "
                         "template, thinking off), score = P(yes) / (P(yes) + P(no)) of the first answer token",
           "premise_note": "shown to the judge: question text + `Correct answer: <correct answer text>` (no wrong "
                           "options, no subject name)",
           "premise": meta["premise"], "groups": GROUPS, "hypotheses": hyps_q, "prompt": {"system": SYSTEM, "user": USER},
           "created": time.strftime("%F %T"), "n": len(ids),
           "by_group": {g: sum(r["group"] == g for r in ids) for g in GROUPS},
           "by_subject": dict(Counter(r["subject"] for r in ids).most_common()), "ids": [r["id"] for r in ids]}
    with open(os.path.join(args.out, "selection.json"), "w", encoding="utf-8") as f:
        json.dump(sel, f, indent=1)

    other = load_nli(args.compare) if args.compare else None
    idsets, sizes, extra = {"LLM": set(sel["ids"])}, {"LLM": sel["n"]}, None
    if other:
        idsets["NLI"], sizes["NLI"] = other["ids"], len(other["ids"])
        extra = compare_section(rows, sel, other)
    elif args.compare:
        print(f"no NLI selection in {args.compare}: comparison skipped")
    runs, excluded = rescore_runs(args.results_root, idsets, args.split)
    write_report(os.path.join(args.out, "report.md"), rows, sel, runs, excluded, sizes, extra)

    print(f"\nai_related (LLM): {sel['n']:,} of {len(rows):,} scored questions at threshold {thr}"
          + (f" (lowered from {args.threshold} for --min-n {args.min_n})" if thr != args.threshold else ""))
    for g, info in GROUPS.items():
        print(f"  [{g}] {'all' if info['hypotheses'] is None else '/'.join(info['hypotheses'])}: {sel['by_group'][g]}")
        for s in info["subjects"]:
            n_tot = sum(r["subject"] == s for r in rows)
            if n_tot:
                print(f"      {s:36s} {sel['by_subject'].get(s, 0):4d} / {n_tot}")
    if other:
        both = len(set(sel["ids"]) & other["ids"])
        print(f"\noverlap with NLI ({len(other['ids'])}): both {both}, LLM only {sel['n'] - both}, "
              f"NLI only {len(other['ids']) - both}")
    print_runs(runs, excluded, list(idsets))
    print(f"\nwrote {args.out}/selection.json, scores.jsonl, report.md")


if __name__ == "__main__":
    main()
