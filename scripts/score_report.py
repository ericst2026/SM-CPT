#!/usr/bin/env python3
"""Leaderboard of the evaluated models (docs/report.md): six accuracies per run, averaged into one score.

    python scripts/score_report.py                  # writes docs/report.md
    python scripts/score_report.py --out /tmp/r.md

Scores (all from files already in eval_results/, no model is run)
    MMLU           5-shot MMLU, every test question                      mmlu_predictions.jsonl
    AI subj. (11)  the 11 hand-picked AI-related MMLU subjects            eval_model.MMLU_AI_SUBJECTS
    LLM-sel.       ai_related subject chosen by the Qwen3-1.7B judge      data/eval/mmlu_ai_related_llm/selection.json
    NLI-sel.       ai_related subject chosen by the zero-shot NLI model   data/eval/mmlu_ai_related/selection.json
    ML             the MMLU machine_learning subject
    MCQ            the pipeline's own MCQ set, 5-shot                     summary.json -> tasks.mcq.accuracy
    score          plain mean of the six; a run missing any of them is listed separately with a partial mean
    AI_Large       the ai_large subject (1,000 ML / AI questions, scripts/build_mmlu_ai_large.py), 5-shot; shown next to
                   the six and in its own section with paired deltas, but not part of the score (so ranks stay comparable)

Runs: only 5-shot, base-prompt-format evaluations of a *-Base checkpoint, alone or with an adapter trained on it
(the rule of scripts/build_mmlu_ai_related.py). Identical adapter-less evaluations of one checkpoint are merged.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_mmlu_ai_related import ci95, run_exclusion, run_setup, short_adapter  # noqa: E402
from eval_model import (MMLU_AI_RELATED, MMLU_AI_SUBJECTS, MMLU_EXTRA_SUBJECTS, RESULTS_ROOT, ROOT,  # noqa: E402
                        load_ai_related)

LLM_SELECTION = os.path.join(ROOT, "data", "eval", "mmlu_ai_related_llm", "selection.json")
OUT = os.path.join(ROOT, "docs", "report.md")
RERUN = re.compile(r"^(?P<orig>.+)_rerun\d*$")  # <tag>_rerun<k> = the same evaluation repeated
SCORES = [("mmlu", "MMLU"), ("ai_subj", "AI subj. (11)"), ("llm_sel", "LLM-sel."), ("nli_sel", "NLI-sel."),
          ("ml", "ML"), ("mcq", "MCQ")]
WHAT = {
    "mmlu": "5-shot MMLU, every test question",
    "ai_subj": "the 11 hand-picked AI-related MMLU subjects (`MMLU_AI_SUBJECTS` in scripts/eval_model.py)",
    "llm_sel": "the `ai_related` subject selected by the Qwen3-1.7B yes/no judge "
               "(`data/eval/mmlu_ai_related_llm/selection.json`, scripts/build_mmlu_ai_related_llm.py)",
    "nli_sel": "the `ai_related` subject selected by the zero-shot NLI model "
               "(`data/eval/mmlu_ai_related/selection.json`, scripts/build_mmlu_ai_related.py)",
    "ml": "the MMLU `machine_learning` subject",
    "mcq": "the pipeline's own MCQ set (MMLU / MMLU-Pro AI items), 5-shot, from the run's summary.json",
}
LARGE = "ai_large"  # extra subject: in the predictions files (merged by scripts/merge_ai_large.py), not in MMLU


def acc(flags) -> float | None:
    flags = list(flags)
    return sum(flags) / len(flags) if flags else None


def collect(results_root: str, split: str) -> tuple[list[dict], list[tuple[str, str]], dict[str, int]]:
    sets = {"nli_sel": load_ai_related(MMLU_AI_RELATED), "llm_sel": load_ai_related(LLM_SELECTION)}
    rows, excluded, ns = [], [], {}
    for sdir in sorted(glob.glob(os.path.join(results_root, "*"))):
        spath = os.path.join(sdir, "summary.json")
        if not os.path.exists(spath):
            continue
        name = os.path.basename(sdir)
        info = run_setup(sdir)
        why = run_exclusion(info)
        if why:
            excluded.append((name, why))
            continue
        ppath = os.path.join(sdir, "mmlu_predictions.jsonl")
        if not os.path.exists(ppath):
            excluded.append((name, "no mmlu_predictions.jsonl"))
            continue
        with open(ppath, encoding="utf-8") as f:
            preds = [json.loads(line) for line in f if line.strip()]
        preds = [p for p in preds if p["id"].split("/")[2] == split]
        large = [p for p in preds if p["subject"] == LARGE]
        preds = [p for p in preds if p["subject"] not in MMLU_EXTRA_SUBJECTS]
        if large and not preds:
            excluded.append((name, f"{LARGE}-only evaluation (merged into the run's own results)"))
            continue
        if not preds:
            excluded.append((name, f"no {split} predictions"))
            continue
        with open(spath, encoding="utf-8") as f:
            summ = json.load(f)
        mcq = summ.get("tasks", {}).get("mcq") or {}
        mpath = os.path.join(sdir, "mcq_predictions.jsonl")
        mcq_preds, mcq_correct = {}, {}
        if os.path.exists(mpath):
            with open(mpath, encoding="utf-8") as f:
                for q in (json.loads(line) for line in f if line.strip()):
                    mcq_preds[q["id"]], mcq_correct[q["id"]] = q["pred"], bool(q["correct"])
        parts = {"mmlu": preds,
                 "ai_subj": [p for p in preds if p["subject"] in MMLU_AI_SUBJECTS],
                 "llm_sel": [p for p in preds if p["id"] in sets["llm_sel"]],
                 "nli_sel": [p for p in preds if p["id"] in sets["nli_sel"]],
                 "ml": [p for p in preds if p["subject"] == "machine_learning"]}
        scores = {k: acc(p["correct"] for p in v) for k, v in parts.items()}
        n = {k: len(v) for k, v in parts.items()}
        scores["mcq"], n["mcq"] = mcq.get("accuracy"), mcq.get("n", 0)
        n[LARGE] = len(large)
        oq = summ.get("tasks", {}).get("open_qa") or {}
        on_chat = bool(summ.get("chat_model")) and "open_qa" in (summ.get("chat_tasks") or [])
        open_qa = ({"rougeL": oq.get("rougeL"), "token_f1": oq.get("token_f1"), "level_fit": oq.get("level_fit"),
                    "n": oq.get("n", 0), "chat_model": os.path.basename(str(summ.get("chat_model")).rstrip("/"))}
                   if oq and on_chat else None)
        n["open_qa"] = open_qa["n"] if open_qa else 0
        for k, v in n.items():
            ns[k] = max(ns.get(k, 0), v)
        rows.append({"run": name, "base": os.path.basename(info["base"].rstrip("/")),
                     "adapter": short_adapter(info["adapter"]), "time": summ.get("time", ""),
                     "scores": scores, "n": n, "aliases": [], "open_qa": open_qa,
                     "large": {p["id"]: (bool(p["correct"]), p.get("topic") or "?") for p in large},
                     "open_qa_note": None if open_qa or not oq else "generated on the base model: not comparable",
                     "preds": {"mmlu": {p["id"]: p["pred"] for p in preds}, "mcq": mcq_preds},
                     "correct": {"mmlu": {p["id"]: bool(p["correct"]) for p in preds}, "mcq": mcq_correct}})
    return rows, excluded, ns, sets


def bootstrap_halfwidth(r: dict, sets: dict[str, set[str]], n_boot: int = 1000, seed: int = 0) -> float | None:
    """95 % half-width of the combined score from resampling the MMLU and MCQ questions with replacement
    (test-set sampling noise; the evaluation itself is deterministic). Same seed for every run = paired draws."""
    import numpy as np

    if not r["correct"]["mcq"]:
        return None
    mm_ids = sorted(r["correct"]["mmlu"])
    mm = np.array([r["correct"]["mmlu"][i] for i in mm_ids], dtype=float)
    subj = [i.split("/")[1] for i in mm_ids]
    masks = {"ai_subj": np.array([x in MMLU_AI_SUBJECTS for x in subj]),
             "llm_sel": np.array([i in sets["llm_sel"] for i in mm_ids]),
             "nli_sel": np.array([i in sets["nli_sel"] for i in mm_ids]),
             "ml": np.array([x == "machine_learning" for x in subj])}
    mc = np.array([v for _, v in sorted(r["correct"]["mcq"].items())], dtype=float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        bi = rng.integers(0, len(mm), len(mm))
        bm = mm[bi]
        parts = [bm.mean()] + [bm[masks[k][bi]].mean() for k in ("ai_subj", "llm_sel", "nli_sel", "ml")
                               if masks[k][bi].any()]
        parts.append(mc[rng.integers(0, len(mc), len(mc))].mean())
        vals.append(float(np.mean(parts)))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(hi - lo) / 2


def split_reruns(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Rows tagged <tag>_rerun<k> are repeats of <tag>: taken out of the ranking and attached to the original as
    "repeats". Returns (rows to rank, names of repeats whose original is not among the rows)."""
    by_name = {r["run"]: r for r in rows}
    kept, orphans = [], []
    for r in rows:
        m = RERUN.match(r["run"])
        if m and m.group("orig") in by_name:
            by_name[m.group("orig")].setdefault("repeats", []).append(r)
        elif m:
            orphans.append(r["run"])
        else:
            kept.append(r)
    return kept, orphans


def changed_answers(a: dict, b: dict, task: str) -> str | None:
    pa, pb = a["preds"].get(task) or {}, b["preds"].get(task) or {}
    common = [i for i in pa if i in pb]
    if not common:
        return None
    ch = sum(pa[i] != pb[i] for i in common)
    return f"{task.upper()} {ch} of {len(common)} ({100 * ch / len(common):.2f} %)"


def merge_duplicates(rows: list[dict]) -> list[dict]:
    """Adapter-less evaluations of the same checkpoint with the same MMLU accuracy are one model: keep the row
    with the most scores (then the earliest) and remember the other run names."""
    out, seen = [], {}
    for r in sorted(rows, key=lambda r: (-(sum(v is not None for v in r["scores"].values())
                                            + (1 if r.get("open_qa") else 0)), r["time"])):
        key = (r["base"], round(r["scores"]["mmlu"], 4)) if r["adapter"] == "-" else None
        if key is not None and key in seen:
            seen[key]["aliases"].append(r["run"])
            if r["large"] and not seen[key]["large"]:  # ai_large merged into only one of the identical evaluations
                seen[key]["large"] = r["large"]
            continue
        if key is not None:
            seen[key] = r
        out.append(r)
    return out


def combined(r: dict) -> tuple[float | None, int]:
    present = [r["scores"][k] for k, _ in SCORES if r["scores"][k] is not None]
    return (mean(present) if present else None), len(present)


def fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.4f}"


def large_acc(r: dict) -> float | None:
    return acc(c for c, _ in r["large"].values())


def large_section(complete: list[dict], partial: list[dict]) -> list[str]:
    """ai_large per run (rank order), paired Δ vs the same checkpoint without an adapter, and per topic."""
    import numpy as np

    rows = [r for r in complete + partial if r["large"]]
    if not rows:
        return []
    ref = {r["base"]: r for r in rows if r["adapter"] == "-"}
    topics = sorted({t for r in rows for _, t in r["large"].values()})
    L = ["", f"## {LARGE} (1,000 ML / AI questions, 5-shot)", "",
         f"`{LARGE}` is a set of 1,000 MMLU-style machine-learning / AI questions written for this project (8 topics x "
         "125, `data/eval/mmlu_ai_large/`, scripts/build_mmlu_ai_large.py; blind-verified, answer letters balanced). "
         "Same 5-shot base-format scoring as MMLU, but not part of the combined score, so the ranking stays comparable "
         "with earlier reports. Δ = paired difference with the same checkpoint without an adapter, on the same "
         "questions, with a 95 % bootstrap interval; a single accuracy on 1,000 questions is ±0.031, a paired Δ "
         "about ±0.02. Rows in ranking order (unranked runs last); per-topic columns have 125 questions each (±0.087).",
         "", "| run | base model | adapter | AI_Large | Δ vs base [95 %] | " + " | ".join(topics) + " |",
         "|---|---|---|---:|---:|" + "---:|" * len(topics)]
    for r in rows:
        b = ref.get(r["base"])
        d = ""
        if b is not None and r is not b:
            ids = [i for i in r["large"] if i in b["large"]]
            x = np.array([r["large"][i][0] for i in ids], float) - np.array([b["large"][i][0] for i in ids], float)
            rng = np.random.default_rng(0)
            lo, hi = np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)], [2.5, 97.5])
            d = f"{x.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"
        per = [acc(c for c, t in r["large"].values() if t == tp) for tp in topics]
        L.append(f"| {r['run']} | {r['base']} | {r['adapter']} | {fmt(large_acc(r))} | {d} | "
                 + " | ".join(fmt(v) for v in per) + " |")
    return L


# --------------------------------------------------------------------------- CPT adapter on the chat model
CHAT_COLS = [("mmlu", "MMLU", lambda p, S: True), ("ai_subj", "AI subj.", lambda p, S: p["subject"] in MMLU_AI_SUBJECTS),
             ("llm_sel", "LLM-sel.", lambda p, S: p["id"] in S["llm_sel"]),
             ("nli_sel", "NLI-sel.", lambda p, S: p["id"] in S["nli_sel"]),
             ("ml", "ML", lambda p, S: p["subject"] == "machine_learning")]


def _read_preds(path: str) -> dict[str, bool]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {q["id"]: (q["correct"], q["subject"]) for q in (json.loads(line) for line in f if line.strip())
                if q["subject"] not in MMLU_EXTRA_SUBJECTS}


def collect_on_chat(results_root: str, sets: dict[str, set[str]]) -> dict[str, list[dict]]:
    """Evaluations made on a chat (non-Base) checkpoint, alone or with an adapter applied to it:
    0-shot chat-format MMLU (tasks.mmlu: generated answer or first-token letter) and thinking-mode MMLU
    (tasks.mmlu_thinking). Grouped by chat model."""
    rows = {}
    for sdir in sorted(glob.glob(os.path.join(results_root, "*"))):
        spath = os.path.join(sdir, "summary.json")
        if not os.path.exists(spath):
            continue
        with open(spath, encoding="utf-8") as f:
            summ = json.load(f)
        info = run_setup(sdir)
        base = os.path.basename((info["base"] or "").rstrip("/"))
        if not base or base.lower().endswith("-base"):
            continue
        t = summ.get("tasks", {})
        mode = None
        if "mmlu_thinking" in t:
            mode, preds = "think", _read_preds(os.path.join(sdir, "mmlu_thinking_predictions.jsonl"))
        elif t.get("mmlu", {}).get("format") == "chat" and t["mmlu"].get("shots") == 0:
            mode = "gen" if t["mmlu"].get("scoring") == "generate" else "nothink"
            preds = _read_preds(os.path.join(sdir, "mmlu_predictions.jsonl"))
        if not mode or not preds:
            continue
        key = (base, short_adapter(info["adapter"]))
        row = rows.setdefault(key, {"base": base, "adapter": key[1], "runs": {}})
        if mode in row["runs"] and len(row["runs"][mode]["preds"]) >= len(preds):
            continue  # same model and mode twice (e.g. a smoke test): keep the larger run
        extra = t["mmlu_thinking"] if mode == "think" else t["mmlu"]
        row["runs"][mode] = {"tag": os.path.basename(sdir), "preds": preds, "extra": extra}
    by_base = defaultdict(list)
    for (base, _), r in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1] != "-", kv[0][1])):
        by_base[base].append(r)
    return by_base


def _paired(a: dict, b: dict, f, S, n_boot: int = 2000, seed: int = 0):
    """accuracy of a, of b, and the paired Δ (a − b) with a 95 % bootstrap interval, on their common questions."""
    import numpy as np

    ids = [i for i in a if i in b and f({"id": i, "subject": a[i][1]}, S)]
    if not ids:
        return None
    x = np.array([a[i][0] for i in ids], float)
    y = np.array([b[i][0] for i in ids], float)
    d = x - y
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"n": len(ids), "a": x.mean(), "b": y.mean(), "d": d.mean(), "lo": lo, "hi": hi}


def on_chat_section(by_base: dict[str, list[dict]], sets: dict[str, set[str]]) -> list[str]:
    if not by_base:
        return []
    L = ["", "## CPT adapters on the chat model (SFT conflict check)", "",
         "The CPT adapters are trained on the `*-Base` checkpoint but are used on the chat (SFT) model. Here each "
         "adapter is applied to the chat model and compared with the chat model alone on 0-shot chat-format MMLU, "
         "the way the tutor is asked questions. A negative Δ means the adapter hurts what SFT built.", "",
         "- **no thinking, generated answer** (the reference): chat template with thinking off, greedy reply that "
         "ends with \"Answer: X\", letter parsed from the text (`scripts/eval_model.py --mmlu-scoring generate`, "
         "simple-evals style). *Parse rate* = replies with a readable letter.",
         "- **no thinking, first-token letter** (not reliable, kept for history): answer letter read from the "
         "probability of the reply's first token. A chat model rarely starts its reply with the bare letter "
         "(*letter mass* well below 1), so this mostly measures whether the adapter makes it do so, not knowledge: "
         "it showed the r64_300mb adapter as +4.4 MMLU points where the generated answer gives −0.8.",
         "- **thinking**: Qwen3 thinking mode, the model reasons and then answers; the letter is parsed from the "
         "answer, or read from the logits after closing the reasoning when it ran out of tokens or gave no letter "
         "(*forced*). Sampled (temperature 0.6, top-p 0.95, seeded), so it carries sampling noise as well "
         "(`scripts/eval_mmlu_thinking.py`). Questions run in a fixed random order, so a partial run is a random "
         "sample and all rows are compared on the questions they have in common.",
         "- Δ = adapter minus the chat model alone on the same questions, with its paired 95 % bootstrap interval."]
    for base, rows in by_base.items():
        ref = next((r for r in rows if r["adapter"] == "-"), None)
        for mode, title in (("gen", "no thinking, generated answer"), ("nothink", "no thinking, first-token letter"),
                            ("think", "thinking")):
            have = [r for r in rows if mode in r["runs"]]
            if not have:
                continue
            L += ["", f"### {base}, {title}", ""]
            if mode == "nothink":
                hdr = "| model | n | letter mass |"
            elif mode == "gen":
                hdr = "| model | n | parse rate |"
            else:
                hdr = "| model | n | forced | think tokens |"
            L += [hdr + "".join(f" {lab} |" for _, lab, _ in CHAT_COLS) + " Δ MMLU [95 %] | Δ AI subj. [95 %] |",
                  "|---|---:|" + ("---:|---:|" if mode == "think" else "---:|") + "---:|" * len(CHAT_COLS) + "---:|---:|"]
            refp = ref["runs"][mode]["preds"] if ref and mode in ref["runs"] else None
            for r in have:
                run = r["runs"][mode]
                name = f"{base} alone" if r["adapter"] == "-" else f"+ {r['adapter']}"
                ex = run["extra"]
                n = len(run["preds"])
                total = ex.get("n_total")
                nstr = f"{n:,}" + (f" of {total:,}" if total and n < total else "")
                lead = (f" {nstr} | {fmt(ex.get('letter_prob_mass'))} |" if mode == "nothink" else
                        f" {nstr} | {fmt(ex.get('parse_rate'))} |" if mode == "gen" else
                        f" {nstr} | {fmt(ex.get('forced_rate'))} | {ex.get('mean_think_tokens') or ''} |")
                common = run["preds"] if refp is None else {i: v for i, v in run["preds"].items() if i in refp}
                cells = []
                for _, _, f in CHAT_COLS:
                    xs = [v[0] for i, v in common.items() if f({"id": i, "subject": v[1]}, sets)]
                    cells.append(fmt(sum(xs) / len(xs)) if xs else "")
                deltas = []
                for col in (0, 1):
                    if refp is None or r["adapter"] == "-":
                        deltas.append("")
                        continue
                    pr = _paired(run["preds"], refp, CHAT_COLS[col][2], sets)
                    deltas.append("" if pr is None else f"{pr['d']:+.4f} [{pr['lo']:+.4f}, {pr['hi']:+.4f}]")
                L.append(f"| {name} |{lead} " + " | ".join(cells) + " | " + " | ".join(deltas) + " |")
            if refp is not None and any(len(r["runs"][mode]["preds"]) != len(refp) for r in have):
                L.append("")
                L.append("Accuracies in this table are on the questions every row has answered.")
    return L


def write_report(path: str, rows: list[dict], excluded: list[tuple[str, str]], ns: dict[str, int],
                 results_root: str, split: str, sets: dict[str, set[str]]) -> tuple[list[dict], list[dict]]:
    complete = sorted((r for r in rows if combined(r)[1] == len(SCORES)), key=lambda r: -combined(r)[0])
    for r in complete:
        r["halfwidth"] = bootstrap_halfwidth(r, sets)
    partial = sorted((r for r in rows if combined(r)[1] < len(SCORES)), key=lambda r: -(combined(r)[0] or 0))
    base_score = {r["base"]: combined(r)[0] for r in complete if r["adapter"] == "-"}
    rel = os.path.relpath(results_root, ROOT)

    L = ["# Model leaderboard: AI-related accuracy", "",
         f"Generated {time.strftime('%F %T')} by `scripts/score_report.py` from the saved evaluations in `{rel}/` "
         f"(MMLU `{split}` split). No model was run; re-run the script after new evaluations or a new `ai_related` "
         "selection.", "",
         "## Scores", "", "| score | what | n |", "|---|---|---:|"]
    L += [f"| {label} | {WHAT[k]} | {ns.get(k, 0):,} |" for k, label in SCORES]
    L.append(f"| Open QA | ROUGE-L, token-F1 and level-fit of generated answers to open questions, with the adapter "
             f"applied to the **chat** model (see the Open QA section); not part of the combined score | {ns.get('open_qa', 0):,} |")
    L += ["", "**score** = plain mean of the six accuracies. A run missing any of them is not ranked; it is listed "
          "below the ranking with the mean of the scores it has. **±** = 95 % bootstrap half-width of the score from "
          "resampling the MMLU and MCQ questions (1,000 replicates): test-set sampling noise. Re-running an "
          "evaluation is deterministic (see Repeatability), so this is the only noise in a score; two runs whose "
          "intervals overlap are not reliably ordered. **AI_Large** = accuracy on the 1,000-question `ai_large` subject "
          "(see its section); shown for reference, not part of the score.", "",
          "Included runs: 5-shot, base-prompt-format evaluations of a pre-trained `*-Base` checkpoint, alone or with a "
          "LoRA adapter trained on it. 0-shot, chat-format and chat-model runs are not comparable and are left out. "
          "Identical adapter-less evaluations of one checkpoint are merged into a single row. Δ vs base = score minus "
          "the score of the same checkpoint without an adapter.", "",
          "## Ranking", "",
          "| # | run | base model | adapter | " + " | ".join(label for _, label in SCORES) + " | **score** | ± | Δ vs base "
          "| AI_Large |",
          "|---:|---|---|---|" + "---:|" * len(SCORES) + "---:|---:|---:|---:|"]
    for i, r in enumerate(complete, 1):
        sc, _ = combined(r)
        d = sc - base_score[r["base"]] if r["base"] in base_score else None
        cells = ([str(i), r["run"], r["base"], r["adapter"]] + [fmt(r["scores"][k]) for k, _ in SCORES]
                 + [f"**{sc:.4f}**", fmt(r.get("halfwidth")), "" if d is None else f"{d:+.4f}", fmt(large_acc(r))])
        L.append("| " + " | ".join(cells) + " |")
    if not complete:
        L.append("| | no run has all six scores | | | " + " | " * len(SCORES) + " | | | |")
    merged = [r for r in rows if r["aliases"]]
    if merged:
        L += ["", "Merged rows: " + "; ".join(f"`{r['run']}` = " + " = ".join(f"`{a}`" for a in r["aliases"])
                                            + f" (same {r['base']} evaluation)" for r in merged) + "."]

    if partial:
        L += ["", "## Runs missing a score (not ranked)", "",
              "| run | base model | adapter | " + " | ".join(label for _, label in SCORES) + " | partial mean | of |",
              "|---|---|---|" + "---:|" * len(SCORES) + "---:|---:|"]
        for r in partial:
            sc, k = combined(r)
            L.append("| " + " | ".join([r["run"], r["base"], r["adapter"]] + [fmt(r["scores"][x]) for x, _ in SCORES]
                                        + [fmt(sc), f"{k}/{len(SCORES)}"]) + " |")
        missing = sorted({label for r in partial for k, label in SCORES if r["scores"][k] is None})
        L += ["", f"Missing: {', '.join(missing)}. The partial means are not comparable with the ranking above "
              "(each score has its own level), so these rows are only informative. Evaluate the missing task with "
              "`scripts/eval_model.py` (5-shot, same adapter) and re-run this script to rank them."]

    L += large_section(complete, partial)

    with_oq = [r for r in complete if r["open_qa"]]
    if with_oq or any(r.get("open_qa_note") for r in complete):
        chat_oq = {r["base"]: r["open_qa"] for r in complete if r["adapter"] == "-" and r["open_qa"]}
        L += ["", "## Open QA (adapter applied to the chat model)", "",
              "Greedy answers (768 new tokens, thinking off) to open questions, scored against references: ROUGE-L, "
              "token-F1, and level-fit (share of answers whose reading grade fits the question's audience level). The "
              "CPT adapters are trained on the `*-Base` checkpoint but are meant to be used on the chat model, so for "
              "this task each adapter is applied to `Qwen3-0.6B` (`Qwen3-1.7b` for the 1.7B row) and the no-adapter "
              "rows are the chat model alone. Δ = token-F1 minus the chat model alone. Rows in ranking order; this "
              "task is not part of the combined score.", "",
              "| # | run | chat model | ROUGE-L | token-F1 | level-fit | Δ token-F1 |", "|---:|---|---|---:|---:|---:|---:|"]
        for i, r in enumerate(complete, 1):
            o = r["open_qa"]
            if o:
                ref = chat_oq.get(r["base"])
                d = "" if not ref or ref["token_f1"] is None or o["token_f1"] is None else f"{o['token_f1'] - ref['token_f1']:+.4f}"
                L.append(f"| {i} | {r['run']} | {o['chat_model']} | {fmt(o['rougeL'])} | {fmt(o['token_f1'])} | "
                         f"{fmt(o['level_fit'])} | {d} |")
            else:
                L.append(f"| {i} | {r['run']} | *{r.get('open_qa_note') or 'not evaluated yet'}* | | | | |")

    L += on_chat_section(collect_on_chat(results_root, sets), sets)

    repeated = [r for r in rows if r.get("repeats")]
    if repeated:
        L += ["", "## Repeatability", "",
              "The same model evaluated again with identical settings under a `<tag>_rerun<k>` tag; repeats are not "
              "ranked. Spread = max − min over the original and its repeats. *Changed answers* = questions whose "
              "predicted letter differs from the original evaluation's."]
        for r in repeated:
            reps_ = sorted(r["repeats"], key=lambda x: x["run"])
            short = [x["run"][len(r["run"]) + 1:] for x in reps_]
            L += ["", f"### {r['run']} ({r['base']}, adapter {r['adapter']}; {len(reps_)} repeat"
                  f"{'s' if len(reps_) != 1 else ''})", "",
                  "| score | original | " + " | ".join(short) + " | spread |", "|---|---:|" + "---:|" * len(reps_) + "---:|"]
            for k, label in SCORES + [("__score", "**score**")]:
                vals = [combined(x)[0] if k == "__score" else x["scores"][k] for x in [r] + reps_]
                vs = [v for v in vals if v is not None]
                spread = f"{max(vs) - min(vs):.4f}" if len(vs) > 1 else ""
                L.append(f"| {label} | " + " | ".join(fmt(v) for v in vals) + f" | {spread} |")
            flips = []
            for x, name in zip(reps_, short):
                parts = [t for t in (changed_answers(r, x, "mmlu"), changed_answers(r, x, "mcq")) if t]
                flips.append(f"{name}: " + (", ".join(parts) if parts else "no predictions to compare"))
            L += ["", "Changed answers vs the original: " + "; ".join(flips) + "."]

    halfw = [f"{label} ±{ci95(0.45, ns[k]):.3f}" for k, label in SCORES if ns.get(k)]
    L += ["", "## Reading the numbers", "",
          "- 95 % half-widths of a single accuracy near 0.45 at these sizes: " + ", ".join(halfw)
          + ". The ± column gives the same for the combined score (bootstrap over questions); the small sets (ML, "
          "MCQ, NLI-sel.) dominate it, so combined scores within about 0.03 of each other are not reliably ordered.",
          f"- LLM-sel. holds {ns.get('llm_sel', 0):,} of the 1,602 pooled math / computer-science / formal-logic / "
          "machine_learning questions: the Qwen3-1.7B judge answers *yes* to almost everything at threshold 0.5, so "
          "this score behaves like \"all of those subjects\" rather than a narrow AI subset; NLI-sel. is the narrow one.",
          "- Several rows are one adapter at different epochs; their ranks mostly tell which epoch to keep."]
    if excluded:
        L += ["", "## Excluded evaluations", ""] + [f"- `{n}`: {why}" for n, why in excluded]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return complete, partial


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default=RESULTS_ROOT)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    rows, excluded, ns, sets = collect(args.results_root, args.split)
    rows, orphans = split_reruns(rows)
    excluded += [(o, "repeat of a run that is not in the ranking") for o in orphans]
    rows = merge_duplicates(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    complete, partial = write_report(args.out, rows, excluded, ns, args.results_root, args.split, sets)

    print(f"{'#':>2s} {'run':42s} {'base':16s}" + "".join(f" {label:>13s}" for _, label in SCORES) + f" {'score':>8s} {'±':>7s}")
    for i, r in enumerate(complete, 1):
        print(f"{i:2d} {r['run']:42s} {r['base']:16s}" + "".join(f" {fmt(r['scores'][k]):>13s}" for k, _ in SCORES)
              + f" {combined(r)[0]:8.4f} {fmt(r.get('halfwidth')):>7s}")
    for r in partial:
        sc, k = combined(r)
        print(f" - {r['run']:42s} {r['base']:16s}" + "".join(f" {fmt(r['scores'][x]):>13s}" for x, _ in SCORES)
              + f" {fmt(sc):>8s} ({k}/{len(SCORES)})")
    print(f"\n{'open_qa (adapter on chat model)':44s} {'chat':12s} {'ROUGE-L':>8s} {'tok-F1':>8s} {'lvl-fit':>8s}")
    for r in complete:
        o = r["open_qa"]
        print(f"{r['run']:44s} " + (f"{o['chat_model']:12s} {fmt(o['rougeL']):>8s} {fmt(o['token_f1']):>8s} {fmt(o['level_fit']):>8s}"
                                     if o else f"-- {r.get('open_qa_note') or 'not evaluated yet'}"))
    for r in rows:
        for x in sorted(r.get("repeats", []), key=lambda x: x["run"]):
            d = {label: (x["scores"][k] or 0) - (r["scores"][k] or 0) for k, label in SCORES if x["scores"][k] is not None and r["scores"][k] is not None}
            print(f" = {x['run']:42s} repeat of {r['run']}: " + ", ".join(f"{k} {v:+.4f}" for k, v in d.items())
                  + f"; {changed_answers(r, x, 'mmlu')}; {changed_answers(r, x, 'mcq')}")
    print(f"\n{len(complete)} ranked, {len(partial)} partial, {len(excluded)} excluded -> {args.out}")


if __name__ == "__main__":
    main()
