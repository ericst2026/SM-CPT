"""MMLU / ai_large 5-shot scoring with the answer-letter preference removed.

The standard loglik score (scripts/eval_model.py) picks the letter with the highest next-token probability. When the
model is unsure, its current preference for a letter decides the answer, and that preference drifts between
checkpoints by tens of answers per 1,000 questions (it moved ai_large by ~2 pts in cpt_qwen06b_rs_adaptive_hq2b).

Here every question is asked n times (n = number of options), each time with the options rotated by one place, so
every option appears once under every letter. For each rotation the log-probabilities of the n letter tokens are
kept (normalised over the letters). Scores reported per subject group:

  std      rotation 0, letter from the bf16 logits = the standard eval_model.py score (reproduces saved results)
  orig     rotation 0, letter from float32 letter logits (no bf16 ties, which eval_model.py breaks towards "A")
  cyclic   each option's log-prob averaged over the n rotations, argmax -> the letter preference cancels out
           (the "permutation debiasing" of Zheng et al. 2023, "Large Language Models Are Not Robust Multiple Choice
           Selectors", with cyclic permutations)
  prior    rotation 0, minus the model's average letter log-prob over the group (one global letter prior per group)
  stable   fraction of questions where the same option wins in all n rotations
  strict   fraction where the right option wins in all n rotations

Prompts are exactly those of eval_model.py (base format, the dev-split shots unchanged; only the test question's
options are rotated).

    python scripts/eval_mmlu_debiased.py --model runs/<run>/epoch0.5/adapter --tag <run>_epoch0.5
    python scripts/eval_mmlu_debiased.py --model base --tag base        # the base model
    python scripts/eval_mmlu_debiased.py --compare base,<tagA>,<tagB>   # table from saved results, no model
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from eval_model import (LETTERS, MMLU_AI_RELATED, MMLU_DIR, MMLU_EXTRA_SUBJECTS, RESULTS_ROOT,  # noqa: E402
                        list_mmlu_subjects, load_ai_related, load_mmlu_split, mmlu_category, mmlu_prompt)
from ai_qa_train.common import load_model, load_tokenizer, resolve_adapter  # noqa: E402

DEFAULT_BASE = "models/Qwen3-0.6B-Base"
AI_SUBJECTS = ["machine_learning", "college_computer_science", "high_school_computer_science", "computer_security",
               "college_mathematics", "high_school_statistics", "econometrics", "formal_logic",
               "electrical_engineering"]


def rotate(r: dict, k: int) -> dict:
    n = len(r["choices"])
    ch = [r["choices"][(j + k) % n] for j in range(n)]  # letter j shows original option (j + k) % n
    return {**r, "choices": ch, "answer": (int(r["answer"]) - k) % n}


@torch.inference_mode()
def letter_logprobs(model, tok, texts: list[str], n_opts: list[int], bs: int, log=print):
    """Per prompt: (log-softmax over its n option-letter tokens " A", " B", ... at the next position, computed in
    float32 from the final hidden state; the letter eval_model.py would pick from the bf16 logits).
    The standard bf16 logits are rounded to steps of ~0.06-0.12, so two letters often tie exactly and argmax then
    takes the earlier letter; the float32 letter logits have no such ties."""
    letter_ids = [tok.encode(" " + L, add_special_tokens=False)[0] for L in LETTERS]
    W = model.get_output_embeddings().weight[letter_ids]  # [10, d] bf16
    out, bf16_pred = [None] * len(texts), [None] * len(texts)
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    tok.padding_side = "left"
    t0, last = time.time(), time.time()
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        h = model.model(**enc).last_hidden_state[:, -1, :]  # after the final norm, bf16
        lg32 = h.float() @ W.float().T
        lg16 = (h @ W.T).float()  # what the bf16 lm_head gives for these rows
        for i, a, b in zip(idx, lg32, lg16):
            n = n_opts[i]
            out[i] = a[:n].log_softmax(-1).tolist()
            bf16_pred[i] = int(b[:n].argmax())
        if time.time() - last > 30:
            last = time.time()
            done = s + len(idx)
            log(f"    {done}/{len(texts)} prompts, {done / (time.time() - t0):.1f}/s")
    tok.padding_side = "right"
    return out, bf16_pred


def score(model_path: str | None, tag: str, args, log=print):
    if model_path:
        base, adapter = resolve_adapter(model_path)
    else:
        base, adapter = os.path.abspath(DEFAULT_BASE), None
    out = os.path.join(RESULTS_ROOT, tag + "_debiased")
    os.makedirs(out, exist_ok=True)
    log(f"[{time.strftime('%H:%M:%S')}] {tag}: base={base} adapter={adapter} -> {out}")
    tok = load_tokenizer(base)
    model = load_model(base, adapter=adapter, merge=True)
    subjects = list_mmlu_subjects(args.mmlu_dir, extra=True)
    if args.subjects != "all":
        subjects = [s for s in subjects if s in args.subjects.split(",")]
    texts, n_opts, meta = [], [], []
    for subj in subjects:
        rows = load_mmlu_split(args.mmlu_dir, subj, "test")
        shots = load_mmlu_split(args.mmlu_dir, subj, "dev")[:args.shots]
        for i, r in enumerate(rows):
            n = len(r["choices"])
            meta.append({"id": f"mmlu/{subj}/test/{i}", "subject": subj, "gold": int(r["answer"]), "n": n,
                         **({"topic": r["topic"]} if r.get("topic") else {})})
            for k in range(n):
                texts.append(mmlu_prompt(rotate(r, k), subj, shots, "base"))
                n_opts.append(n)
    log(f"  {len(meta)} questions, {len(texts)} prompts ({args.shots}-shot, base format)")
    t0 = time.time()
    lps, std = letter_logprobs(model, tok, texts, n_opts, args.bs, log)
    log(f"  scored in {time.time() - t0:.0f}s")
    rows, p = [], 0
    for m in meta:
        rows.append({**m, "std": std[p], "lp": [[round(x, 5) for x in lps[p + k]] for k in range(m["n"])]})
        p += m["n"]
    with open(os.path.join(out, "predictions.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summ = summarise(rows, tag)
    summ.update(model=model_path, base=base, adapter=adapter, shots=args.shots, seconds=round(time.time() - t0))
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    del model
    torch.cuda.empty_cache()
    return summ


def option_scores(r: dict) -> tuple[list[float], list[int]]:
    """(cyclic-averaged log-prob per original option, winning original option in each rotation)."""
    n = r["n"]
    avg = [sum(r["lp"][k][(i - k) % n] for k in range(n)) / n for i in range(n)]
    wins = [(max(range(n), key=lambda j: r["lp"][k][j]) + k) % n for k in range(n)]
    return avg, wins


def group_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    n = rows[0]["n"]
    prior = [sum(r["lp"][k][j] for r in rows for k in range(r["n"])) / sum(r["n"] for r in rows) for j in range(n)]
    std = orig = cyc = pri = stable = strict = 0
    pred0 = Counter()
    for r in rows:
        g = r["gold"]
        lp0 = r["lp"][0]
        p0 = max(range(r["n"]), key=lambda j: lp0[j])
        pred0[LETTERS[p0]] += 1
        orig += p0 == g
        std += r["std"] == g
        pri += max(range(r["n"]), key=lambda j: lp0[j] - prior[j]) == g
        avg, wins = option_scores(r)
        cyc += max(range(r["n"]), key=lambda i: avg[i]) == g
        stable += len(set(wins)) == 1
        strict += all(w == g for w in wins)
    N = len(rows)
    return {"n": N, "std": round(std / N, 4), "orig": round(orig / N, 4), "cyclic": round(cyc / N, 4), "prior": round(pri / N, 4),
            "stable": round(stable / N, 4), "strict": round(strict / N, 4),
            "letter_prior": {LETTERS[j]: round(math.exp(prior[j]), 4) for j in range(n)},
            "pred_distribution_orig": dict(sorted(pred0.items()))}


def groups(rows: list[dict]) -> dict[str, list[dict]]:
    related = load_ai_related(MMLU_AI_RELATED)
    G = defaultdict(list)
    for r in rows:
        s = r["subject"]
        if s in MMLU_EXTRA_SUBJECTS:
            G[s].append(r)
            if r.get("topic"):
                G[f"{s}/{r['topic']}"].append(r)
            continue
        G["mmlu_all"].append(r)
        G[f"mmlu_{mmlu_category(s)}"].append(r)
        if s in AI_SUBJECTS:
            G["mmlu_ai_subjects"].append(r)
        if r["id"] in related:
            G["mmlu_ai_related"].append(r)
        if s == "machine_learning":
            G["machine_learning"].append(r)
    return G


def summarise(rows: list[dict], tag: str) -> dict:
    return {"tag": tag, "groups": {g: group_metrics(rs) for g, rs in sorted(groups(rows).items())}}


def load_rows(tag: str) -> dict[str, dict]:
    with open(os.path.join(RESULTS_ROOT, tag + "_debiased", "predictions.jsonl")) as f:
        return {r["id"]: r for r in map(json.loads, f)}


def paired(a: dict, b: dict, ids: list[str], method: str) -> tuple[int, int, float]:
    """(only a right, only b right, McNemar z of b - a)."""
    def ok(r):
        if method == "std":
            return r["std"] == r["gold"]
        if method == "orig":
            return max(range(r["n"]), key=lambda j: r["lp"][0][j]) == r["gold"]
        avg, _ = option_scores(r)
        return max(range(r["n"]), key=lambda i: avg[i]) == r["gold"]
    x = sum(ok(a[i]) and not ok(b[i]) for i in ids)
    y = sum(ok(b[i]) and not ok(a[i]) for i in ids)
    return x, y, (y - x) / math.sqrt(x + y) if x + y else 0.0


def compare(tags: list[str]) -> str:
    R = {t: load_rows(t) for t in tags}
    S = {t: summarise(list(R[t].values()), t)["groups"] for t in tags}
    G = groups(list(R[tags[0]].values()))
    L = ["| group | n | " + " | ".join(f"{t} std | {t} fp32 | {t} cyclic" for t in tags) + " |",
         "|---|---:|" + "---:|---:|---:|" * len(tags)]
    order = ["ai_large", "machine_learning", "mmlu_ai_subjects", "mmlu_ai_related", "mmlu_all", "mmlu_STEM",
             "mmlu_humanities", "mmlu_social_sciences", "mmlu_other"]
    order += sorted(g for g in S[tags[0]] if g.startswith("ai_large/"))
    for g in order:
        if g not in S[tags[0]]:
            continue
        L.append(f"| {g} | {S[tags[0]][g]['n']} | " +
                 " | ".join(f"{S[t][g]['std']:.4f} | {S[t][g]['orig']:.4f} | {S[t][g]['cyclic']:.4f}" for t in tags) + " |")
    L += ["", "| group | " + " | ".join(f"{t} prior-cal | {t} stable | {t} strict" for t in tags) + " |",
          "|---|" + "---:|---:|---:|" * len(tags)]
    for g in order[:5]:
        if g in S[tags[0]]:
            L.append(f"| {g} | " + " | ".join(f"{S[t][g]['prior']:.4f} | {S[t][g]['stable']:.4f} | {S[t][g]['strict']:.4f}"
                                              for t in tags) + " |")
    L += ["", "| group | " + " | ".join(f"{t} letter prior A/B/C/D | {t} orig preds A/B/C/D" for t in tags) + " |",
          "|---|" + "---|---|" * len(tags)]
    for g in ["ai_large", "machine_learning", "mmlu_all"]:
        if g in S[tags[0]]:
            L.append(f"| {g} | " + " | ".join(
                "/".join(f"{S[t][g]['letter_prior'][c]:.3f}" for c in "ABCD") + " | " +
                "/".join(str(S[t][g]['pred_distribution_orig'].get(c, 0)) for c in "ABCD") for t in tags) + " |")
    L += ["", "Paired comparison (same questions): only-A right / only-B right, z = (B - A) / sqrt(discordant)", "",
          "| group | A -> B | std | cyclic |", "|---|---|---|---|"]
    for g in ["ai_large", "machine_learning", "mmlu_ai_subjects", "mmlu_all"]:
        ids = [r["id"] for r in G[g]]
        for i, a in enumerate(tags):
            for b in tags[i + 1:]:
                o, c = paired(R[a], R[b], ids, "std"), paired(R[a], R[b], ids, "cyclic")
                L.append(f"| {g} | {a} -> {b} | {o[0]} / {o[1]}, z={o[2]:+.2f} | {c[0]} / {c[1]}, z={c[2]:+.2f} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=[],
                    help="adapter dir, or 'base' (repeatable)")
    ap.add_argument("--tag", action="append", default=[], help="result name per --model (or 'base' alone)")
    ap.add_argument("--subjects", default="all", help="all | comma list (ai_large, machine_learning, ...)")
    ap.add_argument("--mmlu-dir", default=MMLU_DIR)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--compare", default=None, help="comma list of tags: print the comparison table, no model")
    ap.add_argument("--out", default=None, help="--compare: also write the table to this file")
    args = ap.parse_args()
    if args.compare:
        table = compare(args.compare.split(","))
        print(table)
        if args.out:
            with open(args.out, "w") as f:
                f.write(table)
        return
    if len(args.model) != len(args.tag):
        raise SystemExit("give one --tag per --model (--model base = the base model)")
    for m, t in zip(args.model, args.tag):
        s = score(None if m == "base" else m, t, args)
        g = s["groups"]
        print(f"[{time.strftime('%H:%M:%S')}] {t}: " + ", ".join(
            f"{k} std {g[k]['std']:.4f} fp32 {g[k]['orig']:.4f} cyclic {g[k]['cyclic']:.4f}"
            for k in ["ai_large", "machine_learning", "mmlu_all"] if k in g), flush=True)


if __name__ == "__main__":
    main()
