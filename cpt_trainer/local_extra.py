"""Extra CPT data (STEM, math, reasoning) from the SFT pipeline's raw downloads in data/raw, as plain text
documents, one parquet per source in cpt_data/extra/ (columns id, text), sized by exact Qwen3 token counts.

    python -m cpt_trainer.local_extra                  # every group below
    python -m cpt_trainer.local_extra --only math      # one group

Rows that the project uses for evaluation, validation or RL are never used: their ids (source/N) are read
from data/final/eval/*.jsonl, data/final/sft_val.jsonl and data/final/rl_*.jsonl and skipped. Afterwards,
run cpt_trainer.decontaminate over all extra parquets (13-gram overlap with MMLU / eval / val texts).

Document format (CPT is plain next-token prediction, no chat template):
    Problem:\\n<question>\\n\\nReasoning:\\n<reasoning trace>\\n\\nSolution:\\n<answer>      (rows with a trace)
    Question:\\n<question>\\n\\nAnswer:\\n<answer>                                          (Q&A rows)
    <answer>                                                                                 (Cosmopedia textbook text)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq

from cpt_trainer.fetch_extra import OUT_DIR, ROOT, TOKENIZER, _take, cut

RAW = os.path.join(ROOT, "data", "raw")

# group -> [(output name, [raw files], style, budget M tokens)]
GROUPS = {
    "reasoning": [
        ("raw_openthoughts", ["openthoughts.jsonl", "openthoughts.replay.jsonl"], "trace", 20),
        ("raw_openscience_reasoning", ["openscience_reasoning2.jsonl", "openscience_reasoning2.replay.jsonl"], "trace", 15),
    ],
    "math": [
        ("raw_nemotron_math", ["nemotron_math.jsonl", "nemotron_math.replay.jsonl"], "trace", 15),
        ("raw_deepmath", ["deepmath_103k.jsonl", "deepmath_103k.replay.jsonl"], "trace", 10),
        ("raw_numina_cot", ["numina_cot.jsonl"], "qa", 10),
        ("raw_openr1_math", ["openr1_math.jsonl"], "trace", 5),
    ],
    "stem": [
        ("raw_cosmo_textbooks", ["cosmo_v2_smollm.jsonl", "cosmo_stanford.jsonl", "cosmo_web_v2.jsonl"], "text", 15),
        ("raw_science_qa", ["megascience.jsonl", "nemotron_science.jsonl", "nemotron_science.replay.jsonl",
                            "nemotron_v1_stem.jsonl", "nemotron_v1_stem.replay.jsonl", "webinstruct_verified.jsonl"], "trace", 15),
        ("raw_stackexchange", ["se_stats.jsonl", "se_cs.jsonl", "se_cstheory.jsonl", "se_datascience.jsonl",
                               "se_philosophy.jsonl"], "qa", 10),
    ],
}


def excluded_ids() -> set[str]:
    ids = set()
    files = glob.glob(os.path.join(ROOT, "data", "final", "eval", "*.jsonl")) + \
        [os.path.join(ROOT, "data", "final", "sft_val.jsonl")] + glob.glob(os.path.join(ROOT, "data", "final", "rl_*.jsonl"))
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    i = json.loads(line).get("id")
                    if i:
                        ids.add(str(i))
    return ids


def render(r: dict, style: str) -> str | None:
    q = (r.get("question") or "").strip()
    a = (r.get("answer") or "").strip()
    t = (r.get("reasoning") or "").strip()
    gt = (r.get("ground_truth") or "").strip() if isinstance(r.get("ground_truth"), str) else ""
    if style == "text":
        return a or None
    if style == "trace" and t:
        sol = a or (f"The final answer is {gt}." if gt else "")
        return f"Problem:\n{q}\n\nReasoning:\n{t}" + (f"\n\nSolution:\n{sol}" if sol else "") if q else None
    if q and (a or gt):
        return f"Question:\n{q}\n\nAnswer:\n{a or gt}"
    return None


def build(name: str, files: list[str], style: str, budget_m: float, tok, skip: set[str], seed: int) -> dict:
    rows = []
    for fn in files:
        p = os.path.join(RAW, fn)
        if not os.path.exists(p):
            print(f"[{name}] missing {fn}, skipped")
            continue
        with open(p, encoding="utf-8") as fh:
            rows += [line for line in fh if line.strip()]
    random.Random(f"{seed}-{name}").shuffle(rows)
    budget = int(budget_m * 1e6)
    ids, texts, n_tok, n_excl, n_empty, seen, batch = [], [], 0, 0, 0, set(), []
    for line in rows:
        r = json.loads(line)
        if str(r.get("id")) in skip:
            n_excl += 1
            continue
        text = render(r, style)
        if not text or len(text) < 300:
            n_empty += 1
            continue
        text = cut(text, 60_000)
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        batch.append((str(r.get("id") or h[:24]), text))
        if len(batch) >= 256:
            n_tok += _take(batch, tok, ids, texts, budget - n_tok)
            batch = []
            if n_tok >= budget:
                break
    if batch and n_tok < budget:
        n_tok += _take(batch, tok, ids, texts, budget - n_tok)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{name}.parquet")
    pq.write_table(pa.table({"id": ids, "text": texts}), out, compression="zstd")
    st = {"name": name, "files": files, "style": style, "docs": len(texts), "tokens": n_tok, "budget": budget,
          "rows": len(rows), "excluded_eval_val_rl": n_excl, "too_short": n_empty, "out": os.path.relpath(out, ROOT)}
    with open(os.path.join(OUT_DIR, f"stats_{name}.json"), "w") as f:
        json.dump(st, f, indent=2)
    print(f"[{name}] {len(texts):,} docs, {n_tok / 1e6:.1f}/{budget / 1e6:.0f} M tokens from {len(rows):,} rows "
          f"({n_excl:,} eval/val/RL rows excluded) -> {out}", flush=True)
    return st


def build_local_long(tok, chunk_chars: int = 128_000, min_chars: int = 30_000) -> dict:
    """Real long documents already on disk: the AI textbooks, survey papers and classic papers of corpus/ai
    and the books of corpus/replay (both from scripts/download_corpus.py), cut into consecutive ~32 k-token
    chunks at paragraph breaks (whole books stay in, one chunk per training sequence)."""
    ids, texts = [], []
    for corpus in ("ai", "replay"):
        mpath = os.path.join(ROOT, "corpus", corpus, "manifest.jsonl")
        if not os.path.exists(mpath):
            continue
        with open(mpath, encoding="utf-8") as fh:
            man = [json.loads(line) for line in fh if line.strip()]
        for d in man:
            if d.get("low_quality") or d.get("kind") not in ("book", "paper", "lecture_notes"):
                continue
            p = os.path.join(ROOT, "corpus", corpus, d["text_path"])
            if not os.path.isfile(p):
                continue
            with open(p, encoding="utf-8", errors="replace") as fh:
                text = fh.read().strip()
            if len(text) < min_chars:
                continue
            k = 0
            while text:
                piece = cut(text, chunk_chars)
                text = text[len(piece):].lstrip()
                if len(piece) >= 2000:
                    ids.append(f"{d['id']}#{k}")
                    texts.append(piece)
                k += 1
    n_tok = sum(len(x) + 1 for x in tok(texts, add_special_tokens=False)["input_ids"]) if texts else 0
    out = os.path.join(OUT_DIR, "local_long.parquet")
    pq.write_table(pa.table({"id": ids, "text": texts}), out, compression="zstd")
    st = {"name": "local_long", "docs": len(texts), "tokens": n_tok, "out": os.path.relpath(out, ROOT)}
    with open(os.path.join(OUT_DIR, "stats_local_long.json"), "w") as f:
        json.dump(st, f, indent=2)
    print(f"[local_long] {len(texts):,} chunks, {n_tok / 1e6:.1f} M tokens -> {out}", flush=True)
    return st


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma list of groups: " + ",".join(GROUPS) + ",long_local")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every token budget (e.g. to fill a shortfall)")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    tok.model_max_length = 10 ** 12
    skip = excluded_ids()
    print(f"{len(skip):,} eval / val / RL ids excluded")
    want = {g.strip() for g in args.only.split(",") if g.strip()} or set(GROUPS) | {"long_local"}
    if "long_local" in want:
        build_local_long(tok)
    for g in GROUPS:
        if g in want:
            for name, files, style, budget in GROUPS[g]:
                build(name, files, style, budget * args.scale, tok, skip, args.seed)


if __name__ == "__main__":
    main()
