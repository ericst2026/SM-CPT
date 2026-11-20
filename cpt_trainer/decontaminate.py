"""Drop CPT documents that overlap evaluation / validation texts (13-gram overlap, like the GPT-3 / Llama
decontamination): any document that shares a 13-word sequence with an MMLU test question (all 57 subjects,
question + options), an MMLU-Pro / AI-MCQ / open-QA / DS-1000 eval item, or an SFT validation conversation
is removed.

    python -m cpt_trainer.decontaminate cpt_data/extra/*.parquet --out cpt_data/extra_clean

Boilerplate is not contamination, so two kinds of 13-grams are ignored: those in 3 or more different
evaluation items, and those in more than --max-train-df of the scanned training documents (prompt templates
like "Make sure to put the answer (and only answer) inside \\boxed{}" followed by the first word of a
problem, licence headers, ...). A copied evaluation item shares 13-grams that are rare on both sides.
Writes <out>/<name>.parquet (same columns) and <out>/decontamination.json with per-file counts and examples.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 13
WORD = re.compile(r"\w+")


def grams(text: str) -> set[int]:
    w = WORD.findall(text.lower())
    return {hash(" ".join(w[i:i + N])) for i in range(len(w) - N + 1)}


def eval_texts() -> list[tuple[str, str]]:
    """(item id, text) of every evaluation / validation item."""
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "eval", "mmlu", "*", "test-*.parquet"))):
        subj = os.path.basename(os.path.dirname(f))
        for i, r in enumerate(pq.read_table(f, columns=["question", "choices"]).to_pylist()):
            out.append((f"mmlu/{subj}/{i}", r["question"] + "\n" + "\n".join(r["choices"])))
    for f in glob.glob(os.path.join(ROOT, "data", "split", "ai_eval", "*.jsonl")) + \
            glob.glob(os.path.join(ROOT, "data", "final", "eval", "*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                parts = [r.get(k) for k in ("question", "reference", "answer") if isinstance(r.get(k), str)]
                out.append((f"{os.path.basename(f)}:{r.get('id')}", "\n".join(parts)))
    with open(os.path.join(ROOT, "data", "final", "sft_val.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out.append((f"sft_val:{r.get('id')}", "\n".join(m.get("content") or "" for m in r["messages"]
                                                            if m.get("role") != "system")))
    return out


_EVAL: dict[int, str] = {}


def _init(eval_grams):
    global _EVAL
    _EVAL = eval_grams


def _scan(args):
    """(row, matching eval 13-gram hashes) for every document of a slice with any match."""
    path, lo, hi = args
    tbl = pq.read_table(path, columns=["text"]).slice(lo, hi - lo)
    hits = []
    for k, t in enumerate(tbl.column("text").to_pylist()):
        common = grams(t) & _EVAL.keys()
        if common:
            hits.append((lo + k, sorted(common)))
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquets", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-train-df", type=int, default=5, help="ignore 13-grams found in more training documents")
    args = ap.parse_args()

    items = eval_texts()
    owners: dict[int, set] = defaultdict(set)
    for iid, text in items:
        for g in grams(text):
            if len(owners[g]) < 3:
                owners[g].add(iid)
    eval_grams = {g: next(iter(o)) for g, o in owners.items() if len(o) < 3}
    print(f"{len(items):,} eval/val items -> {len(eval_grams):,} distinct 13-grams "
          f"({len(owners) - len(eval_grams):,} boilerplate 13-grams ignored)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    all_hits, df = {}, defaultdict(int)
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(eval_grams,)) as ex:
        for path in args.parquets:
            n = pq.ParquetFile(path).metadata.num_rows
            step = max(1, -(-n // (args.workers * 4)))
            all_hits[path] = (n, [h for part in ex.map(_scan, [(path, lo, min(n, lo + step)) for lo in range(0, n, step)])
                                  for h in part])
            for _, gs in all_hits[path][1]:
                for g in gs:
                    df[g] += 1
    rare = {g for g, c in df.items() if c <= args.max_train_df}
    print(f"{len(df):,} eval 13-grams occur in the training documents; {len(rare):,} of them in <= {args.max_train_df} "
          f"documents (the rest is boilerplate)", flush=True)
    report = {}
    for path, (n, hits) in all_hits.items():
        bad = [(r, [g for g in gs if g in rare]) for r, gs in hits]
        bad = [(r, gs) for r, gs in bad if gs]
        drop = {r for r, _ in bad}
        tbl = pq.read_table(path)
        # > 2 GB of text overflows the 32-bit offsets of `string` when take() concatenates the chunks
        tbl = tbl.cast(pa.schema([f.with_type(pa.large_string()) if f.type == pa.string() else f for f in tbl.schema]))
        name = os.path.basename(path)
        pq.write_table(tbl.take([i for i in range(n) if i not in drop]), os.path.join(args.out, name), compression="zstd")
        by_kind = defaultdict(int)
        for _, gs in bad:
            by_kind[eval_grams[gs[0]].split(":")[0].split("/")[0]] += 1
        report[name] = {"docs": n, "dropped": len(drop), "by_eval_set": dict(by_kind),
                        "examples": [{"row": r, "eval_item": eval_grams[gs[0]], "shared_rare_13grams": len(gs)}
                                     for r, gs in bad[:10]]}
        print(f"{name:36s} {n:8,d} docs, dropped {len(drop):6,d} ({100 * len(drop) / max(1, n):.2f} %) {dict(by_kind)}",
              flush=True)
    with open(os.path.join(args.out, "decontamination.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
