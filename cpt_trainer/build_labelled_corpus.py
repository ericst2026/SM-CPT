#!/usr/bin/env python3
"""Build a cpt_trainer corpus folder (manifest.jsonl + text/...) from the large scored parquet corpora, keeping only
documents with the given AI-relevance label(s) (cpt_data/scores/<corpus>/..., zero-shot DeBERTa: AI / near-AI / not-AI)
and sampling them uniformly to a Qwen3 token budget.

    python -m cpt_trainer.build_labelled_corpus cpt_data/corpus_web/web_ai --labels AI --tokens 4e9 \\
        --corpus cpt_data/corpus_ai_fineweb_large --corpus cpt_data/corpus_ai_nemotron_large \\
        --exclude-subsets Translated-Diverse-QA --workers 16

How it works
  1. Planning pass: only the score files are read; documents with a wanted label (and not in an excluded subset) are
     counted per corpus. With the known mean tokens per document of each corpus/label the keep probability is
     p = tokens / (pool docs x mean tokens) x --oversample (every file is thinned by the same p, so the sample is
     uniform over the whole tree).
  2. Writing pass: files in seeded random order in a process pool. A worker joins its score file by position (ids are
     checked), keeps rows with a wanted label / allowed subset / language en / >= --min-chars, draws each with
     probability p, counts its exact Qwen3 tokens, writes text/<source>/<subset>/<file stem>/<doc id>.txt and a
     manifest part. The main process appends parts in completion order and stops dispatching once the exact token
     total reaches the budget (overshoot <= the files in flight).
  Manifest fields follow cpt_trainer/parquet_to_corpus.py (id, source, text_path, bytes, low_quality=False, meta with
  p_ai / label / subset / dump / kw_core ...), plus qwen_tokens. Writes build_stats.json and report.md.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cpt_trainer.parquet_to_corpus import LICENSES, doc_id  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# mean Qwen3 tokens per document by corpus / label (cpt_data/scores/summary.txt, 2026-11-02); used only for planning
MEAN_TOKENS = {"fineweb_edu": {"AI": 999, "near-AI": 968, "not-AI": 1086},
               "nemotron_cc": {"AI": 702, "near-AI": 723, "not-AI": 808}}
_tok = None
_args = None


def source_name(corpus_dir: str) -> str:
    b = os.path.basename(os.path.normpath(corpus_dir))
    return "fineweb_edu" if "fineweb" in b else "nemotron_cc" if "nemotron" in b else b


def list_files(corpus_dir: str) -> list[str]:
    return sorted(os.path.join(dp, f) for dp, _, fs in os.walk(corpus_dir) if "/.git" not in dp for f in fs if f.endswith(".parquet"))


def score_path(corpus_dir: str, f: str) -> str:
    return os.path.join(_args.scores_root, os.path.basename(os.path.normpath(corpus_dir)), os.path.relpath(f, corpus_dir))


def plan_file(job):
    corpus_dir, f = job
    sub = os.path.relpath(f, corpus_dir).split("/")[0]
    if not subset_ok(sub):
        return (corpus_dir, 0)
    lab = pq.read_table(score_path(corpus_dir, f), columns=["label"]).column("label").to_pylist()
    if _args.min_score is not None and "score" in pq.ParquetFile(f).schema_arrow.names:
        sc = pq.read_table(f, columns=["score"]).column("score").to_pylist()
        return (corpus_dir, sum((l in _args.labels) and (v is not None and v >= _args.min_score) for l, v in zip(lab, sc)))
    return (corpus_dir, sum(l in _args.labels for l in lab))


def subset_ok(sub: str) -> bool:
    """Top-level subset dir filter: --exclude-subsets always wins; --include-subsets, when given, is a whitelist."""
    if sub in _args.exclude_subsets:
        return False
    return not _args.include_subsets or sub in _args.include_subsets


def init_worker(args):
    global _tok, _args
    _args = args
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(args.tokenizer).backend_tokenizer


def part_path(fi: int) -> str:
    p = os.path.join(_args.out_dir, "_parts", f"{fi:05d}.jsonl")
    if not os.path.exists(p):
        open(p, "w").close()
    return p


def write_file(job):
    """One source parquet -> text files + a manifest part. Returns stats."""
    fi, corpus_dir, f, p = job
    src = source_name(corpus_dir)
    rel = os.path.relpath(f, corpus_dir)
    sub = rel.split("/")[0]
    stem = os.path.splitext(os.path.basename(rel))[0]
    names = pq.ParquetFile(f).schema_arrow.names
    cols = [c for c in ("id", "text", "url", "dump", "language", "token_count", "score", "int_score", "kw_core", "kw_math", "kw_code", "metadata") if c in names]
    if not subset_ok(sub):
        return {"fi": fi, "file": rel, "source": src, "subset": sub, "rows": 0, "docs": 0, "chars": 0, "tokens": 0,
                "part": part_path(fi)}
    tbl = pq.read_table(f, columns=cols)
    sc = pq.read_table(score_path(corpus_dir, f), columns=["id", "p_ai", "label"])
    assert tbl.num_rows == sc.num_rows and tbl.column("id").to_pylist() == sc.column("id").to_pylist(), f"id mismatch {rel}"
    labels, p_ai = sc.column("label").to_pylist(), sc.column("p_ai").to_pylist()
    rng = random.Random(_args.seed * 1000003 + fi)
    text_dir = os.path.join(_args.out_dir, "text", src, sub, stem)
    os.makedirs(text_dir, exist_ok=True)
    part = part_path(fi)
    kept = chars = tokens = 0
    rows = tbl.to_pylist()
    picked = []
    for i, row in enumerate(rows):
        if labels[i] not in _args.labels:
            continue
        if row.get("language") not in (None, "en"):
            continue
        if _args.min_score is not None and "score" in row and (row["score"] is None or row["score"] < _args.min_score):
            continue
        text = (row.get("text") or "").strip()
        if len(text) < _args.min_chars:
            continue
        if rng.random() >= p:
            continue
        picked.append((i, row, text))
    enc = _tok.encode_batch([t for _, _, t in picked], add_special_tokens=False) if picked else []
    seen = set()
    with open(part, "w", encoding="utf-8") as mf:
        for (i, row, text), e in zip(picked, enc):
            did = doc_id(row, text)
            if did in seen:
                did += "_" + str(i)
            seen.add(did)
            trel = f"text/{src}/{sub}/{stem}/{did}.txt"
            with open(os.path.join(_args.out_dir, trel), "w", encoding="utf-8") as tf:
                tf.write(text + "\n")
            meta = {k: row[k] for k in ("dump", "score", "int_score", "kw_core", "kw_math", "kw_code") if row.get(k) is not None}
            meta.update({"p_ai": round(float(p_ai[i]), 4), "label": labels[i], "subset": sub})
            if isinstance(row.get("metadata"), dict):
                meta.update({k: v for k, v in row["metadata"].items() if v is not None})
            n_tok = len(e.ids) + 1   # + EOS, as cpt_trainer.data adds one per document
            mf.write(json.dumps({"id": f"{src}:{did}", "kind": "web", "title": row.get("url"), "url": row.get("url"),
                                 "license": LICENSES.get(src, "see source dataset card"), "source": src,
                                 "parquet": os.path.relpath(f, _args.out_dir), "text_path": trel,
                                 "bytes": len(text.encode("utf-8")) + 1, "text_chars": len(text),
                                 "text_words": len(text.split()), "qwen_tokens": n_tok, "gpt2_tokens": row.get("token_count"),
                                 "low_quality": False, "meta": meta}, ensure_ascii=False) + "\n")
            kept += 1; chars += len(text); tokens += n_tok
    return {"fi": fi, "file": rel, "source": src, "subset": sub, "rows": tbl.num_rows, "docs": kept, "chars": chars,
            "tokens": tokens, "part": part}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="corpus folder to create (its basename is the corpus name in the trainer)")
    ap.add_argument("--corpus", action="append", required=True, help="scored parquet tree (repeatable)")
    ap.add_argument("--scores-root", default=os.path.join(ROOT, "cpt_data", "scores"))
    ap.add_argument("--labels", default="AI", help="comma list of labels to keep: AI, near-AI, not-AI")
    ap.add_argument("--tokens", type=float, required=True, help="Qwen3 token budget (EOS per document included)")
    ap.add_argument("--exclude-subsets", default="Translated-Diverse-QA", help="comma list of top-level subset dirs to skip")
    ap.add_argument("--include-subsets", default="", help="comma list of top-level subset dirs to use (whitelist; empty = all)")
    ap.add_argument("--min-score", type=float, default=None,
                    help="keep only rows whose parquet `score` column (FineWeb-Edu educational score) is >= this; rows of corpora without that column pass")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--oversample", type=float, default=1.03, help="keep probability safety factor (the exact token count stops the build)")
    ap.add_argument("--limit-files", type=int, default=None, help="smoke tests: only the first N files per corpus")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "models", "Qwen3-0.6B-Base"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.labels = set(args.labels.split(","))
    args.exclude_subsets = set(s for s in args.exclude_subsets.split(",") if s)
    args.include_subsets = set(s for s in args.include_subsets.split(",") if s)
    args.out_dir = os.path.abspath(args.out_dir)
    global _args
    _args = args
    manifest_path = os.path.join(args.out_dir, "manifest.jsonl")
    if os.path.exists(manifest_path):
        raise SystemExit(f"{manifest_path} exists; remove the folder first")
    os.makedirs(os.path.join(args.out_dir, "_parts"), exist_ok=True)

    t0 = time.time()
    jobs = []
    for c in args.corpus:
        fs = [f for f in list_files(c) if subset_ok(os.path.relpath(f, c).split("/")[0])]
        jobs += [(c, f) for f in (fs[:args.limit_files] if args.limit_files else fs)]
    with Pool(args.workers) as pool:
        counts = pool.map(plan_file, jobs, chunksize=4)
    pool_docs, pool_tokens = Counter(), 0.0
    for (c, n) in counts:
        pool_docs[c] += n
    for c, n in pool_docs.items():
        mt = MEAN_TOKENS.get(source_name(c), {})
        pool_tokens += n * (sum(mt.get(l, 900) for l in args.labels) / max(1, len(args.labels)))
    p = min(1.0, args.tokens / max(1.0, pool_tokens) * args.oversample)
    print(f"plan: {len(jobs)} files, labels {sorted(args.labels)}, pool {sum(pool_docs.values()):,} docs "
          f"(~{pool_tokens / 1e9:.2f} B tokens est.) -> keep probability {p:.4f} for a {args.tokens / 1e9:.2f} B budget "
          f"({time.time() - t0:.0f}s)", flush=True)
    if p >= 1.0:
        print("WARNING: the pool is smaller than the budget; every labelled document will be used", flush=True)

    order = list(range(len(jobs)))
    random.Random(args.seed).shuffle(order)
    wjobs = [(fi, jobs[fi][0], jobs[fi][1], p) for fi in order]
    stats, total_tokens, total_docs = [], 0, 0
    t1 = time.time()
    with Pool(args.workers, initializer=init_worker, initargs=(args,)) as pool, open(manifest_path, "w", encoding="utf-8") as mf:
        it = pool.imap_unordered(write_file, wjobs)
        for k, st in enumerate(it):
            with open(st["part"], encoding="utf-8") as pf:
                mf.write(pf.read())
            os.remove(st["part"])
            stats.append(st); total_tokens += st["tokens"]; total_docs += st["docs"]
            if (k + 1) % 20 == 0 or total_tokens >= args.tokens:
                el = time.time() - t1
                print(f"  [{k + 1}/{len(wjobs)}] {total_docs:,} docs, {total_tokens / 1e9:.3f} B tokens, "
                      f"{total_tokens / el / 1e6:.1f} M tok/s, ETA {(min(args.tokens, pool_tokens) - total_tokens) / max(total_tokens / el, 1) / 60:.0f} min", flush=True)
            if total_tokens >= args.tokens:
                pool.terminate()
                break
    for extra in os.listdir(os.path.join(args.out_dir, "_parts")):   # parts of terminated workers: drop their text files
        pth = os.path.join(args.out_dir, "_parts", extra)
        with open(pth, encoding="utf-8") as pf:
            for line in pf:
                try:
                    os.remove(os.path.join(args.out_dir, json.loads(line)["text_path"]))
                except Exception:
                    pass
        os.remove(pth)
    os.rmdir(os.path.join(args.out_dir, "_parts"))

    by_src, by_sub = Counter(), Counter()
    for st in stats:
        by_src[st["source"]] += st["tokens"]; by_sub[(st["source"], st["subset"])] += st["tokens"]
    name = os.path.basename(os.path.normpath(args.out_dir))
    lines = [f"# Corpus `{name}`", "",
             f"- documents: **{total_docs:,}**  |  Qwen3 tokens (incl. EOS): **{total_tokens / 1e9:.3f} B**  |  labels {sorted(args.labels)}  |  "
             f"excluded subsets {sorted(args.exclude_subsets)}  |  included subsets {sorted(args.include_subsets) or 'all'}  |  min score {args.min_score}  |  keep probability {p:.4f}  |  {len(stats)} of {len(jobs)} files  |  "
             f"built {time.strftime('%Y-%m-%d %H:%M')} by cpt_trainer/build_labelled_corpus.py (seed {args.seed})", "",
             "| source | subset | B tokens | share |", "|---|---|---:|---:|"]
    lines += [f"| {s} | {sub} | {t / 1e9:.3f} | {100 * t / max(1, total_tokens):.1f} % |" for (s, sub), t in sorted(by_sub.items())]
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out_dir, "build_stats.json"), "w") as f:
        json.dump({"args": {k: (sorted(v) if isinstance(v, set) else v) for k, v in vars(args).items()},
                   "keep_probability": p, "pool_docs": {source_name(c): n for c, n in pool_docs.items()},
                   "pool_tokens_est": pool_tokens, "docs": total_docs, "tokens": total_tokens,
                   "by_source": dict(by_src), "files": stats}, f, indent=1)
    print("\n".join(lines))
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
