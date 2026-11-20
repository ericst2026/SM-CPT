"""Parquet files -> a corpus folder (manifest.jsonl + text/<source>/<dump>/<id>.txt), taking the HIGHEST-RANKED rows
first until an exact Qwen3 token budget is reached (unlike parquet_to_corpus.py, which samples at random).

Rank = descending tuple of the --rank columns (default: FineWeb-Edu int_score, then the technical-filter outputs
tech_families and tech_density written by cpt_trainer/filter_technical.py). Exact duplicate texts are skipped, keeping the
highest-ranked copy (FineWeb-Edu repeats pages across CommonCrawl dumps: in the technical-filter set only 292 k of 870 k
texts are unique), and so are documents whose id is already in any --exclude-corpus
manifest (e.g. the replay corpora of the same run, so no document is in both).

    python -m cpt_trainer.build_ranked_corpus cpt_data/corpus_ai_high_fw/web_ai_high \
        --parquet-root cpt_data/corpus_ai_high_fw/finweb-edu/data --tokens 1.0e9 \
        --exclude-corpus cpt_data/corpus_replay_hq2b/web_rep_fw
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import time

import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LICENSE = "ODC-By 1.0 (FineWeb-Edu); CommonCrawl terms of use"
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def clean_id(raw: str) -> str:
    return _SAFE.sub("_", str(raw).replace("<urn:uuid:", "").rstrip(">"))[:80]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="corpus folder to create (its basename is the corpus name in the trainer)")
    ap.add_argument("--parquet-root", required=True, help="folder with <dump>/*.parquet")
    ap.add_argument("--tokens", type=float, required=True, help="Qwen3 token budget (EOS per document included)")
    ap.add_argument("--rank", default="int_score,tech_families,tech_density", help="columns, compared in order, descending")
    ap.add_argument("--source", default="fineweb_edu")
    ap.add_argument("--exclude-corpus", action="append", default=[], help="corpus folder whose manifest ids are skipped")
    ap.add_argument("--est-ratio", type=float, default=0.972, help="Qwen tokens per token_count token (only to size pass 2)")
    ap.add_argument("--oversample", type=float, default=1.08)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "models", "Qwen3-0.6B-Base"))
    ap.add_argument("--batch", type=int, default=256, help="documents per tokenizer call")
    args = ap.parse_args()
    if os.path.exists(os.path.join(args.out_dir, "manifest.jsonl")):
        raise SystemExit(f"{args.out_dir} already has a manifest; remove it first")
    t0 = time.time()
    rank_cols = args.rank.split(",")
    files = sorted(glob.glob(os.path.join(args.parquet_root, "*", "*.parquet")))
    excluded = set()
    for c in args.exclude_corpus:
        with open(os.path.join(c, "manifest.jsonl")) as f:
            excluded |= {json.loads(l)["id"] for l in f}
    print(f"{len(files)} parquet files; {len(excluded):,} ids excluded via {args.exclude_corpus}", flush=True)

    # pass 1: rank all rows on their metadata
    rows = []
    for fi, fn in enumerate(files):
        t = pq.read_table(fn, columns=["id", "token_count"] + rank_cols).to_pydict()
        for r in range(len(t["id"])):
            did = f"{args.source}:{clean_id(t['id'][r])}"
            if did in excluded:
                continue
            rows.append((tuple(t[c][r] for c in rank_cols), fi, r, t["token_count"][r]))
    n_all = len(rows)
    rows.sort(key=lambda x: x[0], reverse=True)
    est, cut = 0, 0
    for cut, x in enumerate(rows):
        est += x[3] * args.est_ratio
        if est >= args.tokens * args.oversample:
            break
    cand = rows[:cut + 1]
    print(f"pass 1: {n_all:,} rows ranked; {len(cand):,} candidates (~{est / 1e6:.0f} M est. tokens), "
          f"last rank {cand[-1][0]}, {time.time() - t0:.0f}s", flush=True)

    # pass 2: read the candidates' text per file, exact token counts
    by_file = {}
    for k, (_, fi, r, _) in enumerate(cand):
        by_file.setdefault(fi, []).append((r, k))
    meta = [None] * len(cand)
    texts = [None] * len(cand)
    cols = ["text", "id", "url", "dump", "token_count", "int_score", "score", "tech_families", "tech_density",
            "tech_family_list"]
    for fi, lst in by_file.items():
        t = pq.read_table(files[fi], columns=cols)
        idx = [r for r, _ in lst]
        sub = t.take(idx).to_pydict()
        for j, (_, k) in enumerate(lst):
            texts[k] = sub["text"][j]
            meta[k] = {c: sub[c][j] for c in cols if c != "text"}
    # exact duplicates (keep the highest-ranked copy) and short texts are dropped before tokenising
    seen, drop = set(), [None] * len(cand)
    md5 = [None] * len(cand)
    for k in range(len(cand)):
        if len(texts[k]) < args.min_chars:
            drop[k] = "short"
        else:
            md5[k] = hashlib.md5(texts[k].encode("utf-8")).hexdigest()
            if md5[k] in seen:
                drop[k] = "duplicate"
            seen.add(md5[k])
        if drop[k]:
            texts[k] = None
    keep = [k for k in range(len(cand)) if not drop[k]]
    print(f"pass 2: texts read, {len(cand) - len(keep):,} dropped (duplicates / short), {len(keep):,} kept, "
          f"{time.time() - t0:.0f}s; tokenising", flush=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    counts, last = [0] * len(cand), time.time()
    for s in range(0, len(keep), args.batch):
        ks = keep[s:s + args.batch]
        for k, x in zip(ks, tok([texts[k] for k in ks], add_special_tokens=False)["input_ids"]):
            counts[k] = len(x) + 1  # + EOS, as the trainer adds
        if time.time() - last > 30:
            last = time.time()
            print(f"    {s + len(ks):,}/{len(keep):,} docs, {sum(counts) / 1e6:.0f} M tokens", flush=True)
    print(f"  tokenised {sum(counts) / 1e6:.1f} M tokens, {time.time() - t0:.0f}s", flush=True)

    # pass 3: write the kept documents in rank order until the budget
    os.makedirs(args.out_dir, exist_ok=True)
    total, written = 0, 0
    n_dup, n_short = drop.count("duplicate"), drop.count("short")
    fam_hist, score_hist = {}, {}
    with open(os.path.join(args.out_dir, "manifest.jsonl"), "w") as mf:
        for k in keep:
            if total >= args.tokens:
                break
            text, h = texts[k], md5[k]
            m = meta[k]
            did = clean_id(m["id"])
            rel = os.path.join("text", args.source, m["dump"], did + ".txt")
            os.makedirs(os.path.join(args.out_dir, os.path.dirname(rel)), exist_ok=True)
            with open(os.path.join(args.out_dir, rel), "w", encoding="utf-8") as f:
                f.write(text)
            mf.write(json.dumps({"id": f"{args.source}:{did}", "kind": "web", "title": m["url"], "url": m["url"],
                                 "license": LICENSE, "source": args.source, "dump": m["dump"], "text_path": rel,
                                 "text_chars": len(text), "qwen_tokens": counts[k], "gpt2_tokens": m["token_count"],
                                 "int_score": m["int_score"], "edu_score": m["score"],
                                 "tech_families": m["tech_families"], "tech_density": m["tech_density"],
                                 "tech_family_list": m["tech_family_list"], "md5": h}) + "\n")
            total += counts[k]
            written += 1
            fam_hist[m["tech_families"]] = fam_hist.get(m["tech_families"], 0) + counts[k]
            score_hist[m["int_score"]] = score_hist.get(m["int_score"], 0) + counts[k]
    stats = {"out_dir": args.out_dir, "parquet_root": args.parquet_root, "rank": rank_cols, "budget": args.tokens,
             "docs": written, "qwen_tokens": total, "duplicates_skipped": n_dup, "short_skipped": n_short, "candidates": len(cand),
             "rows_ranked": n_all, "excluded_ids": len(excluded), "exclude_corpus": args.exclude_corpus,
             "last_rank": list(cand[k][0]) if written else None,
             "tokens_by_int_score": {str(a): b for a, b in sorted(score_hist.items())},
             "tokens_by_tech_families": {str(a): b for a, b in sorted(fam_hist.items())},
             "seconds": round(time.time() - t0)}
    with open(os.path.join(args.out_dir, "build_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1), flush=True)
    if total < args.tokens:
        print(f"! only {total / 1e6:.1f} M tokens written (budget {args.tokens / 1e6:.0f} M): raise --oversample")


if __name__ == "__main__":
    main()
