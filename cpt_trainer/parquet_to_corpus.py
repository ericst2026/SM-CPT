"""Parquet files with a `text` column -> a corpus folder for cpt_trainer (manifest.jsonl + text/<source>/<id>.txt).

Every input is sampled uniformly at random (seeded) to a share of its rows; the share is given as an
on-disk size, like cpt_trainer/filter_randomly.py: `@120` on a 936 MB file keeps 120/936 of its rows.
Without `@MB` every row is used. Each kept row becomes one text file plus one manifest line, which is
exactly what cpt_trainer/data.py reads (id, source, text_path, low_quality).

    python -m cpt_trainer.parquet_to_corpus cpt_data/corpus/ai \
        --add fineweb_edu=cpt_data/finewebedu_small_ai.parquet@120 \
        --add nemotron_cc=cpt_data/nemotron_small_ai.parquet@120
    python -m cpt_trainer.parquet_to_corpus cpt_data/corpus/replay \
        --add fineweb_edu=cpt_data/finewebedu_replay_small.recovered.parquet
    python -m cpt_trainer.parquet_to_corpus cpt_data/corpus_web/web_replay --dedupe \
        --add fineweb_edu=cpt_data/finewebedu_replay_large.full.parquet@3216     # ~1.7 B of its 3.77 B tokens
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

LICENSES = {
    "fineweb_edu": "ODC-By 1.0 (FineWeb-Edu); CommonCrawl terms of use",
    "nemotron_cc": "Nemotron-CC-v2 (NVIDIA); CommonCrawl terms of use - see dataset card",
}
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def doc_id(row: dict, text: str) -> str:
    raw = row.get("id")
    if raw:
        raw = str(raw).replace("<urn:uuid:", "").rstrip(">")
        raw = _SAFE.sub("_", raw)[:80]
        if raw:
            return raw
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:24]


def add_source(out_dir: str, source: str, path: str, size_mb: float | None, seed: int, manifest, min_chars: int,
               dedupe: bool = False):
    pf = pq.ParquetFile(path)
    total_rows, total_bytes = pf.metadata.num_rows, os.path.getsize(path)
    n = total_rows if size_mb is None else min(total_rows, round(total_rows * size_mb * 2**20 / total_bytes))
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(total_rows, size=n, replace=False)) if n < total_rows else np.arange(total_rows)
    cols = [c for c in pf.schema_arrow.names if c in ("text", "id", "url", "dump", "token_count", "score", "metadata",
                                                       "kw_core", "kw_math", "kw_code", "language_score")]
    text_dir = os.path.join(out_dir, "text", source)
    os.makedirs(text_dir, exist_ok=True)
    print(f"[{source}] {path}: {total_rows:,} rows, {total_bytes / 2**20:.0f} MB -> keeping {n:,} rows "
          f"({100 * n / max(1, total_rows):.1f} %)", flush=True)
    offset, kept, chars, words, gpt2_tokens, skipped, seen, t0 = 0, 0, 0, 0, 0, 0, set(), time.time()
    for batch in pf.iter_batches(batch_size=4096, columns=cols):
        lo, hi = np.searchsorted(picked, [offset, offset + batch.num_rows])
        if hi > lo:
            sub = batch.take(picked[lo:hi] - offset)
            for row in sub.to_pylist():
                text = (row.get("text") or "").strip()
                if len(text) < min_chars:
                    skipped += 1
                    continue
                did = doc_id(row, text)
                if did in seen:
                    if dedupe:            # content-derived ids (sha1 of the text): an exact duplicate document
                        skipped += 1
                        continue
                    did = did + "_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
                seen.add(did)
                rel = f"text/{source}/{did}.txt"
                with open(os.path.join(out_dir, rel), "w", encoding="utf-8") as f:
                    f.write(text + "\n")
                meta = {k: row[k] for k in ("dump", "score", "kw_core", "kw_math", "kw_code", "language_score")
                        if row.get(k) is not None}
                if isinstance(row.get("metadata"), dict):
                    meta.update({k: v for k, v in row["metadata"].items() if v is not None})
                nw = len(text.split())
                manifest.write(json.dumps({
                    "id": f"{source}:{did}", "kind": "web", "title": row.get("url"), "url": row.get("url"),
                    "license": LICENSES.get(source, "see source dataset card"), "source": source,
                    "parquet": os.path.relpath(path, out_dir), "text_path": rel, "bytes": len(text.encode("utf-8")) + 1,
                    "text_chars": len(text), "text_words": nw, "gpt2_tokens": row.get("token_count"),
                    "low_quality": False, "meta": meta}, ensure_ascii=False) + "\n")
                kept += 1
                chars += len(text)
                words += nw
                gpt2_tokens += row.get("token_count") or 0
        offset += batch.num_rows
    est_tokens = gpt2_tokens if gpt2_tokens else chars / 4.0
    print(f"[{source}] wrote {kept:,} documents ({skipped} shorter than {min_chars} chars skipped): "
          f"{chars / 1e6:.1f} M chars, {words / 1e6:.1f} M words, ~{est_tokens / 1e6:.1f} M tokens "
          f"({'GPT-2 token_count column' if gpt2_tokens else 'chars/4'}) in {time.time() - t0:.0f}s", flush=True)
    return {"source": source, "parquet": path, "rows_total": total_rows, "rows_picked": n, "docs": kept,
            "skipped_short": skipped, "chars": chars, "words": words, "est_tokens": round(est_tokens),
            "size_mb_target": size_mb}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="corpus folder to create (its basename is the corpus name in the trainer)")
    ap.add_argument("--add", action="append", required=True, metavar="SOURCE=PATH[@MB]",
                    help="parquet to sample from; @MB = on-disk megabytes worth of rows to keep (default: all rows)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-chars", type=int, default=50, help="drop documents shorter than this")
    ap.add_argument("--dedupe", action="store_true", help="skip rows whose doc id was already written (exact duplicates "
                    "when the id column is a hash of the text, e.g. the recovered replay parquets)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.jsonl")
    if os.path.exists(manifest_path):
        raise SystemExit(f"{manifest_path} exists; remove the folder first")
    stats = []
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for i, spec in enumerate(args.add):
            source, _, rest = spec.partition("=")
            path, _, mb = rest.partition("@")
            if not source or not path:
                raise SystemExit(f"bad --add {spec!r}; expected SOURCE=PATH[@MB]")
            stats.append(add_source(args.out_dir, source, path, float(mb) if mb else None, args.seed + i, manifest,
                                    args.min_chars, args.dedupe))
    tot_docs, tot_tok = sum(s["docs"] for s in stats), sum(s["est_tokens"] for s in stats)
    lines = [f"# Corpus `{os.path.basename(os.path.normpath(args.out_dir))}`", "",
             f"- documents: **{tot_docs:,}**  |  estimated tokens: **{tot_tok / 1e6:.1f} M**  |  built {time.strftime('%Y-%m-%d %H:%M')} "
             f"by cpt_trainer/parquet_to_corpus.py (seed {args.seed})", "",
             "| source | parquet | rows in file | rows kept | share | docs | M chars | ~M tokens |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {s['source']} | {s['parquet']} | {s['rows_total']:,} | {s['rows_picked']:,} | "
              f"{100 * s['rows_picked'] / max(1, s['rows_total']):.1f} % | {s['docs']:,} | {s['chars'] / 1e6:.1f} | {s['est_tokens'] / 1e6:.1f} |"
              for s in stats]
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out_dir, "build_stats.json"), "w") as f:
        json.dump({"args": vars(args), "sources": stats}, f, indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
