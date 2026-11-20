#!/usr/bin/env python3
"""Draw a uniform random sample of documents from the large AI corpora for the difficulty-classifier comparison.

For each corpus the sample is uniform over documents (file row counts from the parquet footers, then one global
row index per draw). Each sampled row is joined by position with its AI-relevance score file (cpt_data/scores/...,
ids are checked) and gets: the whitespace-collapsed HEAD-char head that every model judge sees, the exact Qwen3 token
count of the full text, and the upstream metadata (fineweb: url, dump, token_count, FineWeb-Edu int_score;
nemotron: subset, category). Output rows are shuffled with the seed and numbered idx = 0..N-1.

usage: python3 -P scripts/sample_difficulty_set.py --per-corpus 500 --seed 20261102 --out cpt_data/difficulty/sample_1k.parquet
"""
import argparse, os, sys, random, json, bisect
from multiprocessing import Pool

import pyarrow as pa, pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from difficulty_common import head  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--corpora", nargs="+", default=["cpt_data/corpus_ai_fineweb_large", "cpt_data/corpus_ai_nemotron_large"])
ap.add_argument("--per-corpus", type=int, default=500)
ap.add_argument("--seed", type=int, default=20261102)
ap.add_argument("--head", type=int, default=1200)
ap.add_argument("--out", default="cpt_data/difficulty/sample_1k.parquet")
ap.add_argument("--tokenizer", default="models/Qwen3-0.6B-Base")
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()


def list_files(corpus):
    return sorted(os.path.join(dp, f) for dp, _, fs in os.walk(corpus) if "/.git" not in dp for f in fs if f.endswith(".parquet"))


def fetch(job):
    """job = (corpus, file, [row indices]) -> list of row dicts."""
    corpus, f, rows = job
    name = os.path.basename(corpus.rstrip("/"))
    rel = os.path.relpath(f, corpus)
    pf = pq.ParquetFile(f)
    names = pf.schema_arrow.names
    cols = ["id", "text"] + [c for c in ("url", "dump", "token_count", "int_score", "score", "kw_core", "metadata") if c in names]
    # row-group boundaries so only the needed groups are read
    starts = [0]
    for g in range(pf.metadata.num_row_groups):
        starts.append(starts[-1] + pf.metadata.row_group(g).num_rows)
    sc_path = os.path.join("cpt_data", "scores", name, rel)
    sc = pq.read_table(sc_path, columns=["id", "p_ai", "p_near", "label"]) if os.path.exists(sc_path) else None
    out = []
    for r in rows:
        g = bisect.bisect_right(starts, r) - 1
        tbl = pf.read_row_group(g, columns=cols).slice(r - starts[g], 1).to_pylist()[0]
        rec = {"corpus": name, "subset": rel.split("/")[0], "file": rel, "row": r, "id": tbl["id"], "text": tbl["text"] or "",
               "url": tbl.get("url"), "dump": tbl.get("dump"), "fw_token_count": tbl.get("token_count"),
               "fw_edu_int_score": tbl.get("int_score"), "fw_edu_score": tbl.get("score"), "kw_core": tbl.get("kw_core"),
               "category": (tbl.get("metadata") or {}).get("category") if isinstance(tbl.get("metadata"), dict) else None}
        if sc is not None:
            s = sc.slice(r, 1).to_pylist()[0]
            assert s["id"] == rec["id"], f"id mismatch {rel}:{r}"
            rec.update(p_ai=s["p_ai"], p_near=s["p_near"], ai_label=s["label"])
        else:
            rec.update(p_ai=None, p_near=None, ai_label=None)
        out.append(rec)
    return out


rng = random.Random(a.seed)
jobs = []
for corpus in a.corpora:
    files = list_files(corpus)
    counts = [pq.ParquetFile(f).metadata.num_rows for f in files]
    cum = [0]
    for c in counts:
        cum.append(cum[-1] + c)
    print(f"{corpus}: {len(files)} files, {cum[-1]:,} docs", file=sys.stderr, flush=True)
    picks = sorted(rng.sample(range(cum[-1]), a.per_corpus))
    by_file = {}
    for p in picks:
        fi = bisect.bisect_right(cum, p) - 1
        by_file.setdefault(files[fi], []).append(p - cum[fi])
    jobs += [(corpus, f, rows) for f, rows in by_file.items()]

with Pool(a.workers) as pool:
    recs = [r for part in pool.imap_unordered(fetch, jobs) for r in part]
print(f"fetched {len(recs)} docs", file=sys.stderr, flush=True)

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(a.tokenizer).backend_tokenizer
for r in recs:
    r["n_tokens"] = len(tok.encode(r["text"], add_special_tokens=False).ids)
    r["head"] = head(r["text"], a.head)

random.Random(a.seed).shuffle(recs)
for i, r in enumerate(recs):
    r["idx"] = i
cols = ["idx", "corpus", "subset", "category", "file", "row", "id", "url", "dump", "n_tokens", "fw_token_count",
        "fw_edu_int_score", "fw_edu_score", "kw_core", "p_ai", "p_near", "ai_label", "head", "text"]
tbl = pa.table({c: [r[c] for r in recs] for c in cols})
os.makedirs(os.path.dirname(a.out), exist_ok=True)
pq.write_table(tbl, a.out)
with open(os.path.splitext(a.out)[0] + ".heads.jsonl", "w", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps({"idx": r["idx"], "head": r["head"]}, ensure_ascii=False) + "\n")
import collections  # noqa: E402
print("per corpus:", collections.Counter(r["corpus"] for r in recs), file=sys.stderr)
print("ai_label:", collections.Counter(r["ai_label"] for r in recs), file=sys.stderr)
print(f"wrote {a.out} ({len(recs)} rows)", file=sys.stderr)
