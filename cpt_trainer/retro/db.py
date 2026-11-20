#!/usr/bin/env python3
"""Build the RETRO retrieval database from the AI documents (config: configs/retro/db_ai_v1.yaml).

    python -m cpt_trainer.retro.db configs/retro/db_ai_v1.yaml --workers 16
    python -m cpt_trainer.retro.db configs/retro/db_ai_v1.yaml --out /tmp/db_smoke --limit-files 2 --limit-docs 2000

Steps (CPU only; resumable: finished parts are kept in <out>/parts and skipped on a re-run):
  1. one job per scored parquet file (rows with a wanted AI label, English, >= min_chars) and per 20 k-document
     slice of each corpus folder; each job tokenises with the Qwen3 tokenizer (+ EOS per document), records the
     document key (the manifest id "source:uuid" that cpt_trainer.build_labelled_corpus also uses), a hash of the
     normalised text, and the rare eval 13-grams it contains -> parts/<job>.u32 + parts/<job>.npz
  2. merge: drop documents contaminated with eval items (13-gram rule of cpt_trainer/decontaminate.py), exact
     duplicates (same text hash) and repeated keys; concatenate the kept tokens into tokens.u32 and cut every
     document into chunks of chunk_len tokens from its start.
Output (<out>/):
    tokens.u32      uint32 memmap, all kept documents back to back (each ends with EOS)
    docs.npz        doc_off [n+1] int64, doc_key / doc_text_hash int64, doc_src int16 (index into meta sources)
    chunks.npz      chunk_start int64 (token offset), chunk_doc int32
    meta.json       counts per source, dedupe / decontamination statistics, settings
A neighbour of database chunk j is tokens[chunk_start[j] : chunk_start[j] + nbr_len], clipped at its document end.
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cpt_trainer.retro.common import doc_key, text_hash  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SLICE = 20_000
_G = {}   # worker globals (fork): tokenizer, eval grams, config


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def eval_gram_table() -> dict[int, str]:
    """13-gram hash -> eval item id, without 13-grams shared by >= 3 eval items (boilerplate)."""
    from cpt_trainer.decontaminate import eval_texts, grams
    owners = defaultdict(set)
    for iid, text in eval_texts():
        for g in grams(text):
            if len(owners[g]) < 3:
                owners[g].add(iid)
    return {g: next(iter(o)) for g, o in owners.items() if len(o) < 3}


def list_jobs(cfg: dict, limit_files: int | None) -> list[dict]:
    jobs = []
    for t in cfg.get("parquet_trees", []):
        d = os.path.join(ROOT, t["dir"])
        files = sorted(glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True))
        files = [f for f in files if os.path.relpath(f, d).split(os.sep)[0] not in set(t.get("exclude_subsets", []))]
        for f in files[:limit_files] if limit_files else files:
            jobs.append({"kind": "parquet", "file": f, "tree": d, "scores": os.path.join(ROOT, t["scores"]),
                         "labels": list(t.get("labels", ["AI"])),
                         "source": "fineweb_edu" if "fineweb" in os.path.basename(d) else
                                   "nemotron_cc" if "nemotron" in os.path.basename(d) else os.path.basename(d)})
    for c in cfg.get("corpus_dirs", []):
        d = os.path.join(ROOT, c)
        with open(os.path.join(d, "manifest.jsonl"), encoding="utf-8") as f:
            man = [json.loads(line) for line in f if line.strip()]
        man = [(m["id"], os.path.join(d, m["text_path"]), m.get("source", "?")) for m in man if not m.get("low_quality")]
        slices = range(0, len(man), SLICE)
        for s in list(slices)[:limit_files] if limit_files else slices:
            jobs.append({"kind": "corpus", "corpus": os.path.basename(os.path.normpath(d)), "docs": man[s:s + SLICE]})
    for i, j in enumerate(jobs):
        j["idx"] = i
    return jobs


def _init(cfg):
    from transformers import AutoTokenizer
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, cfg["tokenizer"]))
    _G["tok"], _G["eos"], _G["cfg"] = tok.backend_tokenizer, tok.eos_token_id, cfg


def _texts_of(job, limit_docs):
    """[(source, key string, text)] of one job."""
    cfg = _G["cfg"]
    out = []
    if job["kind"] == "parquet":
        import pyarrow.parquet as pq
        from cpt_trainer.parquet_to_corpus import doc_id
        f = job["file"]
        names = pq.ParquetFile(f).schema_arrow.names
        cols = [c for c in ("id", "text", "language") if c in names]
        tbl = pq.read_table(f, columns=cols)
        sc = pq.read_table(os.path.join(job["scores"], os.path.relpath(f, job["tree"])), columns=["id", "label"])
        assert tbl.num_rows == sc.num_rows, f"row mismatch {f}"
        labels = sc.column("label").to_pylist()
        want = set(job["labels"])
        for i, row in enumerate(tbl.to_pylist()):
            if labels[i] not in want or row.get("language") not in (None, "en"):
                continue
            text = (row.get("text") or "").strip()
            if len(text) < cfg.get("min_chars", 200):
                continue
            out.append((job["source"], f"{job['source']}:{doc_id(row, text)}", text))
            if limit_docs and len(out) >= limit_docs:
                break
    else:
        for did, path, src in job["docs"]:
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    text = fh.read().strip()
            except OSError:
                continue
            if len(text) < cfg.get("min_chars", 200):
                continue
            out.append((f"{job['corpus']}/{src}", did, text))
            if limit_docs and len(out) >= limit_docs:
                break
    return out


def run_job(args):
    job, out_dir, limit_docs = args
    base = os.path.join(out_dir, "parts", f"{job['idx']:05d}")
    if os.path.exists(base + ".npz"):
        return job["idx"], "cached"
    from cpt_trainer.decontaminate import grams
    docs = _texts_of(job, limit_docs)
    enc = _G["tok"].encode_batch([t for _, _, t in docs], add_special_tokens=False) if docs else []
    eos, ev = _G["eos"], _G.get("eval_grams") or {}
    toks, n_tok, keys, th, srcs, hits = [], [], [], [], [], []
    for (src, key, text), e in zip(docs, enc):
        ids = np.asarray(e.ids + [eos], dtype=np.uint32)
        toks.append(ids); n_tok.append(len(ids))
        keys.append(doc_key(key)); th.append(text_hash(text)); srcs.append(src)
        hits.append(sorted(grams(text) & ev.keys()) if ev else [])
    flat = np.concatenate(toks) if toks else np.zeros(0, np.uint32)
    flat.tofile(base + ".u32.tmp")
    os.replace(base + ".u32.tmp", base + ".u32")
    hit_off = np.cumsum([0] + [len(h) for h in hits]).astype(np.int64)
    np.savez(base + ".tmp.npz", n_tok=np.asarray(n_tok, np.int64), key=np.asarray(keys, np.int64),
             thash=np.asarray(th, np.int64), src=np.asarray(srcs), hit_off=hit_off,
             hits=np.asarray([g for h in hits for g in h], np.int64))
    os.replace(base + ".tmp.npz", base + ".npz")
    return job["idx"], f"{len(docs)} docs {len(flat) / 1e6:.1f} M tok"


def merge(cfg: dict, out_dir: str, jobs: list[dict], eval_grams: dict, log=print) -> dict:
    C, min_chunk = cfg["chunk_len"], cfg.get("min_chunk_tokens", 16)
    parts = [np.load(os.path.join(out_dir, "parts", f"{j['idx']:05d}.npz")) for j in jobs]
    # decontamination: 13-grams by database document frequency
    df = Counter()
    for p in parts:
        df.update(set(p["hits"].tolist()) if len(p["hits"]) else ())
    rare = {g for g, c in df.items() if c <= cfg.get("max_train_df", 5)}
    sources = sorted({s for p in parts for s in p["src"].tolist()})
    src_idx = {s: i for i, s in enumerate(sources)}
    seen_key, seen_text = set(), set()
    stats = defaultdict(Counter)
    keep_masks, examples = [], []
    for j, p in zip(jobs, parts):
        n = len(p["n_tok"])
        keep = np.zeros(n, dtype=bool)
        ho, hits = p["hit_off"], p["hits"]
        for i in range(n):
            s = str(p["src"][i])
            st = stats[s]
            st["docs_in"] += 1; st["tokens_in"] += int(p["n_tok"][i])
            bad = [g for g in hits[ho[i]:ho[i + 1]].tolist() if g in rare]
            if bad:
                st["contaminated"] += 1
                if len(examples) < 20:
                    examples.append({"source": s, "eval_item": eval_grams.get(bad[0]), "rare_13grams": len(bad)})
                continue
            k, t = int(p["key"][i]), int(p["thash"][i])
            if t in seen_text:
                st["dup_text"] += 1; continue
            if k in seen_key:
                st["dup_key"] += 1; continue
            seen_text.add(t); seen_key.add(k)
            keep[i] = True
            st["docs"] += 1; st["tokens"] += int(p["n_tok"][i])
        keep_masks.append(keep)
    n_docs = int(sum(m.sum() for m in keep_masks))
    n_tokens = int(sum(int(p["n_tok"][m].sum()) for p, m in zip(parts, keep_masks)))
    log(f"  merge: {n_docs:,} documents, {n_tokens / 1e9:.3f} B tokens kept")
    tok_path = os.path.join(out_dir, "tokens.u32")
    out = np.memmap(tok_path + ".tmp", dtype=np.uint32, mode="w+", shape=(max(1, n_tokens),))
    doc_off = np.zeros(n_docs + 1, dtype=np.int64)
    key = np.zeros(n_docs, np.int64); th = np.zeros(n_docs, np.int64); src = np.zeros(n_docs, np.int16)
    pos = d = 0
    for j, p, m in zip(jobs, parts, keep_masks):
        if not m.any():
            continue
        flat = np.fromfile(os.path.join(out_dir, "parts", f"{j['idx']:05d}.u32"), dtype=np.uint32)
        off = np.concatenate([[0], np.cumsum(p["n_tok"])])
        for i in np.nonzero(m)[0]:
            seg = flat[off[i]:off[i + 1]]
            out[pos:pos + len(seg)] = seg
            pos += len(seg)
            doc_off[d + 1] = pos
            key[d], th[d], src[d] = p["key"][i], p["thash"][i], src_idx[str(p["src"][i])]
            d += 1
    out.flush(); del out
    os.replace(tok_path + ".tmp", tok_path)
    # chunks: from each document start, every chunk_len tokens; the last one if >= min_chunk tokens
    lens = np.diff(doc_off)
    n_full = lens // C
    tail = (lens % C) >= min_chunk
    per_doc = n_full + tail
    chunk_doc = np.repeat(np.arange(n_docs, dtype=np.int32), per_doc)
    first = np.repeat(np.cumsum(np.concatenate([[0], per_doc[:-1]])), per_doc)
    chunk_start = doc_off[chunk_doc] + (np.arange(len(chunk_doc)) - first) * C
    np.savez(os.path.join(out_dir, "docs.npz"), doc_off=doc_off, doc_key=key, doc_text_hash=th, doc_src=src)
    np.savez(os.path.join(out_dir, "chunks.npz"), chunk_start=chunk_start.astype(np.int64), chunk_doc=chunk_doc)
    meta = {"name": cfg.get("name"), "chunk_len": C, "min_chunk_tokens": min_chunk, "tokenizer": cfg["tokenizer"],
            "docs": n_docs, "tokens": n_tokens, "chunks": int(len(chunk_doc)), "sources": sources,
            "by_source": {s: dict(v) for s, v in sorted(stats.items())},
            "eval_13grams": len(eval_grams), "eval_13grams_in_db": len(df), "rare_eval_13grams_in_db": len(rare),
            "contamination_examples": examples, "config": cfg, "built": time.strftime("%Y-%m-%d %H:%M")}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


class Database:
    """Read side of a built database (memory-mapped tokens)."""

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        self.tokens = np.memmap(os.path.join(path, "tokens.u32"), dtype=np.uint32, mode="r")
        d = np.load(os.path.join(path, "docs.npz"))
        self.doc_off, self.doc_key, self.doc_text_hash = d["doc_off"], d["doc_key"], d["doc_text_hash"]
        c = np.load(os.path.join(path, "chunks.npz"))
        self.chunk_start, self.chunk_doc = c["chunk_start"], c["chunk_doc"]
        self.chunk_len = self.meta["chunk_len"]

    def __len__(self):
        return len(self.chunk_start)

    def neighbours(self, ids: np.ndarray, length: int, offset: int = 0) -> np.ndarray:
        """Token ids [*ids.shape, length] (int64, -1 padding) starting `offset` tokens after the start of each chunk
        in `ids` (-1 = none): offset 0 and length 2C = the neighbour [N, F]; offset C = its continuation F.
        Clipped at the end of the chunk's document."""
        ids = np.asarray(ids)
        flat = ids.reshape(-1).astype(np.int64)
        out = np.full((len(flat), length), -1, dtype=np.int64)
        ok = flat >= 0
        if ok.any():
            f = flat[ok]
            st = self.chunk_start[f] + offset
            end = self.doc_off[self.chunk_doc[f].astype(np.int64) + 1]
            idx = st[:, None] + np.arange(length)[None]
            valid = idx < end[:, None]
            vals = np.asarray(self.tokens[np.where(valid, idx, 0)], dtype=np.int64)
            out[ok] = np.where(valid, vals, -1)
        return out.reshape(ids.shape + (length,))

    def chunk_tokens(self, ids: np.ndarray) -> np.ndarray:
        return self.neighbours(ids, self.chunk_len)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--out", default=None, help="output folder (default: out_dir of the config)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit-files", type=int, default=None, help="smoke tests: jobs per parquet tree / corpus folder")
    ap.add_argument("--limit-docs", type=int, default=None, help="smoke tests: documents per job")
    ap.add_argument("--keep-parts", action="store_true", help="keep parts/ after the merge")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = os.path.abspath(args.out or os.path.join(ROOT, cfg["out_dir"]))
    if os.path.exists(os.path.join(out_dir, "meta.json")):
        raise SystemExit(f"{out_dir} is already built (meta.json exists)")
    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)
    t0 = time.time()
    jobs = list_jobs(cfg, args.limit_files)
    print(f"{len(jobs)} jobs ({sum(j['kind'] == 'parquet' for j in jobs)} parquet files, "
          f"{sum(j['kind'] == 'corpus' for j in jobs)} corpus slices) -> {out_dir}", flush=True)
    eval_grams = eval_gram_table() if cfg.get("decontaminate", True) else {}
    print(f"  {len(eval_grams):,} eval 13-grams for decontamination ({time.time() - t0:.0f}s)", flush=True)
    _G["eval_grams"] = eval_grams                    # inherited by the forked workers (same hash() salt)
    ctx = mp.get_context("fork")
    done = 0
    with ctx.Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
        for idx, msg in pool.imap_unordered(run_job, [(j, out_dir, args.limit_docs) for j in jobs]):
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] job {idx}: {msg} ({time.time() - t0:.0f}s)", flush=True)
    meta = merge(cfg, out_dir, jobs, eval_grams)
    if not args.keep_parts:
        for f in glob.glob(os.path.join(out_dir, "parts", "*")):
            os.remove(f)
        os.rmdir(os.path.join(out_dir, "parts"))
    print(json.dumps({k: meta[k] for k in ("docs", "tokens", "chunks")}), flush=True)
    for s, v in meta["by_source"].items():
        print(f"  {s:28s} kept {v.get('docs', 0):9,d} docs {v.get('tokens', 0) / 1e9:6.3f} B tok | in {v['docs_in']:9,d} | "
              f"contaminated {v.get('contaminated', 0):6,d} dup text {v.get('dup_text', 0):8,d} dup key {v.get('dup_key', 0):6,d}")
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
