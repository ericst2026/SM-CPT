#!/usr/bin/env python3
"""k nearest database chunks for every chunk of the training / validation sets (precomputed, as in RETRO).

    python -m cpt_trainer.retro.neighbours runs/_retro_cache/web1b cpt_data/retro_db/ai_v1

For each query chunk (64 tokens of a training / validation sequence): embed it (frozen bge-small), search the faiss
index for --k-search candidates, drop the leaking ones, keep the first --k-store in rank order.
Leakage filters (RETRO filters neighbours of the same document; the near-duplicate rule is ours):
  same-doc    candidate from the query's own document (same manifest id; the database holds every AI training
              document) -> its continuation would be the text being predicted
  same-text   candidate document with the same normalised text hash (exact duplicate under another id)
  target      the candidate's continuation F contains >= --max-target-overlap of the 8-grams of the query's real
              next chunk (mirrors / near-duplicate copies of the same text: the answer would be copied)
  repeat      a candidate whose chunk tokens equal an already kept neighbour
Overlap between the query chunk and the neighbour chunk N is NOT filtered: a retro-fitted model only learns to use
its neighbours when they overlap with its input (Doostmohammadi & Kuhlmann 2025, arXiv:2505.14309), and N cannot
leak the future. stats.json records the input-overlap histogram of the kept neighbours (activation diagnostic).

Output: <set>/nbr/<db name>/<split>.npy  int32 [n_chunks, k_store] database chunk ids (-1 = none), stats.json.
Shards of --shard-size chunks are written as they finish (resumable).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cpt_trainer.retro.common import overlap_fraction, row_hash  # noqa: E402
from cpt_trainer.retro.db import Database  # noqa: E402
from cpt_trainer.retro.embed import ChunkEmbedder, decode_chunks  # noqa: E402
from cpt_trainer.retro.index import INDEX_NAME, load_qwen_tok, open_index  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def split_chunk_tokens(p, C: int):
    """Query chunk tokens [n, C] and the real next-chunk tokens [n, C] (-1 past the piece end) of a split."""
    seq = np.repeat(np.arange(len(p["chunk_off"]) - 1), np.diff(p["chunk_off"]))
    start = p["seq_off"][seq] + p["chunk_pos"]
    end = p["seq_off"][seq] + p["chunk_end"]
    ar = np.arange(C)[None]
    q = p["tokens"][start[:, None] + ar].astype(np.int64)
    nxt_idx = start[:, None] + C + ar
    nv = nxt_idx < end[:, None]
    nxt = np.where(nv, p["tokens"][np.where(nv, nxt_idx, 0)], -1).astype(np.int64)
    return q, nxt


def filter_candidates(cand: np.ndarray, q_key, q_th, q_tok, q_next, db: Database, k_store: int,
                      max_target_overlap: float, ngram: int = 8, sub: int = 2048):
    """cand [B, K] database chunk ids in rank order (-1 = none). Returns (kept [B, k_store] int32, stats dict)."""
    B, K = cand.shape
    C = db.chunk_len
    st = {"candidates": int((cand >= 0).sum()), "same_doc": 0, "same_text": 0, "target": 0, "repeat": 0}
    valid = cand >= 0
    cdoc = np.where(valid, db.chunk_doc[np.where(valid, cand, 0)], -1)
    same_doc = valid & (db.doc_key[np.maximum(cdoc, 0)] == q_key[:, None])
    same_text = valid & ~same_doc & (db.doc_text_hash[np.maximum(cdoc, 0)] == q_th[:, None])
    st["same_doc"], st["same_text"] = int(same_doc.sum()), int(same_text.sum())
    ok = valid & ~same_doc & ~same_text
    tgt = np.zeros((B, K), np.float32)
    inp = np.zeros((B, K), np.float32)
    h = np.zeros((B, K), np.int64)
    for s in range(0, B, sub):          # token fetch per sub-batch: [sub, K, C] int64 arrays stay small
        e = slice(s, s + sub)
        N = db.neighbours(cand[e], C)                  # neighbour chunks
        Fc = db.neighbours(cand[e], C, offset=C)       # their continuations
        tgt[e] = overlap_fraction(torch.from_numpy(q_next[e]), torch.from_numpy(Fc), ngram).numpy()
        inp[e] = overlap_fraction(torch.from_numpy(q_tok[e]), torch.from_numpy(N), ngram).numpy()
        h[e] = row_hash(N)                             # identity of the chunk tokens
    leak = ok & (tgt >= max_target_overlap)
    st["target"] = int(leak.sum())
    ok &= ~leak
    # repeats: identical chunk tokens to a better-ranked kept candidate (h)
    kept = np.full((B, k_store), -1, np.int32)
    kept_inp = np.full((B, k_store), np.nan, np.float32)
    for b in range(B):
        seen, j = set(), 0
        for c in range(K):
            if not ok[b, c]:
                continue
            if h[b, c] in seen:
                st["repeat"] += 1
                continue
            seen.add(h[b, c])
            kept[b, j] = cand[b, c]; kept_inp[b, j] = inp[b, c]
            j += 1
            if j == k_store:
                break
    st["kept"] = int((kept >= 0).sum())
    st["no_neighbour"] = int((kept[:, 0] < 0).sum())
    first = kept_inp[:, 0][~np.isnan(kept_inp[:, 0])]
    st["input_overlap_hist_top1"] = np.histogram(first, bins=[0, 1e-9, 0.1, 0.25, 0.5, 0.75, 1.01])[0].tolist()
    st["input_overlap_sum_top1"] = float(first.sum())
    return kept, st


def merge_stats(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if isinstance(v, list):
            a[k] = [x + y for x, y in zip(a.get(k, [0] * len(v)), v)]
        else:
            a[k] = a.get(k, 0) + v
    return a


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set", help="training set folder (cpt_trainer.retro.data)")
    ap.add_argument("db", help="database folder (cpt_trainer.retro.db + index)")
    ap.add_argument("--splits", default="all", help="all | comma list (train, val_web_ai, ...)")
    ap.add_argument("--embedder", default=None, help="default: the one the index was built with")
    ap.add_argument("--k-search", type=int, default=20, help="candidates per query (over-retrieve, then filter)")
    ap.add_argument("--k-store", type=int, default=4, help="neighbours kept per chunk (training uses the first k)")
    ap.add_argument("--max-target-overlap", type=float, default=0.5)
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--shard-size", type=int, default=1_000_000)
    ap.add_argument("--embed-batch", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    import faiss
    if args.threads:
        faiss.omp_set_num_threads(args.threads)
    db = Database(os.path.abspath(args.db))
    with open(os.path.join(db.path, "index", "index.json")) as f:
        iinfo = json.load(f)
    index = open_index(os.path.join(db.path, "index", INDEX_NAME), nprobe=args.nprobe)
    emb = ChunkEmbedder(os.path.join(ROOT, args.embedder or iinfo["embedder"]), batch=args.embed_batch)
    qtok = load_qwen_tok(db)
    sdir = os.path.abspath(args.set)
    splits = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(sdir, "*.npz")))
    if args.splits != "all":
        splits = [s for s in splits if s in args.splits.split(",")]
    odir = os.path.join(sdir, "nbr", db.meta["name"] or os.path.basename(db.path))
    os.makedirs(odir, exist_ok=True)
    all_stats = {}
    t0 = time.time()
    for split in splits:
        final = os.path.join(odir, f"{split}.npy")
        if os.path.exists(final):
            continue
        p = np.load(os.path.join(sdir, f"{split}.npz"))
        q_tok, q_next = split_chunk_tokens(p, db.chunk_len)
        q_key, q_th = p["doc_key"][p["chunk_doc"]], p["doc_text_hash"][p["chunk_doc"]]
        n = len(q_tok)
        sh_dir = os.path.join(odir, f"_{split}")
        os.makedirs(sh_dir, exist_ok=True)
        stats = {}
        for k, s in enumerate(range(0, n, args.shard_size)):
            f_np, f_js = os.path.join(sh_dir, f"{k:04d}.npy"), os.path.join(sh_dir, f"{k:04d}.json")
            if os.path.exists(f_np):
                with open(f_js) as f:
                    merge_stats(stats, json.load(f))
                continue
            e = slice(s, min(n, s + args.shard_size))
            x = emb.embed_texts(decode_chunks(qtok, q_tok[e]))
            _, cand = index.search(x, args.k_search)
            kept, st = filter_candidates(cand, q_key[e], q_th[e], q_tok[e], q_next[e], db, args.k_store,
                                         args.max_target_overlap)
            np.save(f_np + ".tmp.npy", kept); os.replace(f_np + ".tmp.npy", f_np)
            with open(f_js, "w") as f:
                json.dump(st, f)
            merge_stats(stats, st)
            print(f"  {split} shard {k}: chunks {s:,}-{e.stop:,} of {n:,}; kept {st['kept']:,}, no neighbour "
                  f"{st['no_neighbour']:,}, filtered same-doc {st['same_doc']:,} same-text {st['same_text']:,} "
                  f"target {st['target']:,} ({time.time() - t0:.0f}s)", flush=True)
        arr = np.concatenate([np.load(f) for f in sorted(glob.glob(os.path.join(sh_dir, "*.npy")))])
        assert len(arr) == n
        np.save(final, arr)
        for f in glob.glob(os.path.join(sh_dir, "*")):
            os.remove(f)
        os.rmdir(sh_dir)
        stats["chunks"] = n
        all_stats[split] = stats
        print(f"{split}: {n:,} chunks done; {stats}", flush=True)
    sp = os.path.join(odir, "stats.json")
    old = json.load(open(sp)) if os.path.exists(sp) else {}
    old.update(all_stats)
    old["settings"] = {**vars(args), "index": iinfo}
    with open(sp, "w") as f:
        json.dump(old, f, indent=2)


if __name__ == "__main__":
    main()
