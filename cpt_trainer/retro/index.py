#!/usr/bin/env python3
"""faiss index over the database chunk embeddings (sharded and resumable).

    python -m cpt_trainer.retro.index cpt_data/retro_db/ai_v1                           # train + add + merge
    python -m cpt_trainer.retro.index <db> --factory "IVF256,PQ16" --train-size 20000 --shard-size 50000   # smoke

  1. train: embed --train-size random database chunks, train the index (OPQ rotation, IVF centroids, PQ codebooks)
     -> <db>/index/trained.index
  2. add:   database chunks in shards of --shard-size: embed (GPU), encode, write index/shards/<s>.index
            (a finished shard is skipped on a re-run)
  3. merge: all shards into index/<name>.index (inverted lists merged; ids = database chunk ids)
Default "OPQ64_256,IVF65536_HNSW32,PQ64": 64 bytes per chunk (~5 GB RAM for 75 M chunks), the Megatron-Retro /
Wang et al. 2023 recipe at a smaller scale. Inner product on L2-normalised embeddings (= cosine).
Cost estimate (RTX 4090, bge-small): ~10 k chunks/s embedding -> ~2 h for 75 M chunks; IVF training ~30 min (CPU).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cpt_trainer.retro.db import Database  # noqa: E402
from cpt_trainer.retro.embed import DEFAULT_EMBEDDER, ChunkEmbedder, decode_chunks  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEX_NAME = "chunks.index"


def load_qwen_tok(db: Database):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(os.path.join(ROOT, db.meta["tokenizer"]))


def embed_db_chunks(db: Database, qtok, emb: ChunkEmbedder, ids: np.ndarray, block: int = 65536, log=print):
    out = np.empty((len(ids), emb.dim), dtype=np.float32)
    t0 = time.time()
    for s in range(0, len(ids), block):
        out[s:s + block] = emb.embed_texts(decode_chunks(qtok, db.chunk_tokens(ids[s:s + block])))
        if s and (s // block) % 20 == 0:
            log(f"    embedded {s + block:,}/{len(ids):,} ({(s + block) / (time.time() - t0):.0f} chunks/s)")
    return out


def open_index(path: str, nprobe: int = 32, ef: int = 64):
    import faiss
    index = faiss.read_index(path)
    ps = faiss.ParameterSpace()
    try:
        ps.set_index_parameter(index, "nprobe", nprobe)
    except RuntimeError:
        pass
    try:
        ps.set_index_parameter(index, "quantizer_efSearch", ef)
    except RuntimeError:
        pass
    return index


def main() -> None:
    import faiss
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db")
    ap.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    ap.add_argument("--factory", default="OPQ64_256,IVF65536_HNSW32,PQ64")
    ap.add_argument("--train-size", type=int, default=3_000_000, help="chunks used to train the index (>= 39 x nlist)")
    ap.add_argument("--shard-size", type=int, default=5_000_000)
    ap.add_argument("--embed-batch", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=0, help="faiss CPU threads (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        faiss.omp_set_num_threads(args.threads)
    db = Database(os.path.abspath(args.db))
    idir = os.path.join(db.path, "index")
    os.makedirs(os.path.join(idir, "shards"), exist_ok=True)
    final = os.path.join(idir, INDEX_NAME)
    if os.path.exists(final):
        raise SystemExit(f"{final} exists")
    qtok = load_qwen_tok(db)
    emb = ChunkEmbedder(args.embedder, batch=args.embed_batch)
    n = len(db)
    print(f"database {db.path}: {n:,} chunks; embedder {emb.name} ({emb.dim}-d, {emb.pooling}) on {emb.device}; "
          f"factory {args.factory}", flush=True)
    t0 = time.time()
    trained = os.path.join(idir, "trained.index")
    if not os.path.exists(trained):
        rng = np.random.default_rng(args.seed)
        ids = np.sort(rng.choice(n, size=min(n, args.train_size), replace=False))
        x = embed_db_chunks(db, qtok, emb, ids)
        print(f"  training sample embedded: {len(ids):,} ({time.time() - t0:.0f}s)", flush=True)
        index = faiss.index_factory(emb.dim, args.factory, faiss.METRIC_INNER_PRODUCT)
        index.train(x)
        faiss.write_index(index, trained + ".tmp")
        os.replace(trained + ".tmp", trained)
        print(f"  index trained ({time.time() - t0:.0f}s)", flush=True)
    shards = list(range(0, n, args.shard_size))
    for k, s in enumerate(shards):
        p = os.path.join(idir, "shards", f"{k:04d}.index")
        if os.path.exists(p):
            continue
        ids = np.arange(s, min(n, s + args.shard_size), dtype=np.int64)
        x = embed_db_chunks(db, qtok, emb, ids)
        index = faiss.read_index(trained)
        index.add_with_ids(x, ids)
        faiss.write_index(index, p + ".tmp")
        os.replace(p + ".tmp", p)
        print(f"  shard {k + 1}/{len(shards)}: chunks {s:,}-{s + len(ids):,} ({time.time() - t0:.0f}s)", flush=True)
    index = faiss.read_index(trained)
    ivf = faiss.extract_index_ivf(index)
    for k in range(len(shards)):
        sh = faiss.read_index(os.path.join(idir, "shards", f"{k:04d}.index"))
        ivf.merge_from(faiss.extract_index_ivf(sh), 0)
    index.ntotal = ivf.ntotal
    assert index.ntotal == n, (index.ntotal, n)
    faiss.write_index(index, final + ".tmp")
    os.replace(final + ".tmp", final)
    with open(os.path.join(idir, "index.json"), "w") as f:
        json.dump({"factory": args.factory, "embedder": os.path.relpath(os.path.abspath(args.embedder), ROOT),
                   "dim": emb.dim, "pooling": emb.pooling, "ntotal": int(index.ntotal), "train_size": args.train_size,
                   "built": time.strftime("%Y-%m-%d %H:%M")}, f, indent=2)
    for f in glob.glob(os.path.join(idir, "shards", "*.index")):
        os.remove(f)
    print(f"  merged -> {final} ({index.ntotal:,} vectors, {(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
