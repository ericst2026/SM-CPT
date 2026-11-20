#!/usr/bin/env python3
"""Training / validation set for the RETRO-fit run: the web5b mix scaled down to a token budget, packed into 16-24 k
document-masked sequences (cpt_trainer/longctx.py), with every sequence's retrieval chunks.

    python -m cpt_trainer.retro.data runs/_retro_cache/web1b --tokens 1e9 \\
        --corpora cpt_data/corpus_web/web_ai cpt_data/corpus_web/web_near cpt_data/corpus_web/web_replay \\
                  cpt_data/corpus_full/stem cpt_data/corpus_full/long \\
        --exclude-docs configs/corpus/corpus_full_exclude_noisy_long.txt

Same as cpt_qwen06b_rs_adaptive_v4_web5b: corpora, train/val split (cpt_trainer.data.split_documents, val_frac 0.01,
seed 42 -> the SAME validation documents and, with the same packing, the same 32 validation sequences per corpus, so
val losses are directly comparable), --exclude-docs, EOS per document, packing algorithm and seed.
Different: the training documents of each corpus are subsampled (seeded shuffle, then a prefix) to
budget x (corpus share of the web5b training tokens), i.e. the same mixing proportions at 1 B instead of 5 B tokens.

Chunks: every document piece of a sequence is cut into chunk_len-token chunks from the piece start; complete chunks
only. For each chunk: its start in the sequence, the end of its piece (CCA spans never cross into the next document),
and the training document it comes from (doc key / text hash, for the neighbour leakage filters).

Output (<out>/):
    train.npz, val_<corpus>.npz   tokens int32, seq_off, piece_len, piece_off, piece_doc (index into the split's docs),
                                  chunk_off [n_seq+1], chunk_pos / chunk_end (offsets in the sequence), chunk_doc,
                                  doc_key, doc_text_hash (per document of the split)
    docs_<split>.txt              corpus/id of each document of the split (line i = doc i)
    meta.json                     token / document counts per corpus and source, settings
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cpt_trainer.data import list_documents, split_documents  # noqa: E402
from cpt_trainer.retro.common import doc_key, text_hash  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pack_tracked(doc_ids: list[np.ndarray], min_len: int, max_len: int, seed: int) -> dict:
    """cpt_trainer.longctx.pack_varlen (identical output) that also records the document of every piece."""
    order = list(range(len(doc_ids)))
    random.Random(seed).shuffle(order)
    total = int(sum(len(d) for d in doc_ids))
    tokens = np.empty(total, dtype=np.int32)
    seq_off, piece_len, piece_off, piece_doc = [0], [], [0], []
    pos, cur = 0, 0

    def close():
        nonlocal cur
        seq_off.append(pos)
        piece_off.append(len(piece_len))
        cur = 0

    for d in order:
        arr, s = doc_ids[d], 0
        while s < len(arr):
            rest, room = len(arr) - s, max_len - cur
            if rest > room and cur >= min_len:
                close()
                continue
            take = min(rest, room)
            tokens[pos:pos + take] = arr[s:s + take]
            pos += take
            cur += take
            s += take
            piece_len.append(take)
            piece_doc.append(d)
            if cur >= max_len:
                close()
    if cur:
        close()
    return {"tokens": tokens[:pos], "seq_off": np.asarray(seq_off, dtype=np.int64),
            "piece_len": np.asarray(piece_len, dtype=np.int32), "piece_off": np.asarray(piece_off, dtype=np.int64),
            "piece_doc": np.asarray(piece_doc, dtype=np.int32)}


def take_first(p: dict, n: int) -> dict:
    """The first n sequences of a packed dict (as longctx.take_sequences(p, range(n)))."""
    n = min(n, len(p["seq_off"]) - 1)
    t_end, p_end = int(p["seq_off"][n]), int(p["piece_off"][n])
    return {"tokens": p["tokens"][:t_end], "seq_off": p["seq_off"][:n + 1], "piece_len": p["piece_len"][:p_end],
            "piece_off": p["piece_off"][:n + 1], "piece_doc": p["piece_doc"][:p_end]}


def add_chunks(p: dict, C: int) -> dict:
    """Complete chunk_len chunks of every piece: chunk_off per sequence, chunk_pos / chunk_end (in the sequence),
    chunk_doc."""
    n_seq = len(p["seq_off"]) - 1
    pl = p["piece_len"].astype(np.int64)
    seq_of_piece = np.repeat(np.arange(n_seq), np.diff(p["piece_off"]))
    # piece start inside its sequence = global piece start - sequence start
    gstart = np.concatenate([[0], np.cumsum(pl)[:-1]])
    pstart = gstart - p["seq_off"][seq_of_piece]
    nch = pl // C
    rep = np.repeat(np.arange(len(pl)), nch)
    k = np.arange(len(rep)) - np.repeat(np.concatenate([[0], np.cumsum(nch)[:-1]]), nch)
    p["chunk_pos"] = (pstart[rep] + k * C).astype(np.int32)
    p["chunk_end"] = (pstart[rep] + pl[rep]).astype(np.int32)
    p["chunk_doc"] = p["piece_doc"][rep]
    per_seq = np.bincount(seq_of_piece[rep], minlength=n_seq) if len(rep) else np.zeros(n_seq, np.int64)
    p["chunk_off"] = np.concatenate([[0], np.cumsum(per_seq)]).astype(np.int64)
    return p


def tokenize(docs: list[dict], tok, log=print, batch: int = 64):
    """Token ids (+ EOS) and text hashes, exactly as cpt_trainer.data.tokenize_documents."""
    eos = tok.eos_token_id
    ids, th = [], []
    old = tok.model_max_length
    tok.model_max_length = 10 ** 12
    t0 = last = time.time()
    n_tok = 0
    try:
        for s in range(0, len(docs), batch):
            texts = []
            for d in docs[s:s + batch]:
                with open(d["path"], encoding="utf-8", errors="replace") as f:
                    texts.append(f.read())
            for t, x in zip(texts, tok(texts, add_special_tokens=False)["input_ids"]):
                a = np.asarray(x + [eos], dtype=np.int32)
                ids.append(a); th.append(text_hash(t)); n_tok += len(a)
            if time.time() - last > 30:
                last = time.time()
                log(f"    tokenised {s + len(texts):,}/{len(docs):,} documents, {n_tok / 1e6:.0f} M tokens "
                    f"({n_tok / (time.time() - t0) / 1e6:.2f} M tok/s)")
    finally:
        tok.model_max_length = old
    return ids, np.asarray(th, dtype=np.int64)


def bytes_per_token(docs: list[dict], tok, n: int = 400, seed: int = 0) -> float:
    rng = random.Random(seed)
    sample = rng.sample(docs, min(n, len(docs)))
    b = sum(d["bytes"] for d in sample)
    t = 0
    for d in sample:
        with open(d["path"], encoding="utf-8", errors="replace") as f:
            t += len(tok(f.read(), add_special_tokens=False)["input_ids"]) + 1
    return b / max(1, t)


def save_split(path: str, p: dict, docs: list[dict], keys: np.ndarray, th: np.ndarray, out_dir: str, name: str):
    np.savez(path, **p, doc_key=keys, doc_text_hash=th)
    with open(os.path.join(out_dir, f"docs_{name}.txt"), "w", encoding="utf-8") as f:
        f.writelines(f"{d['corpus']}/{d['id']}\n" for d in docs)


def main() -> None:
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--corpora", nargs="+", required=True)
    ap.add_argument("--tokens", type=float, default=1e9, help="training token budget (EOS included)")
    ap.add_argument("--exclude-docs", default=None)
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "models", "Qwen3-0.6B-Base"))
    ap.add_argument("--varlen-min", type=int, default=16384)
    ap.add_argument("--varlen-max", type=int, default=24576)
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--max-val-seqs", type=int, default=32)
    ap.add_argument("--chunk-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-docs-per-corpus", type=int, default=None, help="smoke tests: cap the document list")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    if os.path.exists(os.path.join(out, "meta.json")):
        raise SystemExit(f"{out} is already built")
    os.makedirs(out, exist_ok=True)
    log = print
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    t0 = time.time()
    docs = []
    for c in args.corpora:
        found = list_documents(os.path.abspath(c))
        if args.max_docs_per_corpus:
            found = found[:args.max_docs_per_corpus]
        if not found:
            raise SystemExit(f"no documents in {c}")
        docs += found
    train_docs, val_docs = split_documents(docs, args.val_frac, args.seed)     # == cpt_trainer.longctx.build_varlen
    log(f"{len(docs):,} documents -> {len(train_docs):,} train, {len(val_docs):,} held out ({time.time() - t0:.0f}s)")
    if args.exclude_docs:
        with open(args.exclude_docs, encoding="utf-8") as f:
            excl = {k for k in (line.split("#", 1)[0].strip() for line in f) if k}
        n0 = len(train_docs)
        train_docs = [d for d in train_docs if f"{d['corpus']}/{d['id']}" not in excl]
        log(f"  --exclude-docs: {n0 - len(train_docs):,} training documents removed")

    # ---- proportional subsample of the training documents to the budget ----
    by_c = defaultdict(list)
    for d in train_docs:
        by_c[d["corpus"]].append(d)
    est = {}
    for c, ds in by_c.items():
        bpt = bytes_per_token(ds, tok, seed=args.seed)
        est[c] = (bpt, sum(d["bytes"] for d in ds) / bpt)
    total = sum(v[1] for v in est.values())
    frac = min(1.0, args.tokens / total)
    log(f"  estimated training pool {total / 1e9:.2f} B tokens -> keep {100 * frac:.1f} % of every corpus")
    chosen = []
    for c in sorted(by_c):
        ds = list(by_c[c])
        random.Random(f"{args.seed}/{c}").shuffle(ds)
        bpt, tot = est[c]
        target, acc, k = frac * tot, 0.0, 0
        while k < len(ds) and acc < target:
            acc += ds[k]["bytes"] / bpt
            k += 1
        chosen += ds[:k]
        log(f"    {c:10s} {k:9,d} of {len(ds):9,d} docs, ~{acc / 1e6:8.1f} M of ~{tot / 1e6:8.1f} M tokens ({bpt:.2f} bytes/token)")
    train_docs = chosen

    # ---- training split ----
    ids, th = tokenize(train_docs, tok, log=log)
    keys = np.asarray([doc_key(d["id"]) for d in train_docs], dtype=np.int64)
    stats = {"corpora": {}, "sources": {}}
    for d, a in zip(train_docs, ids):
        for k_, name in (("corpora", d["corpus"]), ("sources", f"{d['corpus']}/{d['source']}")):
            s = stats[k_].setdefault(name, {"docs": 0, "tokens": 0})
            s["docs"] += 1; s["tokens"] += int(len(a))
    p = add_chunks(pack_tracked(ids, args.varlen_min, args.varlen_max, args.seed), args.chunk_len)
    del ids
    lens = np.diff(p["seq_off"])
    stats.update({"train_seqs": int(len(lens)), "train_tokens": int(lens.sum()), "train_chunks": int(len(p["chunk_pos"])),
                  "seq_len_quantiles": {q: int(np.quantile(lens, q / 100)) for q in (0, 10, 50, 90, 100)},
                  "pieces_per_seq_mean": float(np.diff(p["piece_off"]).mean())})
    save_split(os.path.join(out, "train.npz"), p, train_docs, keys, th, out, "train")
    log(f"  train: {stats['train_seqs']:,} sequences, {stats['train_tokens'] / 1e9:.3f} B tokens, "
        f"{stats['train_chunks']:,} chunks of {args.chunk_len} ({time.time() - t0:.0f}s)")
    del p

    # ---- validation: per corpus, the first max_val_seqs packed sequences (== build_varlen) ----
    stats["val"] = {}
    for c in sorted({d["corpus"] for d in docs}):
        vd = [d for d in val_docs if d["corpus"] == c]
        if not vd:
            continue
        vids, vth = tokenize(vd, tok, log=log)
        vp = add_chunks(take_first(pack_tracked(vids, args.varlen_min, args.varlen_max, args.seed), args.max_val_seqs),
                        args.chunk_len)
        used = sorted(set(vp["piece_doc"].tolist()))
        remap = np.full(len(vd), -1, np.int32); remap[used] = np.arange(len(used))
        vp["piece_doc"] = remap[vp["piece_doc"]]; vp["chunk_doc"] = remap[vp["chunk_doc"]]
        vdocs = [vd[i] for i in used]
        save_split(os.path.join(out, f"val_{c}.npz"), vp, vdocs,
                   np.asarray([doc_key(d["id"]) for d in vdocs], np.int64), vth[used], out, f"val_{c}")
        stats["val"][c] = {"seqs": int(len(vp["seq_off"]) - 1), "tokens": int(len(vp["tokens"])),
                           "chunks": int(len(vp["chunk_pos"])), "docs": len(vdocs)}
        log(f"  val {c}: {stats['val'][c]}")
    meta = {"args": vars(args), "stats": stats, "built": time.strftime("%Y-%m-%d %H:%M"),
            "corpora": [os.path.relpath(os.path.abspath(c), ROOT) for c in args.corpora]}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
