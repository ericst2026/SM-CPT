"""Corpus folders -> packed token blocks for continued pre-training.

A corpus folder is what scripts/download_corpus.py writes: manifest.jsonl + text/<source>/<id>.txt.
Documents flagged low_quality in the manifest are skipped unless asked otherwise. Every document is
tokenised once, terminated with the tokenizer's EOS token, and the concatenated stream is cut into
fixed-length blocks ("packing"). Per corpus, a small share of the documents is held out before packing,
so the trainer can report a validation loss on AI text and on replay text separately; long documents
(books) are never held out, they stay in training.

Packed blocks are cached as a NumPy .npz keyed by the corpus contents, tokenizer, block length and
split parameters, so later runs on the same data start in seconds.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
def read_manifest(corpus_dir: str) -> list[dict]:
    path = os.path.join(corpus_dir, "manifest.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def list_documents(corpus_dir: str, include_low_quality: bool = False, exclude_sources=()) -> list[dict]:
    """One record per text file: corpus name, source, id, path, size. Uses the manifest when there is one
    (low_quality flag, source names); otherwise every text/**/*.txt."""
    corpus = os.path.basename(os.path.normpath(corpus_dir))
    docs, manifest = [], read_manifest(corpus_dir)
    if manifest:
        for d in manifest:
            if d.get("low_quality") and not include_low_quality:
                continue
            if d.get("source") in exclude_sources:
                continue
            path = os.path.join(corpus_dir, d["text_path"])
            if os.path.isfile(path):
                docs.append({"corpus": corpus, "source": d.get("source", "?"), "id": d["id"], "path": path,
                             "bytes": os.path.getsize(path)})
        return docs
    text_dir = os.path.join(corpus_dir, "text")
    for root, _, files in os.walk(text_dir):
        source = os.path.relpath(root, text_dir).split(os.sep)[0]
        if source in exclude_sources:
            continue
        for fn in sorted(files):
            if fn.endswith(".txt"):
                path = os.path.join(root, fn)
                docs.append({"corpus": corpus, "source": source, "id": f"{source}/{fn[:-4]}", "path": path,
                             "bytes": os.path.getsize(path)})
    return docs


def split_documents(docs: list[dict], val_frac: float, seed: int, max_val_doc_bytes: int = 200_000):
    """Hold out `val_frac` of the documents of each corpus (at least one, only documents up to
    max_val_doc_bytes so that books stay in training). Returns (train_docs, val_docs)."""
    by_corpus = defaultdict(list)
    for d in docs:
        by_corpus[d["corpus"]].append(d)
    train, val = [], []
    rng = random.Random(seed)
    for corpus, ds in sorted(by_corpus.items()):
        cands = [d for d in ds if d["bytes"] <= max_val_doc_bytes]
        rng.shuffle(cands)
        n_val = min(len(cands), max(1, round(len(ds) * val_frac))) if val_frac > 0 else 0
        chosen = {id(d) for d in cands[:n_val]}
        val += [d for d in ds if id(d) in chosen]
        train += [d for d in ds if id(d) not in chosen]
    return train, val


# --------------------------------------------------------------------------- #
# Tokenisation and packing
# --------------------------------------------------------------------------- #
def tokenize_documents(docs: list[dict], tok, batch: int = 16, log=print) -> list[np.ndarray]:
    """Token ids per document (int32), each terminated with EOS."""
    eos = tok.eos_token_id
    out, n_tok, t0, last = [], 0, time.time(), time.time()
    old_max = tok.model_max_length
    tok.model_max_length = 10 ** 12  # whole documents; no "sequence too long" warnings
    try:
        for s in range(0, len(docs), batch):
            texts = []
            for d in docs[s:s + batch]:
                with open(d["path"], encoding="utf-8", errors="replace") as f:
                    texts.append(f.read())
            for ids in tok(texts, add_special_tokens=False)["input_ids"]:
                arr = np.asarray(ids + [eos], dtype=np.int32)
                out.append(arr)
                n_tok += len(arr)
            if time.time() - last > 15:
                last = time.time()
                log(f"    tokenised {s + len(texts)}/{len(docs)} documents, {n_tok / 1e6:.1f} M tokens "
                    f"({n_tok / (time.time() - t0) / 1e6:.2f} M tok/s)")
    finally:
        tok.model_max_length = old_max
    return out


def pack_blocks(doc_ids: list[np.ndarray], seq_len: int, seed: int) -> np.ndarray:
    """Shuffle documents, concatenate, cut into [n_blocks, seq_len] (the tail that does not fill a block
    is dropped)."""
    if not doc_ids:
        return np.zeros((0, seq_len), dtype=np.int32)
    order = list(range(len(doc_ids)))
    random.Random(seed).shuffle(order)
    stream = np.concatenate([doc_ids[i] for i in order])
    n = len(stream) // seq_len
    return stream[: n * seq_len].reshape(n, seq_len)


def build_blocks(corpus_dirs: list[str], tok, seq_len: int, val_frac: float, seed: int, cache_dir: str,
                 include_low_quality: bool = False, exclude_sources=(), max_val_blocks: int = 64,
                 exclude_train=frozenset(), log=print) -> dict:
    """Returns {"train": int32 [n, seq_len], "val": {corpus: int32 [m, seq_len]}, "stats": {...}}.
    exclude_train: "<corpus>/<doc id>" keys dropped from the training split after the train/val split."""
    docs = []
    for c in corpus_dirs:
        found = list_documents(c, include_low_quality, exclude_sources)
        if not found:
            raise SystemExit(f"no documents in {c} (expected manifest.jsonl + text/**/*.txt)")
        docs += found
    fingerprint = {
        "docs": [(d["corpus"], d["id"], d["bytes"], int(os.path.getmtime(d["path"]))) for d in docs],
        "tokenizer": getattr(tok, "name_or_path", str(tok)), "seq_len": seq_len, "val_frac": val_frac,
        "seed": seed, "max_val_blocks": max_val_blocks, "eos": tok.eos_token_id, "version": 1,
    }
    if exclude_train:  # only when set, so the cache keys of earlier runs stay valid
        fingerprint["exclude_train"] = sorted(exclude_train)
    key = hashlib.sha1(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:12]
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"blocks_{key}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        stats = json.loads(str(z["stats"]))
        val = {k[4:]: z[k] for k in z.files if k.startswith("val_")}
        log(f"  packed blocks from cache {cache}: {len(z['train']):,} train blocks")
        return {"train": z["train"], "val": val, "stats": stats}

    train_docs, val_docs = split_documents(docs, val_frac, seed)
    log(f"  {len(docs):,} documents from {len(corpus_dirs)} corpora -> {len(train_docs):,} train, {len(val_docs)} held out")
    if exclude_train:
        n0 = len(train_docs)
        train_docs = [d for d in train_docs if f"{d['corpus']}/{d['id']}" not in exclude_train]
        log(f"  excluded {n0 - len(train_docs):,} of {len(exclude_train):,} listed documents from training")
    t0 = time.time()
    train_ids = tokenize_documents(train_docs, tok, log=log)
    val_ids = tokenize_documents(val_docs, tok, log=log)
    log(f"  tokenised in {time.time() - t0:.0f}s")

    stats = {"seq_len": seq_len, "corpora": {}, "sources": {}}
    for d, ids in zip(train_docs + val_docs, train_ids + val_ids):
        c = stats["corpora"].setdefault(d["corpus"], {"docs": 0, "tokens": 0, "val_docs": 0, "val_tokens": 0})
        s = stats["sources"].setdefault(f"{d['corpus']}/{d['source']}", {"docs": 0, "tokens": 0})
        s["docs"] += 1
        s["tokens"] += int(len(ids))
        if d in val_docs:
            c["val_docs"] += 1
            c["val_tokens"] += int(len(ids))
        else:
            c["docs"] += 1
            c["tokens"] += int(len(ids))

    train = pack_blocks(train_ids, seq_len, seed)
    val = {}
    for corpus in sorted(stats["corpora"]):
        ids = [v for d, v in zip(val_docs, val_ids) if d["corpus"] == corpus]
        blocks = pack_blocks(ids, seq_len, seed)[:max_val_blocks]
        if len(blocks):
            val[corpus] = blocks
        stats["corpora"][corpus]["train_blocks"] = int(sum(len(v) for d, v in zip(train_docs, train_ids)
                                                            if d["corpus"] == corpus) // seq_len)
        stats["corpora"][corpus]["val_blocks"] = int(len(blocks))
    stats["train_blocks"] = int(len(train))
    stats["train_tokens"] = int(train.size)
    stats["val_blocks"] = {k: int(len(v)) for k, v in val.items()}
    np.savez(cache, train=train, stats=np.asarray(json.dumps(stats)), **{f"val_{k}": v for k, v in val.items()})
    log(f"  packed {len(train):,} train blocks of {seq_len} tokens ({train.size / 1e6:.2f} M tokens) -> {cache}")
    return {"train": train, "val": val, "stats": stats}


class BlockDataset(Dataset):
    """Fixed-length token blocks; labels = input_ids (the model shifts them internally)."""

    def __init__(self, blocks: np.ndarray):
        self.blocks = torch.from_numpy(np.ascontiguousarray(blocks).astype(np.int64))

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, i: int) -> dict:
        x = self.blocks[i]
        return {"input_ids": x, "labels": x.clone()}
