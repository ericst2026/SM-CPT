"""Long-context continued pre-training: variable-length packing with per-document attention masks.

Packing ("varlen"): all training documents (every corpus) are tokenised, EOS-terminated, shuffled together
and packed greedily into sequences of `min_len`..`max_len` tokens. A document that does not fit the open
sequence closes it if that sequence is already >= min_len; otherwise the document is split to fill the
sequence up to max_len and its remainder starts the next one. Documents longer than max_len are therefore
cut into max_len pieces, so real long documents (books, papers) fill whole sequences on their own.

Attention: each piece only sees itself (document mask, causal), and its RoPE positions restart at 0, so
packing short documents together costs no cross-document contamination and the attention cost is
sum(piece_len^2), not seq_len^2. Implemented with PyTorch flex attention (a BlockMask built from document
ids per sequence). Sequences are right-padded to a multiple of `pad_multiple` so the compiled kernel only
sees a few distinct shapes; padding is its own "document" and never a target.

Loss: the LM head is applied in chunks under activation checkpointing, so the fp32 logits of a 24 k-token
sequence (152 k vocabulary: ~15 GB) are never materialised at once. The first token of every piece is
never a target (it would be "predicted" from the previous document).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from cpt_trainer.data import list_documents, split_documents, tokenize_documents


# --------------------------------------------------------------------------- #
# Packing
# --------------------------------------------------------------------------- #
def pack_varlen(doc_ids: list[np.ndarray], min_len: int, max_len: int, seed: int) -> dict:
    """Greedy packing into [min_len, max_len] sequences. Returns flat arrays:
    tokens (int32, all sequences concatenated), seq_off (n_seq + 1), piece_len (int32, all pieces),
    piece_off (n_seq + 1; pieces of sequence i are piece_len[piece_off[i]:piece_off[i+1]])."""
    order = list(range(len(doc_ids)))
    random.Random(seed).shuffle(order)
    total = int(sum(len(d) for d in doc_ids))
    tokens = np.empty(total, dtype=np.int32)
    seq_off, piece_len, piece_off = [0], [], [0]
    pos, cur = 0, 0  # write position, tokens in the open sequence

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
            if cur >= max_len:
                close()
    if cur:
        close()
    return {"tokens": tokens[:pos], "seq_off": np.asarray(seq_off, dtype=np.int64),
            "piece_len": np.asarray(piece_len, dtype=np.int32), "piece_off": np.asarray(piece_off, dtype=np.int64)}


def take_sequences(p: dict, idx) -> dict:
    """Sub-select sequences (e.g. the first N validation sequences) from a packed dict."""
    idx = list(idx)
    toks = [p["tokens"][p["seq_off"][i]:p["seq_off"][i + 1]] for i in idx]
    pls = [p["piece_len"][p["piece_off"][i]:p["piece_off"][i + 1]] for i in idx]
    return {"tokens": np.concatenate(toks) if toks else np.zeros(0, np.int32),
            "seq_off": np.cumsum([0] + [len(t) for t in toks]).astype(np.int64),
            "piece_len": np.concatenate(pls) if pls else np.zeros(0, np.int32),
            "piece_off": np.cumsum([0] + [len(x) for x in pls]).astype(np.int64)}


def blocks_as_packed(blocks: np.ndarray) -> dict:
    """Fixed [n, L] blocks (the 2048-token validation sets of earlier runs) as packed sequences of one
    piece each: plain causal attention over the block, exactly how those runs computed their val loss."""
    n, L = blocks.shape
    return {"tokens": blocks.reshape(-1).astype(np.int32), "seq_off": np.arange(n + 1, dtype=np.int64) * L,
            "piece_len": np.full(n, L, dtype=np.int32), "piece_off": np.arange(n + 1, dtype=np.int64)}


def build_varlen(corpus_dirs: list[str], tok, min_len: int, max_len: int, val_frac: float, seed: int, cache_dir: str,
                 include_low_quality: bool = False, exclude_sources=(), max_val_seqs: int = 32,
                 legacy_val: dict | None = None, exclude_train=frozenset(), log=print) -> dict:
    """Corpus folders -> {"train": packed, "val": {corpus: packed}, "stats": {...}} (cached as .npz).

    legacy_val = {"cache": path to an earlier fixed-block .npz, "corpora": [names], "seq_len": 2048,
    "split_seed": seed}: for those corpora, the validation documents and blocks are exactly the ones of the
    earlier run (same split of the same documents), so val losses stay comparable across runs.
    exclude_train: "<corpus>/<doc id>" keys dropped from the training split after the train/val split, so the
    validation sets do not change."""
    docs = []
    for c in corpus_dirs:
        found = list_documents(c, include_low_quality, exclude_sources)
        if not found:
            raise SystemExit(f"no documents in {c} (expected manifest.jsonl + text/**/*.txt)")
        docs += found
    fingerprint = {
        "docs": [(d["corpus"], d["id"], d["bytes"], int(os.path.getmtime(d["path"]))) for d in docs],
        "tokenizer": getattr(tok, "name_or_path", str(tok)), "min_len": min_len, "max_len": max_len,
        "val_frac": val_frac, "seed": seed, "max_val_seqs": max_val_seqs, "eos": tok.eos_token_id,
        "legacy_val": legacy_val, "version": "varlen-1",
    }
    if exclude_train:  # only when set, so the cache keys of earlier runs stay valid
        fingerprint["exclude_train"] = sorted(exclude_train)
    key = hashlib.sha1(json.dumps(fingerprint, sort_keys=True, default=str).encode()).hexdigest()[:12]
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"varlen_{key}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        stats = json.loads(str(z["stats"]))
        out = {"train": {k: z[f"train_{k}"] for k in ("tokens", "seq_off", "piece_len", "piece_off")}, "val": {},
               "stats": stats}
        for c in stats["val_corpora"]:
            out["val"][c] = {k: z[f"val_{c}_{k}"] for k in ("tokens", "seq_off", "piece_len", "piece_off")}
        log(f"  packed sequences from cache {cache}: {len(out['train']['seq_off']) - 1:,} train sequences")
        return out

    legacy_corpora = set((legacy_val or {}).get("corpora", []))
    legacy_docs = [d for d in docs if d["corpus"] in legacy_corpora]
    other_docs = [d for d in docs if d["corpus"] not in legacy_corpora]
    train_docs, val_docs = [], []
    if legacy_docs:
        tr, va = split_documents(legacy_docs, val_frac, legacy_val.get("split_seed", seed))
        train_docs += tr
        val_docs += va
    if other_docs:
        tr, va = split_documents(other_docs, val_frac, seed)
        train_docs += tr
        val_docs += va
    log(f"  {len(docs):,} documents from {len(corpus_dirs)} corpora -> {len(train_docs):,} train, {len(val_docs):,} held out")
    if exclude_train:
        n0 = len(train_docs)
        train_docs = [d for d in train_docs if f"{d['corpus']}/{d['id']}" not in exclude_train]
        log(f"  excluded {n0 - len(train_docs):,} of {len(exclude_train):,} listed documents from training")
    t0 = time.time()
    train_ids = tokenize_documents(train_docs, tok, log=log)
    log(f"  tokenised {len(train_docs):,} train documents in {time.time() - t0:.0f}s")

    stats = {"min_len": min_len, "max_len": max_len, "corpora": {}, "sources": {}}
    for d, ids in zip(train_docs, train_ids):
        c = stats["corpora"].setdefault(d["corpus"], {"docs": 0, "tokens": 0, "long_docs": 0, "long_tokens": 0})
        s = stats["sources"].setdefault(f"{d['corpus']}/{d['source']}", {"docs": 0, "tokens": 0})
        c["docs"] += 1
        c["tokens"] += int(len(ids))
        s["docs"] += 1
        s["tokens"] += int(len(ids))
        if len(ids) >= min_len:
            c["long_docs"] += 1
            c["long_tokens"] += int(len(ids))
    train = pack_varlen(train_ids, min_len, max_len, seed)
    del train_ids
    lens = np.diff(train["seq_off"])
    stats["train_seqs"] = int(len(lens))
    stats["train_tokens"] = int(lens.sum())
    stats["seq_len_quantiles"] = {q: int(np.quantile(lens, q / 100)) for q in (0, 10, 50, 90, 100)}
    stats["pieces_per_seq_mean"] = float(np.diff(train["piece_off"]).mean())

    # validation: legacy corpora = the earlier run's fixed blocks; other corpora = varlen sequences
    val, legacy_blocks = {}, {}
    if legacy_corpora:
        z = np.load(legacy_val["cache"], allow_pickle=False)
        legacy_blocks = {k[4:]: z[k] for k in z.files if k.startswith("val_")}
    for corpus in sorted({d["corpus"] for d in docs}):
        if corpus in legacy_corpora:
            if corpus in legacy_blocks:
                val[corpus] = blocks_as_packed(legacy_blocks[corpus])
            continue
        vd = [d for d in val_docs if d["corpus"] == corpus]
        if not vd:
            continue
        vids = tokenize_documents(vd, tok, log=log)
        p = pack_varlen(vids, min_len, max_len, seed)
        n = min(max_val_seqs, len(p["seq_off"]) - 1)
        if n:
            val[corpus] = take_sequences(p, range(n))
    stats["val_corpora"] = sorted(val)
    stats["val"] = {c: {"seqs": int(len(v["seq_off"]) - 1), "tokens": int(len(v["tokens"])),
                        "fixed_blocks": c in legacy_corpora} for c, v in val.items()}
    arrays = {f"train_{k}": v for k, v in train.items()}
    for c, v in val.items():
        arrays.update({f"val_{c}_{k}": a for k, a in v.items()})
    np.savez(cache, stats=np.asarray(json.dumps(stats)), **arrays)
    log(f"  packed {stats['train_seqs']:,} train sequences ({stats['train_tokens'] / 1e6:.1f} M tokens, length "
        f"quantiles {stats['seq_len_quantiles']}, {stats['pieces_per_seq_mean']:.1f} pieces per sequence) -> {cache}")
    return {"train": train, "val": val, "stats": stats}


class VarlenDataset(Dataset):
    def __init__(self, packed: dict, max_seqs: int | None = None):
        self.p = packed
        self.n = len(packed["seq_off"]) - 1 if max_seqs is None else min(max_seqs, len(packed["seq_off"]) - 1)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        p = self.p
        toks = torch.from_numpy(p["tokens"][p["seq_off"][i]:p["seq_off"][i + 1]].astype(np.int64))
        pieces = torch.from_numpy(p["piece_len"][p["piece_off"][i]:p["piece_off"][i + 1]].astype(np.int64))
        return {"input_ids": toks, "pieces": pieces}

    def lengths(self) -> np.ndarray:
        return np.diff(self.p["seq_off"])[: self.n]


class VarlenCollator:
    """Batch size 1: one packed sequence, right-padded to a multiple of pad_multiple.
    Returns input_ids, labels, position_ids (restart per piece) and doc_ids (piece index; padding = -1)."""

    def __init__(self, pad_id: int, pad_multiple: int = 2048):
        self.pad_id, self.mult = pad_id, pad_multiple

    def __call__(self, batch: list[dict]) -> dict:
        assert len(batch) == 1, "varlen packing uses one sequence per device step (--bs 1)"
        ids, pieces = batch[0]["input_ids"], batch[0]["pieces"]
        L = len(ids)
        Lp = -(-L // self.mult) * self.mult
        doc = torch.repeat_interleave(torch.arange(len(pieces)), pieces)
        starts = torch.cumsum(pieces, 0) - pieces
        pos = torch.arange(L) - torch.repeat_interleave(starts, pieces)
        labels = ids.clone()
        labels[starts] = -100  # first token of every piece is never a target
        pad = Lp - L
        if pad:
            ids = torch.cat([ids, torch.full((pad,), self.pad_id, dtype=ids.dtype)])
            labels = torch.cat([labels, torch.full((pad,), -100, dtype=labels.dtype)])
            pos = torch.cat([pos, torch.arange(pad)])
            doc = torch.cat([doc, torch.full((pad,), -1, dtype=doc.dtype)])
        return {"input_ids": ids[None], "labels": labels[None], "position_ids": pos[None], "doc_ids": doc[None]}


# --------------------------------------------------------------------------- #
# Model side
# --------------------------------------------------------------------------- #
def document_block_mask(doc_ids: torch.Tensor):
    """Causal + same-document BlockMask for one sequence (doc_ids [L], padding = its own document)."""
    from torch.nn.attention.flex_attention import create_block_mask

    doc = doc_ids

    def mask_mod(b, h, q, kv):
        return (doc[q] == doc[kv]) & (q >= kv)

    L = doc.shape[0]
    return create_block_mask(mask_mod, B=1, H=None, Q_LEN=L, KV_LEN=L, device=doc.device, _compile=True)


def _ce_chunk(h: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy((h @ w.t()).float(), y, ignore_index=-100, reduction="sum")


def chunked_lm_loss(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Sum of token cross-entropies; logits are built chunk by chunk and recomputed in backward."""
    from torch.utils.checkpoint import checkpoint

    total = hidden.new_zeros((), dtype=torch.float32)
    for s in range(0, hidden.shape[0], chunk):
        y = targets[s:s + chunk]
        if (y != -100).any():
            total = total + checkpoint(_ce_chunk, hidden[s:s + chunk], weight, y, use_reentrant=False)
    return total


def inner_lm(model):
    """(decoder stack, lm_head weight) of a (PEFT-wrapped) causal LM."""
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return m.model, m.lm_head.weight


def varlen_loss(model, inputs: dict, num_items_in_batch=None, loss_chunk: int = 4096):
    """Mean next-token loss of one packed sequence (or the share of the accumulated batch when
    num_items_in_batch is given), with the document mask."""
    body, w = inner_lm(model)
    doc = inputs["doc_ids"][0]
    mask = document_block_mask(doc)
    hid = body(input_ids=inputs["input_ids"], position_ids=inputs["position_ids"], attention_mask=mask,
               use_cache=False).last_hidden_state[0]
    y = inputs["labels"][0, 1:]
    loss_sum = chunked_lm_loss(hid[:-1], w, y, loss_chunk)
    denom = num_items_in_batch if num_items_in_batch is not None else (y != -100).sum().clamp(min=1)
    return loss_sum / denom
