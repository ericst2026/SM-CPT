"""Small helpers shared by the RETRO pipeline: stable 64-bit hashes, token n-gram overlap."""
from __future__ import annotations

import hashlib
import re

import numpy as np
import torch

_WS = re.compile(r"\s+")


def h64(s: str) -> int:
    """Stable signed 64-bit hash of a string (blake2b; Python's hash() is salted per process)."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=8).digest(), "little", signed=True)


def doc_key(source_id: str) -> int:
    """Document identity: the corpus manifest id ("nemotron_cc:<uuid>", "fineweb_edu:<uuid>", ...). The database
    builder derives the same id from the parquet rows, so a training document and its database copy share it."""
    return h64(source_id)


def text_hash(text: str) -> int:
    """Exact-duplicate identity: lower-cased text with whitespace collapsed."""
    return h64(_WS.sub(" ", text.strip().lower()))


NO_GRAM = torch.iinfo(torch.long).min


def ngram_hashes(tok: torch.Tensor, n: int) -> torch.Tensor:
    """[..., T] int token ids (-1 = padding) -> [..., T-n+1] int64 hashes of the n-grams (NO_GRAM where padding is
    inside; hashes wrap around and may be negative)."""
    t = tok.long()
    T = t.shape[-1]
    if T < n:
        return torch.full(t.shape[:-1] + (0,), NO_GRAM, dtype=torch.long)
    h = torch.zeros(t.shape[:-1] + (T - n + 1,), dtype=torch.long)
    bad = torch.zeros_like(h, dtype=torch.bool)
    P = 1_000_003
    for j in range(n):
        w = t[..., j:T - n + 1 + j]
        h = h * P + (w + 1)
        bad |= w < 0
    return torch.where(bad, torch.full_like(h, NO_GRAM), h)


def overlap_fraction(query: torch.Tensor, cands: torch.Tensor, n: int = 8) -> torch.Tensor:
    """Share of the query's n-grams that occur in each candidate.
    query [B, Tq], cands [B, K, Tc] (token ids, -1 padding) -> [B, K] in [0, 1] (0 when the query has no n-gram)."""
    q = ngram_hashes(query, n)                     # [B, Gq]
    c = ngram_hashes(cands, n)                     # [B, K, Gc]
    if q.shape[-1] == 0 or c.shape[-1] == 0:
        return torch.zeros(cands.shape[:2])
    qv = q != NO_GRAM
    hit = (q[:, None, :, None] == c[:, :, None, :]) & (c[:, :, None, :] != NO_GRAM)   # [B, K, Gq, Gc]
    found = hit.any(-1) & qv[:, None, :]
    return found.sum(-1).float() / qv.sum(-1, keepdim=True).clamp(min=1).float()


def longest_common_run(a: np.ndarray, b: np.ndarray) -> int:
    """Longest common contiguous token run of two 1-D arrays (RETRO's chunk overlap r(C), in tokens)."""
    a = a[a >= 0]; b = b[b >= 0]
    if not len(a) or not len(b):
        return 0
    eq = a[:, None] == b[None, :]
    best = 0
    prev = np.zeros(len(b) + 1, dtype=np.int32)
    for i in range(len(a)):
        cur = np.zeros_like(prev)
        cur[1:] = np.where(eq[i], prev[:-1] + 1, 0)
        best = max(best, int(cur.max()))
        prev = cur
    return best


def row_hash(arr: np.ndarray) -> np.ndarray:
    """64-bit identity hash of the last axis of an int array (padding -1 included), wrapping arithmetic."""
    a = np.asarray(arr, dtype=np.int64) + 2
    p = np.ones(a.shape[-1], dtype=np.int64)
    with np.errstate(over="ignore"):
        for i in range(1, len(p)):
            p[i] = p[i - 1] * 1_000_003
        return (a * p).sum(-1)
