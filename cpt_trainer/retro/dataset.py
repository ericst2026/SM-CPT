"""Training / validation datasets of a RETRO training set (cpt_trainer.retro.data + neighbours) and the collator."""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from cpt_trainer.longctx import VarlenCollator
from cpt_trainer.retro.db import Database


class RetroSplit(Dataset):
    """One split (train.npz / val_<corpus>.npz) with its precomputed neighbours (int32 [n_chunks, k_store])."""

    def __init__(self, set_dir: str, split: str, nbr_name: str | None, max_seqs: int | None = None):
        z = np.load(os.path.join(set_dir, f"{split}.npz"))
        self.p = {k: z[k] for k in ("tokens", "seq_off", "piece_len", "piece_off", "chunk_off", "chunk_pos", "chunk_end")}
        self.n = len(self.p["seq_off"]) - 1 if max_seqs is None else min(max_seqs, len(self.p["seq_off"]) - 1)
        self.nbr = None
        if nbr_name:
            f = os.path.join(set_dir, "nbr", nbr_name, f"{split}.npy")
            if not os.path.exists(f):
                raise FileNotFoundError(f"{f}: run python -m cpt_trainer.retro.neighbours first")
            self.nbr = np.load(f, mmap_mode="r")
            assert len(self.nbr) == len(self.p["chunk_pos"]), "neighbour file does not match the split"

    def __len__(self):
        return self.n

    def lengths(self):
        return np.diff(self.p["seq_off"])[: self.n]

    def __getitem__(self, i: int) -> dict:
        p = self.p
        c0, c1 = int(p["chunk_off"][i]), int(p["chunk_off"][i + 1])
        return {"idx": i,
                "input_ids": torch.from_numpy(p["tokens"][p["seq_off"][i]:p["seq_off"][i + 1]].astype(np.int64)),
                "pieces": torch.from_numpy(p["piece_len"][p["piece_off"][i]:p["piece_off"][i + 1]].astype(np.int64)),
                "chunk_pos": p["chunk_pos"][c0:c1].astype(np.int64), "chunk_end": p["chunk_end"][c0:c1].astype(np.int64),
                "nbr_ids": np.array(self.nbr[c0:c1]) if self.nbr is not None else None}


class RetroCollator:
    """VarlenCollator fields (input_ids, labels, position_ids, doc_ids) + chunk_pos, chunk_end, nbr_tokens.

    mode: "on" (precomputed neighbours), "off" (no chunks: decoder only), "random" (random database chunks; tests
    whether the content of the neighbours is used), "oracle" (every chunk's last neighbour slot = the true chunk and
    its continuation: a copy ceiling).
    oracle_rate: during training, the share of chunks whose last neighbour slot is replaced by the oracle neighbour
    (activation warm-up; the trainer anneals it to 0). nbr_dropout: share of chunks that get no neighbours."""

    def __init__(self, db: Database | None, k: int, nbr_len: int, pad_id: int, pad_multiple: int = 2048,
                 seed: int = 0, mode: str = "on"):
        self.base = VarlenCollator(pad_id, pad_multiple)
        self.db, self.k, self.nbr_len, self.seed, self.mode = db, k, nbr_len, seed, mode
        self.oracle_rate = 0.0
        self.nbr_dropout = 0.0
        self.step = 0

    def __call__(self, batch: list[dict]) -> dict:
        assert len(batch) == 1
        it = batch[0]
        out = self.base([{"input_ids": it["input_ids"], "pieces": it["pieces"]}])
        if self.mode == "off" or self.db is None or it["nbr_ids"] is None or len(it["chunk_pos"]) == 0:
            return out
        rng = np.random.default_rng([self.seed, int(it["idx"]), self.step])
        ids = it["nbr_ids"][:, : self.k].astype(np.int64)
        if ids.shape[1] < self.k:
            ids = np.pad(ids, ((0, 0), (0, self.k - ids.shape[1])), constant_values=-1)
        if self.mode == "random":
            ids = rng.integers(0, len(self.db), size=ids.shape)
        nbr = self.db.neighbours(ids, self.nbr_len)                     # [n, k, T]
        pos, end = it["chunk_pos"], it["chunk_end"]
        rate = 1.0 if self.mode == "oracle" else self.oracle_rate
        if rate > 0:
            pick = rng.random(len(pos)) < rate
            if pick.any():
                seq = it["input_ids"].numpy()
                ar = np.arange(self.nbr_len)
                idx = pos[pick, None] + ar[None]
                ok = idx < end[pick, None]
                nbr[pick, self.k - 1] = np.where(ok, seq[np.where(ok, idx, 0)], -1)
        if self.nbr_dropout > 0:
            drop = rng.random(len(pos)) < self.nbr_dropout
            nbr[drop] = -1
        out["chunk_pos"] = torch.from_numpy(pos)
        out["chunk_end"] = torch.from_numpy(end)
        out["nbr_tokens"] = torch.from_numpy(nbr)
        return out
