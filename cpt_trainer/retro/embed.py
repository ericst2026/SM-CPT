"""Frozen chunk embedder for retrieval keys and queries (RETRO uses a frozen BERT; the retro-fit literature finds
sentence encoders much better than raw BERT at small database sizes, e.g. Retro-li SBERT vs BERT).

Default: BAAI/bge-small-en-v1.5 (33 M params, 384-d, CLS pooling, L2-normalised -> inner product = cosine), in
models/bge-small-en-v1.5. Chunks are Qwen3 token ids; they are decoded to text and re-tokenised by the embedder.
Keys (database chunks) and queries (training / eval chunks) are embedded the same way (symmetric similarity, no
query instruction prefix).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_EMBEDDER = os.path.join(ROOT, "models", "bge-small-en-v1.5")


class ChunkEmbedder:
    def __init__(self, path: str = DEFAULT_EMBEDDER, device: str | None = None, max_len: int = 128, batch: int = 1024,
                 pooling: str | None = None):
        from transformers import AutoModel, AutoTokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(path, dtype=dtype).to(self.device).eval()
        self.max_len, self.batch = max_len, batch
        if pooling is None:   # sentence-transformers pooling config if present, else CLS
            pooling = "cls"
            pc = os.path.join(path, "1_Pooling", "config.json")
            if os.path.exists(pc):
                with open(pc) as f:
                    c = json.load(f)
                pooling = "mean" if c.get("pooling_mode_mean_tokens") else "cls"
        self.pooling = pooling
        self.dim = self.model.config.hidden_size
        self.name = os.path.basename(os.path.normpath(path))

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        order = np.argsort([len(t) for t in texts])           # length-bucketed batches
        for s in range(0, len(texts), self.batch):
            idx = order[s:s + self.batch]
            enc = self.tok([texts[i] for i in idx], padding=True, truncation=True, max_length=self.max_len,
                           return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state
            if self.pooling == "mean":
                m = enc["attention_mask"][..., None].to(h.dtype)
                e = (h * m).sum(1) / m.sum(1).clamp(min=1)
            else:
                e = h[:, 0]
            e = torch.nn.functional.normalize(e.float(), dim=-1)
            out[idx] = e.cpu().numpy()
        return out


def decode_chunks(qwen_tok, chunks: np.ndarray) -> list[str]:
    """Qwen3 token ids [n, T] (-1 padding) -> texts."""
    rows = [r[r >= 0].tolist() for r in chunks]
    return qwen_tok.batch_decode(rows, skip_special_tokens=True)
