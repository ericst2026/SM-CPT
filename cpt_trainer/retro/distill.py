"""Retrieval as a training-only teacher (docs/retro_distill_lora_cpt.md).

The student is the deployed model: Qwen3-0.6B-Base + a LoRA, with no retrieval at inference. During training a
teacher sees the same text PLUS neighbours retrieved from the AI database, and the student is trained on
    (1 - alpha) * CE(student, next token) + alpha * KL(teacher || student)
per token, so that what the neighbours tell the teacher about the next token is pushed into the LoRA weights
(prompt / context distillation: Kujanpaa et al. 2024; kNN-KD: Yang et al. 2022).

Teachers:
  context  the decoder itself (with the current LoRA: "self" teacher, or the frozen base: "base") reading the
           neighbours in its context. The packed sequence is cut into windows of W tokens inside every document piece;
           each window gets a block of up to k neighbours (k * nbr_len tokens) placed right before it. Attention
           (ContextTeacher / teacher_mask): a window's tokens see their own document (all earlier windows, as the
           student does) and their own neighbour block only; neighbour tokens see only their own block. Position ids
           of the document tokens are shifted by a constant, so with no neighbours the teacher equals the student.
  retro    base + the RETRO module of cpt_trainer/retro/model.py (chunked cross-attention, trained in the same run
           on the teacher path with the LoRA switched off). It is dropped after training.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from cpt_trainer.retro.common import overlap_fraction
from cpt_trainer.retro.model import unwrap


# --------------------------------------------------------------------------- #
# LoRA on / off (only the LoRA layers: a RETRO module kept as PEFT modules_to_save stays active)
# --------------------------------------------------------------------------- #
def lora_layers(model) -> list:
    from peft.tuners.lora import LoraLayer
    return [m for m in model.modules() if isinstance(m, LoraLayer)]


@contextmanager
def lora_disabled(layers: list, off: bool = True):
    """Run the base weights only. Sets PEFT's flag directly: enable_adapters() would also flip requires_grad.
    A backward through activations recomputed by gradient checkpointing must run inside the same context."""
    if not off or not layers:
        yield
        return
    old = [m._disable_adapters for m in layers]
    for m in layers:
        m._disable_adapters = True
    try:
        yield
    finally:
        for m, o in zip(layers, old):
            m._disable_adapters = o


# --------------------------------------------------------------------------- #
# Context teacher: sequence layout (CPU, in the collator)
# --------------------------------------------------------------------------- #
class ContextTeacher:
    """Builds the teacher sequence of one packed training sequence.

    Neighbour choice for a window: the precomputed per-chunk neighbours (cpt_trainer/retro/neighbours.py, already
    filtered for same document, same text and copies of the continuation) of the chunks inside the window, taken
    rank by rank with the chunks visited in a spread order (so the k picks cover the whole window), skipping
    database chunks adjacent to one already picked (their 128-token neighbours overlap by half) and neighbours
    whose `overlap_n`-grams lie mostly inside the window (>= max_overlap: a copy of the window teaches nothing
    beyond the cross-entropy). The window may look ahead: the teacher is privileged, only the student is scored."""

    def __init__(self, db, k: int = 4, nbr_len: int = 128, window: int = 1024, sep_ids=(), max_overlap: float = 0.5,
                 overlap_n: int = 8, pad_multiple: int = 2048, max_candidates: int = 4, pad_id: int = 0):
        if window % db.chunk_len:
            raise ValueError(f"window {window} must be a multiple of the chunk length {db.chunk_len}")
        self.db, self.k, self.nbr_len, self.window = db, k, nbr_len, window
        self.sep = np.asarray(list(sep_ids), dtype=np.int64)
        self.max_overlap, self.n, self.mult = max_overlap, overlap_n, pad_multiple
        self.max_candidates = max_candidates           # overlap-checked candidates per accepted neighbour
        self.pad_id = pad_id
        self.block_max = k * (nbr_len + len(self.sep))  # position shift of the document tokens

    def _order(self, n_chunks: int) -> np.ndarray:
        stride = max(1, -(-n_chunks // max(1, self.k)))
        return np.argsort(np.arange(n_chunks) % stride, kind="stable")

    def _pick(self, cand_ids: np.ndarray, window_tok: np.ndarray) -> list[np.ndarray]:
        """Accepted neighbour token rows (padding removed) for one window."""
        seen, ordered = set(), []
        for c in cand_ids.tolist():
            if c < 0 or c in seen:
                continue
            seen.add(c)
            ordered.append(c)
        out, chosen = [], []
        i = 0
        while len(out) < self.k and i < len(ordered):
            batch = []
            while i < len(ordered) and len(batch) < self.max_candidates * (self.k - len(out)):
                c = ordered[i]
                i += 1
                if not any(abs(c - s) <= 1 for s in chosen + batch):
                    batch.append(c)
            if not batch:
                break
            toks = self.db.neighbours(np.asarray(batch), self.nbr_len)          # [m, T]
            if self.max_overlap < 1.0 and len(window_tok) >= self.n:
                ov = overlap_fraction(torch.from_numpy(toks),
                                      torch.from_numpy(np.broadcast_to(window_tok, (len(batch), len(window_tok))).copy())[:, None],
                                      self.n)[:, 0].numpy()
            else:
                ov = np.zeros(len(batch))
            for c, t, o in zip(batch, toks, ov):
                if len(out) == self.k:
                    break
                t = t[t >= 0]
                if o >= self.max_overlap or len(t) == 0 or any(abs(c - s) <= 1 for s in chosen):
                    continue
                chosen.append(c)
                out.append(t)
        return out

    def build(self, item: dict, student: dict) -> dict:
        """item: a RetroSplit item (input_ids, pieces, chunk_pos, nbr_ids); student: the VarlenCollator output.
        Returns t_input_ids [1, Lt], t_position_ids [1, Lt], t_doc / t_blk / t_wb [Lt], t_src [Lp] (teacher index of
        every student position) and nbr statistics."""
        ids = item["input_ids"].numpy()
        pieces = item["pieces"].numpy()
        L, Lp = len(ids), student["input_ids"].shape[1]
        chunk_pos = np.asarray(item["chunk_pos"])
        nbr = item.get("nbr_ids")
        pos = student["position_ids"][0].numpy()
        B = self.block_max
        parts_ids, parts_pos, parts_doc, parts_blk, parts_wb = [], [], [], [], []
        src = np.zeros(Lp, dtype=np.int64)
        cur = 0
        blk_id = 0
        n_win = n_win_nbr = n_nbr = n_nbr_tok = 0

        def add(t, p, d, b, w):
            nonlocal cur
            parts_ids.append(t); parts_pos.append(p)
            parts_doc.append(np.full(len(t), d, np.int64)); parts_blk.append(np.full(len(t), b, np.int64))
            parts_wb.append(np.full(len(t), w, np.int64))
            cur += len(t)

        starts = np.concatenate([[0], np.cumsum(pieces)[:-1]])
        for d, (s0, n) in enumerate(zip(starts.tolist(), pieces.tolist())):
            for a in range(s0, s0 + n, self.window):
                b = min(s0 + n, a + self.window)
                n_win += 1
                wb = -1
                if nbr is not None and self.k > 0:
                    c0, c1 = np.searchsorted(chunk_pos, a), np.searchsorted(chunk_pos, b)
                    if c1 > c0:
                        rows = np.asarray(nbr[c0:c1])[self._order(c1 - c0)]
                        picks = self._pick(rows.T.reshape(-1), ids[a:b])
                        if picks:
                            blk_id += 1
                            wb = blk_id
                            toks = np.concatenate([np.concatenate([t, self.sep]) for t in picks])
                            p_end = int(pos[a]) + B
                            add(toks, np.arange(p_end - len(toks), p_end), d, blk_id, blk_id)
                            n_win_nbr += 1
                            n_nbr += len(picks)
                            n_nbr_tok += len(toks)
                src[a:b] = cur + np.arange(b - a)
                add(ids[a:b].astype(np.int64), pos[a:b].astype(np.int64) + B, d, 0, wb)
        # the student's padding, then the teacher's own padding: one padding "document" (-1)
        Lt0 = cur + (Lp - L)
        Lt = -(-Lt0 // self.mult) * self.mult
        pad = Lt - cur
        if pad:
            src[L:Lp] = cur + np.arange(Lp - L)
            add(np.full(pad, self.pad_id, np.int64), np.arange(pad), -1, 0, -1)
        cat = lambda xs: torch.from_numpy(np.concatenate(xs))  # noqa: E731
        return {"t_input_ids": cat(parts_ids)[None], "t_position_ids": cat(parts_pos)[None], "t_doc": cat(parts_doc),
                "t_blk": cat(parts_blk), "t_wb": cat(parts_wb), "t_src": torch.from_numpy(src),
                "t_stats": torch.tensor([n_win, n_win_nbr, n_nbr, n_nbr_tok], dtype=torch.long)}


def teacher_mask(doc: torch.Tensor, blk: torch.Tensor, wb: torch.Tensor, attn: str):
    """Teacher attention: causal, same document piece; a document token (blk 0) sees document tokens and its window's
    neighbour block (blk == its wb); a neighbour token sees its own block only."""
    def allowed(q_doc, kv_doc, q_blk, kv_blk, q_wb, causal):
        is_doc = q_blk == 0
        return causal & (q_doc == kv_doc) & ((is_doc & ((kv_blk == 0) | (kv_blk == q_wb))) | (~is_doc & (kv_blk == q_blk)))

    L = doc.shape[0]
    if attn == "flex_attention":
        from torch.nn.attention.flex_attention import create_block_mask

        def mask_mod(b, h, q, kv):
            return allowed(doc[q], doc[kv], blk[q], blk[kv], wb[q], q >= kv)

        return create_block_mask(mask_mod, B=1, H=None, Q_LEN=L, KV_LEN=L, device=doc.device, _compile=True)
    causal = torch.ones(L, L, dtype=torch.bool, device=doc.device).tril()
    return allowed(doc[:, None], doc[None, :], blk[:, None], blk[None, :], wb[:, None], causal)[None, None]


def decoder_hidden(model, ids: torch.Tensor, pos_ids: torch.Tensor, attention_mask) -> torch.Tensor:
    """Final hidden states [L, d] of the decoder (as retro_hidden with retrieval off)."""
    _, body = unwrap(model)
    h = body.embed_tokens(ids)
    pe = body.rotary_emb(h, pos_ids)
    for layer in body.layers:
        h = layer(h, attention_mask=attention_mask, position_ids=pos_ids, position_embeddings=pe)
    return body.norm(h)[0]


@torch.no_grad()
def context_teacher_hidden(model, batch: dict, attn: str, layers: list | None = None, use_lora: bool = True) -> torch.Tensor:
    """Teacher hidden states at the student positions [Lp, d] (no gradient)."""
    mask = teacher_mask(batch["t_doc"], batch["t_blk"], batch["t_wb"], attn)
    with lora_disabled(layers or [], off=not use_lora):
        h = decoder_hidden(model, batch["t_input_ids"], batch["t_position_ids"], mask)
    return h.index_select(0, batch["t_src"])


# --------------------------------------------------------------------------- #
# Distillation loss
# --------------------------------------------------------------------------- #
def _kd_chunk(hs, ht, w, y, alpha: float, gate: str):
    zs = (hs @ w.t()).float()
    ls = zs.log_softmax(-1)
    valid = y != -100
    yc = y.clamp(min=0)
    ar = torch.arange(len(y), device=y.device)
    ls_y = ls[ar, yc]
    with torch.no_grad():
        lt = (ht @ w.t()).float().log_softmax(-1)
        pt = lt.exp()
        h_t = (pt * lt).sum(-1)                         # -entropy of the teacher
        lt_y = lt[ar, yc]
    kl = h_t - (pt * ls).sum(-1)                        # KL(teacher || student) per token
    ce = -ls_y
    g = (lt_y > ls_y.detach()).float() if gate == "teacher_better" else torch.ones_like(ce)
    v = valid.float()
    loss = (((1 - alpha) * ce + alpha * g * kl) * v).sum()
    stats = torch.stack([(ce.detach() * v).sum(), (kl.detach() * v).sum(), (-lt_y * v).sum(), (g * v).sum(), v.sum()])
    return loss, stats


def kd_loss(hs: torch.Tensor, ht: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, alpha: float,
            gate: str = "none", chunk: int = 512):
    """Sum over tokens of (1 - alpha) * CE + alpha * gate * KL(teacher || student), with the logits built chunk by
    chunk and recomputed in backward (full-vocabulary float logits: ~0.6 GB per 1 k tokens for each of the four
    [chunk, vocab] tensors). Returns (loss_sum, stats [ce_sum, kl_sum, teacher_ce_sum, gated_tokens, tokens])."""
    from torch.utils.checkpoint import checkpoint

    ht = ht.detach()
    total = hs.new_zeros((), dtype=torch.float32)
    stats = torch.zeros(5, dtype=torch.float32, device=hs.device)
    for s in range(0, hs.shape[0], chunk):
        y = targets[s:s + chunk]
        if (y != -100).any():
            l, st = checkpoint(_kd_chunk, hs[s:s + chunk], ht[s:s + chunk], weight, y, alpha, gate, use_reentrant=False)
            total = total + l
            stats = stats + st.detach()
    return total, stats


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export_student(model, out_dir: str, tok=None) -> str:
    """Save the PEFT adapter without the RETRO module (if any): a plain LoRA adapter that scripts/eval_model.py,
    merging and the device build load on the base model alone."""
    model.save_pretrained(out_dir)
    if tok is not None:
        tok.save_pretrained(out_dir)
    cp = os.path.join(out_dir, "adapter_config.json")
    with open(cp) as f:
        cfg = json.load(f)
    if cfg.get("modules_to_save"):
        cfg["modules_to_save"] = None
        with open(cp, "w") as f:
            json.dump(cfg, f, indent=2)
    wp = os.path.join(out_dir, "adapter_model.safetensors")
    if os.path.exists(wp):
        from safetensors.torch import load_file, save_file
        sd = load_file(wp)
        keep = {k: v for k, v in sd.items() if ".retro." not in f".{k}"}
        if len(keep) != len(sd):
            save_file(keep, wp, metadata={"format": "pt"})
    for f in ("retro_config.json",):
        p = os.path.join(out_dir, f)
        if os.path.exists(p):
            os.remove(p)
    return out_dir
