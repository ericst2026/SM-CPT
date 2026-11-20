"""RETRO layers for a frozen (LoRA-adapted) decoder: neighbour encoder + chunked cross-attention (CCA).

Borgeaud et al. 2022 (RETRO, arXiv:2112.04426), retro-fitting variant: the pretrained decoder is kept (here frozen
and LoRA-adapted), the new modules are added and trained.

Sequence layout (one packed sequence of several documents, see cpt_trainer/longctx.py): every document piece is
cut into chunks of C tokens from the piece start. Chunk u = tokens [s_u, s_u + C) of the sequence; it is retrieved
for, and its k neighbours (each a database chunk N plus its continuation F, nbr_len = 2C tokens) are encoded once.
CCA "span" of chunk u = tokens [s_u + C - 1, s_u + 2C - 1) clipped to the piece end: the last token of the chunk
and the first C - 1 tokens of the next chunk. Those tokens (and only those) attend to the encoded neighbours of
chunk u, so a token never sees neighbours retrieved with text after it, and every token attends to one chunk's
neighbours only (linear cost).

    decoder layers 0 .. cond_layer        (frozen + LoRA)
      -> H = hidden state after cond_layer
      -> E = encoder(neighbour tokens, cross-attending to H of the retrieving chunk)    [trainable]
    after each layer i in cca_layers: h += CCA_i(h, E)                                      [trainable]
    remaining decoder layers, final norm, LM head                                           (frozen + LoRA)

Deviations from the paper (documented in docs/retro_fit_lora_cpt.md): CCA is a separate pre-norm residual
sub-layer after decoder block i (RETRO puts it between the block's attention and FFW, which is the same position
up to the FFW of that block); CCA has no relative position encoding (the encoder output is position-aware through
its RoPE self-attention); the output projection of every CCA layer is zero-initialised, so the untrained model is
exactly the base model (the LoRA B matrices are zero too).
The encoder reads the decoder's own (frozen) token embeddings through a trainable projection.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

RETRO_CONFIG_NAME = "retro_config.json"


@dataclass
class RetroConfig:
    chunk_len: int = 64              # C: tokens per retrieval chunk
    neighbours: int = 2              # k neighbours per chunk
    nbr_len: int = 128               # tokens per neighbour: chunk N (C) + continuation F (C)
    cca_layers: list = field(default_factory=lambda: [5, 8, 11, 14, 17, 20, 23, 26])  # 0-indexed decoder blocks
    cond_layer: int | None = None    # block whose output conditions the encoder (default: cca_layers[0])
    d_model: int = 1024              # decoder hidden size (set from the base config)
    enc_dim: int = 512
    enc_layers: int = 2
    enc_heads: int = 8
    enc_ffn: int = 1536
    cca_heads: int = 8
    cca_head_dim: int = 64
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    dropout: float = 0.0

    def __post_init__(self):
        self.cca_layers = sorted(int(i) for i in self.cca_layers)
        if self.cond_layer is None:
            self.cond_layer = self.cca_layers[0]
        if self.cond_layer > self.cca_layers[0]:
            raise ValueError("cond_layer must not come after the first CCA layer")
        if self.nbr_len < self.chunk_len:
            raise ValueError("nbr_len must be >= chunk_len")

    @classmethod
    def from_file(cls, path: str, **override) -> "RetroConfig":
        if os.path.isdir(path):
            path = os.path.join(path, RETRO_CONFIG_NAME)
        with open(path) as f:
            if path.endswith((".yaml", ".yml")):
                import yaml
                d = yaml.safe_load(f)
            else:
                d = json.load(f)
        d = {k: v for k, v in (d.get("retro", d)).items() if k in cls.__dataclass_fields__}
        d.update({k: v for k, v in override.items() if v is not None})
        return cls(**d)

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, RETRO_CONFIG_NAME), "w") as f:
            json.dump(asdict(self), f, indent=2)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dt)


def rope(x: torch.Tensor, theta: float) -> torch.Tensor:
    """Rotary position embedding over the sequence axis of x [B, H, T, D] (positions 0..T-1)."""
    T, D = x.shape[-2], x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D))
    ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * inv[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def safe_key_mask(valid: torch.Tensor) -> torch.Tensor:
    """[B, T] key validity -> [B, 1, 1, T] boolean SDPA mask; rows without any valid key attend to everything
    (their output is discarded by the caller) so SDPA never produces NaN."""
    none = ~valid.any(-1, keepdim=True)
    return (valid | none)[:, None, None, :]


class Attention(nn.Module):
    """Multi-head attention; self-attention (kv_dim None, optional RoPE) or cross-attention from kv_dim inputs."""

    def __init__(self, dim: int, heads: int, head_dim: int, kv_dim: int | None = None, out_dim: int | None = None,
                 use_rope: bool = False, rope_theta: float = 10000.0, zero_out: bool = False, dropout: float = 0.0):
        super().__init__()
        inner = heads * head_dim
        self.heads, self.head_dim, self.use_rope, self.rope_theta, self.dropout = heads, head_dim, use_rope, rope_theta, dropout
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(kv_dim or dim, inner, bias=False)
        self.to_v = nn.Linear(kv_dim or dim, inner, bias=False)
        self.to_out = nn.Linear(inner, out_dim or dim, bias=False)
        self.q_norm = RMSNorm(head_dim)   # QK-norm as in Qwen3: keeps attention logits bounded for new layers
        self.k_norm = RMSNorm(head_dim)
        if zero_out:
            nn.init.zeros_(self.to_out.weight)

    def forward(self, x, kv=None, key_valid=None):
        B, T, _ = x.shape
        kv = x if kv is None else kv
        S = kv.shape[1]
        q = self.q_norm(self.to_q(x).view(B, T, self.heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.to_k(kv).view(B, S, self.heads, self.head_dim)).transpose(1, 2)
        v = self.to_v(kv).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        if self.use_rope:
            q, k = rope(q, self.rope_theta), rope(k, self.rope_theta)
        mask = safe_key_mask(key_valid) if key_valid is not None else None
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        return self.to_out(o.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class EncoderLayer(nn.Module):
    """Bidirectional self-attention over one neighbour, cross-attention to the retrieving chunk's decoder states, FFW."""

    def __init__(self, cfg: RetroConfig):
        super().__init__()
        d, hd = cfg.enc_dim, cfg.enc_dim // cfg.enc_heads
        self.n_self = RMSNorm(d, cfg.norm_eps)
        self.self_attn = Attention(d, cfg.enc_heads, hd, use_rope=True, rope_theta=cfg.rope_theta, dropout=cfg.dropout)
        self.n_cross = RMSNorm(d, cfg.norm_eps)
        self.n_cond = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.cross_attn = Attention(d, cfg.enc_heads, hd, kv_dim=cfg.d_model, dropout=cfg.dropout)
        self.n_ffw = RMSNorm(d, cfg.norm_eps)
        self.ffw = SwiGLU(d, cfg.enc_ffn)

    def forward(self, x, x_valid, cond, cond_valid):
        x = x + self.self_attn(self.n_self(x), key_valid=x_valid)
        x = x + self.cross_attn(self.n_cross(x), kv=self.n_cond(cond), key_valid=cond_valid)
        return x + self.ffw(self.n_ffw(x))


class ChunkedCrossAttention(nn.Module):
    def __init__(self, cfg: RetroConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg.d_model, cfg.cca_heads, cfg.cca_head_dim, kv_dim=cfg.enc_dim, zero_out=True,
                              dropout=cfg.dropout)


# --------------------------------------------------------------------------- #
# The trainable RETRO module (attached to the base model as `retro`)
# --------------------------------------------------------------------------- #
class RetroModule(nn.Module):
    def __init__(self, cfg: RetroConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.d_model, cfg.enc_dim, bias=False)   # decoder token embedding -> encoder width
        self.enc = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.enc_layers)])
        self.enc_norm = RMSNorm(cfg.enc_dim, cfg.norm_eps)
        self.cca = nn.ModuleDict({str(i): ChunkedCrossAttention(cfg) for i in cfg.cca_layers})
        self.reset_parameters()

    def reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() == 2:
                if name.endswith("attn.to_out.weight") and name.startswith("cca."):
                    nn.init.zeros_(p)
                else:
                    nn.init.normal_(p, std=0.02)

    # PEFT's ModulesToSaveWrapper calls forward(*args, **kwargs) on the active copy; `op` selects the computation.
    def forward(self, op: str, **kw):
        if op == "encode":
            return self.encode(**kw)
        if op == "cca":
            return self.cross_attend(**kw)
        raise ValueError(op)

    def encode(self, nbr_emb, nbr_valid, cond, cond_valid):
        """nbr_emb [n, k, T, d_model] (decoder token embeddings of the neighbours), nbr_valid [n, k, T],
        cond [n, C, d_model] decoder states of the retrieving chunks, cond_valid [n, C] -> E [n, k*T, enc_dim]."""
        n, k, T, _ = nbr_emb.shape
        x = self.in_proj(nbr_emb.reshape(n * k, T, -1))
        xv = nbr_valid.reshape(n * k, T)
        c = cond.repeat_interleave(k, dim=0)
        cv = cond_valid.repeat_interleave(k, dim=0)
        for layer in self.enc:
            x = layer(x, xv, c, cv)
        return self.enc_norm(x).reshape(n, k * T, -1)

    def cross_attend(self, layer: int, h, enc, enc_valid, span_idx, span_valid):
        """h [L, d_model] decoder states of the sequence; enc [n, k*T, enc_dim]; enc_valid [n, k*T];
        span_idx [n, C] token index of each chunk's CCA span (clamped), span_valid [n, C].
        Returns the residual update [L, d_model] (zero outside the spans and for chunks without neighbours)."""
        m = self.cca[str(layer)]
        q_in = h[span_idx]                                            # [n, C, d]
        o = m.attn(m.norm(q_in), kv=enc, key_valid=enc_valid)        # [n, C, d]
        keep = span_valid & enc_valid.any(-1, keepdim=True)          # spans of chunks with >= 1 valid neighbour token
        out = torch.zeros_like(h)
        return out.index_add(0, span_idx[keep], o[keep].to(h.dtype))


# --------------------------------------------------------------------------- #
# Attaching to / reading from a (PEFT-wrapped) Qwen3 causal LM
# --------------------------------------------------------------------------- #
def attach_retro(model, cfg: RetroConfig, dtype=torch.bfloat16):
    """Add model.retro (before get_peft_model, so that LoraConfig(modules_to_save=["retro"]) wraps it)."""
    cfg.d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    if max(cfg.cca_layers) >= n_layers:
        raise ValueError(f"cca_layers {cfg.cca_layers} exceed the {n_layers} decoder layers")
    dev = next(model.parameters()).device
    model.retro = RetroModule(cfg).to(device=dev, dtype=dtype)
    model.retro_config = cfg
    return model


def unwrap(model):
    """(Qwen3ForCausalLM, Qwen3Model) of a plain or PEFT-wrapped model."""
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return m, m.model


def retro_parameters(model):
    """Trainable parameters of the RETRO module (the active modules_to_save copy under PEFT)."""
    return [p for n, p in model.named_parameters() if ".retro." in f".{n}" and p.requires_grad]


def trainable_split(model):
    retro, lora, other = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".retro." in f".{n}":
            retro.append(p)
        elif "lora_" in n:
            lora.append(p)
        else:
            other.append(p)
    return retro, lora, other


# --------------------------------------------------------------------------- #
# Forward pass
# --------------------------------------------------------------------------- #
def chunk_tensors(batch: dict, cfg: RetroConfig, L: int):
    """Index tensors of the chunks of one packed sequence.
    batch: chunk_pos [n] (start token of each chunk), chunk_end [n] (end of its document piece), nbr_tokens
    [n, k, T] (-1 = padding), optional chunk_first [n] (first real token of the chunk; eval left padding)."""
    C = cfg.chunk_len
    pos = batch["chunk_pos"]
    ar = torch.arange(C, device=pos.device)
    tok_idx = pos[:, None] + ar[None]                                   # chunk tokens
    first = batch.get("chunk_first")
    tok_valid = torch.ones_like(tok_idx, dtype=torch.bool) if first is None else tok_idx >= first[:, None]
    span = pos[:, None] + (C - 1) + ar[None]                            # CCA span tokens
    span_valid = (span < batch["chunk_end"][:, None]) & (span < L)
    span = torch.where(span_valid, span, torch.zeros_like(span))
    nbr = batch["nbr_tokens"]
    nbr_valid = nbr >= 0
    return tok_idx.clamp(max=L - 1), tok_valid, span, span_valid, nbr, nbr_valid


def _call(fn, ckpt: bool, *args, **kw):
    return checkpoint(fn, *args, use_reentrant=False, **kw) if ckpt else fn(*args, **kw)


def retro_hidden(model, batch: dict, attention_mask, retrieval: bool = True, grad_ckpt: bool = False) -> torch.Tensor:
    """Final hidden states [L, d_model] of one packed sequence (batch size 1) with the RETRO layers.
    attention_mask: whatever the decoder's attention implementation takes (flex BlockMask or a 4-D bool mask).
    retrieval=False (or no chunks / no neighbours) runs the decoder alone (CCA skipped)."""
    lm, body = unwrap(model)
    cfg: RetroConfig | None = getattr(lm, "retro_config", None)
    ids, pos_ids = batch["input_ids"], batch["position_ids"]
    h = body.embed_tokens(ids)
    pos_emb = body.rotary_emb(h, pos_ids)
    L = ids.shape[1]
    use = cfg is not None and retrieval and batch.get("chunk_pos") is not None and batch["chunk_pos"].numel() > 0
    if use:
        tok_idx, tok_valid, span, span_valid, nbr, nbr_valid = chunk_tensors(batch, cfg, L)
        use = bool(nbr_valid.any())
    enc = enc_valid = None
    for i, layer in enumerate(body.layers):
        h = layer(h, attention_mask=attention_mask, position_ids=pos_ids, position_embeddings=pos_emb)
        if not use:
            continue
        if i == cfg.cond_layer:
            with torch.no_grad():
                nbr_emb = body.embed_tokens(nbr.clamp(min=0))           # frozen decoder embeddings
            cond = h[0][tok_idx]                                        # [n, C, d]
            enc = _call(lm.retro, grad_ckpt, op="encode", nbr_emb=nbr_emb, nbr_valid=nbr_valid, cond=cond,
                        cond_valid=tok_valid)
            enc_valid = nbr_valid.reshape(nbr_valid.shape[0], -1)
        if i in cfg.cca_layers:
            upd = _call(lm.retro, grad_ckpt, op="cca", layer=i, h=h[0], enc=enc, enc_valid=enc_valid,
                        span_idx=span, span_valid=span_valid)
            h = h + upd[None]
    return body.norm(h)[0]


def dense_document_mask(doc_ids: torch.Tensor, dtype=torch.bool) -> torch.Tensor:
    """Causal + same-document 4-D bool mask [1, 1, L, L] for the SDPA / eager attention (CPU tests, short eval
    sequences). The flex BlockMask (cpt_trainer.longctx.document_block_mask) is used for long training sequences."""
    L = doc_ids.shape[0]
    causal = torch.ones(L, L, dtype=torch.bool, device=doc_ids.device).tril()
    return (causal & (doc_ids[:, None] == doc_ids[None, :]))[None, None]


def attention_mask_for(doc_ids: torch.Tensor, attn: str):
    if attn == "flex_attention":
        from cpt_trainer.longctx import document_block_mask
        return document_block_mask(doc_ids)
    return dense_document_mask(doc_ids)


def retro_loss(model, inputs: dict, attn: str, num_items_in_batch=None, loss_chunk: int = 4096,
               retrieval: bool = True, grad_ckpt: bool = False) -> torch.Tensor:
    """Mean next-token loss of one packed sequence (or its share of the accumulated batch), as
    cpt_trainer.longctx.varlen_loss but through retro_hidden."""
    from cpt_trainer.longctx import chunked_lm_loss
    lm, _ = unwrap(model)
    mask = attention_mask_for(inputs["doc_ids"][0], attn)
    hid = retro_hidden(model, inputs, mask, retrieval=retrieval, grad_ckpt=grad_ckpt)
    y = inputs["labels"][0, 1:]
    loss_sum = chunked_lm_loss(hid[:-1], lm.lm_head.weight, y, loss_chunk)
    denom = num_items_in_batch if num_items_in_batch is not None else (y != -100).sum().clamp(min=1)
    return loss_sum / denom


def param_report(cfg: RetroConfig) -> dict:
    m = RetroModule(cfg)
    enc = sum(p.numel() for n, p in m.named_parameters() if not n.startswith("cca."))
    cca = sum(p.numel() for n, p in m.named_parameters() if n.startswith("cca."))
    return {"encoder_M": enc / 1e6, "cca_M": cca / 1e6, "total_M": (enc + cca) / 1e6, "cca_layers": len(cfg.cca_layers)}
