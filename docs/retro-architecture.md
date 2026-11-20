# RETRO architecture

Borgeaud et al., DeepMind, 2021. *Improving language models by retrieving from trillions of tokens* (arXiv:2112.04426).

## Data flow

```
 Input X   [C1][C2][C3] … [C32]     2048 tokens = 32 chunks × 64
               │                               │
               │  each chunk Cu                │  all tokens
               ▼                               ▼
 ┌───────────────────────────┐  ┌─────────────────────────────┐
 │ Frozen BERT               │  │ Token embedding             │
 │ mean-pooled embedding     │  └──────────────┬──────────────┘
 └─────────────┬─────────────┘                 │
               │  query vector                 ▼
               ▼                ┌─────────────────────────────┐
 ┌───────────────────────────┐  │ Layers 1–5                  │
 │ kNN search (ScaNN)        │  │ causal self-attn + FFW      │
 │ over ~2 T-token database  │  └──────────────┬──────────────┘
 │   key   = BERT(N)         │                 ├──▶ Hu  (to encoder)
 │   value = [N, F]          │                 ▼
 └─────────────┬─────────────┘  ┌─────────────────────────────┐
               │  top-k values  │ Layer 6  (RETRO block)      │
               ▼                │   causal self-attention     │
 ┌───────────────────────────┐  │   CCA            ◀── Eu     │
 │ Ret(Cu): k = 2 neighbours │  │   FFW                       │
 │ each [N, F] = 64 + 64 tok │  └──────────────┬──────────────┘
 └─────────────┬─────────────┘                 │
               │                               ▼
               ▼                ┌─────────────────────────────┐
 ┌───────────────────────────┐  │ Layers 7–8: self-attn + FFW │
 │ Neighbour encoder         │  └──────────────┬──────────────┘
 │ bidirectional, per nbr    │                 │
 │ cross-attends to  ◀── Hu  │                 ▼
 └─────────────┬─────────────┘  ┌─────────────────────────────┐
               ▼                │ Layer 9: same as layer 6    │
   Eu = keys/values fed to      └──────────────┬──────────────┘
   every CCA sub-layer                         │
                                               ▼
                                ┌─────────────────────────────┐
                                │ Layers 10 … L               │
                                │ CCA again at 12, 15, …      │
                                └──────────────┬──────────────┘
                                               │
                                               ▼
                                ┌─────────────────────────────┐
                                │ Read-out: next-token logits │
                                └─────────────────────────────┘
```

- `Hu` is the decoder's hidden state for chunk `Cu`, taken at the input to the first CCA layer (layer 6). The encoder cross-attends to it.
- `Eu` is the encoder's output, k neighbours × 128 tokens. It is the key/value input to every CCA sub-layer.
- BERT and the database are never trained. When retro-fitting, the decoder is frozen too, and only the encoder and the CCA sub-layers are trained.

## What a neighbour is, and why it helps predict the next chunk

```
 Database doc:  … [ N: text close to Cu ][ F: the 64 tokens that followed N ] …
                        ▲                               │
                        │ matched via BERT(Cu)          │ encoded, then read by CCA
                        │                               ▼
 Your sequence: … [ Cu  (already seen)  ][ Cu+1  (being predicted)            ] …
```

## CCA alignment (why it stays causal)

```
 token index   0                63 64               127 128              191
               ├─────────────────┼┼──────────────────┼┼──────────────────┼┼───▶
 chunks        [       C1        ][        C2        ][        C3        ] …
                        │                   │                   │
                        ▼ query             ▼ query             ▼ query
 neighbours          Ret(C1)             Ret(C2)             Ret(C3)
                        └─────────┐         └─────────┐         └─────────┐
                                  ▼                   ▼                   ▼
 CCA spans     [ no nbrs: 0–62  ][  C1⁺: 63–126     ][  C2⁺: 127–190    ][ C3⁺ …
```

| CCA span | Tokens | Attends to |
|---|---|---|
| (start) | 0–62 | nothing; self-attention only |
| C1⁺ | 63–126 (last token of C1 + first 63 of C2) | E(Ret(C1)) only |
| C2⁺ | 127–190 | E(Ret(C2)) only |
| Cu⁺ | 64u − 1 … 64u + 62 | E(Ret(Cu)) only |

Each span starts one token before the chunk boundary, because the last token of `Cu` is the one that predicts the first token of `Cu+1`. Each span attends only to its own chunk's neighbours, so the cost grows linearly with sequence length.

## Key numbers

| | |
|---|---|
| Chunk length | 64 tokens |
| Training sequence | 2048 tokens (32 chunks) |
| Neighbour [N, F] | 64 + 64 = 128 tokens |
| Neighbours k | 2 in training |
| CCA layers | 6, 9, 12, …, L |
| Database | MassiveText, up to ~2 T tokens |
| Model sizes | 172 M, 425 M, 1.5 B, 7.5 B |
