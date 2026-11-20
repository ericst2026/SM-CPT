# RETRO-fit + LoRA continued pre-training of Qwen3-0.6B-Base

Prepared 2026-11-11. Nothing has been run yet: everything below is code, configs and estimates. The GPU is busy
with `cpt_qwen06b_rs_adaptive_v4_web5b`. Each stage was smoke-tested on CPU with small data.

> **Superseded as the main run on 2026-11-12** by `docs/retro_distill_lora_cpt.md`. The deployed model must not
> need the database or extra layers, so retrieval is now used only during training, as a teacher distilled into the
> LoRA. The database, index, neighbours and training set below are reused as they are. This design (RETRO layers at
> inference) stays available through `runs/launch_retro_qwen06b_v1_web1b.sh train`.

## 1. Idea

The web5b run puts the AI domain into a large LoRA adapter: ranks 64–512, 158.3 M trainable parameters.

This experiment moves the domain knowledge out of the weights and into a retrieval database:

- **RETRO layers.** A neighbour encoder plus chunked cross-attention (CCA) layers are added to the frozen decoder.
  They learn to read relevant AI text retrieved from a database of every AI document we have.
- **A small LoRA** (the same plan with ranks ÷ 4, i.e. 16–128) adapts the decoder to the new layers and to the
  domain. It does not have to memorise the domain.
- **Data.** The training mix is the web5b mix, scaled to 1 B tokens.

| | web5b (running) | this experiment |
|---|---|---|
| trainable parameters | LoRA 158.3 M | LoRA 39.6 M + RETRO 23.1 M = **62.7 M** |
| adapter on disk (bf16) | 302 MiB | ~120 MiB (+ retrieval database ~19 GB tokens + ~5 GB index) |
| training tokens | 4.97 B | 1.0 B (same mix proportions, same validation sequences) |
| domain knowledge | in the LoRA weights | in the database, read through CCA; updatable without retraining |

## 2. What the literature says

Sources: the papers in `docs/ref/` plus papers found on the web. Numbers were checked against the papers.

| work | what is retro-fitted | decoder | tokens | result / lesson |
|---|---|---|---|---|
| **RETRO** (Borgeaud et al. 2022, 2112.04426), §4.2 retro-fitting | encoder + CCA, every 3rd layer from 6; chunk 64, k=2, neighbour = chunk + 64-token continuation; frozen BERT retriever | **frozen** ("unfreezing degraded retrieval-off performance") | 3 % of pretraining (~12.6 B) | quickly beats the baseline, close to RETRO trained from scratch. WT103 retrofit (d=1024): ppl 21.5 → 18.5 with a 4 B-token Wikipedia database. Excludes neighbours from the same document. |
| **Wang et al. 2023** (2304.06762) | RETRO reproduced from scratch (148 M–9.5 B) | – | 330 B | large perplexity gains, mostly on knowledge-intensive tasks. Inference rule: **left-pad** so the question fills whole chunks. |
| **InstructRetro** (2310.07713) | encoder + CCA on GPT 823 M–43 B | **unfrozen** (better at 823 M with 100 B tokens) | 100 B | matches GPT models ~4× larger in perplexity. Most of the downstream gain survives with the encoder bypassed: the better decoder matters more than the CCA path. |
| **Norlund et al. 2023** (2302.12128) | RETRO, 425 M, from scratch | – | 2.3 B | the loss reduction comes almost entirely from **tokens that overlap the neighbours** (copying). Non-overlapping tokens gain nothing. |
| **Doostmohammadi & Kuhlmann 2025** (2505.14309) | encoder + CCA on GPT-345 M | unfrozen | ≤ 10 k steps × 128 | retrieval only "**activates**" when neighbours overlap the input (≥ ~32/64 tokens). Activation takes ~0.5 M samples. Paraphrased neighbours speed it up 40 %. |
| **Retro-li** (2410.00004) | CCA on GPT-2 small, 6/9/12 | frozen better than unfrozen | 1.1 B | SBERT retrieval is far better than BERT (ppl 24 vs 51). Gains are small and partly not from retrieved content. High variance across seeds. |
| **CEPE** (2402.16617) | encoder + cross-attention in every layer of a frozen LLaMA-2-7B | frozen | 20 B | **zero-initialised output projection**, plus a short copy warm-up before the main stage. |
| **REFRAG** (2509.01092) | chunk encoder + projection | frozen first, then unfrozen | – | bootstrap the new path with a reconstruction stage on a frozen decoder, then train jointly. |
| GPT vs RETRO PEFT (2407.04528) | LoRA / adapters on RETRO for task fine-tuning | – | – | the only RETRO + LoRA work found. **No work trains LoRA jointly with newly added CCA during continued pre-training**, so this setting is new. |

What this means for the experiment:

1. **The main risk is that retrieval never activates.** All positive retrofits used ≥ 10× more tokens than our 1 B.
   - CCA learns fastest from overlapping neighbours.
   - Leakage filters remove exactly that overlap.
   - The LoRA can absorb the domain gain on its own and starve the CCA of gradient.
2. **Expect small gains on non-overlapping text.** Perplexity gains concentrate on tokens that also appear in the
   neighbours. The honest metrics are the gain on non-overlapping tokens and a LoRA-only control run.
3. **Settled design choices:** chunk 64, k = 2, neighbour = chunk + continuation (128 tokens), CCA every 3rd layer
   from layer 6, a 2-layer encoder on shared frozen token embeddings, a frozen sentence-embedding retriever,
   same-document exclusion, zero-init so that step 0 equals the base model, and left padding at inference.

## 3. Design

### 3.1 Model (`cpt_trainer/retro/model.py`, `configs/retro/retro_qwen06b_v1.yaml`)

```
 tokens ─► embed ─► blocks 0..5 ─┬─► CCA ─► blocks 6..8 ─► CCA ─► … ─► CCA (after 26) ─► block 27 ─► norm ─► LM head
            (frozen + LoRA)      │    ▲                   ▲            ▲
                                 │ H  │ E                 │ E          │ E
                                 ▼    │                   │            │
 neighbours [k=2 × 128 tok] ─► frozen embed ─► proj 1024→512 ─► encoder (2 layers: self-attn, cross-attn to H, FFW) ─► E
```

**CCA layers.** There are 8, after decoder blocks 5, 8, 11, 14, 17, 20, 23 and 26 (0-indexed). This is RETRO's
"every 3rd layer from 6" retro-fitting placement.
- Each is a pre-norm residual sub-layer with 8 heads × 64, QK-norm, and a **zero-initialised output projection**.
- 12.6 M parameters in total.

**Encoder.** Two bidirectional layers of width 512 with RoPE.
- It cross-attends to the decoder states of the retrieving chunk, taken after block 5.
- It reads the decoder's frozen token embeddings through a trainable 1024 → 512 projection.
- 10.5 M parameters.

**Causality.**
- Chunk *u* is tokens [64u, 64u+64) of a document piece. Its neighbours are read only by the tokens
  64u+63 … 64u+126: the chunk's last token and the next 63.
- Spans never cross into the next document of a packed sequence.
- Verified: changing chunk *u*'s neighbours changes nothing before token 64u+63 and nothing in other documents.

**PEFT.**
- `LoraConfig(modules_to_save=["retro"])`, so adapters, checkpoints and `--resume` contain both the LoRA and the
  RETRO weights.
- `retro_config.json` is saved next to every adapter.
- Verified: LoRA is not injected into the RETRO layers, and save → reload is bit-identical.

**Deviations from the paper.**
- CCA sits after the whole block rather than between the block's attention and FFW.
- CCA has no relative position encoding. RETRO measured 0.826 vs 0.822 bpb without it; the encoder output carries
  RoPE positions.
- The encoder is narrower: 512 instead of RETRO's 896. This keeps the new modules small, per the request.

### 3.2 LoRA (`configs/lora/cpt_qwen06b_rs_adaptive_v5_r16_128.yaml`)

The plan is v4 with every rank divided by 4: q 16/32, k 16/24, v 16/24, o 16–48, gate 48–128, up 64–96,
down 16–96. That is 39.6 M parameters.

Method (rsLoRA), alpha (16), dropout (0) and layer bands are unchanged. Because rsLoRA scales by alpha/√r, the
smaller ranks get larger scales: 4.0 at r = 16, 1.41 at r = 128.

### 3.3 Training data (`cpt_trainer/retro/data.py`)

The corpora, the train/validation split (1 %, seed 42), `--exclude-docs`, the EOS handling and the 16–24 k
document-masked packing are the same as web5b.

- The packing code was checked to give **bit-identical** output to `cpt_trainer.longctx.pack_varlen`.
- So the **validation sequences are the same 32 per corpus as web5b**, and val losses compare directly.
- The training documents of each corpus are subsampled (seeded) to 1 B tokens × the corpus's web5b share.

| corpus | web5b tokens | share | this run |
|---|---:|---:|---:|
| web_ai | 1,985 M | 40.0 % | ~400 M |
| web_near | 991 M | 19.9 % | ~199 M |
| web_replay | 1,691 M | 34.0 % | ~340 M |
| stem | 189 M | 3.8 % | ~38 M |
| long | 114 M | 2.3 % | ~23 M |
| **total** | 4,969 M | | **~1,000 M**: ~42.4 k sequences, ~5,300 optimizer steps at bs 1 × 8 |

Every complete 64-token chunk of every document piece is recorded as a query chunk: about 15.5 M chunks in
training.

### 3.4 Retrieval database (`cpt_trainer/retro/db.py`, `configs/retro/db_ai_v1.yaml`)

"All AI data" means:

- the AI-labelled documents of the two scored web pools: fineweb_edu 1.89 B tokens, and nemotron_cc 3.72 B
  tokens minus Translated-Diverse-QA, whose labels are inflated (same exclusion as web_ai);
- `corpus_full/ai`;
- `corpus_ai_extra/xtra_ai`.

web_ai is a 40 % sample of the two pools, so it is fully included. The builder then:

- **decontaminates** against every eval / validation item (the 13-gram rule of `cpt_trainer/decontaminate.py`: MMLU
  test incl. ai_large, MCQ, open QA, DS-1000, SFT val);
- removes exact duplicates (normalised-text hash) and repeated document ids;
- tokenises with Qwen3 (EOS per document) and cuts every document into 64-token chunks.

Expected size: ~4.8 B tokens, ~75 M chunks. The exact numbers go to `meta.json`.

### 3.5 Retriever and index (`embed.py`, `index.py`)

- **Retriever:** frozen **bge-small-en-v1.5** (33 M parameters, 384-d, CLS pooling, cosine). It was downloaded to
  `models/bge-small-en-v1.5`. It is a sentence encoder because Retro-li found these far better than raw BERT at
  small database sizes. It is small because ~90 M chunks must be embedded; bge-m3 would take ~17× longer.
- **Index:** faiss `OPQ64_256,IVF65536_HNSW32,PQ64`, the Megatron-Retro recipe at smaller scale. About 5 GB of RAM,
  nprobe 32. It is built in resumable 5 M-chunk shards.
- `faiss-cpu` 1.16 was installed with `pip install --user --no-deps` and is added to `requirements.txt`.

### 3.6 Neighbours and leakage filters (`neighbours.py`)

For each training and validation chunk, the 20 nearest database chunks are retrieved, then filtered in rank order.
Up to 4 are kept; training uses the first k = 2.

| filter | removes | why |
|---|---|---|
| same document | candidates whose document id equals the query's. Every AI training document is also in the database. | its continuation *is* the text being predicted (RETRO's rule) |
| same text | candidates from a document with the same normalised text under another id (fineweb ↔ corpus_full copies) | exact duplicates |
| target overlap | candidates whose continuation F contains ≥ 50 % of the 8-grams of the query's **real next chunk** | mirrors and near-copies that would hand over the answer |
| repeat | candidates with the same tokens as an already kept neighbour | wasted slot |

**Not filtered: overlap between the query chunk and the neighbour chunk N.** That overlap cannot leak the future,
and it is what activates retrieval (Doostmohammadi & Kuhlmann 2025). `stats.json` records its histogram for the
top-1 neighbour. Read it before training: if almost all neighbours have zero 8-gram overlap with their query,
expect slow activation and consider a longer oracle warm-up (§3.7).

Verified on the test database: 1,513 query chunks came from documents that are in the database, and none of the
17,452 kept neighbours came from the query's own document.

### 3.7 Training (`cpt_trainer/retro/train.py`)

**Optimiser.** AdamW with two parameter groups:
- RETRO layers at **2e-4**, the RETRO-fitting learning rate for models ≤ 1.5 B;
- LoRA at **5e-5**, as in web5b.

Both use 3 % warmup and cosine decay to 0. Batch 1 × 8 sequences; one epoch.

**Activation schedule.** This is the answer to the main risk in §2.
- **Oracle-neighbour warm-up:** for the first 5 % of steps, a share of the chunks (50 % → 0, linear) get the chunk
  itself plus its true continuation in the last neighbour slot. The cross-attention learns to align neighbour and
  input, and to read the continuation, within a few hundred steps. Afterwards only real neighbours are used. This
  is the copy / reconstruction warm-up of CEPE and REFRAG.
- **Delayed LoRA:** the LoRA learning rate is 0 for the same first 5 %. The new path is bootstrapped on a frozen
  decoder, as in RETRO, CEPE and REFRAG; then the LoRA joins.

**Evaluation points** are at 0.25, 0.5, 0.75 and 1 epoch.
- **Validation loss** for each of the 5 corpora is measured with retrieval **on**, **off** (decoder + LoRA only)
  and with **random** neighbours. Epoch 0 is copied from web5b, because the sequences are identical.
- `cpt_trainer/retro/eval.py` then runs MMLU 5-shot (incl. ai_large) and the AI MCQ, with retrieval on and off.
- Logs show both learning rates, the oracle rate, and the mean ‖W_o‖ of the CCA output projections. ‖W_o‖ grows
  from 0 as the CCA path comes alive.

**Resume.** `--resume` restores the LoRA, RETRO, optimizer, scheduler, RNG, data order and oracle schedule.
- Verified: a resumed run reproduces the uninterrupted run's steps (loss, grad norm and val losses).

### 3.8 Evaluation (`cpt_trainer/retro/eval.py`)

**MMLU and MCQ** go through `scripts/eval_model.py`'s own prompts and scoring, via a model wrapper that:
- left-pads each prompt to whole chunks, so the answer position attends to the neighbours of the chunk holding
  the question and options;
- retrieves online, cached per chunk.

Verified: with untrained RETRO layers, the wrapper reproduces the plain model's logits on real 5-shot prompts
(max difference 2.6e-5).

**Validation analysis** follows RETRO §2.6 and Norlund et al.:
- Every token predicted inside a CCA span is bucketed by the longest token run its target text shares with the
  chunk's neighbours: r < 8, 8–31, ≥ 32.
- The loss gain of **on** over **off** in the r < 8 bucket is the gain that is not copying.

## 4. How to run (after web5b finishes)

`runs/launch_retro_qwen06b_v1_web1b.sh <stage>`. Every stage is resumable.

| stage | device | time (estimate) | disk | output |
|---|---|---|---|---|
| `db` | CPU, 16 workers | 1–1.5 h | ~19 GB (+ ~23 GB temporary) | `cpt_data/retro_db/ai_v1` |
| `set` | CPU | ~15 min | ~5 GB | `runs/_retro_cache/web1b` |
| `index` | GPU + CPU | ~2–3 h (unmeasured: the log prints chunks/s) | ~5 GB | `<db>/index/chunks.index` |
| `neighbours` | GPU + CPU | ~1–2 h | 0.3 GB | `<set>/nbr/ai_v1/*.npy` + `stats.json` |
| `smoke` | GPU | ~10 min | – | 20 steps: check memory and s/step before the long run |
| `train` | GPU | **~2–2.5 days** | ~2 GB of checkpoints | `runs/retro_qwen06b_v1_web1b` |
| `control` (recommended) | GPU | ~2 days | ~1 GB | `runs/lora_qwen06b_v5_web1b`: the same LoRA on the same 1 B tokens, no retrieval |

- `db` and `set` are CPU-only, so they can run while web5b trains. The rest need the GPU.
- The training time assumes web5b's ~6.2 k tokens/s, a ~10–20 % RETRO overhead and ~25 min per eval point.
  Expected peak GPU memory is ≤ 15 GB: web5b's 15.4 GB minus ~1 GB of optimizer state, plus the encoder/CCA
  activations. The `smoke` stage measures both.
- 125 GB of disk is free; the pipeline peaks at ~55 GB.

## 5. How to read the results

| question | evidence |
|---|---|
| Did retrieval activate? | val loss **on < off** on web_ai / web_near, with the gap growing over the eval points; ‖W_o‖ rising |
| Is the neighbours' *content* used? | **on < random** (random neighbours ≈ off means the content matters) |
| Is it more than copying? | gain of on over off in the **r < 8** bucket of `eval.py --tasks val` |
| Did it help the domain beyond what a small LoRA does alone? | ai_large / AI-subject MMLU: **on** vs the **control** run and vs web5b |
| Forgetting? | general MMLU vs base (0.5249) and vs web5b; retrieval **off** shows what the LoRA did to the decoder |

Expected, from the literature:
- clear on < off gains where the text overlaps the database (web_ai);
- small gains on non-overlapping text;
- uncertain multiple-choice gains: InstructRetro's downstream gains came mostly from the improved decoder.

A null result on the r < 8 bucket and on ai_large would match Norlund et al. and Retro-li at this scale.

## 6. Not covered / open choices

- **Generation tasks** (open QA, DS-1000) are not evaluated with retrieval. That needs re-retrieval every 64
  generated tokens inside `generate`.
- **BM25 re-ranking** of the dense top-k: the cheapest known retrieval gain (Doostmohammadi et al. 2023), but it
  raises lexical overlap and copying. Not implemented.
- **Near-duplicate clusters** (MinHash) across the database are approximated by the target-overlap filter, not
  built explicitly.
- **Encoder width** 512 vs 896–1024, and **CCA from layer 9** (Megatron's rule for > 15 layers): untested
  alternatives, each a config change in `configs/retro/retro_qwen06b_v1.yaml`.
- The database holds only AI text. Non-AI training text (web_replay, stem, long) retrieves loosely related AI
  chunks, which teaches the model to ignore irrelevant neighbours. That matters for non-AI MMLU subjects at
  evaluation.

## 7. Files

| file | purpose |
|---|---|
| `cpt_trainer/retro/model.py` | RetroConfig, encoder, CCA, forward pass / loss |
| `cpt_trainer/retro/db.py` | database builder + `Database` reader |
| `cpt_trainer/retro/data.py` | 1 B training set, same split / packing as web5b, query chunks |
| `cpt_trainer/retro/embed.py`, `index.py` | chunk embedder, sharded faiss index |
| `cpt_trainer/retro/neighbours.py` | neighbour search + leakage filters + overlap statistics |
| `cpt_trainer/retro/dataset.py` | datasets and collator (on / off / random / oracle modes, oracle warm-up, neighbour dropout) |
| `cpt_trainer/retro/train.py` | trainer (two LR groups, delayed LoRA, oracle schedule, eval points, `--no-retro` control) |
| `cpt_trainer/retro/eval.py` | MMLU / ai_large / MCQ with online retrieval; validation by mode and overlap bucket |
| `configs/retro/retro_qwen06b_v1.yaml`, `configs/retro/db_ai_v1.yaml` | architecture, database sources |
| `configs/lora/cpt_qwen06b_rs_adaptive_v5_r16_128.yaml` | LoRA ranks 16–128 |
| `runs/launch_retro_qwen06b_v1_web1b.sh` | staged launcher |
| `docs/retro-architecture.md`, `docs/ref/` | RETRO diagrams; the papers |
