# Retrieval as a training-only teacher for LoRA CPT of Qwen3-0.6B-Base

Prepared 2026-11-12. Nothing has run on the GPU yet; everything waits for `cpt_qwen06b_rs_adaptive_v4_web5b` to finish.
This plan replaces the main run of `docs/retro_fit_lora_cpt.md` (RETRO layers kept at inference). The database,
index, neighbours and 1 B-token training set of that plan are reused unchanged.

## 1. The change in one paragraph

The deployed model must stay a small on-device model: base + LoRA, with no database, no index and no extra layers.
Retrieval is therefore used **only while training**, as a *teacher*. For every training token the teacher sees the
same text **plus** passages retrieved from the AI database; the student (the LoRA model, which never sees them) is
trained to match the teacher's whole next-token distribution as well as the true next token:

    loss per token = (1 - α) · CE(student, true token) + α · KL(teacher ‖ student),   α = 0.5

Whatever the retrieved passages tell the teacher about the next token (the right term, the next step of an
explanation, the usual phrasing in the field) becomes a training signal for the LoRA. Plain CE only says which
single token came next.

| | web5b (running) | RETRO-fit plan (2026-11-11) | **this plan** |
|---|---|---|---|
| deployed model | base + LoRA 158.3 M | base + LoRA 39.6 M + RETRO 23.1 M + database ~19 GB + index ~5 GB | **base + LoRA 39.6 M** (~76 MiB bf16) |
| retrieval at inference | no | yes | **no** |
| training tokens | 4.97 B | 1.0 B | 1.0 B (same mix, same validation sequences) |
| what retrieval does | – | supplies knowledge at run time | **teaches the LoRA faster during training** |

## 2. What the literature says

### 2.1 Training with retrieval and dropping it afterwards does not, by itself, help

| work | finding | consequence for us |
|---|---|---|
| RETRO (Borgeaud et al. 2022, [2112.04426](https://arxiv.org/abs/2112.04426)) | RETRO evaluated **without** retrieval ("RETRO[OFF]") is close to the baseline trained without retrieval: training with retrieval left the decoder neither better nor much worse. | Keeping the RETRO-fit plan and simply removing the layers at inference would most likely give the control run's result, with 2× the training cost. |
| Samuel et al. 2024, "More Room for Language" ([2404.10939](https://arxiv.org/abs/2404.10939), NAACL short) | Pre-training **with** retrieval makes models learn **less world knowledge** (better syntax); the gap grows with scale. | When the context already contains the fact, gradient descent has no reason to store it. A LoRA trained next to RETRO layers would rely on them and learn *less* domain knowledge, the opposite of the goal. |
| InstructRetro (Wang et al. 2023, [2310.07713](https://arxiv.org/abs/2310.07713)) | The encoder can be removed and the decoder alone matches the full model downstream, **but the retrieved passages are then given in the prompt**. | The decoder did not absorb the knowledge; it became better at reading context. Not applicable to a model with no retrieval. |

### 2.2 What does move knowledge into the weights: a teacher that sees the extra text

| work | setting | finding |
|---|---|---|
| **kNN-KD** (Yang et al., NAACL 2022, [2205.00479](https://arxiv.org/abs/2205.00479)) | Machine translation. Retrieval (kNN over a datastore) is done **once, before training**; its distribution is distilled into a plain NMT model. | Matches or beats kNN-MT **with no retrieval at inference** and the base model's decoding speed (+2.1 BLEU De-En over the baseline). The closest precedent for "retrieval only in training". |
| **Prompt distillation** (Kujanpää et al. 2024, [2412.14964](https://arxiv.org/abs/2412.14964), "Efficient Knowledge Injection in LLMs via Self-Distillation") | The teacher is the **same model** with the new document in its prompt; the student is that model + a **LoRA** without the document, trained on the teacher's token distributions. | Knowledge injection on par with RAG and clearly better than plain fine-tuning; the same model as teacher beat 70 B teachers. Our setup is the same idea with retrieved neighbours instead of a given document, and raw text instead of generated questions. |
| **SDFT** (Shenfeld et al. 2026, [2601.19897](https://arxiv.org/abs/2601.19897)) | Context-conditioned copy of the model as its own teacher, on-policy. | Higher new-task accuracy than SFT **and much less forgetting**. Relevant because forgetting is our main cost (web5b: MMLU −0.7 points at 0.5 epoch). |
| Speculative Self-Distillation (NeurIPS 2026 poster) | Self-distillation for knowledge internalisation, teacher used only where its distribution differs from the student's. | Same accuracy with 45 % fewer supervised tokens: the signal sits on a minority of "knowledge-bearing" tokens. Motivates the optional `teacher_better` gate. |
| QUILL (Srinivasan et al. 2022, [2210.15718](https://arxiv.org/abs/2210.15718)) | A retrieval-augmented "professor" distilled into a model without retrieval. | Retrieval gains survive distillation into a retrieval-free student. |

None of these is continued pre-training of a 0.6 B model on raw web text, so the size of the gain here is unknown.
The control run (§5) is what decides it.

### 2.3 Why this should make the LoRA learn faster

- **More signal per token.** CE gives one target token. The teacher's distribution also ranks the plausible
  alternatives, which is where knowledge distillation's speed-up in pre-training comes from.
- **Signal where the student is weakest.** On tokens the neighbours predict (a term, a number, the name of a
  method, a standard phrase of the field), the teacher is confident while the student is not. The KL on those
  tokens is large and points straight at the domain answer.
- **The teacher keeps up.** With `--teacher-weights self` the teacher is the current student plus neighbours, so the
  KL measures only what the neighbours add, never pulls the student back towards the base model, and does not stop
  working once the student has learned the easy material.
- **Less forgetting, probably.** Matching a full distribution produced by (almost) the same model is a gentler
  update than one-hot CE. Self-distillation papers report less forgetting; we measure it with MMLU.

## 3. Design

### 3.1 Teacher choices (`--teacher`)

| teacher | what it is | extra parameters | step cost vs control | risk |
|---|---|---|---|---|
| **`context`** (main run) | The same decoder (base + current LoRA) reading up to 4 retrieved neighbours placed before every 1,024-token window | none | ~1.4× | Qwen3-0.6B-Base must make use of the neighbours in context. The epoch-0 check below measures this before any training. |
| `retro` (optional) | Base + RETRO layers (the 23.1 M module of the old plan), trained in the same run on their own CE with the LoRA off; dropped after training | 23.1 M, training only | ~2.1× | The literature's main risk for 1 B tokens remains: the cross-attention may never "activate", and then the teacher is just the base model. |
| `none` | CE only: the control | – | 1× | – |

`context` is the main run because the decoder already knows how to read context, so the teacher is useful from
step 0. The RETRO layers would first have to learn to read neighbours (§2 of the old report: ≥ 10× more tokens in
every successful retrofit).

### 3.2 The context teacher's sequence (`cpt_trainer/retro/distill.py: ContextTeacher`)

The student's packed 16–24 k sequence is cut into **windows of 1,024 tokens** inside each document piece. Before
each window the teacher gets a **block of up to 4 neighbours × 128 tokens** (each followed by `"\n\n"`):

    student:  [doc A: w1 w2 w3 ...][doc B: w1 w2 ...]
    teacher:  [doc A: N1 w1 N2 w2 N3 w3 ...][doc B: N1 w1 N2 w2 ...]

Attention rules (flex BlockMask; checked against a dense mask):

- A window's tokens see their whole document so far, **as in the student**: all earlier windows of the document,
  plus their own neighbour block. They do not see the neighbour blocks of earlier windows. Those earlier windows'
  tokens have read their own blocks, so later windows get that information indirectly.
- Neighbour tokens see only their own block.
- Document tokens keep their positions, shifted by one constant. With no neighbours the teacher therefore
  reproduces the student exactly (tested: identical hidden states).

Choosing neighbours for a window:

- **Candidates** are the precomputed per-chunk neighbours (`neighbours.py`) of the chunks inside the window. The
  same-document, same-text and copy-of-the-continuation filters are already applied.
- **Order:** candidates are taken rank by rank, with chunks visited in a spread order, so the four picks cover the
  whole window.
- **Skipped:**
  - database chunks adjacent to an already chosen one (their 128-token neighbours overlap by half);
  - neighbours whose 8-grams lie in the window at ≥ 50 % (`--ctx-max-overlap`). A copy of the window only
    reproduces the CE target.
- **Look-ahead is allowed.** A window's neighbours may have been retrieved from its later chunks. The teacher is
  privileged by design; only the student is deployed and scored.

Teacher cost: the teacher sees ~1.5× the student's tokens. It runs forward only, with no gradient and no stored
activations.

### 3.3 Loss (`kd_loss`)

`(1 − α)·CE + α·g·KL(teacher ‖ student)` per target token, full vocabulary, logits built 512 tokens at a time and
recomputed in backward (four float32 [512 × 151,936] tensors ≈ 1.2 GB peak).

- `α = 0.5`. kNN-KD and prompt distillation use comparable mixes. With α = 1 the student could inherit the
  teacher's mistakes.
- `g = 1` by default. `--kd-gate teacher_better` applies the KL only where the teacher gives the true token more
  probability than the student does. This focuses on the tokens the neighbours helped with. Without it, the KL also
  acts as a light regulariser on all other tokens.
- Temperature 1.

### 3.4 Everything else is as planned before

- **LoRA:** `configs/lora/cpt_qwen06b_rs_adaptive_v5_r16_128.yaml` (the v4 plan ÷ 4 = ranks 16–128, rsLoRA α 16,
  39.6 M).
- **Optimisation:** lr 5e-5, cosine with 3 % warm-up, 1 epoch, bs 1 × accum 8, seed 42.
- **Data:** the web5b mix at 1 B tokens: `runs/_retro_cache/web1b`, ~42.4 k sequences, ~5,300 steps. The
  validation sequences are identical to web5b's, so epoch-0 scores carry over and losses compare directly.
- **Database and neighbours:** every AI document, decontaminated against the eval sets: `cpt_data/retro_db/ai_v1`,
  bge-small + faiss, top-4 neighbours per 64-token chunk after the leakage filters.
- **Gradient clipping:** at 1.0, per parameter group. The RETRO teacher's randomly initialised layers would
  otherwise dominate the global norm and shrink the LoRA's step.
- **RETRO teacher only** (as in the old plan):
  - RETRO lr 2e-4;
  - oracle-neighbour warm-up 0.5 → 0 over the first 5 % of steps;
  - KD starts at 5 % (`--kd-start-frac`), because before that the teacher is the base model.

  The teacher's CE is backpropagated inside the LoRA-off context, so gradient-checkpoint recomputation sees the
  same weights.

### 3.5 Evaluation: the student alone, exactly like web5b

- **Every 10 % of the steps:** validation loss of the student and of the teacher on the 5 corpora. The **gap** =
  student − teacher is what the teacher still knows that the student does not. It should shrink if knowledge
  transfers.
- **Before training:** base vs base + neighbours on the same sequences. This is the headroom the teacher offers. If
  it is close to 0 on web_ai, the context teacher has nothing to give, and the run should not start.
- **At 0.25 / 0.5 / 0.75 / 1:**
  - The plain LoRA adapter is exported to `epoch<x>/adapter`, with RETRO weights stripped in the retro mode.
  - It is scored by `scripts/eval_model.py` through `cpt_trainer.mmlu.run_eval`, with web5b's settings: 5-shot
    MMLU incl. ai_large, and the AI MCQ.
  - The final point adds open_qa on Qwen3-0.6B chat, DS-1000 and the SFT val loss.
- **Results:** `eval_history.json` and `summary.md` in the run folder (same table as web5b, plus a teacher-gap
  table).

## 4. How to run (after web5b finishes)

`runs/launch_rkd_qwen06b_v1_web1b.sh <stage>`. The first four stages are those of the RETRO-fit plan and are
shared.

| stage | device | time (estimate) | output |
|---|---|---|---|
| `db` | CPU | 1–1.5 h | `cpt_data/retro_db/ai_v1` (~19 GB) |
| `set` | CPU | ~15 min | `runs/_retro_cache/web1b` (~5 GB) |
| `index` | GPU | ~2–3 h | faiss index (~5 GB) |
| `neighbours` | GPU | ~1–2 h | `<set>/nbr/ai_v1/*.npy` + `stats.json` |
| `smoke` | GPU | ~15 min | 20 steps of each teacher: memory and s/step |
| `control` | GPU | **~1.7 days** | `runs/rkd_none_qwen06b_v1_web1b`: needs only `set` |
| `context` | GPU | **~2.3 days** | `runs/rkd_ctx_qwen06b_v1_web1b`: the main run |
| `retro` | GPU | ~3.5 days | optional |
| `anchor` | GPU | ~2 days | optional ablation: KD to the base model with no neighbours (`--teacher-weights base --ctx-k 0`) |

**Time estimates** are scaled from web5b's measured ~6.3 k tokens/s (r64–512, full gradient checkpointing).
Teacher cost is added from the extra FLOPs: forward-only teacher on ~1.5× tokens, plus two extra LM-head matmuls.
The `smoke` stage gives the real numbers.

**Suggested order:**

1. `db` + `set`: CPU; can start while web5b trains.
2. After web5b: `index`, `neighbours`, `smoke`.
3. `control` and `context`. They see the same sequences in the same order (same seed and sampler), so any
   difference comes from the teacher.
4. `retro` / `anchor` only if `context` beats `control`.

**Disk:** ~30 GB for the database, set and index, plus ~1 GB per run.

## 5. How to read the results

| question | evidence |
|---|---|
| Does the teacher have anything to teach? | Epoch-0 line of `train.log`: base vs base + neighbours on web_ai / web_near. For scale: the RETRO retrofit on WikiText-103 gained ~0.15 nats (perplexity 21.5 → 18.5). A headroom near 0 means the neighbours are not useful in context. |
| Does the LoRA learn faster? | Student validation loss on web_ai / web_near at each 10 % point, `context` vs `control`. **Speed-up** = the share of the control's steps the context run needs to reach the control's final loss. The seed-to-seed noise of these losses has not been measured, so a small difference needs a second seed before it is trusted. |
| Is knowledge transferring? | The teacher-gap table: the gap on web_ai should shrink over training while the student's loss falls faster than the control's. |
| Does it show up on the target? | ai_large (target: Qwen3-1.7B's 0.685; base 0.571; web5b 0.584 at 0.5 epoch with 2.5 B tokens) and the AI MCQ vs `control` and web5b. ai_large has 1,000 questions, so one standard error is ~1.6 points. |
| Forgetting? | General MMLU vs base 0.5249, `control` and web5b (0.5177 at 0.5 epoch). |
| Is the gain from the neighbours, or from distillation as such? | `anchor` (distillation to the base model without neighbours): if it matches `context`, the neighbours do not matter. |

## 6. Risks and open choices

- **Neighbour quality limits everything.** bge-small retrieval over 64-token chunks finds related text, but may
  rarely find the specific fact the next token needs. Check `stats.json` of the neighbours stage and the epoch-0
  headroom first.
- **0.6 B as its own teacher.** A small model reads context less well than the 8 B models of the prompt-distillation
  paper. The teacher could be Qwen3-1.7B-Base reading the same neighbours. It shares the tokenizer, so the
  distributions are directly comparable. It costs ~3× more teacher compute and is not implemented; it is the next
  step if the 0.6 B teacher's headroom is small.
- **Settings not tuned:**
  - α;
  - the window (1,024) and k (4); more neighbours give more information but cost more;
  - the gate;
  - self vs base teacher weights.

  One run per setting is unaffordable at 2 days each. If needed, a 0.1 B-token pilot (`--max-steps 530`) of
  `control` and `context` takes ~5 h each.
- **GPU memory is unmeasured.** The teacher sequence is ≤ 1.5 × 24.6 k tokens, forward only. The KD chunk adds
  ~1.2 GB. The smaller LoRA saves ~1.4 GB of Adam state vs web5b, which used 15.5 GB. Lower `--kd-chunk` if needed.
- **Open QA and code** are only evaluated at the end, as in web5b.

## 7. Tests done (CPU, tiny random Qwen3 with the real tokenizer, smoke data and database)

- Teacher with no neighbours = student, to the bit. The teacher-token map points at the right tokens. All
  positions are ≥ 0.
- Changing one window's neighbour block changes only that window and the rest of its document; no earlier window,
  no other document.
- LoRA-off teacher = the plain base model. The LoRA switch restores its flags and leaves `requires_grad` untouched.
- `kd_loss` with α = 0 equals the existing CE loss. Gradients match a direct full-logits implementation to 1e-6,
  with and without the gate.
- Flex BlockMask = dense mask (identical pair set); CPU flex_attention vs SDPA: 5e-7.
- The trainer runs in all three modes, including eval points, validation-only points, the epoch-0 teacher check,
  checkpoints and the final summary.
- The RETRO-mode student export contains no RETRO weights and no `modules_to_save`. The teacher copy keeps them.
- Resume in the retro mode, the most stateful one (RETRO layers, two learning-rate groups, oracle schedule): resuming
  from checkpoint 3 reproduces step 4 exactly. The loss, both groups' gradient norms and the validation losses all
  match. Per-group clipping logs a separate norm for the LoRA and for the RETRO layers.

Not tested: anything on the GPU. That covers speed, memory, and flex at full length with the compiled kernels,
which is what the `smoke` stage is for.

## 8. Files

| file | purpose |
|---|---|
| `cpt_trainer/retro/distill.py` | context teacher layout, teacher mask, LoRA on/off, KD loss, student export |
| `cpt_trainer/retro/train_distill.py` | trainer: `--teacher context / retro / none`, per-group clipping, evaluation of the student |
| `runs/launch_rkd_qwen06b_v1_web1b.sh` | staged launcher |
| `cpt_trainer/retro/{db,data,embed,index,neighbours,dataset,model}.py` | reused from the RETRO-fit plan |
| `docs/retro_fit_lora_cpt.md` | the RETRO-at-inference plan, its literature review and data / database details |
