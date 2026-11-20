# AI-Domain Q&A Model on Qwen3-1.7B: Task, Training Data, and Data-Split Pipeline

This document covers three things:

1. **The task**: a small, specialised model that answers questions about artificial intelligence for every kind of learner, from primary school children to AI researchers.
2. **The training data**: which public datasets were chosen, what each one contributes, and how they are combined for SFT, RL, and evaluation.
3. **The code** that pulls the AI-related part out of huge general datasets, using dataset metadata, a keyword lexicon, and the local zero-shot model.

---

## 1. Task definition

### 1.1 Goal

Fine-tune **Qwen3-1.7B** into an *AI tutor and assistant*. The model must:

| Capability | Example |
|---|---|
| Explain AI **concepts** | "What is overfitting?", "What does a transformer's attention do?" |
| Describe **applications and use cases** | "How is AI used in hospitals?", "How do recommendation systems work?" |
| Teach **theory and math derivations** | "Derive the gradient of the cross-entropy loss with softmax." |
| Tell the **history** of AI | "What was the Dartmouth workshop?", "Why were there AI winters?" |
| Write and debug **Python for AI** | "Write a PyTorch training loop", "Why does my Keras model not converge?" |
| Discuss **ethics and societal impact** | "Can AI be biased?", "Will AI take my job?" |
| Discuss **research** | "What does the paper *Attention Is All You Need* contribute?" |
| **Reason** step by step (Qwen3 thinking mode) | multi-step math, code, and conceptual reasoning |

### 1.2 Audiences (the "levels")

One model serves eight audiences. A level-specific **system prompt** tells it who it is talking to; see `ai_qa_data/taxonomy.py`.

| Level id | Audience | Expected answer style |
|---|---|---|
| `primary_school` | children aged 6–11 | very simple words, everyday analogies, no formulas |
| `middle_school` | ages 11–14 | clear explanations, relatable examples, basic math |
| `high_school` | high school students | precise terms, simple algebra, small Python snippets |
| `undergrad_non_ai` | university students in other majors | intuition first, light math |
| `ai_student` | CS / AI students | definitions, derivations, code |
| `general_public` | ordinary citizens | no jargon; daily-life impact, benefits, and risks |
| `researcher_non_ai` | scientists from other fields applying AI | methods, data, evaluation, pitfalls |
| `ai_researcher` | AI experts | formal notation, references to methods, trade-offs |

### 1.3 Training stages this data supports

```
          ┌──────────────────────────┐   ┌──────────────────────────────┐   ┌──────────────────┐
Qwen3 ──► │ SFT (sft_train.jsonl)    │──►│ RL / GRPO                     │──►│ Evaluation        │
1.7B      │ level-conditioned QA,    │   │ rl_math.jsonl  (\boxed answer)│   │ eval/mcq.jsonl    │
          │ <think> reasoning traces,│   │ rl_code.jsonl  (pytest tests) │   │ eval/code.jsonl   │
          │ ~10 % general replay     │   │ verifiable rewards            │   │ eval/open_qa.jsonl│
          └──────────────────────────┘   └──────────────────────────────┘   └──────────────────┘
```

* **SFT** teaches knowledge, level-appropriate style, and the reasoning format.
* **RL (GRPO-style)** sharpens the skills that can be checked automatically: math derivations with a final answer, and Python with unit tests.
* **Eval** is held out and decontaminated against every training file.

---

## 2. Training data

### 2.1 Design principles

1. **Use AI-native sources directly** (ai.stackexchange, datascience.stackexchange, ML Q&A sets, ML arXiv abstracts, AI history).
2. **Mine the AI part of huge general corpora** (OpenHermes, Tulu-3, SmolTalk, UltraChat, WildChat, Nemotron, Cosmopedia, …). This is where most of the volume comes from, and it is the main job of the split pipeline (section 3).
3. **Use metadata wherever it exists**: it is free and more precise than any classifier. Examples include the StackExchange site, MMLU `subject`, DeepMath `topic`, Cosmopedia `audience`, OpenHermes `category`, Tulu/SmolTalk `source`, WildChat `language`/`toxic`, and the Magicoder/CodeFeedback `lang` field.
4. **Cover every level.** Cosmopedia's `audience` field (young children, middle school, high school, college, researchers, general) is the main source of explicitly level-tagged explanations. ELI5 and WildChat represent ordinary citizens.
5. **Prefer verifiable data for RL.** Math problems need a final answer; code problems need unit tests.
6. **Keep a small general replay set (~10 %)** so the 1.7B model does not forget general skills.

### 2.2 Source catalog

All of these are defined in `ai_qa_data/sources.py`. Run `python scripts/collect_datasets.py --list` to print the catalog.

Column key:

* **Pool**: `curated` means AI-specific, kept whole. `huge` means a general corpus that is keyword-prefiltered while it is downloaded.
* **Meta rule**: what the source's own fields decide before any keyword or model is used.

#### A. Curated AI Q&A

| Name | HF dataset | Rows | Meta rule | What it gives | License |
|---|---|---:|---|---|---|
| `se_ai` | flax-sentence-embeddings/stackexchange… (`ai`) | 5.8k | site = ai.SE → **accept** | concepts, theory, philosophy, history | CC BY-NC-SA 4.0 (as listed) |
| `se_datascience` | same (`datascience`) | 20.5k | site → **accept** | practical ML/DL | same |
| `se_stats` | same (`stats`) | 116k | filtered | ML theory and derivations (CrossValidated) | same |
| `se_cs`, `se_cstheory`, `se_robotics`, `se_philosophy` | same | 30k / 7.7k / 4.6k / 13k | filtered | learning theory, RL for robots, philosophy of mind | same |
| `ml_qa_mjphayes`, `ai_qa_harikrishnan`, `ds_qa_soufyane`, `ml_qa_prsdm` | small ML Q&A sets | 0.1–1.1k each | **accept** | short concept definitions | unspecified |
| `so_python_dsml` | RazinAleks/SO-Python_QA-Data_Science_and_ML | 6.2k | filtered (*the "DS&ML" tag is noisy*) | Python ML debugging | unspecified |
| `ml_qa_collection` | win-wang/Machine_Learning_QA_Collection | 8.7k | filtered | StackOverflow-style ML Q&A | MIT |
| `ai_interview_kareem` | K-areem/AI-Interview-Questions | 4.7k | filtered | interview Q&A | unspecified |
| `ai_history_timeline` | chronicle-ai-history/ai-history-timeline | 78 | **accept** | AI history milestones, converted to Q&A | CC BY 4.0 |
| `ai_history_llm101_zh` | woohello/llm101-ai-history-sft | 3.7k | **accept** (lang = zh) | Chinese LLM/AI-history pairs, optional | Apache 2.0 |
| `arxiv_ml_abstracts` | CShorten/ML-ArXiv-Papers | 118k (cap 40k) | **accept** | "What is paper X about?", research level | AFL 3.0 |
| `arxiv_dl_code` | AlgorithmicResearchGroup/ArXivDLInstruct | 778k | torch/tf/jax code → **accept** | research-repo DL code | MIT |

#### B. Level-tagged explanations (huge, using the `audience` field)

| Name | HF dataset | Rows | Meta used | Levels it feeds | License |
|---|---|---:|---|---|---|
| `cosmo_stanford` | HuggingFaceTB/cosmopedia (`stanford`) | 1.0M | `audience` ∈ young_children / high_school / college / researchers | primary, high, undergrad, researcher | Apache 2.0 |
| `cosmo_khanacademy` | cosmopedia (`khanacademy`) | 24k | — | middle / high | Apache 2.0 |
| `cosmo_web_v2` | cosmopedia (`web_samples_v2`) | 10.3M | `audience` ∈ general / college | public, undergrad | Apache 2.0 |
| `cosmo_v2_smollm` | HuggingFaceTB/smollm-corpus (`cosmopedia-v2`) | 39M | `audience` ∈ children / middle_school / college / general / … | primary, **middle**, undergrad, public | ODC-BY |
| `eli5` | sentence-transformers/eli5 | 325k | — | general public | (Reddit) |
| `webinstruct` | TIGER-Lab/WebInstructSub | 2.3M | `source` = socratic → high school | high school, undergrad | Apache 2.0 |

Cosmopedia prompts ("Write a long and very detailed course unit for a textbook on 'X' intended for young children…") are rewritten into direct questions such as *Explain "3.1. Simple GANs" (from the course "Generative Adversarial Networks") for high school students.* The original prompt is kept in `meta.orig_prompt`.

#### C. General instruction and chat pools (huge, filtered)

| Name | HF dataset | Rows | Meta rule | License |
|---|---|---:|---|---|
| `openhermes` | teknium/OpenHermes-2.5 | 1.0M | `category` ∈ roleplay/joke/song/… → reject | unspecified (GPT-4 generated) |
| `tulu3_mix` | allenai/tulu-3-sft-mixture | 939k | `source` ∈ jailbreak/guard/aya/… → reject | ODC-BY |
| `smoltalk` | HuggingFaceTB/smoltalk (`all`) | 1.0M | `source` ∈ rewrite/summarize/function-calling → reject | unspecified (Apache 2.0 on card) |
| `ultrachat` | HuggingFaceH4/ultrachat_200k | 208k | — | MIT |
| `wildchat` | allenai/WildChat-1M | 838k | `language` ≠ English or `toxic`/`redacted` → reject | ODC-BY |
| `dolly` | databricks/databricks-dolly-15k | 15k | `category` = creative_writing → reject | CC BY-SA 3.0 |
| `nemotron_science` / `nemotron_chat` / `nemotron_code` | nvidia/Llama-Nemotron-Post-Training-Dataset (SFT) | 621k / 40k / 105k | — (R1 `<think>` traces) | CC BY 4.0 |
| `openthoughts` | open-thoughts/OpenThoughts-114k (`metadata`) | 114k | `domain` ∈ biology/chemistry/puzzle → reject | Apache 2.0 |

These pools supply **reasoning traces**: Nemotron and OpenThoughts include DeepSeek-R1 `<think>` reasoning, which becomes Qwen3 thinking-mode SFT data. WildChat supplies real citizens' questions about AI.

#### D. Python and ML code SFT

| Name | HF dataset | Rows | Meta rule | License |
|---|---|---:|---|---|
| `glaive_code` | glaiveai/glaive-code-assistant-v3 | 950k | — | Apache 2.0 |
| `magicoder_oss` | ise-uiuc/Magicoder-OSS-Instruct-75K | 75k | `lang` ≠ python → reject | MIT |
| `magicoder_evol` | ise-uiuc/Magicoder-Evol-Instruct-110K | 111k | — | Apache 2.0 |
| `codefeedback` | m-a-p/CodeFeedback-Filtered-Instruction | 157k | `lang` ≠ python → reject | Apache 2.0 |
| `self_oss_python` | bigcode/self-oss-instruct-sc2-exec-filter-50k | 51k | — | ODC-BY |
| `python_tested_143k` | Vezora/Tested-143k-Python-Alpaca | 143k | — | Apache 2.0 |
| `python_18k` | iamtarun/python_code_instructions_18k_alpaca | 19k | — | unspecified |

Only the ML / data-science part of these sets is kept: numpy, pandas, sklearn, torch, tf, matplotlib, and ML algorithms.

#### E. RL: verifiable Python code

| Name | HF dataset | Rows | Notes | License |
|---|---|---:|---|---|
| `kodcode_v1` | KodCode/KodCode-V1 | 484k | Python problems + **pytest unit tests**; `subset` Docs/Package covers library usage | **CC BY-NC 4.0** |
| `kodcode_rl10k` | KodCode/KodCode-Light-RL-10K | 10k | RL-curated subset | **CC BY-NC 4.0** |

#### F. RL: verifiable math (AI-relevant)

| Name | HF dataset | Rows | Meta rule | License |
|---|---|---:|---|---|
| `deepmath_103k` | zwhe99/DeepMath-103K | 103k | `topic` ∋ Linear Algebra / Probability / Statistics / Calculus / Optimization → **accept**; Geometry / Number Theory / Abstract Algebra / … → reject | MIT |
| `numina_cot` | AI-MO/NuminaMath-CoT | 860k | filtered; `\boxed{}` answer extracted as ground truth | Apache 2.0 |
| `openr1_math` | open-r1/OpenR1-Math-220k | 94k | `problem_type` ∈ Geometry / Number Theory / Logic → reject | Apache 2.0 |
| `deepscaler` | agentica-org/DeepScaleR-Preview-Dataset | 40k | filtered | MIT |
| `nemotron_math` | Nemotron SFT (`math`) | 277k | filtered; `\boxed{}` answer → ground truth | CC BY 4.0 |

"AI-relevant math" here means the **math foundations of ML**: linear algebra (matrices, eigen-decomposition, SVD), probability and statistics (Bayes, expectation, variance, distributions, MLE), calculus (gradients, chain rule, Jacobians), optimisation (convexity, Lagrange multipliers), and information theory (entropy, KL). Explicit ML derivations, such as the softmax gradient or the ELBO, come mostly from `se_stats`, `se_ai`, and Nemotron science.

#### G. Evaluation (never used for training)

| Name | HF dataset | Meta rule | License |
|---|---|---|---|
| `eval_mmlu` | cais/mmlu (test) | `subject` = machine_learning → accept; CS/stats subjects → filtered; others → reject | MIT |
| `eval_mmlu_pro` | TIGER-Lab/MMLU-Pro (test) | `category` ∈ CS/math/engineering → filtered; others → reject | MIT |
| `eval_ds1000` | xlangai/DS-1000 | accept (numpy/pandas/sklearn/torch/tf with execution tests) | CC BY-SA 4.0 |

`build_training_mix.py` additionally holds out an **open-QA eval** set: up to 150 items per level, never more than 5 % of a level.

> **License note.** KodCode and the StackExchange mirror are **non-commercial** (CC BY-NC). Several small sets have no declared license. OpenHermes and similar sets contain model-generated text that is subject to the generator's terms. If the model will be used commercially, drop those sources (set `enabled=False` in `sources.py`) or replace them. Examples: the official StackExchange dump (CC BY-SA), and self-generated code problems with tests.

### 2.3 Recommended mixture

The defaults in `build_training_mix.py` can all be changed with flags:

| Component | Target share / cap | Why |
|---|---|---|
| AI SFT, curated sources | keep all | highest precision, AI-native |
| AI SFT, mined from general pools | ≤ 50k per source (`--source-cap`) | stops one big pool from dominating style |
| Per level | ≤ 80k (`--level-cap`) | stops `ai_student` (the natural majority) from swamping the children/public levels |
| Reasoning traces (`<think>`) | everything available (`--reasoning think`) | Qwen3 hybrid thinking |
| General replay | 10 % of SFT (`--replay-ratio`) | anti-forgetting |
| RL math | ≤ 60k (`--rl-math-cap`) | GRPO with `\boxed{}` exact-match reward |
| RL code | ≤ 30k (`--rl-code-cap`), AI-relevant first | GRPO with pytest pass-rate reward |

Known gap: open data for **primary and middle school AI explanations** is scarce. Cosmopedia's children / middle-school audiences are the only large source. Recommended next step: generate level-rewrites with a teacher LLM. Take accepted `ai_student` / `general_public` Q&A pairs and ask the teacher to rewrite each answer for `primary_school` and `middle_school`, keeping the facts. Then filter the rewrites with the same zero-shot level classifier (`LEVELS[*]["zs"]`).

---

## 3. The data-split pipeline (code)

### 3.1 Layout

```
SMTrainer/
├── models/deberta-v3-base-zeroshot-v2.0/     # local zero-shot NLI model (MoritzLaurer)
├── ai_qa_data/
│   ├── taxonomy.py     # buckets, AI domains, 7 topics, 8 levels (+ zero-shot hypotheses, system prompts)
│   ├── sources.py      # dataset catalog: HF id, normalizer, metadata rule, level prior, bucket
│   ├── keywords.py     # weighted keyword/regex lexicon -> core / math / code scores
│   ├── zeroshot.py     # batched fp16 NLI zero-shot classifier
│   └── levels.py       # level inference (metadata + readability + technicality + cues + zero-shot)
├── scripts/
│   ├── collect_datasets.py    # step 1: download/stream, normalise, meta rule, keyword prefilter
│   ├── split_ai_data.py       # step 2: meta → keyword → zero-shot cascade, topic/level labels
│   └── build_training_mix.py  # step 3: dedup, decontaminate, balance, format for Qwen3
├── data/
│   ├── raw/      # step 1 output
│   ├── split/    # step 2 output
│   └── final/    # step 3 output (what you train on)
└── docs/AI_QA_DATASET_GUIDE.md
```

### 3.2 Overview

```
 HF Hub ──(parquet shard / stream)──► normalise ──► META RULE ──accept──────────────────────────┐
 (51 sources)                         unified        │                                          │
                                      record         ├─reject──► replay sample                  │
                                                     ▼                                          │
                                        KEYWORD PREFILTER (huge pools only, while downloading)  │
                                                     │ reject ──► replay sample                 │
                                    data/raw/*.jsonl ▼                                          │
                       ┌──────────── KEYWORD SCORE (core / math / code) ─────────────┐          │
                       │ ≥ accept thr                 between             < reject   │          │
                       ▼                                 ▼                     ▼     │          │
                    accept                    ZERO-SHOT NLI (DeBERTa-v3)    reject   │          │
                       │                    P(entail) ≥ thr ─► accept      (replay)  │          │
                       │                    else ──────────► reject (replay)         │          │
                       ▼                                 ▼                           │          ▼
                  ┌──────────────────────── LABEL accepted records ───────────────────────────────┐
                  │ topics: zero-shot (6 hyps) ∪ keyword hints ∪ code regex                        │
                  │ level : meta prior ⊕ Flesch-Kincaid ⊕ math/code/jargon density ⊕ cues ⊕ ZS     │
                  └───────────────────────────────────┬────────────────────────────────────────────┘
                                                      ▼
            data/split/{ai_sft, ai_rl_math, ai_rl_code, ai_eval, general_replay, audit}/<source>.jsonl
                                                      ▼
            build_training_mix: quality ► dedup ► 13-gram decontam ► caps ► Qwen3 chat format
                                                      ▼
            data/final/{sft_train, sft_val, rl_math, rl_code}.jsonl + eval/*.jsonl + stats.md
```

The cascade is ordered by cost. Metadata is free. Keywords cost about 1 ms per record on one CPU core. Zero-shot costs about 1.4 ms per hypothesis pair on an RTX 4090. Each record stops at the first stage that is confident, so the expensive model only sees the ambiguous few percent.

### 3.3 Stage 1: metadata rules (`sources.py`)

Each `Source` has `meta_rule(raw_row) -> "accept" | "reject" | None`. It runs on the **original** row, so any field of the source can be used:

| Source | Field | Rule |
|---|---|---|
| StackExchange | config (site) | `ai`, `datascience` → accept |
| MMLU | `subject` | `machine_learning` → accept; CS/stats subjects → undecided; others → reject |
| MMLU-Pro | `category` | CS/math/engineering → undecided; others → reject |
| DeepMath | `topic` (e.g. "Mathematics -> Algebra -> Linear Algebra -> Matrices") | ML-math topics → accept; geometry/number theory/… → reject |
| OpenR1-Math | `problem_type` | Geometry / Number Theory / Logic → reject |
| OpenHermes | `category` | role-play, jokes, songs, riddles, … → reject |
| Tulu-3 | `source` | jailbreak, guard, multilingual, hard-coded → reject |
| SmolTalk | `source` | rewrite, summarize, constraints, function-calling → reject |
| WildChat | `language`, `toxic`, `redacted` | non-English or toxic → reject |
| Magicoder-OSS, CodeFeedback | `lang` | non-Python → reject |
| OpenThoughts | `domain` | biology, chemistry, puzzle → reject |
| ArXivDLInstruct | `function` code | uses torch/tf/jax/sklearn/numpy → accept |
| Cosmopedia | `audience` | not a relevance rule; it sets the **level prior** |

Metadata also feeds `level_prior(row)`. Examples: Cosmopedia `audience`, MMLU `high_school_*` vs `college_*` subjects, NuminaMath `source` (gsm8k → middle school, cn_k12 → high school), DeepMath `difficulty`, and per-source defaults (ELI5 → general public, arXiv → AI researcher).

**Always validate metadata before trusting it.** During development, the `so_python_dsml` column "Data Science and Machine Learning" turned out to be `1` for every row, including MATLAB and Django questions. That source was therefore moved from "accept" to "let keywords and zero-shot decide".

### 3.4 Stage 2: keyword scoring (`keywords.py`)

There are three independent scores, because "AI-relevant" means different things for prose, math, and code:

| Score | Lexicon (examples) | Used for |
|---|---|---|
| **core** | machine/deep/reinforcement learning, neural network, backpropagation, gradient descent, overfitting, transformer *model*, LLM, GPT, CNN, Turing test, AI winter, Hinton, deepfake, AI safety, … (106 phrase patterns + 10 case-sensitive acronym groups) | every source |
| **math** | eigenvalue, SVD, Jacobian, Hessian, chain rule, Bayes, expectation, variance, MLE, entropy, KL divergence, convex, Lagrange, sigmoid, softmax, least squares, … | `ai_math` sources only |
| **code** | `import torch`, `nn.Module`, `.backward()`, `from sklearn`, `train_test_split`, `tf.keras`, `AutoTokenizer`, `np.linalg`, `pd.DataFrame`, … | every source |

Design details:

* **Weights**: 3 = unambiguous ("backpropagation"), 2 = strong but sometimes used elsewhere ("classifier", "AI"), 1 = weak ("dataset", "epoch"). A record's score is the sum over **distinct** terms, plus +25 % per repeat (capped). The question counts ×1.0 and the answer ×0.5. Long texts are damped by `1/sqrt(words/300)`.
* **Case-sensitive acronyms**: `AI`, `ML`, `NLP`, `CNN`, `GAN`, `LLM`, … are matched case-sensitively, so "said", "html", and "ml of milk" do not fire. `ML` is ignored when a millilitre quantity appears.
* **Negative contexts** subtract from the score: artificial insemination, Adobe Illustrator `.ai` files, CNN *news*, power/voltage *transformers*, Optimus Prime, neural *tube*, Ai Weiwei.
* **Topic hints** are collected at the same time (history names → `history`, bias/regulation → `ethics_society`, LaTeX → `theory_math`, …).
* **Decision** (`keyword_decision`): `core ≥ 6` or `code ≥ 6` → accept. If no relevant score reaches its reject threshold (core 2, code 3, math 3 for math sources) → reject. Otherwise the record is **uncertain** and goes to zero-shot.
* **Speed**: about 450 records/s per core with the v0.2 lexicon (long Q+A texts). Python `re` has no trie, so each term is compiled as one regex per top-level alternative, without a leading `\b`; this keeps the literal-prefix fast scan, and the left boundary is checked in Python. The splitter runs this stage in a multiprocessing pool.

### 3.5 Stage 3: zero-shot NLI (`zeroshot.py`)

Model: `models/deberta-v3-base-zeroshot-v2.0`, an NLI cross-encoder with `entailment` at label index 0. For each (text, hypothesis) pair, the score is P(entailment) = σ(logit_entail − logit_not_entail).

Relevance hypotheses (`taxonomy.DOMAIN_HYPOTHESES`):

| Label | Hypothesis | Default threshold |
|---|---|---:|
| `ai_core` | "This text is about artificial intelligence, machine learning, deep learning, or data science." | 0.85 |
| `ai_math` | "This is a math problem about linear algebra, probability, statistics, calculus, or optimization of the kind used in machine learning." | 0.85 |
| `ai_code` | "This is a programming task about machine learning, neural networks, or data analysis in Python." | 0.80 |

A record is accepted if any applicable label reaches its threshold. `ai_core` is always tested; `ai_math` or `ai_code` is added according to the source's domain. The premise is the question (≤ 1,200 chars) plus the start of the answer (≤ 600 chars), truncated to 256 tokens.

Implementation notes:

* Pairs are batched and sorted by length (fp16 on GPU). This is used instead of `pipeline("zero-shot-classification")`, which is several times slower on large corpora.
* Measured on an RTX 4090: about 730 pairs/s at 256 tokens and about 1,050 pairs/s at 192 tokens.
* **Audit file.** Decisions within ±0.1 of a threshold are written to `data/split/audit/<source>.jsonl` (up to 300 per source). Read them to recalibrate the thresholds per corpus with `--zs-core/--zs-math/--zs-code`.

Sanity check from development:

| Text | P(ai_core) |
|---|---:|
| "How does backpropagation compute gradients in a neural network?" | 0.989 |
| "Who proposed the Turing test and when?" | 0.994 |
| "Write a PyTorch training loop for an MLP." | 0.984 |
| "Compute the eigenvalues of the matrix [[2,1],[1,2]]." | 0.021 (→ that's what `ai_math` is for) |
| "What is the best recipe for chocolate cake?" | 0.004 |
| "The transformer in the power substation overheated." | 0.000 |

### 3.6 Labelling accepted records

**Topics** (multi-label): concepts, applications, theory_math, history, coding, ethics_society, research.

* Zero-shot multi-label with `TOPICS` hypotheses; P ≥ 0.85 (`--topic-thr`) counts as a hit. Strong keyword hints (≥ 4 weighted hits) also count.
* `coding` is **deterministic**: code is present (fenced block, `def`/`import`/`class` lines, method calls), or the task is a code task. The NLI model answers "yes, contains code" for almost any technical text (0.99 on a VAE-vs-GAN prose question), so it is not used for this label.

**Level** (single label), from `levels.infer_level`:

```
score(level) = 3.0 × metadata prior                     (Cosmopedia audience, grade, source default)
             + 1.0 × Flesch-Kincaid fit                  (answer grade vs. level range)
             + Σ pref(level, dim) × density(dim)         (dim ∈ math/LaTeX, code, jargon)
             + 0.5 × plainness bonus                     (plain-language levels only)
             + 1.0 × question cues                       ("ELI5", "for my biology thesis", "will AI take my job")
             + 1.5 × zero-shot P(level)                  (skipped when the metadata prior is ≥ 0.7 confident)
```

Code density favours `ai_student`, formula density favours `ai_researcher`, and plain prose favours `primary_school`, `middle_school`, and `general_public`. Scores are stored in `level_scores`, so you can re-threshold later without rerunning the model.

### 3.7 Record formats

**Unified record** (`data/raw`, `data/split`):

```json
{
  "id": "se_ai/1700", "source": "se_ai", "hf": "flax-…:ai:train",
  "bucket": "ai_sft", "domain": "ai_core", "task": "qa", "lang": "en",
  "question": "...", "answer": "...", "reasoning": null,
  "ground_truth": null, "tests": null, "messages": null,
  "meta": {"...source fields...": "..."},
  "hints": {"meta_decision": "accept", "level_prior": {"ai_student": 0.7, "general_public": 0.3}},
  "ai": {"by": "meta|keyword|zeroshot", "kw": {"core": 9.5, "math": 0, "code": 0, "hits": ["..."], "topic_hints": {}},
         "zs": {"ai_core": 0.97}},
  "topics": ["concepts", "theory_math"], "topic_scores": {"...": 0.0},
  "level": "ai_student", "level_scores": {"...": 0.0}, "level_zs": {"...": 0.0}
}
```

**Final SFT** (`data/final/sft_train.jsonl`), which works with TRL `SFTTrainer` and Qwen3's chat template:

```json
{"messages": [
   {"role": "system", "content": "You are a friendly AI tutor for primary school children ..."},
   {"role": "user", "content": "Explain \"What is a robot?\" for young children."},
   {"role": "assistant", "content": "<think>\n...optional reasoning...\n</think>\n\nA robot is ..."}],
 "level": "primary_school", "topics": ["concepts"], "source": "cosmo_v2_smollm", "has_reasoning": false}
```

**Final RL math** (`rl_math.jsonl`): `prompt` messages end with *"…put your final answer within \boxed{}"*, plus `ground_truth`. The reward is exact or symbolic match on the last `\boxed{}`, for example with `math-verify`.

**Final RL code** (`rl_code.jsonl`): `prompt` messages, `tests` (a pytest file that imports from `solution`), `test_info` (function signature), and `reference`. The reward is the pytest pass rate of the generated code written to `solution.py`, run in a sandbox.

### 3.8 How to run

```bash
pip install -r requirements.txt

# 0) inspect the catalog
python scripts/collect_datasets.py --list

# 1) collect (resumable: finished sources are skipped unless --force)
python scripts/collect_datasets.py --sources all --workers 8
#    bandwidth budget: one random parquet shard per source
python scripts/collect_datasets.py --sources all --max-shards 1
#    only some sources / buckets / pools
python scripts/collect_datasets.py --sources se_ai,eli5
python scripts/collect_datasets.py --sources ai_rl_math
#    crawl more of the huge pools (each source's max_scan × 5)
python scripts/collect_datasets.py --sources huge --scan-scale 5 --force

# 2) split (GPU recommended; resumable per source)
python scripts/split_ai_data.py
python scripts/split_ai_data.py --label-mode fast        # skip zero-shot topic/level labels
python scripts/split_ai_data.py --verify-accepts         # also send keyword-accepts to zero-shot
python scripts/split_ai_data.py --zs-core 0.9 --force    # stricter relevance

# 3) build the training mix
python scripts/build_training_mix.py --reasoning think --replay-ratio 0.1
```

Step 1 reads data in one of two ways:

* **Shard mode** (default): downloads one parquet shard at a time with `hf_hub_download` (hf_xet), reads only the needed columns with pyarrow, then deletes the shard. Peak disk use is one shard per worker. For JSON/JSONL and script-based repos it uses the Hub's auto-converted `refs/convert/parquet` branch, which can be *partial* (first ~5 GB) for very large datasets.
* **Streaming** (`--mode stream`): the fallback.

### 3.9 Extending

* **Add a source**: write a normaliser returning `{"question", "answer", "task", ...}` and append a `Source(...)` to `SOURCES`, with a `meta_rule` and a `level_prior` if the dataset has useful fields.
* **Tune vocabulary**: add `(regex, weight, topic)` entries to `CORE_TERMS` / `MATH_TERMS` / `CODE_TERMS`, and add false-positive contexts to `NEGATIVE_TERMS`.
* **Change label definitions**: edit hypotheses in `taxonomy.py`. Rerun step 2 with `--force` (step 1 does not need rerunning).
* **Different zero-shot model**: `--model path/to/any-NLI-model`. The entailment index is read from `config.label2id`.

### 3.10 Limitations

* The keyword lexicon is English. Non-English records (for example `ai_history_llm101_zh`) rely on metadata or zero-shot, and `--langs en` drops them from the final mix by default.
* NLI zero-shot over-entails on topic-like hypotheses for technical text. Thresholds are high (0.85) and borderline cases are written to `audit/` for inspection. For production precision, label ~500 audit records by hand and fit thresholds per source.
* Readability formulas are rough for code and math. That is why level inference combines them with metadata and zero-shot.
* Dedup is exact on normalised question text. Near-duplicate removal (MinHash) is a recommended addition when crawling several pools at full scale.

---

## 4. Results of the first collection run (2026-10-14)

**Budget.** This run was done on a link of about 1 MB/s, so it used `--max-shards 1`: one randomly chosen parquet shard per source. That is about 3.9 GB of candidates. Small curated sources were taken whole; the huge pools are a random slice of their full size. The full crawl, about 31 GB, is the same commands without `--max-shards`.

### 4.1 Split (step 2)

| | records |
|---|---:|
| Collected candidates (after metadata rules and keyword prefilter while downloading) | 959,828 |
| Accepted by **metadata** | 90,348 |
| Accepted by **keywords** | 93,250 |
| Accepted by **zero-shot** (from the uncertain records) | 55,607 |
| Rejected by keywords / by zero-shot | 645,171 / 75,452 |
| → `ai_sft` / `ai_rl_math` / `ai_rl_code` / `ai_eval` | 231,537 / 6,223 / 259 / 1,186 |

Notable per-source yields:

| source | kept as AI |
|---|---|
| se_stats | 30.5k of 116k |
| OpenHermes | 30.1k of 43.6k prefiltered candidates (1.0M rows scanned) |
| glaive_code | 19.7k |
| Cosmopedia v2 | 10.6k of 26.9k |
| magicoder_evol | 8.9k of 111k |
| python_tested_143k | 6.4k of 143k |
| WildChat | 2.6k |
| ELI5 | 277 |

Most of the zero-shot model's work is on the general pools: it accepted 1.8k of the 5k WildChat candidates and 11.4k of the 43.6k OpenHermes candidates.

### 4.2 Final mix (step 3, defaults)

| file | records |
|---|---:|
| `sft_train.jsonl` + `sft_val.jsonl` | 190,119 (val 3,802), including 2,960 with `<think>` reasoning and 17,283 general replay |
| `rl_math.jsonl` | 6,169 |
| `rl_code.jsonl` | 253 |
| `eval/mcq.jsonl` + `eval/code.jsonl` | 1,186 |
| `eval/open_qa.jsonl` | 984 |

Dropped:

* 31,973 exact duplicates;
* 4,923 failing quality checks (empty, short, or refusal answers);
* 681 contaminated, i.e. sharing a 13-gram with the AI eval items **or with the local MMLU copy** in `data/eval/mmlu`;
* 20,140 over the per-level cap (80k for `ai_student`).

SFT records per level:

| level | SFT | level | SFT |
|---|---:|---|---:|
| primary_school | 760 | ai_student | 80,000 (capped) |
| middle_school | 3,054 | general_public | 5,739 |
| high_school | 3,568 | researcher_non_ai | 853 |
| undergrad_non_ai | 10,499 | ai_researcher | 68,363 |

### 4.3 What the numbers say

* **The young-learner and general-public levels are thin**: 760 / 3.0k / 3.6k / 5.7k, against 80k for `ai_student`. This confirms the gap described in section 2.3. The next step is teacher-LLM level rewrites (section 2.3) and a larger Cosmopedia-v2 crawl (`--sources cosmo_v2_smollm --scan-scale 5 --force`).
* **RL-code data is very small (253)** because only one KodCode shard was read, and AI-specific code problems with tests are rare. Crawl all of `kodcode_v1`. Optionally keep a slice of general Python problems for RL as well: they are in `data/split/general_replay/kodcode_*.jsonl`.
* **History and ethics are rare topics** (698 / 866). Curated sources for them are worth adding, for example AI-history Q&A and AI-ethics course material.

---

## 5. Data versions: v0.1 evaluation and the v0.2 recipe

### 5.1 Versioning workflow

```bash
python scripts/snapshot_data_version.py --version v0.1 --note "..."   # freeze data/final + metadata + code
python scripts/build_training_mix.py --config configs/mix/v0.2.yaml   # build a new recipe into data/final
python scripts/snapshot_data_version.py --version v0.2 --note "..."
python scripts/snapshot_data_version.py --list
python scripts/snapshot_data_version.py --compare v0.1 v0.2
```

`data/versions/<v>/` holds `VERSION.json` (mix stats, split totals, per-source counts, file checksums,
runs and evals that used the data), a full copy of `final/`, hard links to the split files, the split and
collection metadata, and a copy of the pipeline code. Running `build_training_mix.py` without `--config`
reproduces v0.1 exactly (verified: same 190,119 SFT records and per-level counts).

### 5.2 What the v0.1 evaluation showed (lora_r16_v1 vs the chat model)

| | chat model | LoRA v0.1 |
|---|---:|---:|
| validation loss | 2.32 | **1.39** |
| open-QA ROUGE-L / token-F1 | 0.157 / 0.317 | **0.221 / 0.369** |
| MMLU / AI subjects (5-shot) | 57.6 / 50.1 | **58.3 / 52.1** |
| AI MCQ (186) / MMLU machine_learning (112) | 39.2 / 47.3 | 36.6 / 39.3 (both within noise) |
| open-QA **level fit** / judged audience match | **0.60 / 0.33** | 0.47 / 0.24 |

The model learned the references well, but **the references did not match their level labels**:
reference answers labelled `primary_school` had reading grade 10.9 (target ≤ 5.5) and ~450 words; only 3 %
fitted the level. After SFT the model's primary-school answers moved from grade 6.6/214 words to
10.4/439 words. Contributing factors: heuristic level labels (Cosmopedia "young children" textbook units
are long and complex), 78 % of the mix in `ai_student`/`ai_researcher` with 40k arXiv abstracts, all
general replay data carrying the `general_public` system prompt, and almost no reasoning traces (~600 AI).

### 5.3 v0.2 recipe (`configs/mix/v0.2.yaml`, rules in `ai_qa_data/mix_rules.py`)

| change | why |
|---|---|
| **Level fix**: plain-language levels (primary, middle, high school, general public) must fit their reading-grade range (±1.5) and a length cap (220 / 350 / 500 / 450 words); otherwise the example is relabelled up the ladder to the first level it fits (`level_fix: relabel`, or `drop`). Cosmopedia questions get their audience phrase rewritten to match. `ai_researcher` answers that read like plain prose move to `ai_student`. | plain-level fit 57 % → 97 % in a test build on v0.1 data |
| **Replay without a level system prompt** | stop teaching "general_public prompt → technical answer" |
| **Caps**: arXiv abstracts 40k → 4k, OpenHermes 15k, glaive 8k, …; levels `ai_student` 45k, `ai_researcher` 20k | less dense academic style, better level balance |
| **Reasoning**: keep `<think>` traces that fit ~2k tokens (≤ 7,000 chars incl. question and answer); DeepSeek-R1 traces stored with RL-math problems become SFT examples (too-long ones are dropped); other over-long traces keep only the final answer | train the thinking mode with AI-relevant traces |
| **New sources** (`added_in: v0.2`): Nemotron-v1 `stem` (DeepSeek-R1-0528 traces + expected answers), OpenScienceReasoning-2 (science MCQ with `<think>` + answer letter), SmolTalk2 OpenThoughts3 `_think`, MegaScience (cs/math), WebInstruct-verified (CS/math with verified answers), SYNTHETIC-1 verifiable code, Simple English Wikipedia AI/computing articles (plain-language answers for young learners), more KodCode shards | reasoning data, RL data, young-learner text |
| **RL MCQ** (`rl_mcq.jsonl`): half of the MCQ items with a known answer letter go to RL instead of SFT | checkable reward for GRPO |
| **Frozen evaluation**: v0.2 reuses the v0.1 eval files unchanged and excludes them (and the v0.1 validation set) from training | v0.1 vs v0.2 models are compared on identical items |
| **General reasoning bucket** (`reasoning_sft`): rows that fail the AI test but come from a source with `reasoning=True` (or `domain="reasoning"`) and pass `ai_qa_data/reasoning.py` go to `data/split/reasoning_sft/` instead of being dropped. The check: closed `<think>` trace followed by a real answer, question + trace + answer ≤ 12,000 chars at collection time, no repetition loops or refusals, final `\boxed{}` / option letter equal to the ground truth when the source has one. Rows without a trace count only as worked solutions (step-by-step math that ends in the answer, or Python code plus explanation). The mix adds them as `reasoning_ratio` (0.15) of AI SFT, round-robin over math / code / logic / science, max 15k per source, no level system prompt, over-long traces dropped (never stripped) | the AI filter rejected ~99 % of math/code/science reasoning traces (e.g. 3 of 10k OpenThoughts3 rows); general reasoning skill should not depend on the topic |
| **AI keywords extended** to all major AI/ML areas: learning paradigms, classical ML, deep-learning training/architectures, LLM/generative AI, NLP, vision, speech, RL, symbolic AI/search/planning, robotics, MLOps, trustworthy AI, more history; more ML library code terms (peft/trl, diffusers, faiss, gymnasium, sklearn estimators, `nn.*` layers). Words with common non-AI meanings (batch size, perplexity, bagging, discount factor, tool use, ...) only count in an AI phrase; ambiguous acronyms (SAM, NAS, ASR, ...) weigh 2 | AI questions outside the v0.1 lexicon were rejected before reaching zero-shot |

**Known remaining gap**: genuinely young-learner data. After the level fix only ~50 primary-school and
~700 middle-school examples remain from the v0.1 sources; Simple English Wikipedia adds some, but the real
fix is teacher-generated rewrites of the same question for each level (next step).
