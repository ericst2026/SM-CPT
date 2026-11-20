# Training, Evaluating, and Improving the AI Tutor (Qwen3-1.7B + LoRA)

This guide covers the second half of the project. The first half is [AI_QA_DATASET_GUIDE.md](AI_QA_DATASET_GUIDE.md), which builds `data/final/`. This half:

1. trains `models/Qwen3-1.7b` with **LoRA-family adapters** on the AI Q&A mix;
2. **evaluates** the base model and any adapter on the held-out AI eval sets;
3. lets a person **test the model by hand, rate and correct its answers**, and feed those corrections back into training.

```
data/final/sft_train.jsonl ──► train_lora.py ──► runs/<run>/adapter ──► eval_model.py ──► eval_results/<run>/
                                     ▲                    │                                   │
                                     │                    ▼                                   ▼
             data/feedback/feedback_sft.jsonl ◄── chat_feedback.py (chat / review / export) ◄─┘
```

## 1. Files

| File | Purpose |
|---|---|
| `ai_qa_train/common.py` | model and adapter loading, LoRA-variant configs, Qwen3 chat encoding with assistant-only labels, batched generation |
| `ai_qa_train/feedback.py` | feedback store (`data/feedback/feedback.jsonl`), statistics, export to SFT and DPO files |
| `scripts/train_lora.py` | SFT with LoRA / QLoRA / DoRA / QDoRA / rsLoRA / PiSSA / OLoRA / LoRA+ |
| `scripts/eval_model.py` | MMLU (local), MCQ, open-QA (per learner level), DS-1000 code execution, validation loss, optional math; serves a web report at the end |
| `ai_qa_train/report.py` | the self-contained HTML evaluation report + the all-runs index + a local server |
| `scripts/chat_feedback.py` | Gradio web UI or terminal chat: test, rate, correct, review eval answers, export |
| `cpt_trainer/` | **continued pre-training** of `models/Qwen3-0.6B-Base` with a rank-64 LoRA on the document corpora (`corpus/ai` + `corpus/replay`), 5-shot MMLU after every epoch; see `cpt_trainer/README.md` |

Setup:

```bash
pip install -r requirements.txt   # torch, transformers, datasets, peft, bitsandbytes, accelerate, gradio, math-verify
                                  # (+ scikit-learn, scipy, matplotlib for DS-1000)
```

## 2. Training (`scripts/train_lora.py`)

### 2.1 Methods

Every method is selected with `--method`. All of them use the same data, collator, and trainer.

| `--method` | What it is | When to use | Typical `--lr` |
|---|---|---|---|
| `lora` | standard LoRA on the bf16 base | default, fast, solid | 2e-4 |
| `qlora` | LoRA on a 4-bit NF4 base (bitsandbytes), paged 8-bit AdamW | low GPU memory, or long `--max-len` / big batches | 2e-4 |
| `dora` | weight-decomposed LoRA (magnitude + direction) | often a little better than LoRA at the same rank; ~20–30 % slower | 1e-4 |
| `qdora` | DoRA on a 4-bit base | DoRA with less memory | 1e-4 |
| `rslora` | rank-stabilised scaling α/√r | high ranks (r ≥ 64), where plain LoRA under-trains | 2e-4 |
| `pissa` | LoRA initialised from the principal singular vectors of W (adapter dropout forced to 0) | faster convergence; use a lower LR | 2e-5 – 1e-4 |
| `olora` | orthonormal (QR) initialisation | alternative init; stable | 2e-4 |
| `loraplus` | LoRA with lr(B) = ratio × lr(A) (`--loraplus-ratio`, default 16) | faster convergence at the same cost | 1e-4 |

Shared options:

| Option | Meaning |
|---|---|
| `--r` | rank (default 16) |
| `--alpha` | scaling α (default 32) |
| `--dropout` | adapter dropout (default 0.05) |
| `--targets` | which modules get adapters: `all-linear` (q, k, v, o, gate, up, down; default), `attn`, `mlp`, `qv`, or a comma-separated list |

All 8 methods were smoke-tested on this machine (RTX 4090): each trained, ran validation, and saved an adapter. At r = 16 on all linear layers, 17.4M parameters are trainable (1.0 % of 1.74B); DoRA has 18.0M.

### 2.2 Data handling

* Input is chat JSONL with a `messages` list. `data/final/sft_train.jsonl` already includes the **level-specific system prompt**.
* Messages are rendered with **Qwen3's own chat template**. The loss covers **assistant tokens only**: every assistant turn plus its `<|im_end|>`, found through character offsets of the rendered template.
* **Thinking mode.** Samples that have a reasoning trace keep `<think>…</think>` in the final assistant turn. Samples without one get the template's empty `<think>\n\n</think>` block, which is Qwen3's non-thinking format. Training therefore preserves both modes.
* **Length.** Sequences are truncated to `--max-len` (default 2048). A sample is dropped only if its whole assistant answer falls outside that limit.
* **Batching.** Samples of similar length are grouped into a batch (`group_by_length`) to cut padding. Gradient checkpointing is on by default (`--no-grad-ckpt` turns it off).
* **Sample selection.**
  * `--levels primary_school,middle_school` trains on chosen levels only.
  * `--reasoning-frac 0.3` keeps only 30 % of the long reasoning samples.
  * `--max-samples` caps the dataset size.

### 2.3 Commands

```bash
python scripts/train_lora.py --list-methods

# recommended first run
python scripts/train_lora.py --method lora --r 16 --alpha 32 --epochs 1 --lr 2e-4 \
       --bs 4 --grad-accum 8 --max-len 2048 --run-name lora_r16

# other variants
python scripts/train_lora.py --method dora     --lr 1e-4 --run-name dora_r16
python scripts/train_lora.py --method qlora    --r 64 --alpha 128 --bs 8 --run-name qlora_r64
python scripts/train_lora.py --method rslora   --r 128 --alpha 128 --run-name rslora_r128
python scripts/train_lora.py --method pissa    --lr 5e-5 --run-name pissa_r16
python scripts/train_lora.py --method loraplus --lr 1e-4 --loraplus-ratio 16 --run-name loraplus_r16

# quick smoke test
python scripts/train_lora.py --method lora --max-samples 2000 --max-steps 50 --run-name smoke

# resume an interrupted run (same --run-name)
python scripts/train_lora.py --method lora --run-name lora_r16 --resume

# save a merged full model as well (for vLLM, GGUF conversion, ...)
python scripts/train_lora.py --method lora --run-name lora_r16 --merge
```

Effective batch size = `--bs × --grad-accum` (default 32 sequences). On a 24 GB GPU, bf16 LoRA with `--bs 4 --max-len 2048` fits comfortably with gradient checkpointing. Use `qlora` or `qdora` if you need longer contexts or bigger batches.

### 2.4 Outputs (`runs/<run-name>/`)

| Path | Contents |
|---|---|
| `adapter/` | final adapter + tokenizer. Pass this to `eval_model.py` and `chat_feedback.py`. |
| `checkpoint-*/` | the last 3 checkpoints (resumable) |
| `run_config.json` | every CLI argument, plus dataset statistics (records, tokens, level mix) |
| `log_history.json` | training and validation loss curve |
| `merged/` | with `--merge`: the full bf16 model |

## 3. Evaluation (`scripts/eval_model.py`)

```bash
python scripts/eval_model.py --tag base                                   # baseline: the untouched Qwen3-1.7B
python scripts/eval_model.py --model runs/lora_r16/adapter \
       --compare eval_results/base/summary.json                           # adapter, with deltas vs base
python scripts/eval_model.py --model runs/lora_r16 --tasks open_qa --level-judge --limit 300
python scripts/eval_model.py --model runs/lora_r16 --thinking             # generate in thinking mode

# MMLU from the local copy in data/eval/mmlu
python scripts/eval_model.py --tasks mmlu                                  # all 57 subjects, 0-shot
python scripts/eval_model.py --tasks mmlu --mmlu-shots 5                   # standard 5-shot (dev split)
python scripts/eval_model.py --model models/Qwen3-1.7B-Base --tasks mmlu --mmlu-shots 5   # base model: format auto -> base
python scripts/eval_model.py --tasks mmlu --mmlu-subjects ai               # AI-related subjects only
python scripts/eval_model.py --tasks mmlu --mmlu-subjects STEM             # one category
python scripts/eval_model.py --tasks mmlu --mmlu-subjects machine_learning,college_computer_science
python scripts/eval_model.py --tasks mmlu --mmlu-max-per-subject 20        # quick check
```

| Task | Data (`data/final/eval/`) | Metric |
|---|---|---|
| `mmlu` | **local MMLU** in `data/eval/mmlu/<subject>/{dev,validation,test}-*.parquet` (HF `cais/mmlu` layout; `--mmlu-dir` to change) | accuracy. Answer letter by log-likelihood (non-thinking mode); 0-shot or k-shot with examples from `dev` (`--mmlu-shots`). Reports **micro** accuracy, **macro** accuracy (mean over subjects), **AI-related subjects** accuracy (machine_learning, CS, computer_security, statistics, math, econometrics, electrical_engineering, formal_logic), plus per category (STEM / humanities / social sciences / other, the official MMLU mapping) and per subject. Use it to check that AI specialisation does not erode general knowledge. |
| `mcq` | `mcq.jsonl`: MMLU `machine_learning` + filtered CS/stats subjects, plus AI-related MMLU-Pro CS/math/engineering items | accuracy. The answer letter is chosen by **next-token log-likelihood** in non-thinking mode, so scoring is deterministic and fast. Reported per source and per subject. |
| `open_qa` | `open_qa.jsonl`: held-out questions for each of the 8 learner levels, with references | ROUGE-L and token-F1 against the reference; **level fit**; with `--level-judge`, **judged match** |
| `code` | `code.jsonl`: DS-1000 (numpy, pandas, scipy, sklearn, pytorch, tensorflow, matplotlib) | **pass@1**, running the official DS-1000 test harness in a subprocess. Reported per library. Problems whose library is missing are counted under `env_missing_lib`; TensorFlow is not installed here. |
| `val_loss` | `../sft_val.jsonl` | assistant-token loss and perplexity, overall and per level |
| `math` (optional) | `--math-file` with `prompt` + `ground_truth` | `\boxed{}` accuracy with `math-verify`. Only meaningful on problems that were not trained on. |

Open-QA metric definitions:

* **Level fit**: the Flesch-Kincaid grade of the model's answer lies within its level's target range (±1). For example, `primary_school` targets grade 0–5.5 and `ai_researcher` 15+.
* **Judged match** (`--level-judge`): the local DeBERTa zero-shot model classifies which audience the answer is written for, and the score is how often that equals the requested level.

MMLU details:

* Choose subjects with `--mmlu-subjects`: `all` (default), `ai`, a category name (`STEM`, `humanities`, `social_sciences`, `other`), or a comma-separated list of subject folder names. The `all` and `auxiliary_train` folders are ignored.
* `--mmlu-split test|validation|dev` picks the split.
* It is fast: about 180 questions/s on an RTX 4090 at 0-shot, so the full 14,042-question test split takes about 1.5 minutes.
* **Prompt format** (`--mmlu-format`):
  * `auto` (default) picks `base` for pre-trained checkpoints (directory or repo name containing `Base`, e.g. `models/Qwen3-1.7B-Base`, and adapters trained on one) and `chat` otherwise.
  * `chat` scores the model **as the tutor uses it**: chat template, non-thinking mode, answer letter right after the assistant prefix. Few-shot examples are given as **earlier user/assistant turns**.
  * `base` is the classic completion prompt (`…Answer:` → `" A"`/`" B"`/…). This is how the Qwen3 report and lm-eval-harness evaluate base models.
* **Degenerate-scoring guard.** Every run records `letter_prob_mass`, the probability the model puts on the answer letters, and the distribution of predicted letters. If over 70 % of predictions are one letter, or the letter mass is below 20 %, the run prints a warning and the report shows a red banner. The typical cause is the chat format used on a Base model: the Base model was never trained on the chat template or `<think>` block, so it put about 3 % probability on the letters and answered "A" almost every time (~24 % accuracy).

Measured on the full test split (14,042 questions):

| model | format | shots | MMLU | letter mass | reference |
|---|---|---:|---:|---:|---|
| Qwen3-0.6B-**Base** | base | 5 | **52.49 %** | – | paper Table 8: 52.81 % |
| Qwen3-1.7B-**Base** | base (auto) | 5 | **62.73 %** | 0.999 | paper Table 8: **62.63 %** → harness reproduces the report |
| Qwen3-1.7B-Base | chat (forced) | 5 | ~24–35 % | 0.03 | ⚠ degenerate: the warning fires |
| Qwen3-1.7B (post-trained, `models/Qwen3-1.7b`) | chat (auto) | 0 | 53.95 % | ~0.5 | 0-shot chat: the model often wants to start with words, not a letter |
| Qwen3-1.7B | chat (auto), multi-turn few-shot | 5 | **57.53 %** | 0.977 | |
| Qwen3-1.7B | base | 0 | 55.58 % | – | |
| Qwen3-1.7B | base | 5 | 60.16 % | – | |

Notes:

* The paper's 62.63 % is for **Qwen3-1.7B-Base** (Table 8, pre-training evaluation, 5-shot). The post-trained Qwen3-1.7B is not reported on MMLU; the report gives only MMLU-Redux: 64.4 non-thinking, 73.9 thinking (Tables 19–20).
* **Recommended settings:**
  * tracking the tutor (chat model, or adapters on it): `--mmlu-shots 5`, format auto → chat;
  * comparing with published base-model numbers: `--model models/Qwen3-1.7B-Base --mmlu-shots 5`, format auto → base.

  Compare runs only within the same format and shot count (`--compare` shows both).
* If only `data/eval/mmlu.zip` exists, the script tells you to unzip it.

These two metrics answer the question this project cares about: *does the model adapt its explanation to the learner?*

Outputs go to `eval_results/<tag>/`:

* `summary.json` and `summary.md`. With `--compare`, deltas against an earlier summary are shown.
* `*_predictions.jsonl` with every generation. These can be reviewed in the feedback UI.

### 3.1 Web report

When an evaluation finishes, `eval_model.py` writes `eval_results/<tag>/report.html` and serves it at **http://127.0.0.1:7870/<tag>/report.html**, opening it in your browser. Ctrl-C stops the server.

The report has three parts:

* **Score tiles**: MMLU, MMLU AI subjects, AI multiple choice, open-QA level fit and ROUGE-L, DS-1000 pass@1, math, and validation loss. Each shows the change against the `--compare` run (▲/▼).
* **Bar charts** for every score:
  * all headline metrics together;
  * MMLU by category, and by subject (sorted);
  * AI multiple choice by subject;
  * open-QA ROUGE-L, level fit and judged audience match per learner level;
  * a **reading-grade vs target range** chart: for each level, the model's mean reading grade is drawn against that level's target band;
  * DS-1000 by library;
  * validation loss by level.

  With `--compare`, each chart shows both runs side by side, with a legend. Hovering a bar shows the values, the change, and `n`. Every chart has a **Table** view.
* **Examples**, one tab per area (MMLU, AI MCQ, open QA, code, math). You can filter by right/wrong, by subject, level or library, and by free-text search. Each card shows:
  * MMLU: the options, with the correct answer and the model's pick marked;
  * open QA: the model's answer, reasoning trace, reference, ROUGE-L, reading grade and judged audience;
  * code: the problem, the model's code, the error, and the reference solution.

  You can link straight to a tab with `report.html#code`.

`eval_results/index.html` lists every run with its headline metrics and links to each report. The page has no external dependencies and works in light and dark mode and at phone width.

```bash
python scripts/eval_model.py --report-only base                       # reopen / rebuild an earlier run's report
python scripts/eval_model.py --report-only lora_r16 --compare base    # ... side by side with another run
python scripts/eval_model.py --tasks mmlu --no-ui                     # batch jobs: write report.html, don't serve
python scripts/eval_model.py --ui-host 0.0.0.0 --ui-port 8000         # reachable from other machines
python scripts/eval_model.py --no-browser                             # serve without opening a browser (remote / SSH)
```

`--compare` accepts a `summary.json`, a run folder, or just a tag (`--compare base`). `--report-examples N` sets how many examples per area are embedded (default 300, balanced across subjects or levels, and between right and wrong answers).

> DS-1000 runs model-written code in a subprocess, with a timeout, `MPLBACKEND=Agg`, and no GPU. That is **not a security sandbox**. Run it only on a machine where executing generated code is acceptable, or inside a container.

## 4. Manual testing and feedback (`scripts/chat_feedback.py`)

```bash
python scripts/chat_feedback.py --model runs/lora_r16/adapter      # web UI on http://127.0.0.1:7860
python scripts/chat_feedback.py --cli --model runs/lora_r16          # terminal
python scripts/chat_feedback.py --stats
python scripts/chat_feedback.py --export
```

**Chat tab**

* **Model** dropdown: base, or any `runs/*/adapter`, `checkpoint-*`, or `merged`. Click *Load model* to switch without restarting.
* **Learner level**: fills in the level's system prompt, which you can edit. Also set thinking on/off, temperature, top-p, and max tokens.
* Chat over multiple turns. The answer streams in; the reasoning trace appears in a collapsible box.
* Rate the **last answer**:
  * verdict: good / neutral / bad;
  * rating: 1–5;
  * tags: `correct`, `factually_wrong`, `wrong_level`, `too_long`, `too_short`, `unclear`, `code_bug`, `math_error`, `off_topic`, `unsafe`, `great_example`;
  * a comment;
  * an optional **corrected answer** ("Copy model answer" lets you edit the model's answer in place).
* *Save feedback* appends the record to `data/feedback/feedback.jsonl`. Each record stores the model, level, system prompt, generation settings, full conversation, response, thinking, rating, tags, comment, and correction.

**Review tab**

* Load `eval_results/<tag>/open_qa_predictions.jsonl` and step through the items.
* Each screen shows the question, the model answer, the dataset reference, ROUGE-L, and the Flesch-Kincaid grade.
* Rate or correct the item, then *Save & next*. Items you have already reviewed are skipped next time.

**Export tab**: statistics, plus **export**, which writes:

| File | Contents | Use |
|---|---|---|
| `data/feedback/feedback_sft.jsonl` | the corrected answer when one exists; otherwise the model answer, if rated ≥ 4 or "good" | next SFT round |
| `data/feedback/feedback_dpo.jsonl` | `{prompt, chosen=correction, rejected=model answer}` pairs | preference tuning (DPO/ORPO), a later stage |

The terminal mode supports the same flow through commands: `/level`, `/think`, `/temp`, `/rate`, `/reset`, `/stats`, `/export`.

### 4.1 The feedback loop

```bash
# 1. evaluate, then review answers in the UI (Review tab) and chat with the model (Chat tab)
python scripts/eval_model.py --model runs/lora_r16/adapter --tasks open_qa --level-judge
python scripts/chat_feedback.py --model runs/lora_r16/adapter
# 2. export
python scripts/chat_feedback.py --export
# 3. continue the adapter on feedback (upsampled ×3) mixed with a slice of the original data
#    (mixing in original data prevents forgetting)
python scripts/train_lora.py --method lora --init-adapter runs/lora_r16/adapter \
       --extra-data data/feedback/feedback_sft.jsonl --extra-repeat 3 \
       --max-samples 20000 --lr 5e-5 --epochs 1 --run-name lora_r16_fb1
# 4. compare
python scripts/eval_model.py --model runs/lora_r16_fb1/adapter --compare eval_results/lora_r16/summary.json
```

The full continue-with-feedback path was tested end to end: generate, rate, correct, export, then `--init-adapter` with `--extra-data`.

## 5. Recommended workflow

1. `build_training_mix.py`, then check `data/final/stats.md` (level and topic balance).
2. Run `eval_model.py --tag base` to record the baseline (includes full MMLU).
3. Train `lora_r16` (1 epoch). Optionally run `dora` and `pissa` with the same budget and compare.
4. Evaluate each run with `--compare eval_results/base/summary.json`. Watch MMLU (general knowledge must not drop; AI-subject accuracy should rise), MCQ accuracy (AI knowledge), level fit and judged match (audience adaptation), DS-1000 pass@1 (AI coding), and validation loss.
5. Review 100–300 open-QA answers per round in the UI, focusing on the weakest levels. Export, then continue training.
6. Next stage, not part of this code yet: GRPO on `data/final/rl_math.jsonl` (`\boxed{}` + math-verify reward) and `rl_code.jsonl` (pytest pass-rate reward), followed by DPO on `feedback_dpo.jsonl`.
