# Model leaderboard: AI-related accuracy

Generated 2026-10-28 10:36:26 by `scripts/score_report.py` from the saved evaluations in `eval_results/` (MMLU `test` split). No model was run; re-run the script after new evaluations or a new `ai_related` selection.

## Scores

| score | what | n |
|---|---|---:|
| MMLU | 5-shot MMLU, every test question | 14,042 |
| AI subj. (11) | the 11 hand-picked AI-related MMLU subjects (`MMLU_AI_SUBJECTS` in scripts/eval_model.py) | 1,761 |
| LLM-sel. | the `ai_related` subject selected by the Qwen3-1.7B yes/no judge (`data/eval/mmlu_ai_related_llm/selection.json`, scripts/build_mmlu_ai_related_llm.py) | 1,436 |
| NLI-sel. | the `ai_related` subject selected by the zero-shot NLI model (`data/eval/mmlu_ai_related/selection.json`, scripts/build_mmlu_ai_related.py) | 305 |
| ML | the MMLU `machine_learning` subject | 112 |
| MCQ | the pipeline's own MCQ set (MMLU / MMLU-Pro AI items), 5-shot, from the run's summary.json | 186 |
| Open QA | ROUGE-L, token-F1 and level-fit of generated answers to open questions, with the adapter applied to the **chat** model (see the Open QA section); not part of the combined score | 984 |

**score** = plain mean of the six accuracies. A run missing any of them is not ranked; it is listed below the ranking with the mean of the scores it has. **±** = 95 % bootstrap half-width of the score from resampling the MMLU and MCQ questions (1,000 replicates): test-set sampling noise. Re-running an evaluation is deterministic (see Repeatability), so this is the only noise in a score; two runs whose intervals overlap are not reliably ordered. **AI_Large** = accuracy on the 1,000-question `ai_large` subject (see its section); shown for reference, not part of the score.

Included runs: 5-shot, base-prompt-format evaluations of a pre-trained `*-Base` checkpoint, alone or with a LoRA adapter trained on it. 0-shot, chat-format and chat-model runs are not comparable and are left out. Identical adapter-less evaluations of one checkpoint are merged into a single row. Δ vs base = score minus the score of the same checkpoint without an adapter.

## Ranking

| # | run | base model | adapter | MMLU | AI subj. (11) | LLM-sel. | NLI-sel. | ML | MCQ | **score** | ± | Δ vs base | AI_Large |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | mmlu_17base_5shot | Qwen3-1.7B-Base | - | 0.6273 | 0.5883 | 0.5696 | 0.5705 | 0.4643 | 0.3817 | **0.5336** | 0.0290 | +0.0000 | 0.6850 |
| 2 | cpt_qwen06b_r64_300mb_epoch1 | Qwen3-0.6B-Base | cpt_qwen06b_r64_300mb/epoch1 | 0.5214 | 0.4611 | 0.4422 | 0.4689 | 0.4821 | 0.3925 | **0.4614** | 0.0304 | +0.0424 | 0.5670 |
| 3 | cpt_qwen06b_r64_epoch3 | Qwen3-0.6B-Base | cpt_qwen06b_r64/epoch3 | 0.5170 | 0.4611 | 0.4380 | 0.4689 | 0.4464 | 0.3548 | **0.4477** | 0.0292 | +0.0288 | 0.5680 |
| 4 | cpt_qwen06b_r128_full_epoch1 | Qwen3-0.6B-Base | cpt_qwen06b_r128_full/epoch1 | 0.5100 | 0.4480 | 0.4325 | 0.4852 | 0.4464 | 0.3441 | **0.4444** | 0.0292 | +0.0254 | 0.5650 |
| 5 | cpt_qwen06b_r64_epoch2 | Qwen3-0.6B-Base | cpt_qwen06b_r64/epoch2 | 0.5232 | 0.4656 | 0.4415 | 0.4623 | 0.4196 | 0.3441 | **0.4427** | 0.0294 | +0.0238 | 0.5720 |
| 6 | cpt_qwen06b_r64_epoch1 | Qwen3-0.6B-Base | cpt_qwen06b_r64/epoch1 | 0.5221 | 0.4628 | 0.4429 | 0.4721 | 0.4196 | 0.3280 | **0.4413** | 0.0284 | +0.0223 | 0.5760 |
| 7 | cpt_qwen06b_rs_adaptive_long_epoch1 | Qwen3-0.6B-Base | cpt_qwen06b_rs_adaptive_long/epoch1 | 0.5199 | 0.4611 | 0.4401 | 0.4492 | 0.4464 | 0.3280 | **0.4408** | 0.0287 | +0.0219 | 0.5860 |
| 8 | cpt_qwen06b_r64_epoch4 | Qwen3-0.6B-Base | cpt_qwen06b_r64/epoch4 | 0.5171 | 0.4600 | 0.4325 | 0.4721 | 0.4107 | 0.3333 | **0.4376** | 0.0295 | +0.0187 | 0.5740 |
| 9 | cpt_qwen06b_rs_adaptive_q4_100m_epoch0.5 | Qwen3-0.6B-Base | cpt_qwen06b_rs_adaptive_q4_100m/epoch0.5 | 0.5247 | 0.4622 | 0.4422 | 0.4393 | 0.4018 | 0.3495 | **0.4366** | 0.0295 | +0.0177 | 0.5680 |
| 10 | cpt_qwen06b_rs_adaptive_long_epoch0.5 | Qwen3-0.6B-Base | cpt_qwen06b_rs_adaptive_long/epoch0.5 | 0.5179 | 0.4497 | 0.4338 | 0.4459 | 0.4464 | 0.3172 | **0.4352** | 0.0283 | +0.0162 | 0.5850 |
| 11 | cpt_qwen06b_ml_rs_adaptive_v1_epoch0.5 | Qwen3-0.6B-Base | cpt_qwen06b_ml_rs_adaptive_v1/epoch0.5 | 0.5045 | 0.4543 | 0.4338 | 0.4623 | 0.3929 | 0.3548 | **0.4338** | 0.0286 | +0.0148 | 0.5760 |
| 12 | cpt_qwen06b_rs_adaptive_q4_100m_epoch1 | Qwen3-0.6B-Base | cpt_qwen06b_rs_adaptive_q4_100m/epoch1 | 0.5234 | 0.4566 | 0.4373 | 0.4426 | 0.3839 | 0.3495 | **0.4322** | 0.0286 | +0.0133 | 0.5760 |
| 13 | cpt_qwen06b_r128_full_epoch2 | Qwen3-0.6B-Base | cpt_qwen06b_r128_full/epoch2 | 0.5072 | 0.4452 | 0.4283 | 0.4557 | 0.3839 | 0.3280 | **0.4247** | 0.0275 | +0.0058 | 0.5600 |
| 14 | cpt_qwen06b_r128_full_epoch0 | Qwen3-0.6B-Base | - | 0.5249 | 0.4600 | 0.4408 | 0.4262 | 0.3661 | 0.2957 | **0.4189** | 0.0297 | +0.0000 | 0.5710 |
| 15 | cpt_qwen06b_rs128_long_epoch0.5 | Qwen3-0.6B-Base | cpt_qwen06b_rs128_long/epoch0.5 | 0.5182 | 0.4407 | 0.4276 | 0.4328 | 0.3571 | 0.3333 | **0.4183** | 0.0267 | -0.0007 | 0.5870 |

Merged rows: `cpt_qwen06b_r128_full_epoch0` = `exp_06base_base5` = `cpt_qwen06b_r64_epoch0` = `cpt_qwen06b_r64_300mb_epoch0` (same Qwen3-0.6B-Base evaluation).

## ai_large (1,000 ML / AI questions, 5-shot)

`ai_large` is a set of 1,000 MMLU-style machine-learning / AI questions written for this project (8 topics x 125, `data/eval/mmlu_ai_large/`, scripts/build_mmlu_ai_large.py; blind-verified, answer letters balanced). Same 5-shot base-format scoring as MMLU, but not part of the combined score, so the ranking stays comparable with earlier reports. Δ = paired difference with the same checkpoint without an adapter, on the same questions, with a 95 % bootstrap interval; a single accuracy on 1,000 questions is ±0.031, a paired Δ about ±0.02. Rows in ranking order (unranked runs last); per-topic columns have 125 questions each (±0.087).

| run | base model | adapter | AI_Large | Δ vs base [95 %] | applied_responsible | architectures | classical_ml | foundations | llm_nlp | optimization_training | probabilistic_math | rl_classical_ai |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mmlu_17base_5shot | Qwen3-1.7B-Base | - | 0.6850 |  | 0.6640 | 0.6640 | 0.7120 | 0.6640 | 0.7520 | 0.6560 | 0.6880 | 0.6800 |
| cpt_qwen06b_r64_300mb_epoch1 | Qwen3-0.6B-Base | cpt_qwen06b_r64_300mb/epoch1 | 0.5670 | -0.0040 [-0.0260, +0.0190] | 0.6480 | 0.4960 | 0.5680 | 0.5760 | 0.6080 | 0.5520 | 0.5520 | 0.5360 |
| cpt_qwen06b_r64_epoch3 | Qwen3-0.6B-Base | cpt_qwen06b_r64/epoch3 | 0.5680 | -0.0030 [-0.0250, +0.0180] | 0.6560 | 0.5200 | 0.6160 | 0.5520 | 0.5760 | 0.5520 | 0.5280 | 0.5440 |
| cpt_qwen06b_r128_full_epoch1 | Qwen3-0.6B-Base | cpt_qwen06b_r128_full/epoch1 | 0.5650 | -0.0060 [-0.0330, +0.0210] | 0.6320 | 0.5040 | 0.5760 | 0.5360 | 0.6080 | 0.5680 | 0.5600 | 0.5360 |
| cpt_qwen06b_r64_epoch2 | Qwen3-0.6B-Base | cpt_qwen06b_r64/epoch2 | 0.5720 | +0.0010 [-0.0210, +0.0220] | 0.6560 | 0.4960 | 0.6080 | 0.5520 | 0.5600 | 0.5680 | 0.5680 | 0.5680 |
| cpt_qwen06b_r64_epoch1 | Qwen3-0.6B-Base | cpt_qwen06b_r64/epoch1 | 0.5760 | +0.0050 [-0.0130, +0.0240] | 0.6400 | 0.5120 | 0.6160 | 0.5840 | 0.5760 | 0.5520 | 0.5840 | 0.5440 |
| cpt_qwen06b_rs_adaptive_long_epoch1 | Qwen3-0.6B-Base | cpt_qwen06b_rs_adaptive_long/epoch1 | 0.5860 | +0.0150 [-0.0060, +0.0360] | 0.6400 | 0.5440 | 0.6160 | 0.5520 | 0.5840 | 0.5840 | 0.5520 | 0.6160 |
| cpt_qwen06b_r64_epoch4 | Qwen3-0.6B-Base | cpt_qwen06b_r64/epoch4 | 0.5740 | +0.0030 [-0.0190, +0.0260] | 0.6560 | 0.5040 | 0.6160 | 0.5760 | 0.5680 | 0.5760 | 0.5360 | 0.5600 |
| cpt_qwen06b_rs_adaptive_q4_100m_epoch0.5 | Qwen3-0.6B-Base | cpt_qwen06b_rs_adaptive_q4_100m/epoch0.5 | 0.5680 | -0.0030 [-0.0230, +0.0170] | 0.6640 | 0.4800 | 0.5840 | 0.5680 | 0.5760 | 0.5280 | 0.5440 | 0.6000 |
| cpt_qwen06b_rs_adaptive_long_epoch0.5 | Qwen3-0.6B-Base | cpt_qwen06b_rs_adaptive_long/epoch0.5 | 0.5850 | +0.0140 [-0.0070, +0.0330] | 0.6400 | 0.5360 | 0.6080 | 0.5600 | 0.5840 | 0.5600 | 0.5680 | 0.6240 |
| cpt_qwen06b_ml_rs_adaptive_v1_epoch0.5 | Qwen3-0.6B-Base | cpt_qwen06b_ml_rs_adaptive_v1/epoch0.5 | 0.5760 | +0.0050 [-0.0210, +0.0320] | 0.6720 | 0.5680 | 0.6320 | 0.5600 | 0.5680 | 0.5360 | 0.5120 | 0.5600 |
| cpt_qwen06b_rs_adaptive_q4_100m_epoch1 | Qwen3-0.6B-Base | cpt_qwen06b_rs_adaptive_q4_100m/epoch1 | 0.5760 | +0.0050 [-0.0150, +0.0250] | 0.6720 | 0.4960 | 0.6000 | 0.6000 | 0.5920 | 0.5360 | 0.5120 | 0.6000 |
| cpt_qwen06b_r128_full_epoch2 | Qwen3-0.6B-Base | cpt_qwen06b_r128_full/epoch2 | 0.5600 | -0.0110 [-0.0380, +0.0150] | 0.6480 | 0.4960 | 0.6000 | 0.5280 | 0.5840 | 0.5760 | 0.5440 | 0.5040 |
| cpt_qwen06b_r128_full_epoch0 | Qwen3-0.6B-Base | - | 0.5710 |  | 0.6320 | 0.5200 | 0.6240 | 0.5680 | 0.6000 | 0.5360 | 0.5360 | 0.5520 |
| cpt_qwen06b_rs128_long_epoch0.5 | Qwen3-0.6B-Base | cpt_qwen06b_rs128_long/epoch0.5 | 0.5870 | +0.0160 [-0.0070, +0.0390] | 0.6720 | 0.5280 | 0.6160 | 0.5440 | 0.5680 | 0.6240 | 0.5280 | 0.6160 |

## Open QA (adapter applied to the chat model)

Greedy answers (768 new tokens, thinking off) to open questions, scored against references: ROUGE-L, token-F1, and level-fit (share of answers whose reading grade fits the question's audience level). The CPT adapters are trained on the `*-Base` checkpoint but are meant to be used on the chat model, so for this task each adapter is applied to `Qwen3-0.6B` (`Qwen3-1.7b` for the 1.7B row) and the no-adapter rows are the chat model alone. Δ = token-F1 minus the chat model alone. Rows in ranking order; this task is not part of the combined score.

| # | run | chat model | ROUGE-L | token-F1 | level-fit | Δ token-F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | mmlu_17base_5shot | Qwen3-1.7b | 0.1572 | 0.3165 | 0.6022 | +0.0000 |
| 2 | cpt_qwen06b_r64_300mb_epoch1 | Qwen3-0.6B | 0.1778 | 0.3335 | 0.6389 | +0.0132 |
| 3 | cpt_qwen06b_r64_epoch3 | Qwen3-0.6B | 0.1663 | 0.3140 | 0.6509 | -0.0063 |
| 4 | cpt_qwen06b_r128_full_epoch1 | Qwen3-0.6B | 0.1819 | 0.3384 | 0.6291 | +0.0181 |
| 5 | cpt_qwen06b_r64_epoch2 | Qwen3-0.6B | 0.1684 | 0.3155 | 0.6553 | -0.0048 |
| 6 | cpt_qwen06b_r64_epoch1 | Qwen3-0.6B | 0.1699 | 0.3191 | 0.6345 | -0.0012 |
| 7 | cpt_qwen06b_rs_adaptive_long_epoch1 | Qwen3-0.6B | 0.1688 | 0.3170 | 0.6311 | -0.0033 |
| 8 | cpt_qwen06b_r64_epoch4 | Qwen3-0.6B | 0.1664 | 0.3130 | 0.6249 | -0.0073 |
| 9 | cpt_qwen06b_rs_adaptive_q4_100m_epoch0.5 | Qwen3-0.6B | 0.1719 | 0.3228 | 0.6489 | +0.0025 |
| 10 | cpt_qwen06b_rs_adaptive_long_epoch0.5 | Qwen3-0.6B | 0.1696 | 0.3201 | 0.6337 | -0.0002 |
| 11 | cpt_qwen06b_ml_rs_adaptive_v1_epoch0.5 | Qwen3-0.6B | 0.1599 | 0.3105 | 0.6055 | -0.0098 |
| 12 | cpt_qwen06b_rs_adaptive_q4_100m_epoch1 | Qwen3-0.6B | 0.1732 | 0.3252 | 0.6267 | +0.0049 |
| 13 | cpt_qwen06b_r128_full_epoch2 | Qwen3-0.6B | 0.1812 | 0.3361 | 0.6068 | +0.0158 |
| 14 | cpt_qwen06b_r128_full_epoch0 | Qwen3-0.6B | 0.1700 | 0.3203 | 0.6389 | +0.0000 |
| 15 | cpt_qwen06b_rs128_long_epoch0.5 | Qwen3-0.6B | 0.1688 | 0.3193 | 0.6227 | -0.0010 |

## CPT adapters on the chat model (SFT conflict check)

The CPT adapters are trained on the `*-Base` checkpoint but are used on the chat (SFT) model. Here each adapter is applied to the chat model and compared with the chat model alone on 0-shot chat-format MMLU, the way the tutor is asked questions. A negative Δ means the adapter hurts what SFT built.

- **no thinking, generated answer** (the reference): chat template with thinking off, greedy reply that ends with "Answer: X", letter parsed from the text (`scripts/eval_model.py --mmlu-scoring generate`, simple-evals style). *Parse rate* = replies with a readable letter.
- **no thinking, first-token letter** (not reliable, kept for history): answer letter read from the probability of the reply's first token. A chat model rarely starts its reply with the bare letter (*letter mass* well below 1), so this mostly measures whether the adapter makes it do so, not knowledge: it showed the r64_300mb adapter as +4.4 MMLU points where the generated answer gives −0.8.
- **thinking**: Qwen3 thinking mode, the model reasons and then answers; the letter is parsed from the answer, or read from the logits after closing the reasoning when it ran out of tokens or gave no letter (*forced*). Sampled (temperature 0.6, top-p 0.95, seeded), so it carries sampling noise as well (`scripts/eval_mmlu_thinking.py`). Questions run in a fixed random order, so a partial run is a random sample and all rows are compared on the questions they have in common.
- Δ = adapter minus the chat model alone on the same questions, with its paired 95 % bootstrap interval.

### Qwen3-0.6B, no thinking, generated answer

| model | n | parse rate | MMLU | AI subj. | LLM-sel. | NLI-sel. | ML | Δ MMLU [95 %] | Δ AI subj. [95 %] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B alone | 14,042 | 0.9932 | 0.4732 | 0.5241 | 0.5299 | 0.4295 | 0.3214 |  |  |
| + cpt_qwen06b_r64_300mb/epoch1 | 14,042 | 0.9979 | 0.4651 | 0.4543 | 0.4513 | 0.4164 | 0.3304 | -0.0080 [-0.0137, -0.0023] | -0.0698 [-0.0909, -0.0483] |
| + cpt_qwen06b_rs_adaptive_long/epoch1 | 14,042 | 0.9954 | 0.4445 | 0.4407 | 0.4471 | 0.3738 | 0.3125 | -0.0287 [-0.0355, -0.0216] | -0.0835 [-0.1056, -0.0630] |

### Qwen3-0.6B, no thinking, first-token letter

| model | n | letter mass | MMLU | AI subj. | LLM-sel. | NLI-sel. | ML | Δ MMLU [95 %] | Δ AI subj. [95 %] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B alone | 14,042 | 0.6755 | 0.3922 | 0.2993 | 0.2890 | 0.3279 | 0.3304 |  |  |
| + cpt_qwen06b_r64_300mb/epoch1 | 14,042 | 0.9246 | 0.4365 | 0.3708 | 0.3677 | 0.3639 | 0.3214 | +0.0443 [+0.0367, +0.0516] | +0.0716 [+0.0494, +0.0937] |
| + cpt_qwen06b_rs_adaptive_long/epoch1 | 14,042 | 0.5903 | 0.3994 | 0.3101 | 0.2960 | 0.3541 | 0.3661 | +0.0072 [+0.0017, +0.0127] | +0.0108 [-0.0017, +0.0227] |

## Repeatability

The same model evaluated again with identical settings under a `<tag>_rerun<k>` tag; repeats are not ranked. Spread = max − min over the original and its repeats. *Changed answers* = questions whose predicted letter differs from the original evaluation's.

### cpt_qwen06b_r64_300mb_epoch1 (Qwen3-0.6B-Base, adapter cpt_qwen06b_r64_300mb/epoch1; 2 repeats)

| score | original | rerun1 | rerun2 | spread |
|---|---:|---:|---:|---:|
| MMLU | 0.5214 | 0.5214 | 0.5214 | 0.0000 |
| AI subj. (11) | 0.4611 | 0.4611 | 0.4611 | 0.0000 |
| LLM-sel. | 0.4422 | 0.4422 | 0.4422 | 0.0000 |
| NLI-sel. | 0.4689 | 0.4689 | 0.4689 | 0.0000 |
| ML | 0.4821 | 0.4821 | 0.4821 | 0.0000 |
| MCQ | 0.3925 | 0.3925 | 0.3925 | 0.0000 |
| **score** | 0.4614 | 0.4614 | 0.4614 | 0.0000 |

Changed answers vs the original: rerun1: MMLU 0 of 14042 (0.00 %), MCQ 0 of 186 (0.00 %); rerun2: MMLU 0 of 14042 (0.00 %), MCQ 0 of 186 (0.00 %).

## Reading the numbers

- 95 % half-widths of a single accuracy near 0.45 at these sizes: MMLU ±0.008, AI subj. (11) ±0.023, LLM-sel. ±0.026, NLI-sel. ±0.056, ML ±0.092, MCQ ±0.071. The ± column gives the same for the combined score (bootstrap over questions); the small sets (ML, MCQ, NLI-sel.) dominate it, so combined scores within about 0.03 of each other are not reliably ordered.
- LLM-sel. holds 1,436 of the 1,602 pooled math / computer-science / formal-logic / machine_learning questions: the Qwen3-1.7B judge answers *yes* to almost everything at threshold 0.5, so this score behaves like "all of those subjects" rather than a narrow AI subset; NLI-sel. is the narrow one.
- Several rows are one adapter at different epochs; their ranks mostly tell which epoch to keep.

## Excluded evaluations

- `base`: chat prompt format
- `base06_ai_large_5shot`: ai_large-only evaluation (merged into the run's own results)
- `base17_ai_large_5shot`: ai_large-only evaluation (merged into the run's own results)
- `chat06b_0shot`: 0-shot
- `chat06b_ai_large_gen`: 0-shot
- `chat06b_mmlu_gen`: 0-shot
- `chat06b_mmlu_gen_smoke`: 0-shot
- `chat_baseline`: chat prompt format
- `cpt_qwen06b_ml_rs_adaptive_v1_epoch0.5_ai_large_5shot`: ai_large-only evaluation (merged into the run's own results)
- `cpt_qwen06b_r128_full_epoch1_ai_large_5shot`: ai_large-only evaluation (merged into the run's own results)
- `cpt_qwen06b_r128_full_epoch2_ai_large_5shot`: ai_large-only evaluation (merged into the run's own results)
- `cpt_qwen06b_r64_300mb_epoch1_ai_large_5shot`: ai_large-only evaluation (merged into the run's own results)
- `cpt_qwen06b_r64_300mb_epoch1_onchat_0shot`: 0-shot
- `cpt_qwen06b_r64_300mb_epoch1_onchat_gen`: 0-shot
- `cpt_qwen06b_r64_epoch1_ai_large_5shot`: ai_large-only evaluation (merged into the run's own results)
- `cpt_qwen06b_r64_epoch2_ai_large_5shot`: ai_large-only evaluation (merged into the run's own results)
- `cpt_qwen06b_r64_epoch3_ai_large_5shot`: ai_large-only evaluation (merged into the run's own results)
- `cpt_qwen06b_r64_epoch4_ai_large_5shot`: ai_large-only evaluation (merged into the run's own results)
- `cpt_qwen06b_rs128_long_epoch0.5_ai_large_5shot`: ai_large-only evaluation (merged into the run's own results)
- `cpt_qwen06b_rs_adaptive_long_epoch0.5_ai_large_5shot`: ai_large-only evaluation (merged into the run's own results)
- `cpt_qwen06b_rs_adaptive_long_epoch0_chat`: no MMLU result in summary.json
- `cpt_qwen06b_rs_adaptive_long_epoch1_ai_large_5shot`: ai_large-only evaluation (merged into the run's own results)
- `cpt_qwen06b_rs_adaptive_long_epoch1_onchat_0shot`: 0-shot
- `cpt_qwen06b_rs_adaptive_long_epoch1_onchat_gen`: 0-shot
- `cpt_qwen06b_rs_adaptive_q4_100m_epoch0.5_ai_large_5shot`: ai_large-only evaluation (merged into the run's own results)
- `cpt_qwen06b_rs_adaptive_q4_100m_epoch1_ai_large_5shot`: ai_large-only evaluation (merged into the run's own results)
- `exp_17_base0`: 0-shot
- `exp_17_base5`: Qwen3-1.7b is not a base checkpoint
- `lora_r16_v1`: chat prompt format
- `lora_r16_v2`: chat prompt format
- `mmlu_17chat_5shot`: chat prompt format
