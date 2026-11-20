# cpt_trainer - continued pre-training with a high-rank LoRA

Continued pre-training (CPT) of `models/Qwen3-0.6B-Base` on the document corpora built by
`scripts/download_corpus.py`: `corpus/ai` (AI textbooks, surveys, papers, Wikipedia; ~16 M tokens) plus
`corpus/replay` (general non-AI web, encyclopedia, books, math, code; ~4 M tokens, so ~20 % replay against
forgetting). A LoRA adapter of rank 64 on all linear layers is trained with the plain next-token loss on
packed 2048-token blocks; after every epoch the adapter is saved and `scripts/eval_model.py` runs the
evaluation tasks (`--eval-tasks`, default `mmlu,mcq,open_qa,code,val_loss`: 5-shot MMLU, the AI MCQ set,
open QA per learner level, DS-1000 code execution, assistant-token loss on `data/final/sft_val.jsonl`).

```bash
python -m cpt_trainer.train --run-name cpt_qwen06b_r64            # 4 epochs, evaluation after each (+ epoch 0 baseline)
python -m cpt_trainer.train --run-name cpt_qwen06b_r64 --resume   # continue an interrupted run
python -m cpt_trainer.train --run-name smoke --max-blocks 64 --epochs 1 --grad-accum 4 --mmlu-max-per-subject 2 --eval-limit 8
python -m cpt_trainer.train --help                                # every option

# corpora from the cpt_data parquet files (120 MB worth of each AI file + the whole replay file), 2 epochs
python -m cpt_trainer.parquet_to_corpus cpt_data/corpus/ai --add fineweb_edu=cpt_data/finewebedu_small_ai.parquet@120 --add nemotron_cc=cpt_data/nemotron_small_ai.parquet@120
python -m cpt_trainer.parquet_to_corpus cpt_data/corpus/replay --add fineweb_edu=cpt_data/finewebedu_replay_small.recovered.parquet
python -m cpt_trainer.train --corpora cpt_data/corpus/ai cpt_data/corpus/replay --epochs 2 --run-name cpt_qwen06b_r64_300mb

# every row of the three parquet files (~620 M tokens per epoch), rank 128, 2 epochs
python -m cpt_trainer.recover_truncated_parquet recover cpt_data/finewebedu_replay_small.parquet cpt_data/finewebedu_replay_small.recovered.parquet
python -m cpt_trainer.parquet_to_corpus cpt_data/corpus_full/ai --add fineweb_edu=cpt_data/finewebedu_small_ai.parquet --add nemotron_cc=cpt_data/nemotron_small_ai.parquet
python -m cpt_trainer.parquet_to_corpus cpt_data/corpus_full/replay --add fineweb_edu=cpt_data/finewebedu_replay_small.recovered.parquet
python -m cpt_trainer.train --corpora cpt_data/corpus_full/ai cpt_data/corpus_full/replay --r 128 --alpha 256 --epochs 2 --run-name cpt_qwen06b_r128_full
```

| file | role |
|---|---|
| `data.py` | corpus folders -> documents (manifest, low_quality filter) -> tokens -> packed blocks, per-corpus validation hold-out, `.npz` cache |
| `mmlu.py` | runs `scripts/eval_model.py --tasks <eval-tasks> --mmlu-shots 5 --mmlu-format base --no-ui` in a subprocess, reads `eval_results/<tag>/summary.json`, formats the per-epoch table |
| `lora_plan.py` | per-layer / per-module rank plans (`configs/lora/*.yaml`) -> PEFT `rank_pattern`/`alpha_pattern`; `python -m cpt_trainer.lora_plan <yaml>` prints the rank table and parameter count; `train.py --lora-config <yaml>` uses it and checks the ranks in the model |
| `train.py` | the CLI: HF `Trainer` + PEFT LoRA (`ai_qa_train.common.build_lora_config`), epoch-end callback (save adapter, evaluate), `summary.md` |
| `parquet_to_corpus.py` | parquet files with a `text` column -> corpus folder (`SOURCE=PATH@MB` keeps that many on-disk MB worth of rows; e.g. `cpt_data/corpus/ai`, `cpt_data/corpus/replay`) |
| `filter_randomly.py` | random row sample of all parquet files under a folder, sized in on-disk MB |
| `recover_truncated_parquet.py` | reads the `text` column out of a parquet file whose footer was never written (copied mid-write) |

Outputs: `runs/<run>/epoch<N>/adapter`, `runs/<run>/adapter` (final), `eval_history.json`, `summary.md`,
`train.log`; evaluation details in `eval_results/<run>_epoch<N>/`. The validation loss is reported per corpus
(`eval_ai_loss`, `eval_replay_loss`), so a rising replay loss with a falling AI loss shows forgetting. A failed
epoch-end evaluation is logged and training continues; the adapter of that epoch is already saved, so the
evaluation can be re-run by hand with `scripts/eval_model.py`.

Per-layer / per-module ranks: `--lora-config configs/lora/cpt_qwen06b_rs_adaptive_v1.yaml` (rsLoRA, alpha 16,
ranks 32-256, 79 M params: low rank for attention in layers 0-2 and for k/v, most of the budget in the MLP of
layers 8-26). The YAML's method/alpha/dropout/targets/default_r replace the CLI flags; the resolved plan is
saved as `runs/<run>/lora_plan.json` and in `run_config.json`.

```bash
python -m cpt_trainer.lora_plan configs/lora/cpt_qwen06b_rs_adaptive_v1.yaml      # inspect the plan
python -m cpt_trainer.train --lora-config configs/lora/cpt_qwen06b_rs_adaptive_v1.yaml --corpora ... --epochs 1 --run-name cpt_qwen06b_rs_adaptive_v1
```

Defaults that matter: lr 1e-4 cosine with 3 % warmup, 2 x 16 x 2048 = 65 k tokens per optimizer step,
gradient checkpointing on, bf16, AdamW (fused). `--method rslora` is the usual alternative at rank 64.
