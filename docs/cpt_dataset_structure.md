# CPT dataset: structure and sizes

Snapshot of 2026-11-12, measured on disk. Token counts are exact Qwen3 counts unless marked "est." (characters / 4,
from the corpus builder). "Disk" is space used on disk; "apparent" is the sum of file sizes. The two differ for
folders with millions of small files.

## 1. How the data flows

```
source pools (parquet)                          labels                     corpus folders (1 .txt per doc)          packed cache                 training
------------------------------------------      --------------------       ------------------------------------      -------------------------    -----------------
cpt_data/corpus_ai_fineweb_large  (9.0 M docs) ┐                           cpt_data/corpus_web/web_ai    (AI)    ┐
cpt_data/corpus_ai_nemotron_large (15.0 M docs)┴─> cpt_data/scores ──────> cpt_data/corpus_web/web_near  (near-AI)│
cpt_data/finewebedu_replay_large.full.parquet ───────────────────────────> cpt_data/corpus_web/web_replay        ├─> runs/_cpt_cache/          ──> cpt_qwen06b_rs_adaptive_v4_web5b
cpt_data/extra_clean/*.parquet ──────────────────────────────────────────> cpt_data/corpus_full/stem, long       ┘    varlen_a164452919c3.npz
                                                (zero-shot DeBERTa:          (sampled to a token budget,               (tokenised, 1 % val split,
                                                 AI / near-AI / not-AI)       manifest.jsonl + text/)                   16-24 k document packing)
```

The current run (web5b) reads only the packed cache. It needs the corpus folders again only to rebuild the cache.

## 2. Disk use of `cpt_data/` and the cache

| path | disk | apparent | files | role |
|---|---:|---:|---:|---|
| `corpus_web/web_ai` | 17 GB | 12 GB | 2,605,453 | web5b corpus (AI label) |
| `corpus_web/web_near` | 8.1 GB | 5.6 GB | 1,228,590 | web5b corpus (near-AI label) |
| `corpus_web/web_replay` | 12 GB | 8.1 GB | 1,676,175 | web5b corpus (FineWeb-Edu replay) |
| `corpus_full/stem` | 1.1 GB | 744 MB | 152,291 | web5b corpus (STEM) |
| `corpus_full/long` | 395 MB | 384 MB | 4,848 | web5b corpus (long documents) |
| `corpus_full/ai` | 3.9 GB | 2.7 GB | 584,036 | earlier runs' AI corpus (not in web5b) |
| `corpus_full/replay` | 714 MB | 495 MB | 101,097 | earlier runs' small replay (not in web5b) |
| `corpus_ai_extra/xtra_ai` | 1.1 GB | 620 MB | 193,887 | extra AI text (not in web5b) |
| `corpus_ai_fineweb_large` | 19 GB | 19 GB | 485 | source pool, parquet |
| `corpus_ai_nemotron_large` | 21 GB | 21 GB | 467 | source pool, parquet |
| `finewebedu_replay_large.full.parquet` | 6.9 GB | 6.9 GB | 1 | source of web_replay (recovered from the footer-less upstream file) |
| `scores` | 1.1 GB | 1.1 GB | 959 | AI-relevance labels and token counts of the two pools |
| `extra_clean` | 349 MB | 349 MB | 14 | parquet sources of corpus_full/stem and long |
| `extra_ai_clean` | 196 MB | 196 MB | 3 | parquet sources of xtra_ai |
| `difficulty` | 41 MB | 41 MB | 56 | reading-difficulty classifier study (not training data) |
| `corpus_mix100m` | 2.3 MB | 2.3 MB | 4 | manifests of an earlier 100 M-token long / STEM selection (points at corpus_full) |
| **`cpt_data/` total** | **91 GB** | **77 GB** | | |
| `runs/_cpt_cache` | 22 GB | 22 GB | 5 | packed token caches (§5) |

The corpus folders keep one file per document: 6.5 M files in total. The file system's minimum allocation per file
adds about 14 GB on top of the text itself.

## 3. The web5b training mix (current run)

From the packed cache's statistics: training documents after the 1 % validation split and the noisy-long exclusion
list (`configs/corpus/corpus_full_exclude_noisy_long.txt`, 169 entries).

| corpus | built from | train docs | train tokens | share | docs ≥ 16 k tokens |
|---|---|---:|---:|---:|---:|
| web_ai | AI-labelled pool docs, 40.3 % random sample, Translated-Diverse-QA excluded | 2,579,383 | 1,985.2 M | 40.0 % | 1,974 |
| web_near | near-AI-labelled pool docs, 28.5 % sample, same exclusion | 1,216,289 | 990.7 M | 19.9 % | 1,188 |
| web_replay | FineWeb-Edu replay: 1,692,044 of the parquet's 3,678,443 rows | 1,659,410 | 1,690.7 M | 34.0 % | 3,851 |
| stem | corpus_full/stem (10 sources) | 150,672 | 188.5 M | 3.8 % | 319 |
| long | corpus_full/long (3 sources) | 4,762 | 113.7 M | 2.3 % | 3,198 |
| **total** | | **5,610,516** | **4,968.9 M** | | |

By source:

| corpus | source | docs | tokens |
|---|---|---:|---:|
| web_ai | nemotron_cc | 1,878,174 | 1,282.8 M |
| web_ai | fineweb_edu | 701,209 | 702.5 M |
| web_near | nemotron_cc | 778,491 | 565.9 M |
| web_near | fineweb_edu | 437,798 | 424.8 M |
| web_replay | fineweb_edu | 1,659,410 | 1,690.7 M |
| long | arxiv | 2,525 | 57.4 M |
| long | doab (open-access books) | 1,779 | 44.6 M |
| long | ai_books_papers | 458 | 11.7 M |
| stem | code | 22,010 | 32.5 M |
| stem | openthoughts | 5,513 | 27.3 M |
| stem | openscience_reasoning | 8,522 | 20.6 M |
| stem | nemotron_math | 5,684 | 20.4 M |
| stem | science_qa | 16,587 | 20.1 M |
| stem | cosmo_textbooks | 25,121 | 19.4 M |
| stem | numina_cot | 26,611 | 13.8 M |
| stem | stackexchange | 24,789 | 13.8 M |
| stem | deepmath | 2,157 | 13.7 M |
| stem | openr1_math | 13,678 | 6.9 M |

Packing:
- **Sequences:** 210,880 sequences of 16,384–24,576 tokens. The median is 24,089 tokens. The shortest is 5,873, the last leftover.
- **Documents per sequence:** 26.7 pieces on average. A piece is a document, or part of one, separated by the
  document attention mask.
- **Training budget:** at bs 1 × grad accum 8 that is 26,360 optimizer steps for one epoch.

Validation:
- 32 packed sequences per corpus, ~747 k tokens each, 3.74 M tokens in total.
- Documents come from a 1 % split. Corpus names sort after "stem", so long and stem keep the split of the earlier
  rs_adaptive_long run.

## 4. Source pools

### 4.1 Labelled web pools (`corpus_ai_fineweb_large`, `corpus_ai_nemotron_large`)

Parquet columns:
- **FineWeb-Edu:** `text, id, dump, url, file_path, language, language_score, token_count, score, int_score,
  kw_core, kw_math, kw_code`.
- **Nemotron-CC:** `id, text, metadata, kw_core, kw_math, kw_code`.

Labels are in `scores/<pool>/<same path>.parquet`. They come from a zero-shot DeBERTa classifier, with AI = p_ai ≥ 0.5.
Exact token counts are in `scores/<pool>.token_counts.json`.

| pool | files | label | docs | tokens |
|---|---:|---|---:|---:|
| FineWeb-Edu (85 CC dumps, 2013-20 to 2022-21) | 484 | AI | 1,890,939 | 1.889 B |
| | | near-AI | 1,625,147 | 1.573 B |
| | | not-AI | 5,483,146 | 5.952 B |
| Nemotron-CC | 466 | AI | 5,303,917 | 3.724 B |
| | | near-AI | 2,881,327 | 2.083 B |
| | | not-AI | 6,781,691 | 5.480 B |
| **total** | 950 | | 23,966,167 | **20.70 B** |

Nemotron-CC by subset, in Qwen3 tokens:

| subset | disk | AI | near-AI | not-AI |
|---|---:|---:|---:|---:|
| High-Quality-Synthetic | 5.7 GB | 0.991 B (2.38 M docs) | 0.573 B | 1.293 B |
| Diverse-QA | 2.9 GB | 0.658 B | 0.307 B | 0.692 B |
| High-Quality | 2.5 GB | 0.482 B | 0.250 B | 0.603 B |
| Medium-Quality | 6.1 GB | 0.442 B | 0.681 B | 2.159 B |
| Medium-High-Quality | 1.6 GB | 0.188 B | 0.188 B | 0.506 B |
| Translated-Diverse-QA (**excluded**: AI labels inflated) | 1.5 GB | 0.964 B | 0.085 B | 0.228 B |

Usable pools, with Translated-Diverse-QA excluded:
- **AI:** 4.65 B tokens. web_ai uses ~2.0 B of it.
- **near-AI:** 3.57 B tokens. web_near uses ~1.0 B of it.

### 4.2 Replay (`finewebedu_replay_large.full.parquet`)

- **Contents:** 3,678,443 rows with columns `row, id, text`, ~3.77 B tokens, mean 1,053 tokens per document.
- **Origin:** every row recovered from the upstream file, which lacks a parquet footer. The recovery script is
  `cpt_data/recover_replay_large.py`, which uses the same page walker as `cpt_trainer/recover_truncated_parquet.py`.
- **Use:** web_replay takes 1.69 M of its rows.

### 4.3 Curated parquets (`extra_clean`, `extra_ai_clean`)

- **`extra_clean`:** decontaminated parquets for arxiv, doab, local long documents, code and the STEM sets. They
  feed corpus_full/long and corpus_full/stem.
- **`extra_ai_clean`:** genesis_ml prose and SFT AI answers as text. They feed xtra_ai.

## 5. Packed token cache (`runs/_cpt_cache`)

| file | size | content |
|---|---:|---|
| `varlen_a164452919c3.npz` (web5b) | 19.0 GB | `train_tokens` int32 [4,968,859,845] + `train_seq_off`, `train_piece_len`, `train_piece_off`; the same four arrays per validation corpus (`val_<corpus>_*`); `stats` (the numbers of §3) |
| `blocks_1d520d942f93.npz` | 2.3 GB | fixed-block cache from an October run, not used by web5b |
| `blocks_dfe5d193d22d.npz` | 0.3 GB | fixed-block cache from an October run, not used by web5b |

The npz is stored uncompressed, so its arrays can be memory-mapped in place.

## 6. Other corpus folders (on disk, not in web5b)

| folder | docs | tokens | content |
|---|---:|---:|---|
| `corpus_full/ai` | 584,033 | ~565 M est. | AI documents of the earlier small FineWeb-Edu (242 k) and Nemotron (342 k) samples. Mostly contained in the large pools' AI label. |
| `corpus_full/replay` | 101,094 | ~118 M est. | earlier small FineWeb-Edu replay |
| `corpus_ai_extra/xtra_ai` | 193,884 | ~143 M est. (~129 M after cleaning) | genesis_ml_prose 71,875 docs (~91 M), sft_ai_text 122,009 docs (~52 M) |

## 7. Corpus folder format

```
<corpus>/
  manifest.jsonl     one line per document: id ("<source>:<doc id>"), source, parquet (origin file), text_path,
                     bytes, text_chars, text_words, qwen_tokens (web_* corpora), gpt2_tokens, license, meta, ...
  text/<source>/<subset>/<part>/<doc id>.txt      one UTF-8 file per document
  build_stats.json   builder arguments, sampling probability, docs / tokens per source and file
  report.md          human-readable build summary
```

The manifest `id` is the document identity used across the project: the val split, the exclusion list,
deduplication, and the RETRO database's same-document filter.

## 8. Redundancy (relevant to disk planning)

- **Text exists three times.**
  - In the source parquets: compressed, 40 GB for 20.7 B tokens of the pools.
  - In the corpus folders: plain text plus small-file overhead, 37 GB for the 4.7 B tokens of web_ai / web_near /
    web_replay.
  - In the packed cache: int32 tokens, 19 GB for the 4.97 B training tokens.
- **The corpus folders can be rebuilt** from the parquets and their `build_stats.json` (`cpt_trainer/build_labelled_corpus.py`, `parquet_to_corpus.py`).
- **The cache can be rebuilt** from the corpus folders by the trainer's packing step, which is deterministic for a
  given seed.
- **Not in `cpt_data/` (evaluation data):** `data/final` and `data/eval` hold the evaluation data used by the CPT
  evals: MMLU incl. ai_large, the MCQ set, open QA and the SFT validation set.
