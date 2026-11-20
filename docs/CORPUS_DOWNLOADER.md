# Corpus Downloader (books, lecture notes, papers, encyclopedia)

`scripts/download_corpus.py` + `corpus_kit/` build a **document corpus for one task** from a YAML spec.
The first spec is `configs/corpus/ai.yaml` (≈1 GB of AI textbooks, lecture notes, survey/tutorial papers,
highly-cited papers and Wikipedia articles). The same code works for any other domain: write a new spec.
The second, `configs/corpus/replay.yaml`, is the opposite: ≈4 M tokens of general **non-AI** text (web,
encyclopedia, books, math, code) streamed from Hugging Face, used as replay data next to the AI corpus in
continued pre-training (section 7).

Typical uses of the result: a **RAG index** for the tutor, source material for **teacher-generated
Q&A** at different learner levels, or **continued pre-training** text.

## 1. Quick start

```bash
python scripts/download_corpus.py --spec configs/corpus/ai.yaml --dry-run     # what would be fetched
python scripts/download_corpus.py --spec configs/corpus/ai.yaml               # download (~1 GB)
python scripts/download_corpus.py --spec configs/corpus/ai.yaml --only books  # one source
python scripts/download_corpus.py --spec configs/corpus/ai.yaml --max-gb 5    # raise the budget later
python scripts/download_corpus.py --spec configs/corpus/ai.yaml --report      # rebuild report.md
python scripts/download_corpus.py --spec configs/corpus/ai.yaml --export corpus/ai/corpus.jsonl
```

Runs in the background the usual way:
`nohup python scripts/download_corpus.py --spec configs/corpus/ai.yaml > data/logs/corpus_ai.log 2>&1 &`

**Resumable.** Every kept document is appended to `manifest.jsonl` immediately; failures go to
`failed.jsonl`. Re-running the same command skips both (use `--retry-failed` to retry failures), so you
can stop with Ctrl-C at any time or raise `--max-gb` and continue.

## 2. Output

```
corpus/ai/
├── raw/<source>/<id>.pdf | <id>/pNN.html | <id>.txt     original files
├── text/<source>/<id>.txt                               extracted + cleaned text (pdftotext / HTML parser)
├── manifest.jsonl                                       one record per document (below)
├── failed.jsonl                                         what could not be fetched and why
└── report.md                                            counts and sizes per source / kind / license
```

Manifest record:

```json
{"id": "arxiv:2106.11342", "kind": "paper", "title": "...", "authors": ["..."], "year": 2021,
 "license": "...", "level": "ai_researcher", "url": "https://export.arxiv.org/pdf/2106.11342",
 "source": "arxiv_surveys", "bytes": 2841120, "sha1": "...", "pages": 34,
 "raw_path": "raw/arxiv_surveys/2106.11342.pdf", "text_path": "text/arxiv_surveys/2106.11342.txt",
 "text_chars": 112345, "text_words": 17321, "alpha_ratio": 0.79, "low_quality": false,
 "relevance": 18.5, "abstract": "...", "query": "...", "fetched_at": "2026-10-15 05:12:03"}
```

`low_quality` flags extractions with < 2,000 characters or < 50 % letters (scanned PDFs, broken
encodings); `--export` leaves them out.

## 3. Source types

| `type` | What it does | Key options |
|---|---|---|
| `urls` | curated list of books / lecture notes / online books | per item: `url`, or `resolve: {page, pattern}` (regex on a landing page, for files whose name changes), or `github_release: {repo, asset}` (latest release asset); `type: pdf\|html`; `pages:` extra HTML pages to concatenate; `license`, `level`, `kind` |
| `arxiv` | arXiv API search, PDFs from `export.arxiv.org` (arXiv's endpoint for automated access) | `queries: [{query, max, level, sort}]` with arXiv query syntax (`cat:cs.LG AND ti:survey`) |
| `openalex` | citation-ranked open-access works; if a work has an arXiv copy, that copy is downloaded | `filter` (OpenAlex filter string), `sort`, `max` |
| `wikipedia` | category crawl + plain-text extracts via the MediaWiki API | `lang` (`en`, `simple`, …), `categories`, `depth`, `max_pages`, `min_chars` |
| `hf` | rows of a Hugging Face dataset, **streamed** (nothing is downloaded whole); needs `pip install datasets`, `HF_TOKEN` in `.env` raises the rate limits | `dataset`, `config`, `split`, `text_field`; `id_field` / `title_field` / `url_field` (a column, a dotted path into a JSON column such as `METADATA.title`, or a list of columns joined with `/`); `where: {col: {eq|in|min|max}}`; `min_chars`, `max_chars` (skip longer rows), `truncate_chars` (cut long rows at a paragraph boundary); `strip_gutenberg`; `clean: false` (keep whitespace as is; the default for `kind: code`); `shuffle: {seed, buffer, shards}`; `skip`, `max_rows` |

Every source has `budget_gb` (its share), `max_file_mb` (skip larger files) and optionally `threshold`
(relevance threshold override). The global `max_gb` (or `--max-gb`) caps the total.

**Token budgets.** With `tokenizer: <HF name or local path>` in the spec (e.g. `models/Qwen3-1.7B-Base`),
every document gets `text_tokens` in the manifest, a source may set `budget_tokens` instead of (or next to)
`budget_gb`, and `max_tokens` (or `--max-tokens`) caps the whole corpus. The report then shows M tokens
per source.

## 4. Filtering, de-duplication, politeness

* **Relevance** (non-curated items only): `relevance.scorer: builtin:ai` reuses the AI lexicon of the
  data pipeline (`ai_qa_data/keywords.py`) on title + abstract (papers) or the page start (Wikipedia);
  `builtin:ai_core` is the same lexicon without the math vocabulary. `mode: include` (default) keeps
  documents scoring **at or above** the threshold; `mode: exclude` keeps those **below** it, which turns the
  same lexicon into a filter that removes AI material from a general corpus (the replay spec).
  Other tasks use `scorer: terms` with their own weighted regexes:
  ```yaml
  relevance:
    scorer: terms
    threshold: 3
    terms:
      - {pattern: '\bprotein\b', weight: 2}
      - {pattern: 'molecular dynamics', weight: 3}
  ```
* **Duplicates**: the same arXiv paper found by both arXiv and OpenAlex gets the same id (`arxiv:<id>`);
  identical files under different URLs are caught by SHA-1; near-identical titles are skipped.
* **Politeness**: per-host spacing (arXiv 3.1 s, Wikipedia 0.2 s, OpenAlex 0.15 s, others 1 s),
  exponential back-off on 429/5xx and on arXiv's "Rate exceeded" replies, a size cap per file, and
  PDF magic-byte checks (an HTML error page is never stored as a PDF). Wikimedia and OpenAlex ask
  clients to identify themselves: set `user_agent` in the spec (or `--user-agent`) to something like
  `"MyProject/1.0 (https://my-site or my e-mail)"`.

## 5. Licenses

Every document carries the license recorded at download time:

* arXiv papers: authors choose per paper (often arXiv's non-exclusive distribution license; some CC BY).
* OpenAlex: the license OpenAlex reports for the open-access copy (often missing → `unknown`).
* Wikipedia: CC BY-SA 4.0.
* Books: as published by the authors (see `configs/corpus/ai.yaml`). Many are **free to read but not
  openly licensed** (`free online …`) or **non-commercial / no-derivatives** (CC BY-NC-ND).

Fine for private research (RAG experiments, analysis). Before **redistributing files** or **training a
model you will release commercially**, filter by `license` in the manifest.
Some sites refuse automated downloads (e.g. the ESL/ISLR book site returns 403/404 to scripts); such
books are deliberately not in the list.

## 6. A new task in 5 minutes

1. `cp configs/corpus/ai.yaml configs/corpus/<task>.yaml`, set `task: <task>`.
2. Replace the book list (open textbooks of the field), the arXiv queries (`cat:` codes of the field +
   `ti:survey OR ti:tutorial`), the OpenAlex filter (OpenAlex topic / subfield ids) and the Wikipedia
   categories.
3. Switch relevance to `scorer: terms` with the field's vocabulary.
4. `--dry-run` to check what passes the filters, then run it.

## 7. Replay text for continued pre-training

`configs/corpus/replay.yaml` builds `corpus/replay/`: ≈4 M Qwen3 tokens of general English text in
roughly pre-training proportions, with anything the AI lexicon recognises as AI / ML removed
(`relevance: {scorer: builtin:ai_core, mode: exclude, threshold: 2.5}`):

| source | dataset (streamed) | share | notes |
|---|---|---:|---|
| `fineweb_edu` | `HuggingFaceFW/fineweb-edu` (sample-10BT) | 45 % | educational web pages, 2–25 k chars |
| `wiki_en` | `wikimedia/wikipedia` (20231101.en) | 20 % | random articles, cut to ≈12 k chars |
| `gutenberg` | `sedthh/gutenberg_english` | 15 % | public-domain books, first ≈60 k chars of each |
| `open_web_math` | `open-web-math/open-web-math` | 10 % | math web pages (math vocabulary is not an exclusion reason) |
| `github_code` | `codeparrot/github-code-clean` | 10 % | permissive licences, 13 common languages, 2–15 k chars |

```bash
python scripts/download_corpus.py --spec configs/corpus/replay.yaml                # ~4 M tokens, resumable
python scripts/download_corpus.py --spec configs/corpus/replay.yaml --max-tokens 8e6   # more replay later
python scripts/download_corpus.py --spec configs/corpus/replay.yaml --export corpus/replay/corpus.jsonl
```

Raising the budget and re-running appends new documents (the streams are re-shuffled with the same seed,
so already-kept rows are skipped by id). If a stream stalls (DNS errors / read timeouts in the log while
`hf_hub_download` works), fetch one shard with `huggingface_hub.hf_hub_download(repo, "data/<shard>.parquet",
repo_type="dataset")` and run the source from a spec copy whose `data_files` is that local file (set `label`
so the manifest still names the dataset); the `github_code` slice of `corpus/replay` was built that way. Mixed 1:4 with `corpus/ai/text` this gives a CPT set with ≈20 %
replay; the manifest's `text_tokens` lets a mix builder weight sources by tokens rather than bytes.
