#!/usr/bin/env python3
"""Download a domain corpus (books, lecture notes, papers, encyclopedia articles) from a YAML task spec.

    python scripts/download_corpus.py --spec configs/corpus/ai.yaml                  # build the AI corpus (~1 GB)
    python scripts/download_corpus.py --spec configs/corpus/ai.yaml --dry-run        # list what would be fetched
    python scripts/download_corpus.py --spec configs/corpus/ai.yaml --only books,wiki_en
    python scripts/download_corpus.py --spec configs/corpus/ai.yaml --max-gb 5       # bigger budget
    python scripts/download_corpus.py --spec configs/corpus/ai.yaml --report         # rebuild report.md only
    python scripts/download_corpus.py --spec configs/corpus/ai.yaml --export corpus/ai/corpus.jsonl
    python scripts/download_corpus.py --spec configs/corpus/replay.yaml --max-tokens 4e6   # general replay text

Runs are resumable: documents already in manifest.jsonl are skipped (failures too, unless --retry-failed),
so you can stop with Ctrl-C and run the same command again, or raise --max-gb later.

A new task = a new spec file (copy configs/corpus/ai.yaml): change the sources (book list, arXiv queries,
OpenAlex filter, Wikipedia categories) and the relevance terms. See docs/CORPUS_DOWNLOADER.md.
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_qa_data.env import load_env  # noqa: E402
from corpus_kit.fetch import DEFAULT_UA, Fetcher  # noqa: E402
from corpus_kit.runner import CorpusBuilder  # noqa: E402

load_env()  # HF_TOKEN from .env -> authenticated Hub requests for `hf` sources (higher rate limits)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="YAML task spec, e.g. configs/corpus/ai.yaml")
    ap.add_argument("--out", default=None, help="output dir (default: corpus/<task>)")
    ap.add_argument("--max-gb", type=float, default=None, help="total raw-size budget (overrides spec max_gb)")
    ap.add_argument("--max-tokens", type=float, default=None,
                    help="total token budget (overrides spec max_tokens; the spec needs `tokenizer:`)")
    ap.add_argument("--only", default=None, help="comma list of source names from the spec")
    ap.add_argument("--dry-run", action="store_true", help="list candidates that pass the filters, download nothing")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--report", action="store_true", help="only rebuild report.md from the manifest")
    ap.add_argument("--export", default=None, help="write all texts to one JSONL file and exit")
    ap.add_argument("--user-agent", default=None,
                    help="HTTP User-Agent; Wikimedia/OpenAlex ask for a contact URL or e-mail in it")
    args = ap.parse_args()

    with open(args.spec) as f:
        spec = yaml.safe_load(f)
    out = args.out or os.path.join(ROOT, "corpus", spec.get("task", "corpus"))
    ua = args.user_agent or spec.get("user_agent") or DEFAULT_UA
    tok = spec.get("tokenizer")
    if tok and not os.path.isabs(tok) and os.path.isdir(os.path.join(ROOT, tok)):
        spec["tokenizer"] = os.path.join(ROOT, tok)        # local tokenizer path relative to the repo root
    builder = CorpusBuilder(spec, out, args.max_gb, Fetcher(user_agent=ua), retry_failed=args.retry_failed,
                            max_tokens=int(args.max_tokens) if args.max_tokens else None)

    if args.report:
        builder.write_report()
        return
    if args.export:
        n = builder.export_jsonl(args.export)
        print(f"exported {n} documents -> {args.export}")
        return
    print(f"task={spec.get('task')} out={out} budget={builder.max_bytes / 1e9:.2f} GB"
          + (f" / {builder.max_tokens / 1e6:.2f} M tokens" if builder.max_tokens else "")
          + f" (already {builder.total_bytes / 1e9:.2f} GB"
          + (f" / {builder.total_tokens / 1e6:.2f} M tokens" if builder.tok else "") + f", {len(builder.docs)} docs)")
    try:
        builder.run(only=set(args.only.split(",")) if args.only else None, dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("\ninterrupted - progress is saved in manifest.jsonl; run the same command to resume")
        builder.write_report()
    if any(src.get("type") == "hf" for src in spec.get("sources", [])):
        # Streamed (hf) sources can leave pyarrow prefetch threads behind when a stream is stopped at its
        # budget; interpreter finalisation then aborts (exit 134) although every file is written and
        # closed. Nothing else is pending, so skip finalisation.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
