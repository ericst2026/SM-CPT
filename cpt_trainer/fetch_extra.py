"""Download extra CPT data (STEM, math, code, reasoning, and real long documents) from public Hugging Face
datasets into one parquet per source (columns id, text), sized by exact Qwen3 token counts.

    python -m cpt_trainer.fetch_extra                       # every source below -> cpt_data/extra/<name>.parquet
    python -m cpt_trainer.fetch_extra --only finemath,code  # some of them
    python -m cpt_trainer.fetch_extra --scale 0.01          # 1 % of every budget (smoke test)

Shards are fetched one at a time with hf_hub_download (plain HTTP downloads are reliable on this machine,
`datasets` streaming is not) in a seeded random order, rows inside a shard are visited in random order,
and a source stops as soon as its token budget is reached. Downloads are cached in the HF hub cache, so a
re-run only re-reads them. Then build corpus folders for the trainer:

    python -m cpt_trainer.parquet_to_corpus cpt_data/corpus_full/long --add arxiv=cpt_data/extra/arxiv.parquet ...
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import random
import re
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

# Xet transfers stall on this machine's (VPN) connection; plain HTTP downloads are steadier.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "cpt_data", "extra")
TOKENIZER = os.path.join(ROOT, "models", "Qwen3-0.6B-Base")
PERMISSIVE = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc", "unlicense", "cc0-1.0", "mpl-2.0"}
CODE_LANGS = {"Python", "JavaScript", "TypeScript", "Java", "C", "C++", "GO", "Rust", "Shell", "SQL", "Ruby", "PHP", "C#",
              "Julia", "Scala", "Haskell", "Lua", "R", "FORTRAN", "Markdown"}
LONG_MAX_CHARS = 128_000  # ~32 k tokens: one document never fills more than one max-length training sequence


def _reasoning_text(r: dict) -> str | None:
    p, t, s = r.get("problem"), r.get("deepseek_reasoning"), r.get("deepseek_solution")
    if not (p and t and s):
        return None
    return f"Problem:\n{p.strip()}\n\nReasoning:\n{t.strip()}\n\nSolution:\n{s.strip()}"


def _code_text(r: dict) -> str | None:
    if r.get("license") not in PERMISSIVE or r.get("language") not in CODE_LANGS:
        return None
    return f"# File: {r.get('repo_name', '?')}/{r.get('path', '?')}\n\n{r['code']}"


# name, group, repo, file regex, text (column name or function), budget (M tokens), min/max chars
SOURCES = [  # fetch order: cheapest first (the code shard is usually cached), arXiv last
    dict(name="code", group="stem", repo="codeparrot/github-code-clean", files=r"^data/train-.*\.parquet$", text=_code_text,
         columns=["code", "repo_name", "path", "language", "license"], budget=35, min_chars=500, max_chars=60_000,
         license="permissive OSS licences (per file)"),
    dict(name="libretexts", group="stem", repo="common-pile/libretexts", files=r"\.json\.gz$", text="text",
         budget=15, min_chars=500, max_chars=60_000, license="openly licensed textbooks (LibreTexts, Common Pile v0.1)"),
    dict(name="reasoning", group="stem", repo="open-thoughts/OpenThoughts-114k", files=r"^metadata/.*\.parquet$",
         text=_reasoning_text, columns=["problem", "deepseek_reasoning", "deepseek_solution", "domain"], budget=35,
         min_chars=500, max_chars=None, license="Apache 2.0 (OpenThoughts-114k)"),
    dict(name="pg19", group="long", repo="emozilla/pg19", files=r"^data/train-.*\.parquet$", text="text",
         budget=30, min_chars=30_000, max_chars=LONG_MAX_CHARS, license="public domain (Project Gutenberg, PG-19)"),
    dict(name="finemath", group="stem", repo="HuggingFaceTB/finemath", files=r"^finemath-4plus/.*\.parquet$", text="text",
         budget=40, min_chars=500, max_chars=60_000, license="ODC-By 1.0 (FineMath)"),
    dict(name="cosmo_openstax", group="stem", repo="HuggingFaceTB/cosmopedia", files=r"^data/openstax/.*\.parquet$",
         text="text", budget=10, min_chars=500, max_chars=None, license="Apache 2.0 (Cosmopedia)"),
    dict(name="cosmo_stanford", group="stem", repo="HuggingFaceTB/cosmopedia", files=r"^data/stanford/.*\.parquet$",
         text="text", budget=10, min_chars=500, max_chars=None, license="Apache 2.0 (Cosmopedia)"),
    dict(name="cosmo_khan", group="stem", repo="HuggingFaceTB/cosmopedia", files=r"^data/khanacademy/.*\.parquet$",
         text="text", budget=5, min_chars=500, max_chars=None, license="Apache 2.0 (Cosmopedia)"),
    dict(name="doab", group="long", repo="common-pile/doab", files=r"\.json\.gz$", text="text",
         budget=45, min_chars=30_000, max_chars=LONG_MAX_CHARS, license="openly licensed books (DOAB, Common Pile v0.1)"),
    dict(name="arxiv", group="long", repo="common-pile/arxiv_papers", files=r"\.jsonl\.gz$", text="text",
         budget=75, min_chars=30_000, max_chars=LONG_MAX_CHARS, license="openly licensed arXiv papers (Common Pile v0.1)"),
]


def cut(text: str, max_chars: int | None) -> str:
    """Cut at the last paragraph (or line) break before max_chars."""
    if not max_chars or len(text) <= max_chars:
        return text
    head = text[:max_chars]
    for sep in ("\n\n", "\n"):
        k = head.rfind(sep)
        if k > max_chars * 0.8:
            return head[:k]
    return head


def iter_rows(path: str, columns: list[str] | None, rng: random.Random):
    """Rows of one shard in random order (parquet row groups shuffled; json lines shuffled)."""
    if path.endswith(".parquet"):
        pf = pq.ParquetFile(path)
        groups = list(range(pf.metadata.num_row_groups))
        rng.shuffle(groups)
        cols = [c for c in (columns or []) if c in pf.schema_arrow.names] or None
        for g in groups:
            rows = pf.read_row_group(g, columns=cols).to_pylist()
            rng.shuffle(rows)
            yield from rows
    else:  # json lines, gzip
        with gzip.open(path, "rt", encoding="utf-8") as f:
            buf = []
            for line in f:
                if line.strip():
                    buf.append(line)
                if len(buf) >= 2000:
                    rng.shuffle(buf)
                    yield from (json.loads(x) for x in buf)
                    buf = []
            rng.shuffle(buf)
            yield from (json.loads(x) for x in buf)


def fetch(src: dict, tok, scale: float, seed: int, log=print) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=False)
    files = sorted(f for f in api.list_repo_files(src["repo"], repo_type="dataset", token=False) if re.search(src["files"], f))
    if not files:
        raise SystemExit(f"[{src['name']}] no files matching {src['files']} in {src['repo']}")
    rng = random.Random(f"{seed}-{src['name']}")
    rng.shuffle(files)
    # shards already in the local HF cache first (the connection here is slow)
    from huggingface_hub import try_to_load_from_cache
    cached = {f for f in files if isinstance(try_to_load_from_cache(src["repo"], f, repo_type="dataset"), str)}
    files = [f for f in files if f in cached] + [f for f in files if f not in cached]
    if cached:
        log(f"[{src['name']}] {len(cached)} of {len(files)} shards already cached")
    budget = int(src["budget"] * 1e6 * scale)
    out_path = os.path.join(OUT_DIR, f"{src['name']}.parquet")
    ids, texts, n_tok, n_seen, n_skip, t0 = [], [], 0, 0, 0, time.time()
    seen = set()
    for fi, fname in enumerate(files):
        if n_tok >= budget:
            break
        t1 = time.time()
        # The connection here (VPN) drops for minutes at a time: keep retrying; partial files resume.
        local = None
        for attempt in range(200):
            try:
                local = hf_hub_download(src["repo"], fname, repo_type="dataset", token=False, etag_timeout=60)
                break
            except Exception as e:  # noqa: BLE001
                wait = min(300, 15 * 2 ** min(attempt, 5))
                log(f"[{src['name']}] download of {fname} failed ({type(e).__name__}: {str(e)[:120]}); retry {attempt + 1} in {wait}s")
                time.sleep(wait)
        if local is None:
            continue
        log(f"[{src['name']}] shard {fi + 1}/{len(files)} {fname} ({os.path.getsize(local) / 2**20:.0f} MB, {time.time() - t1:.0f}s)")
        batch = []
        for row in iter_rows(local, src.get("columns") or ([src["text"], "id"] if isinstance(src["text"], str) else None), rng):
            n_seen += 1
            text = src["text"](row) if callable(src["text"]) else row.get(src["text"])
            if not text or len(text) < src["min_chars"]:
                n_skip += 1
                continue
            text = cut(text.strip(), src.get("max_chars"))
            h = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            batch.append((str(row.get("id") or h[:24]), text))
            if len(batch) >= 64:
                n_tok += _take(batch, tok, ids, texts, budget - n_tok)
                batch = []
                if n_tok >= budget:
                    break
        if batch and n_tok < budget:
            n_tok += _take(batch, tok, ids, texts, budget - n_tok)
        log(f"[{src['name']}]   {len(texts):,} docs, {n_tok / 1e6:.1f}/{budget / 1e6:.1f} M tokens "
            f"({n_skip:,} of {n_seen:,} rows filtered) {time.time() - t0:.0f}s")
    os.makedirs(OUT_DIR, exist_ok=True)
    pq.write_table(pa.table({"id": ids, "text": texts}), out_path, compression="zstd")
    stats = {"name": src["name"], "group": src["group"], "repo": src["repo"], "license": src["license"], "docs": len(texts),
             "tokens": n_tok, "budget": budget, "rows_seen": n_seen, "rows_filtered": n_skip, "out": os.path.relpath(out_path, ROOT)}
    log(f"[{src['name']}] wrote {out_path}: {len(texts):,} docs, {n_tok / 1e6:.1f} M tokens")
    return stats


def _take(batch, tok, ids, texts, room: int) -> int:
    lens = [len(x) for x in tok([t for _, t in batch], add_special_tokens=False)["input_ids"]]
    got = 0
    for (i, t), n in zip(batch, lens):
        if got >= room:
            break
        ids.append(i)
        texts.append(t)
        got += n + 1  # + EOS added by the trainer
    return got


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma list of source names")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every token budget (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    tok.model_max_length = 10 ** 12
    want = {s.strip() for s in args.only.split(",") if s.strip()}
    stats = []
    for src in SOURCES:
        if want and src["name"] not in want:
            continue
        stats.append(fetch(src, tok, args.scale, args.seed))
        with open(os.path.join(OUT_DIR, f"stats_{src['name']}.json"), "w") as f:
            json.dump(stats[-1], f, indent=2)
    print("\n| source | group | docs | M tokens |\n|---|---|---:|---:|")
    for s in stats:
        print(f"| {s['name']} | {s['group']} | {s['docs']:,} | {s['tokens'] / 1e6:.1f} |")


if __name__ == "__main__":
    main()
