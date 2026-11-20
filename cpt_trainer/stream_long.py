"""Stream real long documents (arXiv papers, open-access books, PG-19 books) from Hugging Face without
downloading whole shards, for a slow connection.

    python -m cpt_trainer.stream_long --deadline-min 90            # all three sources in parallel
    python -m cpt_trainer.stream_long --only arxiv --budget 60

Gzipped JSON-lines shards are decompressed while they arrive and abandoned as soon as the budget is reached;
parquet shards are read one row group at a time with HTTP range requests (HfFileSystem). Every document of at
least --min-chars characters is kept whole, cut into consecutive ~32 k-token chunks at paragraph breaks
(one chunk = at most one 24 k-32 k training sequence), so no downloaded text is wasted. Each source stops at
its token budget or at the deadline and writes cpt_data/extra/<name>.parquet (id, text) with what it has;
partial results are also flushed every few minutes, so an interruption loses little.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from cpt_trainer.fetch_extra import OUT_DIR, ROOT, TOKENIZER, cut  # noqa: E402

CHUNK_CHARS = 128_000
_TOK_LOCK = threading.Lock()  # fast tokenizers are not safe to call from several threads

SOURCES = {
    "arxiv": dict(repo="common-pile/arxiv_papers", files=r"\.jsonl\.gz$", fmt="jsonl.gz", text="text", budget=60,
                  license="openly licensed arXiv papers (Common Pile v0.1)"),
    "doab": dict(repo="common-pile/doab", files=r"\.json\.gz$", fmt="jsonl.gz", text="text", budget=45,
                 license="openly licensed books (DOAB, Common Pile v0.1)"),
    "pg19": dict(repo="emozilla/pg19", files=r"^data/train-.*\.parquet$", fmt="parquet", text="text", budget=30,
                 license="public domain (Project Gutenberg, PG-19)"),
}


def chunks(text: str, chunk_chars: int = CHUNK_CHARS):
    while text:
        piece = cut(text, chunk_chars)
        text = text[len(piece):].lstrip()
        if len(piece) >= 2000:
            yield piece


def iter_jsonl_gz(repo: str, fname: str, log):
    """JSON lines of a gzip shard, decompressed while the bytes arrive (multi-member gzip handled)."""
    import zlib

    import httpx
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo, fname, repo_type="dataset")
    with httpx.stream("GET", url, follow_redirects=True, timeout=httpx.Timeout(120.0, connect=30.0)) as r:
        r.raise_for_status()
        d, buf = zlib.decompressobj(16 + zlib.MAX_WBITS), b""
        for raw in r.iter_raw(1 << 16):
            data = d.decompress(raw)
            while d.unused_data:  # next gzip member
                rest = d.unused_data
                d = zlib.decompressobj(16 + zlib.MAX_WBITS)
                data += d.decompress(rest)
            buf += data
            *lines, buf = buf.split(b"\n")
            for line in lines:
                if line.strip():
                    yield json.loads(line)
        if buf.strip():
            yield json.loads(buf)


def iter_parquet(repo: str, fname: str, rng: random.Random, log):
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=False)
    with fs.open(f"datasets/{repo}/{fname}", "rb", block_size=4 * 2**20) as f:
        pf = pq.ParquetFile(f)
        groups = list(range(pf.metadata.num_row_groups))
        rng.shuffle(groups)
        for g in groups:
            for attempt in range(5):
                try:
                    rows = pf.read_row_group(g, columns=["text"]).to_pylist()
                    break
                except Exception as e:  # noqa: BLE001
                    log(f"    row group {g} failed ({type(e).__name__}); retry {attempt + 1}")
                    time.sleep(20)
            else:
                continue
            yield from rows


def stream(name: str, src: dict, tok, budget_m: float, min_chars: int, deadline: float, seed: int) -> dict:
    import re

    from huggingface_hub import HfApi

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] [{name}] {msg}", flush=True)

    files = [f for f in HfApi(token=False).list_repo_files(src["repo"], repo_type="dataset") if re.search(src["files"], f)]
    rng = random.Random(f"{seed}-{name}")
    rng.shuffle(files)
    budget = int(budget_m * 1e6)
    out = os.path.join(OUT_DIR, f"{name}.parquet")
    ids, texts, n_tok, n_docs, n_short, t0, last_flush = [], [], 0, 0, 0, time.time(), time.time()

    def flush():
        pq.write_table(pa.table({"id": ids, "text": texts}), out, compression="zstd")

    for fname in files:
        if n_tok >= budget or time.time() > deadline:
            break
        log(f"streaming {fname}")
        try:
            it = iter_jsonl_gz(src["repo"], fname, log) if src["fmt"] == "jsonl.gz" else iter_parquet(src["repo"], fname, rng, log)
            for row in it:
                text = (row.get(src["text"]) or "").strip()
                if len(text) < min_chars:
                    n_short += 1
                    continue
                doc_id = str(row.get("id") or f"{fname}:{n_docs}")
                pieces = list(chunks(text))
                with _TOK_LOCK:
                    lens = [len(x) + 1 for x in tok(pieces, add_special_tokens=False)["input_ids"]]
                for k, (p, n) in enumerate(zip(pieces, lens)):
                    if n_tok >= budget:
                        break
                    ids.append(f"{doc_id}#{k}")
                    texts.append(p)
                    n_tok += n
                n_docs += 1
                if time.time() - last_flush > 300:
                    flush()
                    last_flush = time.time()
                    log(f"{n_docs:,} docs -> {len(texts):,} chunks, {n_tok / 1e6:.1f}/{budget / 1e6:.0f} M tokens "
                        f"({n_short:,} shorter than {min_chars} chars) {time.time() - t0:.0f}s")
                if n_tok >= budget or time.time() > deadline:
                    break
        except Exception as e:  # noqa: BLE001 - connection drops: keep what we have, go on with the next shard
            log(f"stream of {fname} broke off ({type(e).__name__}: {str(e)[:150]}); next shard")
            time.sleep(15)
    flush()
    st = {"name": name, "repo": src["repo"], "license": src["license"], "docs": n_docs, "chunks": len(texts),
          "tokens": n_tok, "budget": budget, "stopped": "budget" if n_tok >= budget else "deadline",
          "seconds": round(time.time() - t0), "out": os.path.relpath(out, ROOT)}
    with open(os.path.join(OUT_DIR, f"stats_{name}.json"), "w") as f:
        json.dump(st, f, indent=2)
    log(f"done ({st['stopped']}): {n_docs:,} docs -> {len(texts):,} chunks, {n_tok / 1e6:.1f} M tokens -> {out}")
    return st


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma list of: " + ",".join(SOURCES))
    ap.add_argument("--budget", type=float, default=None, help="M tokens per source (default: per-source budget)")
    ap.add_argument("--min-chars", type=int, default=30_000, help="only documents at least this long (~8 k tokens)")
    ap.add_argument("--deadline-min", type=float, default=90.0, help="stop every source after this many minutes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    tok.model_max_length = 10 ** 12
    os.makedirs(OUT_DIR, exist_ok=True)
    want = [s.strip() for s in args.only.split(",") if s.strip()] or list(SOURCES)
    deadline = time.time() + 60 * args.deadline_min
    with ThreadPoolExecutor(len(want)) as ex:
        futs = [ex.submit(stream, n, SOURCES[n], tok, args.budget or SOURCES[n]["budget"], args.min_chars, deadline, args.seed)
                for n in want]
        stats = [f.result() for f in futs]
    print("\n| source | docs | chunks | M tokens | stopped |\n|---|---:|---:|---:|---|")
    for s in stats:
        print(f"| {s['name']} | {s['docs']:,} | {s['chunks']:,} | {s['tokens'] / 1e6:.1f} | {s['stopped']} |")


if __name__ == "__main__":
    main()
