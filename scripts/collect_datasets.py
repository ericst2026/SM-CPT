#!/usr/bin/env python3
"""Step 1 - collect: stream every source in the catalog into unified JSONL.

    python scripts/collect_datasets.py --list
    python scripts/collect_datasets.py --sources all --workers 6
    python scripts/collect_datasets.py --sources se_ai,eli5 --scan-scale 0.1   # quick test
    python scripts/collect_datasets.py --sources all --no-prefilter            # keep everything (needs disk!)

Output (per source, in --out, default data/raw):
    <name>.jsonl          candidate records (meta-accepted, curated, or keyword-plausible), plus - for
                          sources with reasoning=True - non-AI rows that pass ai_qa_data/reasoning.py
                          (hints.reasoning holds the check; split_ai_data.py routes them to reasoning_sft)
    <name>.replay.jsonl   reservoir sample of rejected rows (general replay data, anti-forgetting)
    <name>.stats.json     counts + timing; its presence marks the source as done (use --force to redo)

Why streaming + prefilter: the general pools (OpenHermes, Tulu, WildChat, Cosmopedia, ...) are
tens of GB. We never materialise them: rows are streamed, normalised, scored by the cheap keyword
model, and only plausible AI candidates (typically 1-10 %) are written. The expensive zero-shot
model then runs on this much smaller candidate set in step 2 (split_ai_data.py).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_qa_data.env import hf_auth_headers, load_env  # noqa: E402

load_env()  # HF_TOKEN from .env -> authenticated Hub requests (higher rate limits); before any HF import

from ai_qa_data.keywords import keyword_decision, score_record  # noqa: E402
from ai_qa_data.reasoning import DEFAULT_MAX_CHARS, reasoning_check  # noqa: E402
from ai_qa_data.sources import SOURCES, SOURCES_BY_NAME, Source  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_Q_CHARS, MAX_A_CHARS = 12000, 40000


# --------------------------------------------------------------------------- #
# Row readers
# --------------------------------------------------------------------------- #
def _parse_hf_url(url: str):
    """hf://datasets/<org>/<name>@<rev>/<path>  ->  (repo_id, rev, path)"""
    body = url.split("hf://datasets/", 1)[1]
    repo_rev, path = body.split("/", 2)[:2], body.split("/", 2)[2]
    org, name_rev = repo_rev
    name, _, rev = name_rev.partition("@")
    return f"{org}/{name}", (rev or None), path


def list_parquet_shards(src: Source) -> list[tuple[str, str | None, str]]:
    """(repo_id, revision, path_in_repo) of the parquet shards for this config/split.
    1) native parquet files of the repo (resolved exactly like `datasets` does)
    2) otherwise the Hub's auto-converted parquet branch (refs/convert/parquet), which also works for
       JSON(L) repos and script-based repos (note: it can be *partial* for datasets > 5 GB)."""
    try:
        from datasets import load_dataset_builder
        b = load_dataset_builder(src.hf_id, src.config)
        files = list((b.config.data_files or {}).get(src.split, []))
        if files and all(str(f).endswith(".parquet") for f in files):
            return [_parse_hf_url(str(f)) for f in files]
    except Exception:  # noqa: BLE001
        pass
    import urllib.parse
    import urllib.request
    url = "https://datasets-server.huggingface.co/parquet?dataset=" + urllib.parse.quote(src.hf_id)
    with urllib.request.urlopen(urllib.request.Request(url, headers=hf_auth_headers()), timeout=60) as r:
        info = json.load(r)
    out = []
    for f in info.get("parquet_files", []):
        if f["config"] == (src.config or "default") and f["split"] == src.split:
            path = urllib.parse.unquote(f["url"].split("refs%2Fconvert%2Fparquet/", 1)[1])
            out.append((src.hf_id, "refs/convert/parquet", path))
    return out


NET_RETRIES = 8


def _retry(fn, what: str, tries: int = NET_RETRIES):
    """Call fn() with exponential backoff - the Hub connection drops / times out / loses DNS a lot."""
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            wait = min(5 * 2 ** k, 120)
            print(f"  retry {k + 1}/{tries - 1} {what}: {type(e).__name__}: {str(e)[:100]} (sleep {wait}s)", flush=True)
            time.sleep(wait)


def iter_shards(src: Source, cache_dir: str, shuffle_seed: int | None, max_shards: int | None = None):
    """Read parquet shards straight from the Hub with HTTP range requests, one row group at a time,
    only the needed columns. Nothing is downloaded beyond what is actually scanned, so a max_scan
    budget on a 15 GB single-file dataset costs a few hundred MB, and a dropped connection only
    re-fetches one row group (with retry/backoff) instead of restarting a whole-file download."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    shards = _retry(lambda: list_parquet_shards(src), "list shards")
    if not shards:
        raise RuntimeError("no parquet shards found")
    if shuffle_seed is not None:  # partial scans should sample the whole dataset, not just shard 0
        random.Random(shuffle_seed).shuffle(shards)
    if max_shards:
        shards = shards[:max_shards]
    fs = HfFileSystem()

    for repo, rev, path in shards:
        url = f"datasets/{repo}@{rev}/{path}" if rev else f"datasets/{repo}/{path}"
        url = url.replace("refs/convert/parquet", "refs%2Fconvert%2Fparquet")
        state = {}

        def reopen():
            if state.get("f"):
                try:
                    state["f"].close()
                except Exception:  # noqa: BLE001
                    pass
            fs.invalidate_cache()
            state["f"] = fs.open(url, "rb", block_size=4 << 20)
            state["pf"] = pq.ParquetFile(state["f"])

        _retry(reopen, f"open {path}")
        pf = state["pf"]
        names = set(pf.schema_arrow.names)
        cols = [c for c in src.columns if c in names] if src.columns else None
        try:
            for i in range(pf.num_row_groups):
                def read(i=i):
                    try:
                        return state["pf"].read_row_group(i, columns=cols)
                    except Exception:
                        reopen()  # fresh connection for the next attempt
                        raise
                table = _retry(read, f"{path} row group {i}/{pf.num_row_groups}")
                for batch in table.to_batches(max_chunksize=1024):
                    yield from batch.to_pylist()
                del table
        finally:
            try:
                state["f"].close()
            except Exception:  # noqa: BLE001
                pass


def iter_stream(src: Source):
    """Fallback: `datasets` streaming (slow for big text corpora, but works for anything)."""
    from datasets import load_dataset

    kwargs = dict(split=src.split, streaming=True)
    if src.columns:
        kwargs["columns"] = src.columns
    try:
        yield from load_dataset(src.hf_id, src.config, **kwargs)
    except (TypeError, ValueError) as e:
        if "column" not in str(e).lower():
            raise
        kwargs.pop("columns")
        yield from load_dataset(src.hf_id, src.config, **kwargs)


def open_rows(src: Source, mode: str, cache_dir: str, shuffle_seed: int | None, max_shards: int | None = None):
    if mode in ("auto", "shard"):
        try:
            gen = iter_shards(src, cache_dir, shuffle_seed, max_shards)
            first = next(gen)
            yield first
            yield from gen
            return
        except StopIteration:
            return
        except Exception as e:  # noqa: BLE001
            if mode == "shard":
                raise
            print(f"[{src.name}] shard mode failed ({type(e).__name__}: {str(e)[:120]}) -> streaming", flush=True)
    yield from iter_stream(src)


def build_record(src: Source, n: int, row: dict, norm: dict, meta_decision: str | None) -> dict | None:
    q, a = norm.get("question") or "", norm.get("answer") or ""
    if not q.strip() or len(q) > MAX_Q_CHARS or len(a) > MAX_A_CHARS:
        return None
    try:
        prior = src.level_prior(row)
    except Exception:  # noqa: BLE001
        prior = None
    return {
        "id": f"{src.name}/{n}",
        "source": src.name,
        "hf": src.hf,
        "bucket": src.bucket,
        "domain": src.domain,
        "task": norm.get("task", "qa"),
        "lang": src.lang,
        "question": q,
        "answer": a or None,
        "reasoning": norm.get("reasoning"),
        "ground_truth": norm.get("ground_truth"),
        "tests": norm.get("tests"),
        "messages": norm.get("messages"),
        "meta": norm.get("meta") or {},
        "hints": {"meta_decision": meta_decision, "level_prior": prior},
    }


def collect_one(name: str, out_dir: str, scan_scale: float, prefilter: bool, replay_n: int,
                force: bool, seed: int, mode: str = "auto", max_shards: int | None = None,
                reasoning_max_chars: int = DEFAULT_MAX_CHARS, reasoning_cap: int = 40000) -> dict:
    src = SOURCES_BY_NAME[name]
    stats_path = os.path.join(out_dir, f"{name}.stats.json")
    if os.path.exists(stats_path) and not force:
        with open(stats_path) as f:
            st = json.load(f)
        st["skipped"] = True
        return st

    rng = random.Random(seed)
    max_scan = int(src.max_scan * scan_scale) if src.max_scan else None
    if scan_scale < 1.0 and max_scan is None and src.pool == "huge":
        max_scan = int(200000 * scan_scale)
    if max_shards and not max_scan:
        # a shard budget must also bound the streaming fallback (used when shard download fails)
        max_scan = 150000 * max_shards
    max_rows = src.max_rows
    do_prefilter = prefilter and src.pool == "huge"

    st = {"source": name, "hf": src.hf, "pool": src.pool, "bucket": src.bucket, "scanned": 0, "written": 0,
          "norm_failed": 0, "meta_accept": 0, "meta_reject": 0, "kw_reject": 0, "replay": 0,
          "reasoning_keep": 0, "reasoning_fail": {}}
    t0 = time.time()
    tmp_main = os.path.join(out_dir, f"{name}.jsonl.tmp")
    replay: list[dict] = []
    seen_rejects = 0

    def add_replay(rec):
        nonlocal seen_rejects
        seen_rejects += 1
        if replay_n <= 0 or rec is None:
            return
        if len(replay) < replay_n:
            replay.append(rec)
        else:
            j = rng.randrange(seen_rejects)
            if j < replay_n:
                replay[j] = rec

    try:
        cache_dir = os.path.join(out_dir, ".shard_cache", name)
        it = open_rows(src, mode, cache_dir, seed if (max_scan or max_shards) else None, max_shards)
        with open(tmp_main, "w", encoding="utf-8") as fout:
            for n, row in enumerate(it):
                if max_scan and n >= max_scan:
                    break
                if max_rows and st["written"] >= max_rows:
                    break
                st["scanned"] += 1
                try:
                    md = src.meta_rule(row)
                    norm = src.normalize(row)
                except Exception:  # noqa: BLE001
                    md, norm = None, None
                if not norm:
                    st["norm_failed"] += 1
                    continue
                for k, nd in enumerate(norm if isinstance(norm, list) else [norm]):
                    rec = build_record(src, n if k == 0 else f"{n}.{k}", row, nd, md)
                    if rec is None:
                        st["norm_failed"] += 1
                        continue
                    chk = None
                    if src.reasoning:
                        chk = reasoning_check(rec, reasoning_max_chars)
                        rec["hints"]["reasoning"] = chk
                        if not chk["ok"]:
                            st["reasoning_fail"][chk["why"]] = st["reasoning_fail"].get(chk["why"], 0) + 1

                    def keep_for_reasoning():
                        # not AI, but a good general reasoning example -> reasoning_sft (decided again in split)
                        if chk and chk["ok"] and st["reasoning_keep"] < reasoning_cap:
                            st["reasoning_keep"] += 1
                            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            st["written"] += 1
                            return True
                        return False

                    if md == "reject":
                        st["meta_reject"] += 1
                        if not keep_for_reasoning():
                            add_replay(rec)
                        continue
                    if md == "accept":
                        st["meta_accept"] += 1
                    elif do_prefilter:
                        kw = score_record(rec["question"], rec["answer"] or (rec.get("reasoning") or "")[:3000])
                        if keyword_decision(kw, src.domain) == "reject":
                            st["kw_reject"] += 1
                            if not keep_for_reasoning():
                                add_replay(rec)
                            continue
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    st["written"] += 1
                if st["scanned"] % 50000 == 0:
                    print(f"[{name}] scanned={st['scanned']:,} written={st['written']:,} "
                          f"reasoning-only={st['reasoning_keep']:,} ({time.time() - t0:.0f}s)", flush=True)
        os.replace(tmp_main, os.path.join(out_dir, f"{name}.jsonl"))
        with open(os.path.join(out_dir, f"{name}.replay.jsonl"), "w", encoding="utf-8") as f:
            for rec in replay:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        st["replay"] = len(replay)
        st["seconds"] = round(time.time() - t0, 1)
        st["max_scan"] = max_scan
        st["max_shards"] = max_shards
        st["exhausted"] = not (max_scan and st["scanned"] >= max_scan) and not (max_rows and st["written"] >= max_rows)
        with open(stats_path, "w") as f:
            json.dump(st, f, indent=2)
        print(f"[{name}] DONE scanned={st['scanned']:,} written={st['written']:,} "
              f"(reasoning-only {st['reasoning_keep']:,}) replay={st['replay']} ({st['seconds']}s)", flush=True)
        return st
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        st["error"] = f"{type(e).__name__}: {e}"
        return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="all", help="comma list of source names, 'all', or a bucket name "
                                                     "(ai_sft, ai_rl_math, ai_rl_code, ai_eval) or pool (curated, huge)")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scan-scale", type=float, default=1.0,
                    help="multiply every source's max_scan (e.g. 0.05 for a smoke test, 10 for a bigger crawl)")
    ap.add_argument("--no-prefilter", action="store_true", help="write all rows of huge pools (much more disk)")
    ap.add_argument("--replay-per-source", type=int, default=2000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--include-disabled", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["auto", "shard", "stream"], default="auto",
                    help="auto = parquet shards (fast) with streaming fallback")
    ap.add_argument("--max-shards", type=int, default=None,
                    help="download at most N (randomly chosen) parquet shards per source - a bandwidth budget")
    ap.add_argument("--reasoning-max-chars", type=int, default=DEFAULT_MAX_CHARS,
                    help="reasoning_sft rows: max question+trace+answer characters kept at collection time")
    ap.add_argument("--reasoning-per-source", type=int, default=40000,
                    help="cap on non-AI reasoning rows kept per source (sources with reasoning=True)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for s in SOURCES:
            flag = "" if s.enabled else "  [disabled]"
            print(f"{s.name:24s} {s.bucket:11s} {s.pool:8s} {s.hf:70s}{flag}\n{'':24s} {s.desc}")
        return

    sel = args.sources.split(",")
    if sel == ["all"]:
        names = [s.name for s in SOURCES if s.enabled or args.include_disabled]
    else:
        names = []
        for x in sel:
            if x in SOURCES_BY_NAME:
                names.append(x)
            else:
                names += [s.name for s in SOURCES if (s.bucket == x or s.pool == x) and s.enabled]
    if not names:
        sys.exit(f"no sources match {args.sources!r}; use --list")
    os.makedirs(args.out, exist_ok=True)
    print("HF auth: " + ("HF_TOKEN set (authenticated, higher rate limits)" if os.environ.get("HF_TOKEN")
                         else "no HF_TOKEN - anonymous, low rate limits"), flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(collect_one, n, args.out, args.scan_scale, not args.no_prefilter,
                          args.replay_per_source, args.force, args.seed, args.mode, args.max_shards,
                          args.reasoning_max_chars, args.reasoning_per_source): n for n in names}
        for fu in as_completed(futs):
            results.append(fu.result())

    print("\n=== collection summary ===")
    print(f"{'source':24s} {'scanned':>10s} {'written':>9s} {'reason':>8s} {'meta+':>7s} {'meta-':>8s} {'kw-':>9s}  status")
    for st in sorted(results, key=lambda s: s["source"]):
        status = "ERROR " + st["error"][:60] if "error" in st else ("cached" if st.get("skipped") else "ok")
        print(f"{st['source']:24s} {st.get('scanned', 0):>10,} {st.get('written', 0):>9,} {st.get('reasoning_keep', 0):>8,} "
              f"{st.get('meta_accept', 0):>7,} "
              f"{st.get('meta_reject', 0):>8,} {st.get('kw_reject', 0):>9,}  {status}")


if __name__ == "__main__":
    main()
