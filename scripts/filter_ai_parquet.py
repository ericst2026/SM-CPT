#!/usr/bin/env python3
"""Keyword-filter a tree of parquet files down to AI / ML rows (no model).

Every *.parquet under SRC (recursively) is written to the same relative path under DST,
containing only the rows whose text the keyword scorer in ai_qa_data/keywords.py accepts.

Speed:
  * Each regex alternative in keywords.py gets a literal "anchor" (a substring every match must
    contain). One Aho-Corasick pass over the lower-cased text finds which anchors occur, and only
    those terms' regexes are run. Rows with no anchor hit are rejected without running any regex.
    The scores are identical to keywords.score_text (minus topic hints, which are unused here).
  * Files are split into chunks of row groups and scored in a process pool; the main process
    writes each output file in input order.

Budget:
  Files are processed in random order and the run stops once DST holds --max-gb of parquet
  (outputs from earlier runs count), so a capped collection is spread over many sources instead
  of exhausting the first folders alphabetically. The file being written when the cap is hit is
  closed as it is (truncated at a chunk boundary).

Example:
  python scripts/filter_ai_parquet.py /data/raw /data/ai_only --workers 16
  python scripts/filter_ai_parquet.py SRC DST --columns question:1,answer:0.5 --keep uncertain --add-scores
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import ahocorasick
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import re._parser as sre_parse  # Python 3.11+
except ImportError:  # pragma: no cover
    import sre_parse

DEFAULT_KEYWORDS = Path(__file__).resolve().parents[1] / "ai_qa_data" / "keywords.py"
GROUPS = ("core", "core_cs", "math", "code")

# Per-process state, filled by _init_worker.
KW = None          # the keywords module
AUTO = None        # Aho-Corasick automaton: anchor -> tuple of (group, term_idx)
ALWAYS = None      # group -> term indices without a usable anchor (always run)
REGEX = None       # group -> list of compiled alternatives per term (from keywords.py)
TERMS = None       # group -> list of (pattern, weight, topic)
OPTS = None        # dict of filter options


# --------------------------------------------------------------------------- #
# Keyword index
# --------------------------------------------------------------------------- #
def load_keywords(path: str):
    # Load the file directly so the package __init__ (and its imports) is not executed.
    spec = importlib.util.spec_from_file_location("ai_keywords", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses needs the module registered
    spec.loader.exec_module(mod)
    return mod


def _rank(cands: set[str]) -> tuple[int, int]:
    """Prefer anchor sets whose shortest member is longest, then fewer members."""
    return min(map(len, cands)), -len(cands)


def _anchor_set(seq) -> set[str] | None:
    """A set of lower-cased literals such that every match of `seq` contains at least one."""
    best: set[str] | None = None
    run: list[str] = []

    def consider(cands):
        nonlocal best
        if cands and all(cands) and (best is None or _rank(cands) > _rank(best)):
            best = cands

    def flush():
        if run:
            consider({"".join(run)})
            run.clear()

    for op, av in seq:
        if op == sre_parse.LITERAL:
            run.append(chr(av).lower())
            continue
        flush()
        if op == sre_parse.SUBPATTERN:
            consider(_anchor_set(av[-1]))
        elif op == sre_parse.BRANCH:
            subs = [_anchor_set(s) for s in av[1]]
            if all(subs):
                consider(set().union(*subs))
        elif op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT) and av[0] >= 1:
            consider(_anchor_set(av[2]))
    flush()
    return best


def build_index(kw):
    regex = {"core": kw._CORE_RE, "core_cs": kw._CORE_CS_RE, "math": kw._MATH_RE, "code": kw._CODE_RE}
    terms = {"core": kw.CORE_TERMS, "core_cs": kw.CORE_TERMS_CS, "math": kw.MATH_TERMS, "code": kw.CODE_TERMS}
    auto = ahocorasick.Automaton()
    always: dict[str, list[int]] = {g: [] for g in GROUPS}
    for g in GROUPS:
        for idx, alts in enumerate(regex[g]):
            anchors = set()
            for rx in alts:
                a = _anchor_set(sre_parse.parse(rx.pattern))
                if not a or _rank(a)[0] < 2:
                    anchors = None
                    break
                anchors |= a
            if anchors is None:
                always[g].append(idx)
                continue
            for s in anchors:
                auto.add_word(s, auto.get(s, ()) + ((g, idx),))
    auto.make_automaton()
    return auto, always, regex, terms


# --------------------------------------------------------------------------- #
# Scoring (mirrors keywords.score_text, running only the anchored candidate terms)
# --------------------------------------------------------------------------- #
def score_field(text: str, weight: float, r) -> None:
    if not text:
        return
    kw = KW
    text = text[:kw.MAX_CHARS]
    low = text.lower()
    cand = {g: set(ALWAYS[g]) for g in GROUPS}
    for _, hits in AUTO.iter(low):
        for g, idx in hits:
            cand[g].add(idx)
    if not any(cand.values()):
        return

    def sub(g):
        ids = sorted(cand[g])
        return [REGEX[g][i] for i in ids], [TERMS[g][i] for i in ids]

    hits: list[str] = []
    th: dict[str, float] = {}
    core = kw._score(*sub("core"), low, hits, th) if cand["core"] else 0.0
    if cand["core_cs"]:
        has_ml_unit = bool(kw._ML_UNIT_RE.search(low))
        core += kw._score(*sub("core_cs"), text, hits, th,
                          skip=(lambda m: has_ml_unit and m.group(0) == "ML"))
    neg = sum(w for rx, w in kw._NEG_RE if rx.search(low)) if core else 0.0
    math_s = kw._score(*sub("math"), low, hits, th) if cand["math"] else 0.0
    code_s = kw._score(*sub("code"), text, hits, th, left_boundary=False) if cand["code"] else 0.0

    n_words = max(1, text.count(" ") + 1)
    damp = 1.0 if n_words < 300 else max(0.5, 1.0 / (n_words / 300) ** 0.5)
    r.core += weight * max(0.0, core - neg) * damp
    r.math += weight * math_s * damp
    r.code += weight * code_s * damp
    r.neg += weight * neg


# --------------------------------------------------------------------------- #
# Arrow helpers
# --------------------------------------------------------------------------- #
def _is_text(t: pa.DataType) -> bool:
    return pa.types.is_string(t) or pa.types.is_large_string(t) or (
        hasattr(pa.types, "is_string_view") and pa.types.is_string_view(t))


def _has_text(t: pa.DataType) -> bool:
    if _is_text(t):
        return True
    if pa.types.is_dictionary(t):
        return _has_text(t.value_type)
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        return _has_text(t.value_type)
    if pa.types.is_struct(t):
        return any(_has_text(t.field(i).type) for i in range(t.num_fields))
    if pa.types.is_map(t):
        return _has_text(t.key_type) or _has_text(t.item_type)
    return False


def _flatten(v, out: list[str]) -> None:
    if isinstance(v, str):
        out.append(v)
    elif isinstance(v, dict):
        for x in v.values():
            _flatten(x, out)
    elif isinstance(v, (list, tuple)):
        for x in v:
            _flatten(x, out)


def column_texts(col: pa.ChunkedArray) -> list:
    t = col.type
    if pa.types.is_dictionary(t) and _is_text(t.value_type):
        col = col.cast(t.value_type)
        t = col.type
    if _is_text(t):
        return col.to_pylist()
    texts = []
    for v in col.to_pylist():  # nested (e.g. chat messages): join all string leaves
        parts: list[str] = []
        _flatten(v, parts)
        texts.append("\n".join(parts))
    return texts


def pick_columns(schema: pa.Schema, spec: list[tuple[str, float]] | None) -> list[tuple[str, float]]:
    if spec is None:
        return [(f.name, 1.0) for f in schema if _has_text(f.type)]
    return [(c, w) for c, w in spec if c in schema.names]


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def _init_worker(keywords_path: str, opts: dict) -> None:
    global KW, AUTO, ALWAYS, REGEX, TERMS, OPTS
    pa.set_cpu_count(1)
    pa.set_io_thread_count(2)
    KW = load_keywords(keywords_path)
    AUTO, ALWAYS, REGEX, TERMS = build_index(KW)
    OPTS = opts


def plan_file(path: str):
    try:
        md = pq.ParquetFile(path).metadata
        return path, [md.row_group(i).num_rows for i in range(md.num_row_groups)], None
    except Exception as e:  # noqa: BLE001
        return path, [], f"{type(e).__name__}: {e}"


def filter_chunk(task):
    file_idx, chunk_idx, path, rg_ids = task
    try:
        pf = pq.ParquetFile(path)
        table = pf.read_row_groups(rg_ids, use_threads=False) if rg_ids else pf.schema_arrow.empty_table()
        cols = pick_columns(table.schema, OPTS["columns"])
        texts = [(column_texts(table.column(c)), w) for c, w in cols]
        keep_uncertain, domain = OPTS["keep"] == "uncertain", OPTS["domain"]
        mask, core, math_s, code = [], [], [], []
        for i in range(table.num_rows):
            r = KW.KeywordResult()
            for col_texts, w in texts:
                score_field(col_texts[i], w, r)
            d = KW.keyword_decision(r, domain)
            mask.append(d == "accept" or (keep_uncertain and d == "uncertain"))
            if OPTS["add_scores"]:
                core.append(round(r.core, 2)), math_s.append(round(r.math, 2)), code.append(round(r.code, 2))
        if OPTS["add_scores"]:
            for name, vals in (("kw_core", core), ("kw_math", math_s), ("kw_code", code)):
                table = table.append_column(name, pa.array(vals, pa.float32()))
        out = table.filter(pa.array(mask, pa.bool_()))
        return file_idx, chunk_idx, out, table.num_rows, None
    except Exception as e:  # noqa: BLE001
        return file_idx, chunk_idx, None, 0, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_columns(s: str | None):
    if not s:
        return None
    out = []
    for part in s.split(","):
        name, _, w = part.strip().partition(":")
        out.append((name, float(w) if w else 1.0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="input folder (searched recursively for *.parquet)")
    ap.add_argument("dst", type=Path, help="output folder (mirrors the input tree)")
    ap.add_argument("--columns", default=None,
                    help="text columns with optional weights, e.g. 'question:1,answer:0.5'. "
                         "Default: every column containing strings, weight 1")
    ap.add_argument("--keep", choices=("accept", "uncertain"), default="accept",
                    help="'accept' keeps rows keywords.py auto-accepts; 'uncertain' also keeps borderline rows")
    ap.add_argument("--domain", choices=("ai_core", "ai_math"), default="ai_core",
                    help="ai_math lets math vocabulary count as signal (only matters with --keep uncertain)")
    ap.add_argument("--add-scores", action="store_true", help="add kw_core / kw_math / kw_code columns")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--chunk-rows", type=int, default=20_000, help="target rows per work unit")
    ap.add_argument("--compression", default="zstd")
    ap.add_argument("--overwrite", action="store_true", help="re-process files that already exist in dst")
    ap.add_argument("--max-gb", type=float, default=10.0,
                    help="stop once dst holds this many GB of parquet (0 = no limit)")
    ap.add_argument("--seed", type=int, default=None, help="seed for the random file order")
    ap.add_argument("--keywords", default=str(DEFAULT_KEYWORDS), help="path to keywords.py")
    args = ap.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    files = sorted(p for p in src.rglob("*.parquet") if p.is_file() and dst not in p.parents)
    todo = [p for p in files if args.overwrite or not (dst / p.relative_to(src)).exists()]
    random.Random(args.seed).shuffle(todo)  # spread a capped collection across many sources
    print(f"{len(files)} parquet files found, {len(files) - len(todo)} already done, {len(todo)} to process")

    limit = int(args.max_gb * 1024 ** 3) if args.max_gb > 0 else None
    todo_set = set(todo)
    done_bytes = sum((dst / p.relative_to(src)).stat().st_size for p in files
                     if p not in todo_set and (dst / p.relative_to(src)).exists())
    if limit:
        print(f"output budget {args.max_gb:g} GB, {done_bytes / 1024 ** 3:.2f} GB already in dst")
    if not todo or (limit and done_bytes >= limit):
        return

    opts = {"columns": parse_columns(args.columns), "keep": args.keep,
            "domain": args.domain, "add_scores": args.add_scores}
    with Pool(args.workers, initializer=_init_worker, initargs=(args.keywords, opts)) as pool:
        # Plan: split every file into chunks of whole row groups.
        tasks, n_chunks, plans = [], [], []
        for path, rg_rows, err in pool.imap(plan_file, map(str, todo), chunksize=8):
            if err:
                print(f"  SKIP {path}: {err}", file=sys.stderr)
                continue
            fi = len(plans)
            plans.append(Path(path))
            chunks, cur, cur_rows = [], [], 0
            for rg, n in enumerate(rg_rows):
                cur.append(rg)
                cur_rows += n
                if cur_rows >= args.chunk_rows:
                    chunks.append(cur)
                    cur, cur_rows = [], 0
            if cur or not chunks:
                chunks.append(cur)
            n_chunks.append(len(chunks))
            tasks.extend((fi, ci, path, c) for ci, c in enumerate(chunks))

        total_rows = kept_rows = files_done = 0
        writer, tmp_path, out_path, failed = None, None, None, set()
        budget_hit = False
        t0 = last_log = time.time()
        # imap is ordered, so each file's chunks arrive contiguously and in order.
        for fi, ci, table, n_rows, err in pool.imap(filter_chunk, tasks, chunksize=1):
            if fi in failed:
                continue
            out_path = dst / plans[fi].relative_to(src)
            if err:
                print(f"  FAIL {plans[fi]}: {err}", file=sys.stderr)
                failed.add(fi)
                if writer:
                    writer.close()
                    writer = None
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink()
                continue
            if ci == 0:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = out_path.with_name(out_path.name + ".tmp")
                writer = pq.ParquetWriter(tmp_path, table.schema, compression=args.compression)
            writer.write_table(table)
            total_rows += n_rows
            kept_rows += table.num_rows
            # write_table flushes whole row groups, so the tmp size tracks the bytes written so far.
            budget_hit = bool(limit) and done_bytes + tmp_path.stat().st_size >= limit
            if ci == n_chunks[fi] - 1 or budget_hit:
                writer.close()
                writer = None
                done_bytes += tmp_path.stat().st_size
                os.replace(tmp_path, out_path)
                files_done += 1
                if budget_hit:
                    if ci < n_chunks[fi] - 1:
                        print(f"  budget reached; {out_path} truncated after chunk {ci + 1}/{n_chunks[fi]}")
                    break  # leaving the `with` terminates the pool

            now = time.time()
            if now - last_log > 10:
                last_log = now
                print(f"  {files_done}/{len(plans)} files | {total_rows:,} rows scanned | "
                      f"{kept_rows:,} kept ({kept_rows / max(1, total_rows):.2%}) | "
                      f"{total_rows / (now - t0):,.0f} rows/s", flush=True)

    dt = time.time() - t0
    print(f"{'stopped at budget' if budget_hit else 'done'}: {files_done} files, {total_rows:,} rows -> "
          f"{kept_rows:,} kept ({kept_rows / max(1, total_rows):.2%}) in {dt:.1f}s; {len(failed)} failed; "
          f"dst holds {done_bytes / 1024 ** 3:.2f} GB")


if __name__ == "__main__":
    main()
