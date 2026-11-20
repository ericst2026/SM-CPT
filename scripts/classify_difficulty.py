#!/usr/bin/env python3
"""Reading-difficulty classifier (5 levels: primary / middle / high school / university / expert) for parquet corpora.

Two measures (difficulty_common.py):
  zs      zero-shot NLI (models/deberta-v3-base-zeroshot-v2.0) over 5 level hypotheses, softmax across the 5 ->
          <T>_lo1..<T>_lo5 (entailment log-odds), <T>_p1..p5 = softmax(lo / ZS_TEMPERATURE), <T>_level = argmax (1..5),
          <T>_score = expected level (1.0..5.0) under those probabilities, per template T
  length  sentence / word / document length statistics on the full text -> avg_sent_len, max_sent_len, p90_sent_len,
          avg_word_len, long_word_ratio, fk_grade, n_words, n_sents, n_chars, n_tokens (upstream token column, else
          n_words * TOKENS_PER_WORD) and threshold levels len_level_fk / len_level_sent / len_level_max / len_level_tok
  final   zs_level_cal = default template's expected level through the calibrated thresholds (ZS_CAL_THRESHOLDS);
          difficulty_score / difficulty_level = ridge COMBO of the default template's log-odds + length features.
          Both were fitted against Claude labels on the 1k sample (cpt_data/difficulty/report.md); use difficulty_level.

usage (corpus, resumable, one output parquet per input parquet):
  python3 -P scripts/classify_difficulty.py --corpus cpt_data/corpus_ai_fineweb_large --out cpt_data/difficulty/corpus_ai_fineweb_large
usage (single file, chunked + resumable via <out>.parts/; e.g. the replay corpus, or all templates for the comparison):
  python3 -P scripts/classify_difficulty.py --input cpt_data/finewebedu_replay_small.recovered.parquet --out cpt_data/difficulty/finewebedu_replay_small.recovered.parquet
  python3 -P scripts/classify_difficulty.py --input cpt_data/difficulty/sample_1k.parquet --out cpt_data/difficulty/sample_1k.auto.parquet --templates all
options: --methods zs,length  --templates audience|understand|level|all  --head 1200 --max-len 512 --batch 64 --limit-files N --device cuda
"""
from __future__ import annotations

import argparse, os, sys, time

import pyarrow as pa, pyarrow.parquet as pq
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from difficulty_common import (COMBO, DEFAULT_ZS_TEMPLATE, LENGTH_FEATURES, LENGTH_THRESHOLDS, TOKENS_PER_WORD,  # noqa: E402
                               ZS_CAL_THRESHOLDS, ZS_TEMPERATURE, ZS_TEMPLATES, combo_score, head, length_features,
                               level_from_thresholds)

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--corpus"); g.add_argument("--input")
ap.add_argument("--out", required=True)
ap.add_argument("--methods", default="zs,length")
ap.add_argument("--templates", default=DEFAULT_ZS_TEMPLATE, help="comma list of ZS_TEMPLATES keys or 'all'")
ap.add_argument("--model", default=os.path.join(ROOT, "models", "deberta-v3-base-zeroshot-v2.0"))
ap.add_argument("--head", type=int, default=1200); ap.add_argument("--max-len", type=int, default=512)
ap.add_argument("--batch", type=int, default=64); ap.add_argument("--limit-files", type=int, default=None)
ap.add_argument("--rows-per-chunk", type=int, default=5000, help="--input mode: rows per resumable part file")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
methods = set(a.methods.split(","))
templates = list(ZS_TEMPLATES) if a.templates == "all" else a.templates.split(",")
for t in templates:
    assert t in ZS_TEMPLATES, f"unknown template {t}, have {list(ZS_TEMPLATES)}"

zs = None
if "zs" in methods:
    from ai_qa_data.zeroshot import ZeroShotClassifier
    zs = ZeroShotClassifier(a.model, device=a.device, batch_size=a.batch, max_length=a.max_len, fp16=True)


def classify_table(tbl: pa.Table) -> dict[str, list]:
    cols: dict[str, list] = {"id": tbl.column("id").to_pylist()}
    texts = tbl.column("text").to_pylist()
    if zs is not None:
        heads = [head(t, a.head) for t in texts]
        for T in templates:
            pairs = [(h, hyp) for h in heads for hyp in ZS_TEMPLATES[T]]
            lo = torch.tensor(zs._entail_logits(pairs)).view(len(heads), 5)      # entailment log-odds per level
            p = torch.softmax(lo / ZS_TEMPERATURE, -1)
            for i in range(5):
                cols[f"{T}_lo{i + 1}"] = lo[:, i].tolist(); cols[f"{T}_p{i + 1}"] = p[:, i].tolist()
            cols[f"{T}_level"] = (lo.argmax(-1) + 1).tolist()
            cols[f"{T}_score"] = (p * torch.arange(1, 6, dtype=p.dtype)).sum(-1).tolist()
        if DEFAULT_ZS_TEMPLATE in templates:
            cols["zs_level"] = cols[f"{DEFAULT_ZS_TEMPLATE}_level"]; cols["zs_score"] = cols[f"{DEFAULT_ZS_TEMPLATE}_score"]
            cols["zs_level_cal"] = [level_from_thresholds(x, ZS_CAL_THRESHOLDS[DEFAULT_ZS_TEMPLATE]) for x in cols["zs_score"]]
    if "length" in methods:
        feats = [length_features(t) for t in texts]
        for k in LENGTH_FEATURES:
            cols[k] = [f[k] for f in feats]
        tokcol = next((c for c in ("n_tokens", "token_count") if c in tbl.schema.names), None)
        if tokcol:
            cols["n_tokens"] = tbl.column(tokcol).to_pylist()
        else:
            cols["n_tokens"] = [round(f["n_words"] * TOKENS_PER_WORD) for f in feats]      # estimate, see TOKENS_PER_WORD
        cols["len_level_fk"] = [level_from_thresholds(f["fk_grade"], LENGTH_THRESHOLDS["fk_grade"]) for f in feats]
        cols["len_level_sent"] = [level_from_thresholds(f["avg_sent_len"], LENGTH_THRESHOLDS["avg_sent_len"]) for f in feats]
        cols["len_level_max"] = [level_from_thresholds(f["max_sent_len"], LENGTH_THRESHOLDS["max_sent_len"]) for f in feats]
        cols["len_level_tok"] = [level_from_thresholds(n or 0, LENGTH_THRESHOLDS["n_tokens"]) for n in cols["n_tokens"]]
    if all(f in cols for f in COMBO["features"]):
        n = len(cols["id"])
        cols["difficulty_score"] = [combo_score({f: cols[f][i] for f in COMBO["features"]}) for i in range(n)]
        cols["difficulty_level"] = [level_from_thresholds(x, COMBO["thresholds"]) for x in cols["difficulty_score"]]
    return cols


def write(cols, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    pq.write_table(pa.table(cols), tmp); os.replace(tmp, path)


t0 = time.time()
if a.input:
    # single file: processed in chunks of --rows-per-chunk, one part file per chunk under <out>.parts/ (resumable),
    # then concatenated into --out
    pf = pq.ParquetFile(a.input)
    n_rows = pf.metadata.num_rows
    names = pf.schema_arrow.names
    parts_dir = a.out + ".parts"; os.makedirs(parts_dir, exist_ok=True)
    cols_in = ["id", "text"] + [c for c in ("token_count", "n_tokens") if c in names]
    parts, done = [], 0
    for ci, batch in enumerate(pf.iter_batches(batch_size=a.rows_per_chunk, columns=cols_in)):
        part = os.path.join(parts_dir, f"part_{ci:05d}.parquet")
        if os.path.exists(part) and pq.ParquetFile(part).metadata.num_rows == batch.num_rows:
            parts.append(part); done += batch.num_rows; continue
        cols = classify_table(pa.Table.from_batches([batch]))
        write(cols, part); parts.append(part); done += batch.num_rows
        el = time.time() - t0
        lv = cols.get("difficulty_level") or cols.get("zs_level_cal") or cols.get("len_level_fk")
        dist = " ".join(f"{k}:{sum(x == k for x in lv) / len(lv):.2f}" for k in range(1, 6))
        print(f"[chunk {ci + 1}] {done:,}/{n_rows:,} docs  levels {dist}  {done / el:.0f} docs/s  "
              f"ETA {(n_rows - done) / max(done / el, 1e-9) / 60:.0f} min", file=sys.stderr, flush=True)
    tbl = pa.concat_tables([pq.read_table(p) for p in parts])
    write({c: tbl.column(c) for c in tbl.schema.names}, a.out)
    print(f"{a.input}: {tbl.num_rows} docs -> {a.out} ({(time.time() - t0) / 60:.1f} min; parts kept in {parts_dir})", file=sys.stderr)
    sys.exit(0)

files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(a.corpus) if "/.git" not in dp for f in fs if f.endswith(".parquet"))
if a.limit_files:
    files = files[:a.limit_files]
total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
done = scored = skipped = 0
print(f"{len(files)} files, {total:,} docs, methods={sorted(methods)}, templates={templates}", file=sys.stderr, flush=True)
for fi, f in enumerate(files):
    rel = os.path.relpath(f, a.corpus); of = os.path.join(a.out, rel)
    n_in = pq.ParquetFile(f).metadata.num_rows
    if os.path.exists(of):
        try:
            if pq.ParquetFile(of).metadata.num_rows == n_in:
                skipped += 1; done += n_in; continue
        except Exception:
            pass
    names = pq.ParquetFile(f).schema_arrow.names
    tbl = pq.read_table(f, columns=["id", "text"] + [c for c in ("token_count",) if c in names])
    cols = classify_table(tbl)
    write(cols, of)
    done += n_in; scored += n_in
    el = time.time() - t0; rate = scored / el if el else 0
    lv = cols.get("difficulty_level") or cols.get("zs_level_cal") or cols.get("len_level_fk")
    dist = " ".join(f"{k}:{sum(x == k for x in lv) / len(lv):.2f}" for k in range(1, 6))
    print(f"[{fi + 1}/{len(files)}] {rel}: {n_in} docs  levels {dist}  cum {done:,}/{total:,}  {rate:.0f} docs/s  "
          f"ETA {(total - done) / max(rate, 1e-9) / 3600:.1f} h", file=sys.stderr, flush=True)
print(f"done: {scored:,} docs this run, {skipped} files skipped, {(time.time() - t0) / 3600:.2f} h", file=sys.stderr)
