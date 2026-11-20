#!/usr/bin/env python3
"""Freeze the current training data + its metadata as a named version, so later mixes can be compared.

    python scripts/snapshot_data_version.py --version v0.1 --note "first mix (heuristic levels)"
    python scripts/snapshot_data_version.py --list
    python scripts/snapshot_data_version.py --compare v0.1 v0.2

data/versions/<version>/
    VERSION.json        when/what: stats of the final mix, split totals, per-source counts, checksums,
                        mix parameters, runs/evals that used this data
    final/              full copy of data/final (train/val/RL/eval files) - copied, never linked
    split/              hard links to data/split/<bucket>/*.jsonl (no extra disk; the pipeline always
                        replaces files atomically, so the links keep the old content)
    meta/               split report, per-source _done stats, collection *.stats.json
    code/               copy of the pipeline code that produced it (ai_qa_data/, scripts/)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSIONS = os.path.join(ROOT, "data", "versions")
CODE_FILES = ["ai_qa_data", "scripts/collect_datasets.py", "scripts/split_ai_data.py",
              "scripts/build_training_mix.py", "scripts/snapshot_data_version.py"]


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def snapshot(version: str, note: str):
    dst = os.path.join(VERSIONS, version)
    if os.path.exists(dst):
        raise SystemExit(f"{dst} already exists")
    os.makedirs(dst)
    # final: real copy (build_training_mix rewrites these files in place)
    shutil.copytree(os.path.join(ROOT, "data", "final"), os.path.join(dst, "final"))
    # split: hard links
    n_links = 0
    for p in glob.glob(os.path.join(ROOT, "data", "split", "*", "*.jsonl")):
        rel = os.path.relpath(p, os.path.join(ROOT, "data", "split"))
        q = os.path.join(dst, "split", rel)
        os.makedirs(os.path.dirname(q), exist_ok=True)
        os.link(p, q)
        n_links += 1
    # meta
    meta = os.path.join(dst, "meta")
    os.makedirs(os.path.join(meta, "split_done"), exist_ok=True)
    os.makedirs(os.path.join(meta, "collect_stats"), exist_ok=True)
    for f in ("report.json", "report.md"):
        p = os.path.join(ROOT, "data", "split", f)
        if os.path.exists(p):
            shutil.copy2(p, meta)
    for p in glob.glob(os.path.join(ROOT, "data", "split", "_done", "*.json")):
        shutil.copy2(p, os.path.join(meta, "split_done"))
    for p in glob.glob(os.path.join(ROOT, "data", "raw", "*.stats.json")):
        shutil.copy2(p, os.path.join(meta, "collect_stats"))
    # code
    for c in CODE_FILES:
        src = os.path.join(ROOT, c)
        out = os.path.join(dst, "code", c)
        if os.path.isdir(src):
            shutil.copytree(src, out, ignore=shutil.ignore_patterns("__pycache__"))
        elif os.path.exists(src):
            os.makedirs(os.path.dirname(out), exist_ok=True)
            shutil.copy2(src, out)
    # version record
    final = os.path.join(dst, "final")
    files = {}
    for p in sorted(glob.glob(os.path.join(final, "**", "*.jsonl"), recursive=True)):
        files[os.path.relpath(p, final)] = {"records": count_lines(p), "bytes": os.path.getsize(p), "sha256": sha256(p)}
    stats = json.load(open(os.path.join(final, "stats.json"))) if os.path.exists(os.path.join(final, "stats.json")) else {}
    split_report = json.load(open(os.path.join(meta, "report.json"))) if os.path.exists(os.path.join(meta, "report.json")) else []
    runs = []
    for rc in glob.glob(os.path.join(ROOT, "runs", "*", "run_config.json")):
        try:
            a = json.load(open(rc)).get("args", {})
            if any("data/final" in str(x) for x in (a.get("data") or [])):
                runs.append(os.path.basename(os.path.dirname(rc)))
        except Exception:  # noqa: BLE001
            pass
    rec = {
        "version": version, "created": time.strftime("%F %T"), "note": note,
        "final_files": files, "mix_stats": stats,
        "split_totals": {k: sum(r.get(k, 0) for r in split_report) for k in
                         ("in", "meta_accept", "kw_accept", "zs_accept", "kw_reject", "zs_reject",
                          "out_ai_sft", "out_ai_rl_math", "out_ai_rl_code", "out_ai_eval", "out_general_replay")},
        "split_per_source": {r["source"]: {k: r.get(k, 0) for k in ("in", "out_ai_sft", "out_ai_rl_math",
                                                                     "out_ai_rl_code", "out_ai_eval")}
                             for r in split_report},
        "split_files_linked": n_links,
        "runs_trained_on_data_final": sorted(runs),
        "eval_results": sorted(os.path.basename(os.path.dirname(p)) for p in
                               glob.glob(os.path.join(ROOT, "eval_results", "*", "summary.json"))),
    }
    with open(os.path.join(dst, "VERSION.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print(f"snapshot {version} -> {dst}  ({len(files)} final files, {n_links} split files linked)")


def list_versions():
    for v in sorted(glob.glob(os.path.join(VERSIONS, "*", "VERSION.json"))):
        r = json.load(open(v))
        s = r.get("mix_stats", {})
        print(f"{r['version']:8s} {r['created']}  sft={s.get('sft_total')} rl_math={s.get('rl_math')} "
              f"rl_code={s.get('rl_code')}  {r.get('note', '')}")


def compare(a: str, b: str):
    ra, rb = (json.load(open(os.path.join(VERSIONS, v, "VERSION.json"))) for v in (a, b))
    sa, sb = ra["mix_stats"], rb["mix_stats"]
    print(f"{'':28s} {a:>12s} {b:>12s}")
    for k in ("sft_total", "sft_with_reasoning", "replay", "rl_math", "rl_code", "rl_mcq", "eval_open_qa"):
        print(f"{k:28s} {str(sa.get(k)):>12s} {str(sb.get(k)):>12s}")
    print("\nSFT per level")
    for lv in sorted(set(sa.get("levels", {})) | set(sb.get("levels", {}))):
        print(f"  {lv:26s} {sa.get('levels', {}).get(lv, 0):>12,} {sb.get('levels', {}).get(lv, 0):>12,}")
    print("\nSFT per source (top 25 by new version)")
    src_a, src_b = sa.get("sources", {}), sb.get("sources", {})
    for s in sorted(set(src_a) | set(src_b), key=lambda x: -src_b.get(x, 0))[:25]:
        print(f"  {s:26s} {src_a.get(s, 0):>12,} {src_b.get(s, 0):>12,}")
    for k in ("level_fit_rate", "plain_level_fit_rate", "reasoning_share", "ai_sft_with_reasoning"):
        if k in sb or k in sa:
            print(f"\n{k}: {sa.get(k)} -> {sb.get(k)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version")
    ap.add_argument("--note", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    a = ap.parse_args()
    if a.list:
        list_versions()
    elif a.compare:
        compare(*a.compare)
    elif a.version:
        snapshot(a.version, a.note)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
