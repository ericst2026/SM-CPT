#!/usr/bin/env python3
"""The extra AI text found locally beyond cpt_data/corpus_full (2026-10-28): two parquet files with a `text` column,
to be decontaminated (cpt_trainer.decontaminate) and turned into a corpus folder (cpt_trainer.parquet_to_corpus).

    python -m cpt_trainer.build_ai_extra --out cpt_data/extra_ai
    python -m cpt_trainer.decontaminate cpt_data/extra_ai/*.parquet --out cpt_data/extra_ai_clean
    python -m cpt_trainer.parquet_to_corpus cpt_data/corpus_ai_extra/xtra_ai \\
        --add genesis_ml_prose=cpt_data/extra_ai_clean/genesis_ml_prose.parquet \\
        --add sft_ai_text=cpt_data/extra_ai_clean/sft_ai_text.parquet

The corpus is named xtra_ai so it sorts after the corpus_full names (ai, long, replay, stem):
cpt_trainer.data.split_documents draws every corpus's validation documents from one seeded generator in name
order, so those four keep exactly the train / val split they have without it.

genesis_ml_prose.parquet
    The synthetic genesis_ml shard (cpt_data/corpus/ml_clean, already decontaminated) without the documents that
    hold a multiple-choice question / answer letter or a User / Assistant dialogue: those are what made
    cpt_qwen06b_ml_rs_adaptive_v1 forget MMLU and favour letter A, and they would teach the answer format of
    the MCQ evaluations rather than knowledge. ~72 k documents, ~81 M tokens.
sft_ai_text.parquet
    Real AI / ML text from the SFT pool (data/split/ai_sft): arXiv ML abstracts and code papers, StackExchange
    ai / datascience / stats / cs / cstheory / robotics, ML Q&A collections. One document per item: question,
    blank line, answer. Items of the SFT validation set and of the evaluation sets are left out. ~49 M tokens.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENESIS = os.path.join(ROOT, "cpt_data", "corpus", "ml_clean", "train-00008-of-00020.parquet")
SFT_AI = os.path.join(ROOT, "data", "split", "ai_sft")
SFT_SOURCES = ["arxiv_ml_abstracts", "arxiv_dl_code", "se_ai", "se_datascience", "se_stats", "se_cs", "se_cstheory",
               "se_robotics", "ml_qa_collection", "ml_qa_mjphayes", "ml_qa_prsdm", "so_python_dsml", "simplewiki_ai",
               "ai_qa_harikrishnan", "ai_history_timeline", "ai_interview_kareem"]
# a stated answer / lettered option line / "option B", or a chat transcript
MCQ = re.compile(r"(?im)^\s*(\*\*)?\s*(correct answer|answer)\s*[:：]|^\s*[A-D][\.\)]\s+\S|\boption [A-D]\b")
DIALOGUE = re.compile(r"\*\*(User|Assistant)\*\*|^(User|Assistant):", re.M)


def held_out_ids() -> set[str]:
    """Ids of the SFT validation items and of every evaluation item (same id space as data/split)."""
    ids = set()
    paths = [os.path.join(ROOT, "data", "final", "sft_val.jsonl")] + \
        glob.glob(os.path.join(ROOT, "data", "final", "eval", "*.jsonl")) + \
        glob.glob(os.path.join(ROOT, "data", "split", "ai_eval", "*.jsonl"))
    for p in paths:
        with open(p, encoding="utf-8") as f:
            ids.update(json.loads(line).get("id") for line in f if line.strip())
    return ids


def genesis_prose(out: str) -> None:
    t = pq.read_table(GENESIS, columns=["text", "pipeline", "style"])
    keep = [i for i, x in enumerate(t.column("text").to_pylist()) if not MCQ.search(x) and not DIALOGUE.search(x)]
    sub = t.take(keep)
    pq.write_table(sub, os.path.join(out, "genesis_ml_prose.parquet"), compression="zstd")
    chars = sum(len(x) for x in sub.column("text").to_pylist())
    print(f"genesis_ml_prose: {len(keep):,} of {t.num_rows:,} documents, {chars / 1e6:.0f} M chars")


def sft_ai_text(out: str) -> None:
    held = held_out_ids()
    rows, skipped = [], 0
    for src in SFT_SOURCES:
        with open(os.path.join(SFT_AI, f"{src}.jsonl"), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("id") in held:
                    skipped += 1
                    continue
                q, a = (r.get("question") or "").strip(), (r.get("answer") or "").strip()
                if q and a:
                    rows.append({"id": r["id"], "text": f"{q}\n\n{a}", "source": src})
    pq.write_table(pa.Table.from_pylist(rows), os.path.join(out, "sft_ai_text.parquet"), compression="zstd")
    print(f"sft_ai_text: {len(rows):,} documents, {sum(len(r['text']) for r in rows) / 1e6:.0f} M chars "
          f"({skipped} held-out items left out)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "cpt_data", "extra_ai"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    genesis_prose(args.out)
    sft_ai_text(args.out)


if __name__ == "__main__":
    main()
