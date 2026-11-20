#!/usr/bin/env python3
"""Random document sample of a corpus folder, up to a token budget -> a new corpus folder that points at the
original text files (no copies).

    python -m cpt_trainer.sample_corpus cpt_data/corpus_full/stem cpt_data/corpus_mix100m/stem --tokens 6.8e6 \\
        --exclude-docs configs/corpus/corpus_full_exclude_noisy_long.txt

Documents are drawn uniformly at random (seeded) and kept until the budget is reached, so every source gets a
share of the sample proportional to its share of the corpus tokens. Tokens are counted with the base model's
tokenizer (EOS included, as cpt_trainer.data adds one per document). Documents flagged low_quality or listed in
--exclude-docs ("<corpus>/<id>" lines) are never picked. The output folder has the usual manifest.jsonl, with
text_path rewritten relative to it, plus selection.json (per-source docs / tokens of the sample vs the corpus).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_qa_train.common import load_tokenizer  # noqa: E402
from cpt_trainer.data import read_manifest  # noqa: E402

DEFAULT_TOKENIZER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "Qwen3-0.6B-Base")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="corpus folder (manifest.jsonl + text/)")
    ap.add_argument("out", help="output corpus folder; its name is the corpus name in training (e.g. .../stem)")
    ap.add_argument("--tokens", type=float, required=True, help="token budget, e.g. 6.8e6")
    ap.add_argument("--exclude-docs", default=None, help="'<corpus>/<id>' lines (# comments) never picked")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src, out = os.path.abspath(args.src), os.path.abspath(args.out)
    corpus = os.path.basename(os.path.normpath(src))
    if os.path.basename(os.path.normpath(out)) != corpus:
        print(f"! output name {os.path.basename(out)!r} differs from the source corpus {corpus!r}: "
              f"--exclude-docs keys and validation-loss names use the folder name")
    exclude = set()
    if args.exclude_docs:
        with open(args.exclude_docs, encoding="utf-8") as f:
            exclude = {k for k in (line.split("#", 1)[0].strip() for line in f) if k}
    docs = [d for d in read_manifest(src)
            if not d.get("low_quality") and f"{corpus}/{d['id']}" not in exclude
            and os.path.isfile(os.path.join(src, d["text_path"]))]
    random.Random(args.seed).shuffle(docs)
    tok = load_tokenizer(args.tokenizer)

    t0, budget, total = time.time(), int(args.tokens), 0
    picked, by_src = [], defaultdict(lambda: [0, 0])
    for d in docs:
        if total >= budget:
            break
        with open(os.path.join(src, d["text_path"]), encoding="utf-8") as f:
            n = len(tok(f.read(), add_special_tokens=False)["input_ids"]) + 1  # + EOS
        picked.append({**d, "text_path": os.path.relpath(os.path.join(src, d["text_path"]), out), "qwen_tokens": n})
        total += n
        by_src[d["source"]][0] += 1
        by_src[d["source"]][1] += n

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "manifest.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in picked)
    chars = defaultdict(int)  # corpus share per source by characters (no full tokenisation of the source)
    for d in docs:
        chars[d["source"]] += d.get("text_chars") or 0
    all_chars = sum(chars.values()) or 1
    stats = {"src": os.path.relpath(src, out), "seed": args.seed, "budget_tokens": budget, "tokens": total,
             "docs": len(picked), "candidate_docs": len(docs), "excluded": len(exclude),
             "tokenizer": os.path.basename(os.path.normpath(args.tokenizer)), "seconds": round(time.time() - t0, 1),
             "sources": {s: {"docs": v[0], "tokens": v[1], "share": round(v[1] / max(1, total), 4),
                             "corpus_char_share": round(chars[s] / all_chars, 4)}
                         for s, v in sorted(by_src.items(), key=lambda kv: -kv[1][1])}}
    with open(os.path.join(out, "selection.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"{corpus}: {len(picked):,} of {len(docs):,} docs, {total / 1e6:.2f} M tokens -> {out}")
    for s, v in stats["sources"].items():
        print(f"  {s:24s} {v['docs']:6,d} docs {v['tokens'] / 1e6:6.2f} M  sample {v['share']:.1%}  corpus {v['corpus_char_share']:.1%}")


if __name__ == "__main__":
    main()
