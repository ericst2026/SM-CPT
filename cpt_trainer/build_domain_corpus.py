#!/usr/bin/env python3
"""Build a domain-knowledge corpus folder (manifest.jsonl + text/<source>/<id>.txt) for cpt_trainer from
parquet files plus keyword-filtered documents of existing corpus folders, with part of the tokens grouped
into long, topic-coherent documents.

    python -m cpt_trainer.build_domain_corpus cpt_data/corpus_ml/ai \
        --parquet genesis_ml=cpt_data/corpus/ml_clean/train-00008-of-00020.parquet \
        --filter-corpus cpt_data/corpus_full/ai --min-hits 2 --max-tokens 600e6 --long-frac 0.3

1. Parquet rows (a `text` column) are all kept, as long as they fit --max-tokens (else a random sample).
2. Documents of every --filter-corpus are kept when explicit ML/AI keywords ("machine learning", "neural
   network", case-sensitive "AI" / "ML", "LLM", "gradient descent", ...) occur at least --min-hits times in
   total. Kept documents fill the rest of the token budget densest first (keyword hits per 1000 words).
   Their text files are hard-linked, not copied.
3. --long-frac of the tokens (a seeded random choice of documents) is concatenated into long documents of
   --long-min..--long-max tokens: documents are grouped by their dominant ML topic (reinforcement learning,
   NLP/LLMs, vision, ...) and joined with blank lines inside a group. cpt_trainer's varlen packing masks
   attention per document, so a long document is attended as one context of 16-24 k tokens instead of
   being cut into independent pieces. Group leftovers too short to reach --long-min stay single documents.

Token counts are estimated from characters with a chars/token ratio measured per source on a sample
(Qwen3 tokenizer); the trainer reports the exact counts. Writes build_stats.json and report.md.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpt_trainer.data import read_manifest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------- #
# Keywords
# --------------------------------------------------------------------------- #
# Explicit ML/AI terms. Ambiguous ones are left out ("CNN" = the news network, "transformer" = electrical,
# "algorithm", "model", "regression" alone) or made specific ("convolutional neural", "transformer model").
ML_PHRASES = [
    "machine learning", "artificial intelligence", "deep learning", "neural networks?", "neural nets?",
    "reinforcement learning", "supervised learning", "unsupervised learning", "semi-supervised", "self-supervised",
    "large language models?", "language models?", "transformer (?:models?|architectures?)", "attention mechanisms?",
    "natural language processing", "computer vision", "gradient descent", "backpropagation", "back-propagation",
    "convolutional neural", "recurrent neural", "generative adversarial", "diffusion models?", "chatbots?",
    "chatgpt", "data science", "data scientists?", "training data", "training sets?", "overfitting",
    "underfitting", "hyperparameters?", "feature engineering", "logistic regression", "linear regression",
    "decision trees?", "random forests?", "support vector machines?", "k-means", "clustering algorithms?",
    "loss functions?", "activation functions?", "word embeddings?", "fine-tun(?:e|ed|ing)", "pre-?train(?:ed|ing)",
    "pytorch", "tensorflow", "scikit-learn", "keras", "hugging ?face", "openai", "deepmind", "speech recognition",
    "image recognition", "facial recognition", "predictive models?", "classification models?", "bayesian inference",
    "markov decision", "q-learning", "policy gradients?", "dimensionality reduction", "principal component analysis",
    "cross-validation", "bias-variance", "learning rates?", "stochastic gradient", "autoencoders?",
]
ML_ACRONYMS = ["AI", "A\\.I\\.", "AGI", "LLMs?", "NLP", "GPT-?\\d?", "BERT", "GANs?", "RNNs?", "LSTMs?", "SVMs?"]
_PHRASES = re.compile(r"\b(?:" + "|".join(ML_PHRASES) + r")\b", re.IGNORECASE)
_ACRONYMS = re.compile(r"\b(?:" + "|".join(ML_ACRONYMS) + r")(?![\w.])")
_ML = re.compile(r"(?<![\d.])(?<!\d )\bML\b")          # not "5 ML" (millilitres)
_INSEMINATION = re.compile(r"insemination", re.IGNORECASE)  # "AI" = artificial insemination
_WORD = re.compile(r"\w+")


def ml_hits(text: str) -> int:
    n = len(_PHRASES.findall(text)) + len(_ML.findall(text))
    if not _INSEMINATION.search(text):
        n += len(_ACRONYMS.findall(text))
    return n


# Dominant topic per document, for grouping documents into long ones.
TOPICS = {
    "reinforcement_learning": r"reinforcement learning|reward|policy gradient|q-learning|markov decision|agents? learn",
    "nlp_llm": r"language models?|llms?|natural language|nlp|transformers?|tokens?|tokeniz\w*|attention|chatgpt|gpt|bert|text generation",
    "vision": r"computer vision|images?|convolutional|object detection|segmentation|pixels?|cnns?",
    "deep_learning": r"neural networks?|deep learning|backpropagation|activation functions?|layers?|neurons?|weights",
    "classical_supervised": r"regression|classification|classifiers?|decision trees?|random forests?|support vector|svms?|naive bayes|labels?",
    "unsupervised": r"clustering|k-means|dimensionality reduction|pca|principal component|anomaly detection|unsupervised",
    "probabilistic": r"bayesian|probabilit\w*|priors?|posteriors?|likelihood|distributions?|stochastic|markov",
    "optimization_training": r"gradient descent|optimi[sz]\w*|learning rates?|loss functions?|convergence|hyperparameters?|regulari[sz]\w*",
    "evaluation": r"overfitting|cross-validation|precision|recall|accuracy|bias-variance|validation sets?|test sets?|metrics?",
    "generative": r"generative|gans?|diffusion|autoencoders?|vaes?|synthetic data",
    "data_engineering": r"datasets?|data preprocessing|feature engineering|data cleaning|pipelines?|data science|big data",
    "ai_society": r"ethic\w*|fairness|privacy|regulation|jobs|society|governance|responsible ai|safety",
    "robotics_agents": r"robots?|robotics|autonomous|self-driving|planning|multi-agent",
}
_TOPICS = {k: re.compile(r"\b(?:" + v + r")\b", re.IGNORECASE) for k, v in TOPICS.items()}


def topic_of(text: str) -> str:
    sample = text[:20000]
    counts = {k: len(p.findall(sample)) for k, p in _TOPICS.items()}
    best = max(counts, key=counts.get)
    return best if counts[best] else "general"


def _scan_files(paths: list[str]) -> list[tuple[int, int, int, str]]:
    out = []
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as f:
            t = f.read()
        out.append((ml_hits(t), len(_WORD.findall(t)), len(t), topic_of(t)))
    return out


def _scan_texts(texts: list[str]) -> list[tuple[int, int, int, str]]:
    return [(0, 0, len(t), topic_of(t)) for t in texts]


def chunks(xs: list, n: int) -> list[list]:
    return [xs[i:i + n] for i in range(0, len(xs), n)]


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="corpus folder to create (its basename is the corpus name in the trainer)")
    ap.add_argument("--parquet", action="append", default=[], metavar="SOURCE=PATH", help="parquet with a text column")
    ap.add_argument("--filter-corpus", action="append", default=[], help="corpus folder to keyword-filter")
    ap.add_argument("--min-hits", type=int, default=2, help="minimum total ML/AI keyword occurrences")
    ap.add_argument("--max-tokens", type=float, default=600e6, help="token budget of the whole corpus")
    ap.add_argument("--long-frac", type=float, default=0.3, help="share of tokens grouped into long documents")
    ap.add_argument("--long-min", type=int, default=16384, help="min tokens of a grouped long document")
    ap.add_argument("--long-max", type=int, default=24576, help="max tokens of a grouped long document")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "models", "Qwen3-0.6B-Base"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    if os.path.exists(os.path.join(args.out_dir, "manifest.jsonl")):
        sys.exit(f"{args.out_dir} already has a manifest.jsonl; remove the folder to rebuild it")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rng = random.Random(args.seed)

    def ratio(texts: list[str]) -> float:
        texts = [t for t in texts if t][:1000]
        n_tok = sum(len(x) for x in tok(texts, add_special_tokens=False)["input_ids"])
        return sum(len(t) for t in texts) / max(1, n_tok)

    docs: list[dict] = []  # {source, text | path, chars, tokens, topic, hits, words, origin}
    ex = ProcessPoolExecutor(args.workers)

    # ---- 1. parquet rows ----
    pq_stats = {}
    for spec in args.parquet:
        source, _, path = spec.partition("=")
        texts = [t for t in pq.read_table(path, columns=["text"]).column("text").to_pylist() if t and t.strip()]
        r = ratio(rng.sample(texts, min(1000, len(texts))))
        scanned = [x for part in ex.map(_scan_texts, chunks(texts, 2000)) for x in part]
        for i, (t, (_, _, chars, topic)) in enumerate(zip(texts, scanned)):
            docs.append({"source": source, "text": t, "chars": chars, "tokens": chars / r, "topic": topic,
                         "origin": f"{os.path.basename(path)}#{i}"})
        pq_stats[source] = {"path": os.path.relpath(path, ROOT), "rows": len(texts), "chars_per_token": round(r, 3)}
        print(f"{source}: {len(texts):,} rows, ~{sum(d['tokens'] for d in docs if d['source'] == source) / 1e6:.1f} M "
              f"tokens ({r:.2f} chars/token)", flush=True)
    parquet_tokens = sum(d["tokens"] for d in docs)
    if parquet_tokens > args.max_tokens:
        rng.shuffle(docs)
        kept, acc = [], 0.0
        for d in docs:
            if acc + d["tokens"] > args.max_tokens:
                continue
            kept.append(d)
            acc += d["tokens"]
        print(f"parquet rows exceed the budget: sampled {len(kept):,} of {len(docs):,}", flush=True)
        docs, parquet_tokens = kept, acc

    # ---- 2. keyword-filtered corpus documents ----
    filt_stats = {}
    budget = args.max_tokens - parquet_tokens
    cands = []
    for cdir in args.filter_corpus:
        man = [d for d in read_manifest(cdir) if not d.get("low_quality")]
        paths = [os.path.join(cdir, d["text_path"]) for d in man]
        print(f"scanning {len(paths):,} documents of {cdir} ...", flush=True)
        scanned = [x for part in ex.map(_scan_files, chunks(paths, 500)) for x in part]
        by_src_ratio = {}
        for src in sorted({d["source"] for d in man}):
            sp = [p for p, d in zip(paths, man) if d["source"] == src]
            sample = []
            for p in rng.sample(sp, min(300, len(sp))):
                with open(p, encoding="utf-8", errors="replace") as f:
                    sample.append(f.read())
            by_src_ratio[src] = ratio(sample)
        hist = Counter()
        n_pass = 0
        for d, p, (hits, words, chars, topic) in zip(man, paths, scanned):
            hist[min(hits, 10)] += 1
            if hits >= args.min_hits:
                n_pass += 1
                cands.append({"source": d["source"], "path": p, "chars": chars, "tokens": chars / by_src_ratio[d["source"]],
                              "topic": topic, "hits": hits, "words": words, "origin": f"{os.path.basename(cdir)}:{d['id']}",
                              "density": 1000 * hits / max(words, 1)})
        filt_stats[cdir] = {"docs": len(man), "passed": n_pass, "hits_histogram(10=10+)": dict(sorted(hist.items())),
                            "chars_per_token": {k: round(v, 3) for k, v in by_src_ratio.items()}}
        print(f"  {n_pass:,} of {len(man):,} documents have >= {args.min_hits} ML/AI keyword hits", flush=True)
    ex.shutdown()
    cands.sort(key=lambda d: -d["density"])
    pass_tokens = sum(d["tokens"] for d in cands)
    taken, acc = [], 0.0
    for d in cands:
        if acc + d["tokens"] > budget:
            break
        taken.append(d)
        acc += d["tokens"]
    min_density = taken[-1]["density"] if taken else None
    print(f"filtered documents: {len(cands):,} pass (~{pass_tokens / 1e6:.1f} M tokens); {len(taken):,} fit the remaining "
          f"budget of ~{budget / 1e6:.1f} M tokens (density >= {min_density and round(min_density, 2)} hits / 1000 words)",
          flush=True)
    docs += taken
    total = sum(d["tokens"] for d in docs)

    # ---- 3. long documents grouped by topic ----
    order = list(range(len(docs)))
    rng.shuffle(order)
    chosen, acc = [], 0.0
    for i in order:
        if acc >= args.long_frac * total:
            break
        if docs[i]["tokens"] < args.long_min:  # already-long documents need no grouping
            chosen.append(i)
            acc += docs[i]["tokens"]
    by_topic = defaultdict(list)
    for i in chosen:
        by_topic[docs[i]["topic"]].append(i)
    groups, grouped = [], set()
    for topic, idx in sorted(by_topic.items()):
        cur, cur_tok, target = [], 0.0, rng.uniform(args.long_min * 1.05, args.long_max)
        for i in idx:
            cur.append(i)
            cur_tok += docs[i]["tokens"]
            if cur_tok >= target:
                groups.append((topic, cur))
                grouped.update(cur)
                cur, cur_tok, target = [], 0.0, rng.uniform(args.long_min * 1.05, args.long_max)
    long_tokens = sum(docs[i]["tokens"] for _, g in groups for i in g)
    print(f"long documents: {len(groups):,} of ~{long_tokens / max(1, len(groups)) / 1e3:.1f} k tokens from "
          f"{len(grouped):,} documents = {100 * long_tokens / total:.1f} % of ~{total / 1e6:.1f} M tokens", flush=True)

    # ---- 4. write the corpus folder ----
    def text_of(d: dict) -> str:
        if "text" in d:
            return d["text"]
        with open(d["path"], encoding="utf-8", errors="replace") as f:
            return f.read()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []
    for n, d in enumerate(docs):
        if n in grouped:
            continue
        rel = os.path.join("text", d["source"], f"{n:07d}.txt")
        dst = os.path.join(args.out_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if "path" in d:
            try:
                os.link(d["path"], dst)
            except OSError:
                with open(dst, "w", encoding="utf-8") as f:
                    f.write(text_of(d))
        else:
            with open(dst, "w", encoding="utf-8") as f:
                f.write(d["text"])
        manifest.append({"id": f"{d['source']}:{n:07d}", "source": d["source"], "kind": "document", "text_path": rel,
                         "bytes": os.path.getsize(dst), "tokens_est": round(d["tokens"]), "topic": d["topic"],
                         "ml_hits": d.get("hits"), "origin": d["origin"], "low_quality": False})
    for g, (topic, idx) in enumerate(groups):
        rel = os.path.join("text", "long_grouped", f"{topic}_{g:05d}.txt")
        dst = os.path.join(args.out_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            f.write("\n\n\n".join(text_of(docs[i]).strip() for i in idx))
        manifest.append({"id": f"long_grouped:{topic}_{g:05d}", "source": "long_grouped", "kind": "grouped_long",
                         "text_path": rel, "bytes": os.path.getsize(dst),
                         "tokens_est": round(sum(docs[i]["tokens"] for i in idx)), "topic": topic,
                         "members": [docs[i]["origin"] for i in idx],
                         "member_sources": dict(Counter(docs[i]["source"] for i in idx)), "low_quality": False})
    with open(os.path.join(args.out_dir, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    src_tok = defaultdict(float)
    for d in docs:
        src_tok[d["source"]] += d["tokens"]
    stats = {"args": vars(args), "parquet": pq_stats, "filtered": filt_stats,
             "filtered_selected": {"docs": len(taken), "tokens_est": round(acc), "min_density": min_density},
             "tokens_est_by_source": {k: round(v) for k, v in src_tok.items()}, "tokens_est_total": round(total),
             "long": {"docs": len(groups), "member_docs": len(grouped), "tokens_est": round(long_tokens),
                      "share": round(long_tokens / total, 4),
                      "by_topic": dict(Counter(t for t, _ in groups).most_common())},
             "topics_all_docs": dict(Counter(d["topic"] for d in docs).most_common()),
             "files": len(manifest), "seconds": round(time.time() - t0)}
    with open(os.path.join(args.out_dir, "build_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    lines = [f"# Domain corpus `{os.path.basename(os.path.normpath(args.out_dir))}`", "",
             f"~{total / 1e6:.1f} M tokens (estimated; the trainer logs exact counts) in {len(manifest):,} files.", "",
             "| source | ~M tokens |", "|---|---:|"]
    lines += [f"| {k} | {v / 1e6:.1f} |" for k, v in sorted(src_tok.items(), key=lambda x: -x[1])]
    lines += ["", f"Long documents: {len(groups):,} grouped files ({long_tokens / 1e6:.1f} M tokens, "
              f"{100 * long_tokens / total:.1f} %), {args.long_min}-{args.long_max} tokens each, grouped by topic:", ""]
    lines += [f"- {t}: {n}" for t, n in Counter(t for t, _ in groups).most_common()]
    lines += ["", "Keyword filter (>= %d hits):" % args.min_hits, ""]
    for c, s in filt_stats.items():
        lines.append(f"- `{os.path.relpath(c, ROOT)}`: {s['passed']:,} of {s['docs']:,} documents pass; "
                     f"{len(taken):,} selected by keyword density (>= {min_density and round(min_density, 2)} hits / 1000 words)")
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"done in {time.time() - t0:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
