"""Corpus builder: iterate sources -> relevance check -> download -> text extraction -> manifest.

Output layout (out_dir):
  raw/<source>/<slug>.(pdf|html|txt)     original files
  text/<source>/<slug>.txt               extracted, cleaned text
  manifest.jsonl                         one line per kept document (append-only => resumable)
  failed.jsonl                           download / extraction failures (retried with --retry-failed)
  report.md                              sizes and counts per source, kind, license
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from collections import Counter, defaultdict

from .extract import html_to_text, pdf_to_text, text_quality
from .fetch import Fetcher, FetchError
from .relevance import Relevance
from .sources import SOURCE_TYPES, slug


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class CorpusBuilder:
    def __init__(self, spec: dict, out_dir: str, max_gb: float | None, fetcher: Fetcher,
                 retry_failed: bool = False, log=print, max_tokens: int | None = None):
        self.spec, self.out, self.fx, self.log = spec, out_dir, fetcher, log
        self.max_bytes = int((max_gb if max_gb is not None else spec.get("max_gb", 1.0)) * 1e9)
        # Optional token accounting: `tokenizer:` in the spec (a HF tokenizer name or path) enables
        # `text_tokens` per document, `budget_tokens` per source and `max_tokens` for the whole corpus.
        self.tok = None
        if spec.get("tokenizer"):
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(spec["tokenizer"])
            self.tok.model_max_length = 10 ** 12          # whole documents, no "sequence too long" warnings
        mt = max_tokens if max_tokens is not None else spec.get("max_tokens")
        self.max_tokens = int(mt) if mt else None
        if (self.max_tokens or any(src.get("budget_tokens") for src in spec.get("sources", []))) and not self.tok:
            raise ValueError("token budgets (max_tokens / budget_tokens) need `tokenizer:` in the spec")
        self.rel = Relevance(spec.get("relevance"))
        self.manifest_path = os.path.join(out_dir, "manifest.jsonl")
        self.failed_path = os.path.join(out_dir, "failed.jsonl")
        os.makedirs(out_dir, exist_ok=True)
        self.docs = _read_jsonl(self.manifest_path)
        self.done = {d["id"] for d in self.docs}
        self.hashes = {d["sha1"] for d in self.docs if d.get("sha1")}
        self.titles = {self._norm_title(d.get("title", "")) for d in self.docs}
        self.failed = set() if retry_failed else {d["id"] for d in _read_jsonl(self.failed_path)}
        self.total_bytes = sum(d.get("bytes", 0) for d in self.docs)
        self.total_tokens = sum(d.get("text_tokens") or 0 for d in self.docs)

    def _count_tokens(self, text: str) -> int | None:
        return len(self.tok(text, add_special_tokens=False)["input_ids"]) if self.tok else None

    @staticmethod
    def _norm_title(t: str) -> str:
        return "".join(ch for ch in t.lower() if ch.isalnum())[:120]

    def _source_bytes(self, name):
        return sum(d.get("bytes", 0) for d in self.docs if d["source"] == name)

    def _source_tokens(self, name):
        return sum(d.get("text_tokens") or 0 for d in self.docs if d["source"] == name)

    # ------------------------------------------------------------------ #
    def run(self, only: set[str] | None = None, dry_run: bool = False):
        for src in self.spec["sources"]:
            name = src["name"]
            if only and name not in only:
                continue
            budget = int(float(src.get("budget_gb", 1e9)) * 1e9)
            budget_tok = int(src["budget_tokens"]) if src.get("budget_tokens") else None
            used, used_tok = self._source_bytes(name), self._source_tokens(name)
            self.log(f"\n== source {name} ({src['type']}) budget {budget / 1e9:.2f} GB"
                     + (f" / {budget_tok / 1e6:.2f} M tokens" if budget_tok else "")
                     + f", already {used / 1e6:.0f} MB" + (f" / {used_tok / 1e6:.2f} M tokens" if self.tok else ""))
            stats = Counter()
            it = SOURCE_TYPES[src["type"]](src, self.fx, self.log)
            for doc in it:
                if (used >= budget or self.total_bytes >= self.max_bytes
                        or (budget_tok and used_tok >= budget_tok)
                        or (self.max_tokens and self.total_tokens >= self.max_tokens)):
                    self.log(f"  budget reached for {name} (source {used / 1e6:.0f} MB / {used_tok / 1e6:.2f} M tokens, "
                             f"total {self.total_bytes / 1e6:.0f} MB / {self.total_tokens / 1e6:.2f} M tokens)")
                    break
                stats["candidates"] += 1
                if doc["id"] in self.done or doc["id"] in self.failed:
                    stats["already"] += 1
                    continue
                nt = self._norm_title(doc.get("title", ""))
                if nt and nt in self.titles and not doc.get("curated"):
                    stats["dup_title"] += 1
                    continue
                if not doc.get("curated"):
                    ok, score = self.rel.ok(doc.get("screen", doc.get("title", "")), src.get("threshold"))
                    doc["relevance"] = score
                    if not ok:
                        stats["excluded" if self.rel.mode == "exclude" else "irrelevant"] += 1
                        continue
                if dry_run:
                    if doc["id"] in self.done:
                        stats["already"] += 1
                        continue
                    self.done.add(doc["id"])          # in-memory only: de-duplicate across queries
                    self.titles.add(nt)
                    self.log(f"  [dry] {doc['id']:45s} rel={doc.get('relevance', '-')!s:6s} {doc['title'][:70]}")
                    stats["would_download"] += 1
                    continue
                try:
                    rec = self._fetch(src, doc)
                except (FetchError, RuntimeError, OSError, ValueError) as e:
                    stats["failed"] += 1
                    self._append(self.failed_path, {"id": doc["id"], "source": name, "url": doc.get("url"),
                                                    "error": str(e)[:300], "time": time.strftime("%F %T")})
                    self.failed.add(doc["id"])
                    self.log(f"  ! {doc['id']}: {str(e)[:120]}")
                    continue
                if rec is None:
                    stats["dup_content"] += 1
                    continue
                used += rec["bytes"]
                self.total_bytes += rec["bytes"]
                used_tok += rec.get("text_tokens") or 0
                self.total_tokens += rec.get("text_tokens") or 0
                stats["kept"] += 1
                self.docs.append(rec)
                self.done.add(rec["id"])
                self.hashes.add(rec["sha1"])
                self.titles.add(nt)
                self._append(self.manifest_path, rec)
                size = (f"{rec['text_tokens'] / 1e3:7.1f} kTok" if rec.get("text_tokens") is not None
                        else f"{rec['bytes'] / 1e6:6.1f} MB  {rec.get('pages') or '':>4} p")
                total = (f"{self.total_tokens / 1e6:.2f} M tokens" if self.tok else f"{self.total_bytes / 1e9:.2f} GB")
                self.log(f"  + {rec['id'][:45]:45s} {size}  {rec['title'][:60]}  [total {total}]")
            # Close the source generator now rather than at garbage collection: it logs its own statistics
            # in order, and a streamed dataset releases its readers (an abandoned parquet prefetch thread
            # otherwise can abort the interpreter at exit).
            if hasattr(it, "close"):
                it.close()
                gc.collect()
            self.log(f"  {name}: {dict(stats)}")
        self.write_report()

    # ------------------------------------------------------------------ #
    def _fetch(self, src: dict, doc: dict) -> dict | None:
        name = src["name"]
        base = slug(doc["id"].split(":", 1)[-1])
        raw_dir = os.path.join(self.out, "raw", name)
        txt_path = os.path.join(self.out, "text", name, base + ".txt")
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        max_file = int(float(src.get("max_file_mb", 60)) * 1e6)
        pages = 0
        if "text" in doc:                                   # already text (Wikipedia)
            raw_path = os.path.join(raw_dir, base + ".txt")
            os.makedirs(raw_dir, exist_ok=True)
            data = doc["text"].encode("utf-8")
            with open(raw_path, "wb") as f:
                f.write(data)
            n_bytes, text = len(data), doc["text"]
        elif doc.get("expect") == "html":
            raw_path = os.path.join(raw_dir, base)
            urls = [doc["url"]] + list(doc.get("pages") or [])
            texts, n_bytes, data = [], 0, b""
            for i, u in enumerate(urls):
                p = os.path.join(raw_path, f"p{i:02d}.html")
                info = self.fx.download(u, p, max_file, expect="html")
                n_bytes += info["bytes"]
                with open(p, "rb") as f:
                    b = f.read()
                data += b
                texts.append(html_to_text(b.decode("utf-8", "replace")))
            text = "\n\n".join(texts)
        else:
            raw_path = os.path.join(raw_dir, base + ".pdf")
            info = self.fx.download(doc["url"], raw_path, max_file, expect="pdf")
            n_bytes = info["bytes"]
            with open(raw_path, "rb") as f:
                data = f.read()
            text, pages = pdf_to_text(raw_path)
        sha1 = hashlib.sha1(data if "text" not in doc else doc["text"].encode()).hexdigest()
        if sha1 in self.hashes:                             # same file under another id / url
            if os.path.isfile(raw_path):
                os.remove(raw_path)
            return None
        q = text_quality(text)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        rec = {k: doc.get(k) for k in ("id", "kind", "title", "authors", "year", "license", "level", "url",
                                       "categories", "cited_by", "topic", "query", "note", "hf") if doc.get(k) is not None}
        rec.update({
            "source": name, "bytes": n_bytes, "sha1": sha1, "pages": pages,
            "raw_path": os.path.relpath(raw_path, self.out), "text_path": os.path.relpath(txt_path, self.out),
            "text_chars": q["chars"], "text_words": q["words"], "text_tokens": self._count_tokens(text),
            "alpha_ratio": q["alpha_ratio"],
            # letters < 50 % flags scanned PDFs / broken encodings; source code is symbol-heavy by nature
            "low_quality": q["chars"] < 2000 or (q["alpha_ratio"] < 0.5 and doc.get("kind") != "code"),
            "relevance": doc.get("relevance"), "abstract": (doc.get("abstract") or "")[:2000] or None,
            "fetched_at": time.strftime("%F %T"),
        })
        return rec

    @staticmethod
    def _append(path, rec):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ #
    def write_report(self):
        by_src, by_kind, by_lic = defaultdict(lambda: [0, 0, 0, 0]), Counter(), Counter()
        for d in self.docs:
            s = by_src[d["source"]]
            s[0] += 1
            s[1] += d.get("bytes", 0)
            s[2] += d.get("text_chars", 0)
            s[3] += d.get("text_tokens") or 0
            by_kind[d.get("kind", "?")] += 1
            lic = str(d.get("license", "unknown"))
            by_lic[lic if len(lic) < 60 else lic[:57] + "..."] += 1
        failed = _read_jsonl(self.failed_path)
        L = [f"# Corpus report: {self.spec.get('task', '')}", "",
             f"- documents: **{len(self.docs):,}**  |  raw size: **{self.total_bytes / 1e9:.2f} GB**  |  "
             f"text: **{sum(d.get('text_chars', 0) for d in self.docs) / 1e6:.1f} M chars**  |  "
             + (f"tokens: **{self.total_tokens / 1e6:.2f} M** ({self.spec.get('tokenizer')})  |  " if self.total_tokens else "")
             + f"failed: {len(failed)}  |  low-quality extractions: {sum(1 for d in self.docs if d.get('low_quality'))}",
             "", "| source | docs | raw MB | text M chars | M tokens |", "|---|---:|---:|---:|---:|"]
        L += [f"| {k} | {v[0]:,} | {v[1] / 1e6:,.0f} | {v[2] / 1e6:,.1f} | {v[3] / 1e6:,.2f} |" if v[3] else
              f"| {k} | {v[0]:,} | {v[1] / 1e6:,.0f} | {v[2] / 1e6:,.1f} | - |" for k, v in by_src.items()]
        L += ["", "| kind | docs |", "|---|---:|"] + [f"| {k} | {v:,} |" for k, v in by_kind.most_common()]
        L += ["", "| license | docs |", "|---|---:|"] + [f"| {k} | {v:,} |" for k, v in by_lic.most_common(25)]
        books = [d for d in self.docs if d.get("kind") in ("book", "lecture_notes")]
        if books:
            L += ["", "## Books & lecture notes", "", "| title | pages | MB | license |", "|---|---:|---:|---|"]
            L += [f"| {d['title']} | {d.get('pages') or ''} | {d['bytes'] / 1e6:.1f} | {d.get('license', '')} |"
                  for d in books]
        with open(os.path.join(self.out, "report.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(L) + "\n")
        self.log("\n" + "\n".join(L[:8 + len(by_src)]))

    def export_jsonl(self, path: str, include_low_quality: bool = False):
        """One JSON line per document with its full text, for RAG indexing or continued pre-training."""
        n = 0
        with open(path, "w", encoding="utf-8") as out:
            for d in self.docs:
                if d.get("low_quality") and not include_low_quality:
                    continue
                with open(os.path.join(self.out, d["text_path"]), encoding="utf-8") as f:
                    text = f.read()
                meta = {k: d.get(k) for k in ("id", "kind", "title", "authors", "year", "license", "level",
                                              "source", "url", "hf", "text_tokens")}
                out.write(json.dumps({**meta, "text": text}, ensure_ascii=False) + "\n")
                n += 1
        return n
