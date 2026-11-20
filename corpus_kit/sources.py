"""Corpus sources. Each source yields candidate documents (dicts); the runner downloads, extracts,
checks budget/duplicates and writes the manifest.

Candidate document fields:
  id          stable unique id ("arxiv:1706.03762", "book:d2l", "wiki:en:Backpropagation")
  kind        book | lecture_notes | paper | encyclopedia | ...
  title, authors (list), year, license, level (optional audience hint), source (source name)
  url         what to download (pdf/html)    -- or --   text: already-extracted text (wikipedia)
  pages       list of extra HTML page urls to concatenate (multi-page online books)
  expect      pdf | html | any
  screen      text used for the relevance check before downloading (title + abstract)
  curated     True -> skip relevance check (hand-picked items)
  hf          "dataset:config:split" for rows taken from a Hugging Face dataset
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter

from .extract import clean_text
from .fetch import Fetcher, FetchError

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
_ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf)/([^\s?#v]+(?:v\d+)?)", re.I)


def arxiv_base_id(s: str) -> str:
    s = s.strip().split("/abs/")[-1].split("/pdf/")[-1]
    s = s.replace(".pdf", "")
    return re.sub(r"v\d+$", "", s)


def slug(s: str, n: int = 80) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")[:n] or "doc"


# --------------------------------------------------------------------------- #
# 1. Curated URL list (books, lecture notes, online textbooks)
# --------------------------------------------------------------------------- #
def iter_url_list(cfg: dict, fx: Fetcher, log):
    for it in cfg.get("items", []):
        doc = {
            "id": f"{cfg.get('id_prefix', 'book')}:{it['id']}", "kind": it.get("kind", "book"),
            "title": it["title"], "authors": it.get("authors", []), "year": it.get("year"),
            "license": it.get("license", "unknown"), "level": it.get("level"),
            "expect": it.get("type", "pdf"), "curated": True, "pages": it.get("pages", []),
            "note": it.get("note"),
        }
        try:
            doc["url"] = resolve_url(it, fx)
        except FetchError as e:
            log(f"  ! cannot resolve {it['id']}: {e}")
            continue
        yield doc


def resolve_url(it: dict, fx: Fetcher) -> str:
    """url | resolve: {page, pattern} (first regex match on a landing page) |
    github_release: {repo, asset} (asset-name regex in the latest release)."""
    if it.get("url"):
        return it["url"]
    if it.get("github_release"):
        gr = it["github_release"]
        rel = fx.get_json(f"https://api.github.com/repos/{gr['repo']}/releases/latest")
        for a in rel.get("assets", []):
            if re.search(gr.get("asset", r"\.pdf$"), a["name"]):
                return a["browser_download_url"]
        raise FetchError(f"no asset matching {gr.get('asset')} in {gr['repo']}")
    if it.get("resolve"):
        rv = it["resolve"]
        page = fx.get_text(rv["page"])
        m = re.search(rv["pattern"], page)
        if not m:
            raise FetchError(f"pattern {rv['pattern']!r} not found on {rv['page']}")
        return urllib.parse.urljoin(rv["page"], m.group(1) if m.groups() else m.group(0))
    raise FetchError("item needs url, resolve or github_release")


# --------------------------------------------------------------------------- #
# 2. arXiv API (search queries; PDFs from the export.arxiv.org mirror for automated access)
# --------------------------------------------------------------------------- #
def iter_arxiv(cfg: dict, fx: Fetcher, log):
    page = int(cfg.get("page_size", 100))
    for q in cfg.get("queries", []):
        query, limit = q["query"], int(q.get("max", 100))
        sort = q.get("sort", cfg.get("sort", "relevance"))
        got = 0
        for start in range(0, limit, page):
            url = ("https://export.arxiv.org/api/query?" + urllib.parse.urlencode({
                "search_query": query, "start": start, "max_results": min(page, limit - start),
                "sortBy": sort, "sortOrder": "descending"}))
            try:
                root = ET.fromstring(fx.get_text(url))
            except (FetchError, ET.ParseError) as e:
                log(f"  ! arXiv query failed ({query}): {e}")
                break
            entries = root.findall(f"{ATOM}entry")
            if not entries:
                break
            for e in entries:
                aid = arxiv_base_id(e.findtext(f"{ATOM}id", ""))
                title = " ".join(e.findtext(f"{ATOM}title", "").split())
                abstract = " ".join(e.findtext(f"{ATOM}summary", "").split())
                prim = e.find(f"{ARXIV_NS}primary_category")
                lic = None
                for ln in e.findall(f"{ATOM}link"):
                    if ln.get("rel") == "license":
                        lic = ln.get("href")
                got += 1
                yield {
                    "id": f"arxiv:{aid}", "kind": q.get("kind", "paper"), "title": title,
                    "authors": [a.findtext(f"{ATOM}name", "") for a in e.findall(f"{ATOM}author")][:20],
                    "year": int(e.findtext(f"{ATOM}published", "0")[:4] or 0),
                    "license": lic or "arXiv (see paper; default arXiv non-exclusive license)",
                    "url": f"https://export.arxiv.org/pdf/{aid}", "expect": "pdf",
                    "screen": f"{title}\n{abstract}", "abstract": abstract,
                    "categories": [prim.get("term")] if prim is not None else [], "query": query,
                    "level": q.get("level"),
                }
            if len(entries) < min(page, limit - start):
                break
        log(f"  arXiv query {query!r}: {got} candidates")


# --------------------------------------------------------------------------- #
# 3. OpenAlex (citation-ranked open-access works; arXiv copy preferred when available)
# --------------------------------------------------------------------------- #
def _abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = {p: w for w, ps in inv.items() for p in ps}
    return " ".join(pos[i] for i in sorted(pos))


def iter_openalex(cfg: dict, fx: Fetcher, log):
    """Page-based paging (page=1..N, OpenAlex allows up to 10,000 results this way). Cursor paging now
    requires an API key; set OPENALEX_API_KEY to raise the free daily credit limit if needed."""
    import os
    limit = int(cfg.get("max", 200))
    per = min(200, int(cfg.get("per_page", 100)))
    got, page_no = 0, 1
    select = "id,display_name,publication_year,cited_by_count,authorships,best_oa_location,locations,abstract_inverted_index,primary_topic"
    key = os.environ.get("OPENALEX_API_KEY")
    while got < limit and page_no * per <= 10000:
        params = {"filter": cfg["filter"], "sort": cfg.get("sort", "cited_by_count:desc"), "per-page": per,
                  "page": page_no, "select": select}
        if key:
            params["api_key"] = key
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
        page_no += 1
        try:
            d = fx.get_json(url)
        except (FetchError, ValueError) as e:
            log(f"  ! OpenAlex failed: {e}")
            break
        if not d.get("results"):
            break
        for w in d.get("results", []):
            arx = None
            for loc in [w.get("best_oa_location")] + (w.get("locations") or []):
                for u in ((loc or {}).get("landing_page_url"), (loc or {}).get("pdf_url")):
                    m = _ARXIV_ID.search(u or "")
                    if m:
                        arx = arxiv_base_id(m.group(1))
                        break
                if arx:
                    break
            best = w.get("best_oa_location") or {}
            pdf = f"https://export.arxiv.org/pdf/{arx}" if arx else best.get("pdf_url")
            if not pdf:
                continue
            title = w.get("display_name") or ""
            abstract = _abstract(w.get("abstract_inverted_index"))
            got += 1
            yield {
                "id": f"arxiv:{arx}" if arx else f"openalex:{w['id'].rsplit('/', 1)[-1]}",
                "kind": "paper", "title": title, "year": w.get("publication_year"),
                "authors": [a.get("author", {}).get("display_name", "") for a in (w.get("authorships") or [])][:20],
                "license": best.get("license") or "unknown (open access; see source)",
                "url": pdf, "expect": "pdf", "screen": f"{title}\n{abstract}", "abstract": abstract,
                "cited_by": w.get("cited_by_count"), "openalex_id": w["id"],
                "topic": ((w.get("primary_topic") or {}).get("display_name")),
            }
            if got >= limit:
                break
    log(f"  OpenAlex: {got} candidates")


# --------------------------------------------------------------------------- #
# 4. Wikipedia category crawl (plain-text extracts via the MediaWiki API)
# --------------------------------------------------------------------------- #
def iter_wikipedia(cfg: dict, fx: Fetcher, log):
    lang = cfg.get("lang", "en")
    api = f"https://{lang}.wikipedia.org/w/api.php"
    depth_max = int(cfg.get("depth", 1))
    limit = int(cfg.get("max_pages", 500))
    seen_cat, titles = set(), []
    frontier = [(c if c.startswith("Category:") else f"Category:{c}", 0) for c in cfg.get("categories", [])]
    while frontier and len(titles) < limit * 3:
        cat, depth = frontier.pop(0)
        if cat in seen_cat:
            continue
        seen_cat.add(cat)
        cont = None
        while True:
            params = {"action": "query", "list": "categorymembers", "cmtitle": cat, "cmlimit": "500",
                      "cmtype": "page|subcat", "format": "json", "formatversion": "2"}
            if cont:
                params["cmcontinue"] = cont
            try:
                d = fx.get_json(api + "?" + urllib.parse.urlencode(params))
            except (FetchError, ValueError) as e:
                log(f"  ! wiki category {cat}: {e}")
                break
            for m in d.get("query", {}).get("categorymembers", []):
                if m["ns"] == 14 and depth < depth_max:
                    frontier.append((m["title"], depth + 1))
                elif m["ns"] == 0 and m["title"] not in titles:
                    titles.append(m["title"])
            cont = d.get("continue", {}).get("cmcontinue")
            if not cont:
                break
    log(f"  Wikipedia ({lang}): {len(titles)} candidate pages from {len(seen_cat)} categories")
    got = 0
    for t in titles:
        if got >= limit:
            break
        params = {"action": "query", "prop": "extracts", "explaintext": "1", "titles": t,
                  "format": "json", "formatversion": "2", "redirects": "1"}
        try:
            d = fx.get_json(api + "?" + urllib.parse.urlencode(params))
        except (FetchError, ValueError):
            continue
        pages = d.get("query", {}).get("pages", [])
        if not pages or not pages[0].get("extract"):
            continue
        p = pages[0]
        text = p["extract"]
        if len(text) < int(cfg.get("min_chars", 1500)):
            continue
        got += 1
        yield {
            "id": f"wiki:{lang}:{p['title']}", "kind": "encyclopedia", "title": p["title"],
            "license": "CC BY-SA 4.0 (Wikipedia)", "level": cfg.get("level"),
            "url": f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(p['title'].replace(' ', '_'))}",
            "text": text, "screen": text[:6000],
        }


# --------------------------------------------------------------------------- #
# 5. Hugging Face datasets (streaming): rows of an existing text corpus become documents
# --------------------------------------------------------------------------- #
_GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", re.I)
_GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.I)
_URN = re.compile(r"^<?urn:uuid:")


def _get(row: dict, path: str):
    """Field lookup with dotted paths; a JSON string on the way is parsed ("METADATA.title")."""
    v = row
    for part in path.split("."):
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except ValueError:
                return None
        if not isinstance(v, dict):
            return None
        v = v.get(part)
    return v


def _join_fields(row: dict, spec) -> str:
    fields = [spec] if isinstance(spec, str) else list(spec or [])
    return "/".join(str(_get(row, f) or "") for f in fields).strip("/")


def _row_ok(row: dict, where: dict | None) -> bool:
    """where: {field: {eq: x | in: [..] | min: n | max: n}} - all conditions must hold."""
    for field, cond in (where or {}).items():
        v = _get(row, field)
        if v is None:
            return False
        if "eq" in cond and v != cond["eq"]:
            return False
        if "in" in cond and v not in cond["in"]:
            return False
        if "min" in cond or "max" in cond:
            try:
                x = float(v)
            except (TypeError, ValueError):
                return False
            if ("min" in cond and x < float(cond["min"])) or ("max" in cond and x > float(cond["max"])):
                return False
    return True


def _truncate(text: str, n: int) -> str:
    """Keep the first ~n characters, cutting at a paragraph (else line) boundary."""
    if len(text) <= n:
        return text
    cut = text.rfind("\n\n", int(n * 0.6), n)
    if cut < 0:
        cut = text.rfind("\n", int(n * 0.6), n)
    return text[:cut if cut > 0 else n].rstrip()


def _strip_gutenberg(text: str) -> str:
    m = _GUTENBERG_START.search(text)
    if m:
        text = text[m.end():]
    m = _GUTENBERG_END.search(text)
    if m:
        text = text[:m.start()]
    return text


def _row_id(row: dict, id_field, text: str) -> str:
    val = _join_fields(row, id_field) if id_field else ""
    if not val:
        val = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16]
    val = _URN.sub("", val).strip("<>")
    if len(val) > 60:                                     # keep file names short but unique
        val = val[:48] + "-" + hashlib.sha1(val.encode()).hexdigest()[:8]
    return val


def iter_hf(cfg: dict, fx: Fetcher, log):
    """Stream rows of a Hugging Face dataset (nothing is downloaded whole).

    cfg: dataset, config, split (train), data_dir/data_files/revision (optional), text_field (text),
         id_field / title_field / url_field (a field name, a dotted path into a JSON field, or a list of
         fields joined with "/"), where ({field: {eq|in|min|max}}), min_chars (2000), max_chars (skip
         longer rows), truncate_chars (cut long rows at a paragraph boundary), strip_gutenberg, clean
         (PDF/HTML-style whitespace clean-up; default true except for kind: code),
         shuffle ({seed, buffer, shards=1}: random shard order + buffer shuffle; `shards` > 1 reads several
         shards at once, which costs bandwidth and memory for datasets with big rows), skip, max_rows,
         license, kind, level, label (provenance string recorded in the manifest instead of dataset:config:split).
    Needs the `datasets` package; HF_TOKEN (from .env) raises the Hub rate limits.
    """
    from datasets import load_dataset  # lazy: only this source type needs it

    name, dataset, config, split = cfg["name"], cfg["dataset"], cfg.get("config"), cfg.get("split", "train")
    kw = {k: cfg[k] for k in ("data_dir", "data_files", "revision") if cfg.get(k)}
    prov = cfg.get("label") or (f"{dataset}:{config or 'default'}:{split}"
                                + (f" {kw['data_files']}" if kw.get("data_files") else ""))
    ds = load_dataset(dataset, config, split=split, streaming=True, **kw)
    sh = cfg.get("shuffle")
    if sh:   # random shard order + a reservoir of `buffer` rows; one shard open at a time unless `shards` says otherwise
        ds = ds.shuffle(seed=int(sh.get("seed", 0)), buffer_size=int(sh.get("buffer", 1000)),
                        max_buffer_input_shards=int(sh.get("shards", 1)))
    if cfg.get("skip"):
        ds = ds.skip(int(cfg["skip"]))
    text_field, id_field = cfg.get("text_field", "text"), cfg.get("id_field")
    title_field, url_field = cfg.get("title_field", "title"), cfg.get("url_field", "url")
    min_chars, max_chars = int(cfg.get("min_chars", 2000)), cfg.get("max_chars")
    trunc, where, max_rows = cfg.get("truncate_chars"), cfg.get("where"), int(cfg.get("max_rows", 10 ** 9))
    stats = Counter()
    try:
        for n, row in enumerate(ds):
            if n >= max_rows:
                break
            stats["rows"] += 1
            if not _row_ok(row, where):
                stats["where"] += 1
                continue
            text = _get(row, text_field) or ""
            if cfg.get("strip_gutenberg"):
                text = _strip_gutenberg(text)
            if len(text) < min_chars:
                stats["short"] += 1
                continue
            if max_chars and len(text) > int(max_chars):
                stats["long"] += 1
                continue
            if trunc and len(text) > int(trunc):
                stats["truncated"] += 1
                text = _truncate(text, int(trunc))
            if cfg.get("clean", cfg.get("kind") != "code"):   # clean_text collapses indentation: never for code
                text = clean_text(text)
            else:
                text = text.replace("\r", "").strip()
            rid = _row_id(row, id_field, text)
            url = _get(row, url_field) if isinstance(url_field, str) else None
            yield {
                "id": f"{cfg.get('id_prefix', name)}:{rid}", "kind": cfg.get("kind", "web"),
                "title": _join_fields(row, title_field) or rid, "license": cfg.get("license", "see dataset card"),
                "level": cfg.get("level"), "url": url or f"https://huggingface.co/datasets/{dataset}",
                "text": text, "screen": text[:6000], "hf": prov,
            }

    finally:   # also when the runner stops the stream early (budget reached)
        log(f"  HF {prov}: {dict(stats)}")


SOURCE_TYPES = {"urls": iter_url_list, "arxiv": iter_arxiv, "openalex": iter_openalex, "wikipedia": iter_wikipedia,
                "hf": iter_hf}
