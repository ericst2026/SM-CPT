"""Text extraction: PDF (poppler `pdftotext`) and HTML (stdlib parser), plus light clean-up."""
from __future__ import annotations

import html
import re
import shutil
import subprocess
from html.parser import HTMLParser

_SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "noscript", "svg", "form", "button"}
_BLOCK_TAGS = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "pre", "blockquote", "section",
               "article", "table", "ul", "ol", "dd", "dt", "figcaption"}


class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self.skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self.skip:
            self.skip -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_to_text(raw: str) -> str:
    p = _TextParser()
    p.feed(raw)
    return clean_text(html.unescape("".join(p.parts)))


def pdf_to_text(path: str, timeout: int = 600) -> tuple[str, int]:
    """Returns (text, n_pages). Requires poppler-utils (`pdftotext`, `pdfinfo`)."""
    if not shutil.which("pdftotext"):
        raise RuntimeError("pdftotext not found: install poppler-utils")
    r = subprocess.run(["pdftotext", "-enc", "UTF-8", "-nopgbrk", path, "-"], capture_output=True, timeout=timeout)
    text = r.stdout.decode("utf-8", "replace")
    pages = 0
    if shutil.which("pdfinfo"):
        info = subprocess.run(["pdfinfo", path], capture_output=True, timeout=60).stdout.decode("utf-8", "replace")
        m = re.search(r"^Pages:\s+(\d+)", info, re.M)
        pages = int(m.group(1)) if m else 0
    return clean_text(text), pages


_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_NL = re.compile(r"\n{3,}")
_SPACES = re.compile(r"[ \t ]{2,}")


def clean_text(text: str) -> str:
    text = text.replace("\r", "")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)          # re-join words hyphenated across lines
    text = "\n".join(ln.rstrip() for ln in text.split("\n"))
    text = _SPACES.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def text_quality(text: str) -> dict:
    """Cheap signals to spot bad extractions (scanned PDFs, garbage encodings)."""
    n = len(text)
    if not n:
        return {"chars": 0, "alpha_ratio": 0.0, "words": 0}
    alpha = sum(c.isalpha() for c in text[:200000]) / min(n, 200000)
    return {"chars": n, "alpha_ratio": round(alpha, 3), "words": len(text.split())}
