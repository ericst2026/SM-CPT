"""Polite HTTP client (standard library only).

* per-host minimum interval between requests (arXiv: 3 s, Wikipedia: 0.2 s, default 1 s)
* retries with exponential back-off on network errors, 429/5xx and arXiv's plain-text "Rate exceeded"
* streaming download to a temp file with a size cap, then atomic rename
* content sniffing (PDF magic bytes / HTML) so a login page is never saved as "book.pdf"
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_UA = "SMTrainer-corpus/0.1 (research corpus builder; python-urllib)"

# seconds between requests to the same host (override with Fetcher(intervals={...}))
HOST_INTERVALS = {
    "export.arxiv.org": 3.1,
    "arxiv.org": 3.1,
    "api.openalex.org": 0.15,
    "en.wikipedia.org": 0.2,
    "simple.wikipedia.org": 0.2,
    "api.github.com": 1.0,
}


class FetchError(Exception):
    pass


class Fetcher:
    def __init__(self, user_agent: str = DEFAULT_UA, timeout: int = 90, retries: int = 4,
                 intervals: dict | None = None, default_interval: float = 1.0):
        self.ua = user_agent
        self.timeout = timeout
        self.retries = retries
        self.intervals = {**HOST_INTERVALS, **(intervals or {})}
        self.default_interval = default_interval
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def _wait(self, host: str):
        gap = self.intervals.get(host, self.default_interval)
        with self._lock:
            now = time.time()
            ready = self._last.get(host, 0.0) + gap
            if ready > now:
                time.sleep(ready - now)
            self._last[host] = time.time()

    def _open(self, url: str, headers: dict | None = None):
        host = urllib.parse.urlparse(url).netloc
        req = urllib.request.Request(url, headers={"User-Agent": self.ua, **(headers or {})})
        delay = 5.0
        for attempt in range(self.retries + 1):
            self._wait(host)
            try:
                return urllib.request.urlopen(req, timeout=self.timeout)
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    ra = e.headers.get("Retry-After")
                    time.sleep(float(ra) if ra and ra.isdigit() else delay + random.random())
                    delay *= 2
                    continue
                raise FetchError(f"HTTP {e.code} for {url}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                if attempt < self.retries:
                    time.sleep(delay + random.random())
                    delay *= 2
                    continue
                raise FetchError(f"{type(e).__name__}: {e} for {url}") from None
        raise FetchError(f"giving up on {url}")

    # ------------------------------------------------------------------ #
    def get_text(self, url: str, headers: dict | None = None, max_bytes: int = 20_000_000) -> str:
        """GET a small text/JSON/XML response. Retries arXiv's 200-with-'Rate exceeded' bodies."""
        delay = 30.0
        for attempt in range(self.retries + 1):
            with self._open(url, headers) as r:
                body = r.read(max_bytes).decode(r.headers.get_content_charset() or "utf-8", "replace")
            if body.strip().startswith("Rate exceeded") and attempt < self.retries:
                time.sleep(delay)
                delay *= 2
                continue
            return body
        raise FetchError(f"rate limited: {url}")

    def get_json(self, url: str, headers: dict | None = None):
        return json.loads(self.get_text(url, headers))

    def download(self, url: str, dest: str, max_bytes: int, expect: str = "any") -> dict:
        """Stream `url` to `dest`. expect: 'pdf' | 'html' | 'any'. Returns {bytes, content_type, final_url}.
        Raises FetchError when the body is too large or is not what was expected."""
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".part"
        with self._open(url) as r:
            ctype = (r.headers.get("Content-Type") or "").lower()
            length = r.headers.get("Content-Length")
            if length and length.isdigit() and int(length) > max_bytes:
                raise FetchError(f"too large ({int(length) / 1e6:.0f} MB > {max_bytes / 1e6:.0f} MB): {url}")
            n = 0
            head = b""
            with open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    if not head:
                        head = chunk[:16]
                    n += len(chunk)
                    if n > max_bytes:
                        f.close()
                        os.remove(tmp)
                        raise FetchError(f"too large (> {max_bytes / 1e6:.0f} MB): {url}")
                    f.write(chunk)
            final = r.geturl()
        is_pdf = head.startswith(b"%PDF")
        if expect == "pdf" and not is_pdf:
            os.remove(tmp)
            raise FetchError(f"not a PDF (content-type {ctype or '?'}): {url}")
        if expect == "html" and is_pdf:
            os.remove(tmp)
            raise FetchError(f"expected HTML, got PDF: {url}")
        os.replace(tmp, dest)
        return {"bytes": n, "content_type": "application/pdf" if is_pdf else ctype, "final_url": final}
