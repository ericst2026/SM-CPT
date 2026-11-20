"""Load KEY=VALUE pairs from the project .env into os.environ (no python-dotenv dependency).

Already-set environment variables win, so `HF_TOKEN=... python scripts/...` still overrides the file.
huggingface_hub / datasets pick HF_TOKEN up from the environment automatically.
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env(path: str = os.path.join(ROOT, ".env")) -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.removeprefix("export ").partition("=")
            key, val = key.strip(), val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            if key and val:
                os.environ.setdefault(key, val)


def hf_auth_headers() -> dict:
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}
