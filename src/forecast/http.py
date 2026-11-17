"""Small, polite HTTP helpers (standard library only)."""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

from . import config


def get_bytes(url: str, retries: int = 3, pause: float = 2.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except Exception as err:  # network hiccups: wait and retry
            last_err = err
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"Could not download {url}: {last_err}")


def get_json(url: str):
    return json.loads(get_bytes(url).decode("utf-8"))


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(get_bytes(url))
    return dest
