"""Fetch the current Spanish voting-intention table from English Wikipedia (CC BY-SA 4.0).

Writes data/raw/spain_polls_2026_latest.json with rows
    ["2026", 0, pollster, fieldwork date text, sample size text, party (joint columns joined by "+"), value text]
which spain/data.py uses in place of the 2026 current table in spain_data.json.

This mirrors the browser extraction used to build spain_data.json: header party names come from the
logo links' titles, a data cell spanning several party columns becomes a joint "A+B" figure, the value
is the text before the line break (the seat projection after it is ignored), and footnote markers
are removed.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone

import requests
from bs4 import BeautifulSoup

from .. import config

PAGE = "Opinion_polling_for_the_2026_Spanish_general_election"
API = "https://en.wikipedia.org/w/api.php"
OUT = config.ROOT / "data" / "raw" / "spain_polls_2026_latest.json"
USER_AGENT = "MachineLearningWindsForecasts/1.0 (https://forecasts.machinelearningwinds.com; research.fellow@machinelearningwinds.com)"


def _text(node) -> str:
    return re.sub(r"\s+", " ", node.get_text()).strip()


def _before_break(cell) -> str:
    parts = []
    for child in cell.children:
        if getattr(child, "name", None) == "br":
            break
        parts.append(child.get_text() if hasattr(child, "get_text") else str(child))
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def parse(html: str) -> list[list]:
    soup = BeautifulSoup(html, "html.parser")
    for sup in soup.select("sup.reference"):
        sup.decompose()
    tables = [t for t in soup.find_all("table") if "Fieldwork" in _text(t)[:600]]
    if not tables:
        raise ValueError("no voting-intention table found")
    t = tables[0]
    rows = t.find_all("tr")
    names = []
    for c in rows[0].find_all(["th", "td"], recursive=False):
        a = c.find("a")
        nm = a.get("title") if a is not None and a.get("title") else _text(c)
        names += [nm] * int(c.get("colspan", 1) or 1)
    out = []
    for r in rows[1:]:
        cells = r.find_all(["th", "td"], recursive=False)
        if len(cells) < 6:
            continue
        rec, col = [], 0
        for c in cells:
            cs = int(c.get("colspan", 1) or 1)
            rec.append((col, cs, c))
            col += cs
        get = {c0: c for c0, _, c in rec}
        if 0 not in get:
            continue
        pollster, when = _text(get[0]), _text(get[1]) if 1 in get else ""
        n = _text(get[2]) if 2 in get else ""
        for c0, cs, c in rec:
            if c0 < 4:
                continue
            nm = list(dict.fromkeys(names[c0:c0 + cs]))
            if "Lead" in nm:
                continue
            out.append(["2026", 0, pollster, when, n, "+".join(nm), _before_break(c)])
    return out


def fetch() -> dict:
    r = requests.get(API, params={"action": "parse", "page": PAGE, "prop": "text|revid", "format": "json",
                                  "formatversion": "2"}, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    j = r.json()["parse"]
    rows = parse(j["text"])
    return {"retrieved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": f'English Wikipedia, "{PAGE.replace("_", " ")}" (CC BY-SA 4.0), current voting-intention table',
            "page_rev": j["revid"], "rows": rows}


def validate(data: dict) -> None:
    """Refuse to overwrite good data with a broken parse (page restructured, vandalism, etc.)."""
    rows = data["rows"]
    polls = {(r[2], r[3]) for r in rows if r[2]}
    parties = {r[5] for r in rows}
    if len(polls) < 20:
        raise ValueError(f"only {len(polls)} polls parsed")
    for must in ("PP", "PSOE", "Vox"):
        if must not in parties:
            raise ValueError(f"column {must} missing; parties found: {sorted(parties)}")
    nums = [float(r[6]) for r in rows if r[5] == "PP" and re.fullmatch(r"\d+(\.\d+)?", r[6] or "")]
    if not nums or not all(10 < x < 60 for x in nums):
        raise ValueError("PP values out of range")


def update() -> int:
    data = fetch()
    validate(data)
    OUT.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return len({(r[2], r[3]) for r in data["rows"] if r[2]})
