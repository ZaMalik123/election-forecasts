"""Generic-ballot polls copied from Wikipedia's polling tables (CC BY-SA 4.0).

Poll numbers are facts; we store them with the source and credit Wikipedia. File format (tab-separated):
pollster, dates (e.g. "October 28 – November 2, 2026"), sample (e.g. "1,212 (LV)"), dem, rep.
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from .. import config, db

FILE = "wiki_generic_ballot_2026.tsv"
DASH = re.compile(r"\s*[–—-]\s*")


def parse_dates(text: str) -> tuple[str, str]:
    text = text.strip()
    parts = DASH.split(text)
    end = parts[-1]
    year = re.search(r"(\d{4})$", end).group(1)
    if re.match(r"^\d+, \d{4}$", end):                     # "November 1–2, 2026"
        month = re.match(r"^([A-Z][a-z]+)", parts[0]).group(1)
        end = f"{month} {end}"
    start = parts[0] if len(parts) > 1 else end
    if not re.search(r"\d{4}$", start):
        start = f"{start}, {year}"
    fmt = "%B %d, %Y"
    return (datetime.strptime(start, fmt).date().isoformat(), datetime.strptime(end, fmt).date().isoformat())


def _sample(text: str) -> tuple[int | None, str]:
    n = re.search(r"([\d,]+)", text)
    pop = re.search(r"\((LV|RV|A|V)\)", text)
    size = int(n.group(1).replace(",", "")) if n else None
    return size, (pop.group(1).lower() if pop else "")


def _partisan(pollster: str) -> int:
    d, r = "(D)" in pollster, "(R)" in pollster
    return int(d != r)                                     # one-party sponsor = partisan; bipartisan teams aren't


def load(con: sqlite3.Connection, path: Path | None = None) -> int:
    path = Path(path or config.DATA_RAW / FILE)
    df = pd.read_csv(path, sep="\t", comment="#", dtype=str)
    rows = []
    for r in df.itertuples():
        try:
            start, end = parse_dates(r.dates)
        except Exception:
            print(f"  skipped unparseable date: {r.dates!r} ({r.pollster})")
            continue
        size, pop = _sample(str(r.sample))
        rows.append(("wikipedia", r.pollster, start, end, size, pop, _partisan(r.pollster),
                     float(r.dem), float(r.rep)))
    con.execute("DELETE FROM generic_ballot_polls WHERE source = 'wikipedia'")
    con.executemany("INSERT INTO generic_ballot_polls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    db.record_source(con, "Wikipedia: 2026 United States elections polling (generic ballot)",
                     "https://en.wikipedia.org/wiki/2026_United_States_elections_polling",
                     "CC BY-SA 4.0 - credit Wikipedia; poll numbers are facts from the cited pollsters")
    con.commit()
    return len(rows)


def polls_for_model(con: sqlite3.Connection) -> pd.DataFrame:
    """One row per poll: prefer likely voters, then registered voters, then adults."""
    g = pd.read_sql("SELECT * FROM generic_ballot_polls", con)
    if g.empty:
        return g
    g["rank"] = g["population"].map({"lv": 0, "rv": 1, "v": 1, "a": 2}).fillna(3)
    key = ["source", "pollster", "start_date", "end_date"]
    best = g[g["rank"] == g.groupby(key)["rank"].transform("min")]
    best = best.assign(two_party=best["dem"] / (best["dem"] + best["rep"]))
    out = best.groupby(key, as_index=False).agg(two_party=("two_party", "mean"), samplesize=("sample_size", "max"),
                                                 partisan_flag=("partisan", "max"))
    out["partisan_flag"] = out["partisan_flag"].astype(bool)
    out["poll_id"] = out["pollster"] + "|" + out["end_date"]
    return out
