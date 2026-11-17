"""FiveThirtyEight historical polls (pollster-ratings archive, CC BY 4.0)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .. import config, db, http

COLS = ["poll_id", "question_id", "race_id", "cycle", "location", "type_simple", "race", "pollster",
        "methodology", "partisan", "polldate", "electiondate", "time_to_election", "samplesize",
        "cand1_name", "cand1_party", "cand1_pct", "cand1_actual", "cand2_name", "cand2_party",
        "cand2_pct", "cand2_actual", "margin_poll", "margin_actual"]


def download(force: bool = False) -> Path:
    dest = config.DATA_RAW / config.FTE_RAW_POLLS["local_name"]
    if dest.exists() and not force:
        return dest
    return http.download_file(config.FTE_RAW_POLLS["url"], dest)


def load(con: sqlite3.Connection, path: Path | None = None) -> int:
    path = Path(path or config.DATA_RAW / config.FTE_RAW_POLLS["local_name"])
    df = pd.read_csv(path, low_memory=False)
    df = df[[c for c in COLS if c in df.columns]]
    con.execute("DELETE FROM historical_polls")
    df.to_sql("historical_polls", con, if_exists="append", index=False)
    db.record_source(con, "FiveThirtyEight pollster-ratings raw polls", config.FTE_RAW_POLLS["url"],
                     config.FTE_RAW_POLLS["license"])
    con.commit()
    return len(df)
