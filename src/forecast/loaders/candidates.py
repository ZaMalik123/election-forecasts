"""Candidate -> party table, kept by hand in data/manual/candidates.csv (committed to git)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .. import config, db

PATH = config.DATA_MANUAL / "candidates.csv"


def load(con: sqlite3.Connection, path: Path | None = None) -> int:
    df = pd.read_csv(path or PATH, dtype=str, keep_default_na=False)
    df["district"] = pd.to_numeric(df["district"], errors="coerce")
    con.execute("DELETE FROM candidates")
    df.to_sql("candidates", con, if_exists="append", index=False)
    db.record_source(con, "Candidate party lookup (FEC + manual)", "https://api.open.fec.gov",
                     "FEC data is public domain; manual entries are our own")
    con.commit()
    return len(df)
