"""Economic series from FRED (public CSV endpoint, no API key needed)."""
from __future__ import annotations

import io
import sqlite3

import pandas as pd

from .. import config, db, http


def fetch(series_id: str) -> pd.DataFrame:
    raw = http.get_bytes(config.FRED_CSV.format(series=series_id))
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = ["date", "value"]  # first column is the date whatever FRED names it
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["series_id"] = series_id
    return df.dropna(subset=["value"])


def load_all(con: sqlite3.Connection, series: dict[str, str] | None = None) -> dict[str, int]:
    counts = {}
    for sid in (series or config.FRED_SERIES):
        df = fetch(sid)
        con.execute("DELETE FROM econ_series WHERE series_id = ?", (sid,))
        df[["series_id", "date", "value"]].to_sql("econ_series", con, if_exists="append", index=False)
        counts[sid] = len(df)
    db.record_source(con, "FRED (Federal Reserve Bank of St. Louis)", "https://fred.stlouisfed.org",
                     "Public data; check each series' notes for third-party copyright")
    con.commit()
    return counts


def load_file(con: sqlite3.Connection, path) -> tuple[str, int]:
    """Load a FRED CSV saved by hand (first column date, second column the series)."""
    from pathlib import Path
    df = pd.read_csv(Path(path))
    sid = str(df.columns[1])
    df.columns = ["date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).assign(series_id=sid)
    con.execute("DELETE FROM econ_series WHERE series_id = ?", (sid,))
    df[["series_id", "date", "value"]].to_sql("econ_series", con, if_exists="append", index=False)
    db.record_source(con, "FRED (Federal Reserve Bank of St. Louis)", "https://fred.stlouisfed.org",
                     "Public data; check each series' notes for third-party copyright")
    con.commit()
    return sid, len(df)
