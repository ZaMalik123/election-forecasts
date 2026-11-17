"""Import any CSV or Excel file into its own SQLite table.

Big spreadsheets are read in one pass and written in chunks, so you can query them with SQL
instead of keeping them open in Excel.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from .. import db


def _table_name(path: Path) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", path.stem.lower()).strip("_") or "imported"


def import_file(con: sqlite3.Connection, path: Path, table: str | None = None,
                sheet: str | int | None = 0, replace: bool = True,
                source_url: str = "", license: str = "unknown - check before publishing") -> int:
    path = Path(path)
    table = table or _table_name(path)
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet)
    elif path.suffix.lower() in (".csv", ".txt", ".tab", ".tsv"):
        sep = "\t" if path.suffix.lower() in (".tab", ".tsv") else ","
        df = pd.read_csv(path, sep=sep, low_memory=False)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    df.columns = [re.sub(r"[^a-z0-9_]+", "_", str(c).strip().lower()).strip("_") for c in df.columns]
    df.to_sql(table, con, if_exists="replace" if replace else "append", index=False, chunksize=50_000)
    db.record_source(con, f"file:{path.name}", source_url, license, notes=f"table={table}")
    return len(df)
