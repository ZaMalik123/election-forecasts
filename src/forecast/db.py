"""SQLite helpers: one database file holds every dataset."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from . import config

SCHEMA = Path(__file__).with_name("schema.sql")


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = Path(path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    # TRUNCATE empties the rollback journal instead of deleting it: safe everywhere, and works in
    # synced or sandboxed folders (OneDrive, Claude's workspace) where deleting files can fail.
    con.execute("PRAGMA journal_mode = TRUNCATE")
    return con


def init_db(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA.read_text(encoding="utf-8"))
    con.commit()


def record_source(con: sqlite3.Connection, name: str, url: str, license: str, notes: str = "") -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con.execute(
        "INSERT INTO sources (name, url, license, retrieved_at, notes) VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(name) DO UPDATE SET url=excluded.url, license=excluded.license, "
        "retrieved_at=excluded.retrieved_at, notes=excluded.notes",
        (name, url, license, now, notes),
    )
    con.commit()
