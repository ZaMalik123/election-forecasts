"""VoteHub Polling API (CC BY 4.0: credit VoteHub and cite each poll's source URL)."""
from __future__ import annotations

import json
import sqlite3
import urllib.parse
from datetime import datetime, timezone

from .. import config, db, http


def poll_types() -> list[str]:
    return http.get_json(f"{config.VOTEHUB_BASE}/poll-types")


def fetch(poll_type: str | None = None, **filters) -> list[dict]:
    params = {k: v for k, v in {"poll_type": poll_type, **filters}.items() if v is not None}
    url = f"{config.VOTEHUB_BASE}/polls"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = http.get_json(url)
    return data.get("polls", data) if isinstance(data, dict) else data


def _int_or_none(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def store(con: sqlite3.Connection, polls: list[dict]) -> int:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n = 0
    for p in polls:
        pid = p.get("id")
        if not pid:
            continue
        con.execute(
            "INSERT OR REPLACE INTO polls (poll_id, source, poll_type, subject, seat_name, pollster, "
            "sponsors, start_date, end_date, created_at, sample_size, population, internal, partisan, "
            "url, fetched_at) VALUES (?, 'votehub', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pid, p.get("poll_type"), p.get("subject"), p.get("seat_name"), p.get("pollster"),
             json.dumps(p.get("sponsors") or []), p.get("start_date"), p.get("end_date"),
             p.get("created_at"), _int_or_none(p.get("sample_size")), p.get("population"),
             int(bool(p.get("internal"))), p.get("partisan"), p.get("url"), now),
        )
        con.execute("DELETE FROM poll_answers WHERE poll_id = ?", (pid,))
        for a in p.get("answers") or []:
            con.execute(
                "INSERT OR REPLACE INTO poll_answers (poll_id, choice, party, pct) VALUES (?, ?, ?, ?)",
                (pid, a.get("choice"), a.get("party"), a.get("pct")),
            )
        n += 1
    db.record_source(con, "VoteHub Polling API", config.VOTEHUB_BASE, config.VOTEHUB_LICENSE)
    con.commit()
    return n


def fetch_all(con: sqlite3.Connection) -> dict[str, int]:
    """Fetch every poll type VoteHub offers and store it."""
    counts = {}
    for pt in poll_types():
        counts[pt] = store(con, fetch(pt))
    return counts


def load_file(con: sqlite3.Connection, path) -> dict[str, int]:
    """Load a saved VoteHub export: {"polls_by_type": {type: [poll, ...]}} or a plain list of polls."""
    from pathlib import Path
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "polls_by_type" in data:
        return {pt: store(con, polls) for pt, polls in data["polls_by_type"].items()}
    polls = data.get("polls", data) if isinstance(data, dict) else data
    return {"all": store(con, polls)}
