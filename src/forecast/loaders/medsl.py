"""MIT Election Data + Science Lab statewide returns (Senate and President).

Files come from Harvard Dataverse (CC0 1.0). If the download fails, download the CSV
by hand from the dataset page and save it to data/raw/ under the `local_name` in config.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .. import config, db, http

SENATE_COLS = ["year", "state", "state_po", "special", "stage", "candidate", "party_detailed",
               "party_simplified", "writein", "candidatevotes", "totalvotes"]
PRESIDENT_COLS = ["year", "state", "state_po", "candidate", "party_detailed",
                  "party_simplified", "writein", "candidatevotes", "totalvotes"]


def _resolve_csv_url(doi: str, must_contain: str) -> str:
    """Find the main CSV file in a Dataverse dataset by DOI."""
    meta = http.get_json(f"{config.DATAVERSE_BASE}/api/datasets/:persistentId/?persistentId={doi}")
    files = meta["data"]["latestVersion"]["files"]
    for f in files:
        df = f["dataFile"]
        name = (df.get("originalFileName") or df.get("filename") or "").lower()
        if must_contain in name and (name.endswith(".csv") or name.endswith(".tab")):
            return f"{config.DATAVERSE_BASE}/api/access/datafile/{df['id']}?format=original"
    raise RuntimeError(f"No data file containing '{must_contain}' found in {doi}")


def download(kind: str, force: bool = False) -> Path:
    spec = config.MEDSL_DATASETS[kind]
    dest = config.DATA_RAW / spec["local_name"]
    if dest.exists() and not force:
        print(f"Using existing {dest.name}")
        return dest
    url = _resolve_csv_url(spec["doi"], kind)
    print(f"Downloading {spec['title']} ...")
    return http.download_file(url, dest)


def _to_bool_int(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper().isin(["TRUE", "1", "T", "YES"]).astype(int)


def _clean(path: Path, cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in cols if c not in df.columns and c not in ("special", "stage")]
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {missing}")
    if "special" in cols:
        df["special"] = _to_bool_int(df["special"]) if "special" in df else 0
    if "stage" in cols and "stage" not in df:
        df["stage"] = "gen"
    df["writein"] = _to_bool_int(df["writein"])
    df["party_simplified"] = df["party_simplified"].fillna("OTHER").str.upper()
    for c in ("candidatevotes", "totalvotes", "year"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df[cols]


def load(con: sqlite3.Connection, kind: str, path: Path | None = None) -> int:
    spec = config.MEDSL_DATASETS[kind]
    path = Path(path or config.DATA_RAW / spec["local_name"])
    cols = SENATE_COLS if kind == "senate" else PRESIDENT_COLS
    table = f"{kind}_results"
    df = _clean(path, cols)
    con.execute(f"DELETE FROM {table}")  # full reload keeps the table in sync with the file
    df.to_sql(table, con, if_exists="append", index=False)
    db.record_source(con, f"MEDSL {spec['title']}", f"https://doi.org/{spec['doi'][4:]}", spec["license"])
    con.commit()
    return len(df)
