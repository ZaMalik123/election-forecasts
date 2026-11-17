"""Turn raw VoteHub race polls into one row per poll with Democratic and Republican shares."""
from __future__ import annotations

import sqlite3

import pandas as pd

STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}
PLACEHOLDERS = {"Dem": "D", "Rep": "R"}


def _parse_subject(subject: str) -> tuple[int, str | None, str | None]:
    """'2026 North Carolina' -> (2026, 'NC', None); '2026 Texas Democratic' -> (2026, 'TX', 'D')."""
    year, rest = int(subject[:4]), subject[5:]
    primary = None
    for word, p in ((" Democratic", "D"), (" Republican", "R")):
        if rest.endswith(word):
            rest, primary = rest[: -len(word)], p
    return year, STATES.get(rest), primary


def senate_polls(con: sqlite3.Connection) -> pd.DataFrame:
    """General-election Senate polls with dem/rep shares (top candidate of each party)."""
    polls = pd.read_sql("SELECT * FROM polls WHERE poll_type = 'us-senator'", con)
    answers = pd.read_sql("SELECT a.* FROM poll_answers a JOIN polls p USING (poll_id) "
                          "WHERE p.poll_type = 'us-senator'", con)
    cands = pd.read_sql("SELECT name, state, party FROM candidates WHERE office = 'S'", con)

    parsed = polls["subject"].map(_parse_subject)
    polls["year"] = parsed.str[0]
    polls["state_po"] = parsed.str[1]
    polls["primary_party"] = parsed.str[2]
    general = polls[polls["primary_party"].isna()]

    a = answers.drop(columns=["party"]).merge(general[["poll_id", "state_po"]], on="poll_id")
    a = a.merge(cands, left_on=["choice", "state_po"], right_on=["name", "state"], how="left")
    a["party"] = a["party"].where(a["party"].notna() & (a["party"] != ""), a["choice"].map(PLACEHOLDERS))

    unmatched = a[a["party"].isna() | (a["party"] == "")][["state_po", "choice"]].drop_duplicates()
    top = (a[a["party"].isin(["D", "R"])].sort_values("pct", ascending=False)
           .groupby(["poll_id", "party"]).first()["pct"].unstack())
    out = general.merge(top.reset_index(), on="poll_id", how="inner").rename(columns={"D": "dem_pct", "R": "rep_pct"})
    out = out.dropna(subset=["dem_pct", "rep_pct"])
    out["dem_two_party"] = out["dem_pct"] / (out["dem_pct"] + out["rep_pct"])
    out.attrs["unmatched_names"] = unmatched
    cols = ["poll_id", "year", "state_po", "pollster", "start_date", "end_date", "sample_size", "population",
            "internal", "partisan", "dem_pct", "rep_pct", "dem_two_party", "url"]
    return out[cols].sort_values(["state_po", "end_date"])
