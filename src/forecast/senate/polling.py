"""Poll averages for the Senate model.

Weighting (each poll, after averaging multiple questions from the same poll, e.g. LV and RV):
  sqrt(sample size, clipped to 100..3000) x 0.5 ** (days before election / HALF_LIFE)
  x 0.5 for partisan (campaign- or party-sponsored) polls,
  then divided by that pollster's number of polls in the race, so each pollster counts about once.

Shares are Democratic two-party shares, D / (D + R).
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

HALF_LIFE_DAYS = 14


def _prepare(p: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    p = p[(p["cand1_party"] == "DEM") & (p["cand2_party"] == "REP")].copy()
    p["two_party"] = p["cand1_pct"] / (p["cand1_pct"] + p["cand2_pct"])
    p["partisan"] = p["partisan"].fillna("")
    p = (p.groupby(keys + ["poll_id", "pollster", "partisan"], as_index=False)
         .agg(two_party=("two_party", "mean"), samplesize=("samplesize", "max"),
              time_to_election=("time_to_election", "min")))
    w = np.sqrt(p["samplesize"].fillna(600).clip(100, 3000)) * 0.5 ** (p["time_to_election"] / HALF_LIFE_DAYS)
    w = w * np.where(p["partisan"].isin(["DEM", "REP"]), 0.5, 1.0)
    p["w"] = w / p.groupby(keys + ["pollster"])["poll_id"].transform("count")
    p["wy"] = p["w"] * p["two_party"]
    return p


def _average(p: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = p.groupby(keys)
    out = pd.DataFrame({"poll_avg": g["wy"].sum() / g["w"].sum(), "n_polls": g.size(),
                        "n_pollsters": g["pollster"].nunique()})
    return out.reset_index()


def senate_race_averages(con: sqlite3.Connection) -> pd.DataFrame:
    p = pd.read_sql("SELECT * FROM historical_polls WHERE type_simple = 'Sen-G'", con)
    kind = p["race"].str.split("_").str[1]
    p = p[kind.isin(["Sen-G", "Sen-GS"])].copy()          # general elections; runoffs excluded
    p["special"] = (kind == "Sen-GS").astype(int)
    p = p[p["cycle"] % 2 == 0]
    keys = ["cycle", "location", "special"]
    return _average(_prepare(p, keys), keys).rename(columns={"cycle": "year", "location": "state_po"})


def generic_ballot(con: sqlite3.Connection) -> pd.DataFrame:
    p = pd.read_sql("SELECT * FROM historical_polls WHERE type_simple = 'House-G-US'", con)
    p = p[p["cycle"] % 2 == 0]
    out = _average(_prepare(p, ["cycle"]), ["cycle"])
    return out.rename(columns={"cycle": "year", "poll_avg": "gb_two_party", "n_polls": "gb_n_polls"})[
        ["year", "gb_two_party", "gb_n_polls"]]


def add_polls(races: pd.DataFrame, con: sqlite3.Connection) -> pd.DataFrame:
    races = races.merge(generic_ballot(con), on="year", how="left")
    races["gb_lean"] = races["gb_two_party"] - 0.5
    return races.merge(senate_race_averages(con), on=["year", "state_po", "special"], how="left")
