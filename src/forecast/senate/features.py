"""Turn raw candidate-level returns into one row per Senate race with model features.

Features (all known before Election Day):
  pres_lean  state's Democratic two-party share in the previous presidential election,
             minus the national Democratic two-party share that year
  incumbent  +1 Democratic incumbent running, -1 Republican incumbent, 0 open seat
  pres_sign  +1 if the sitting president is a Democrat, -1 if a Republican
  midterm    1 in midterm years, 0 in presidential years
Target:
  dem_share  Democratic share of the two-party (D + R) vote
"""
from __future__ import annotations

import re
import sqlite3

import numpy as np
import pandas as pd

from .. import config

MIN_TWO_PARTY = 0.75  # skip races where independents/third parties took >25% (e.g. Maine 2018)
SUFFIXES = {"JR", "SR", "II", "III", "IV"}


def _last_name(name: str) -> str:
    tokens = [t for t in re.split(r"[\s,.]+", str(name).upper()) if t and t not in SUFFIXES]
    return tokens[-1] if tokens else ""


def presidential_lean(con: sqlite3.Connection) -> pd.DataFrame:
    pres = pd.read_sql("SELECT year, state_po, party_simplified AS party, candidatevotes AS votes "
                       "FROM president_results", con)
    pres = pres[pres["party"].isin(["DEMOCRAT", "REPUBLICAN"])]
    by_state = pres.pivot_table(index=["year", "state_po"], columns="party", values="votes",
                                aggfunc="sum", fill_value=0).reset_index()
    by_state["state_share"] = by_state["DEMOCRAT"] / (by_state["DEMOCRAT"] + by_state["REPUBLICAN"])
    nat = by_state.groupby("year")[["DEMOCRAT", "REPUBLICAN"]].sum()
    nat_share = (nat["DEMOCRAT"] / nat.sum(axis=1)).rename("national_share")
    by_state = by_state.join(nat_share, on="year")
    by_state["pres_lean"] = by_state["state_share"] - by_state["national_share"]
    return by_state[["year", "state_po", "pres_lean"]].rename(columns={"year": "pres_year"})


def senate_races(con: sqlite3.Connection) -> pd.DataFrame:
    raw = pd.read_sql("SELECT * FROM senate_results", con)
    raw = raw[raw["stage"].fillna("gen").str.lower().str.startswith("gen")]
    key = ["year", "state_po", "special"]

    totals = raw.groupby(key)["totalvotes"].max().rename("totalvotes")
    party = raw.pivot_table(index=key, columns="party_simplified", values="candidatevotes",
                            aggfunc="sum", fill_value=0)
    for p in ("DEMOCRAT", "REPUBLICAN"):
        if p not in party:
            party[p] = 0
    races = party[["DEMOCRAT", "REPUBLICAN"]].join(totals).reset_index()
    races = races.rename(columns={"DEMOCRAT": "dem_votes", "REPUBLICAN": "rep_votes"})
    two = races["dem_votes"] + races["rep_votes"]
    races["dem_share"] = np.where(two > 0, races["dem_votes"] / two.replace(0, np.nan), np.nan)
    denom = races["totalvotes"].where(races["totalvotes"] > 0, two)
    races["two_party_frac"] = two / denom.replace(0, np.nan)
    races["contested"] = ((races["dem_votes"] > 0) & (races["rep_votes"] > 0)
                          & (races["two_party_frac"] >= MIN_TWO_PARTY)).astype(int)

    # Winners, used to detect incumbents in later races.
    idx = raw.groupby(key)["candidatevotes"].idxmax()
    winners = raw.loc[idx, key + ["candidate", "party_simplified"]].copy()
    winners["last"] = winners["candidate"].map(_last_name)

    raw = raw.assign(last=raw["candidate"].map(_last_name))
    inc = []
    for (year, st, special), grp in raw.groupby(key):
        prior = winners[(winners["state_po"] == st) & (winners["year"] < year) & (winners["year"] >= year - 6)]
        sign = 0
        for _, c in grp[grp["party_simplified"].isin(["DEMOCRAT", "REPUBLICAN"])].iterrows():
            hit = prior[(prior["last"] == c["last"]) & (prior["party_simplified"] == c["party_simplified"])]
            if len(hit):
                sign = 1 if c["party_simplified"] == "DEMOCRAT" else -1
                break
        inc.append((year, st, special, sign))
    races = races.merge(pd.DataFrame(inc, columns=key + ["incumbent"]), on=key, how="left")

    # Previous presidential election strictly before this year (never peek at same-day results).
    races["pres_year"] = ((races["year"] - 1) // 4) * 4
    races = races.merge(presidential_lean(con), on=["pres_year", "state_po"], how="left")

    races["pres_sign"] = races["year"].map(lambda y: 1 if config.president_party(int(y)) == "D" else -1)
    races["midterm"] = (races["year"] % 4 == 2).astype(int)
    races["dem_won"] = (races["dem_votes"] > races["rep_votes"]).astype(int)
    return races
