"""Daily, as-of-date Senate forecasts with no lookahead.

For a forecast dated D, h = days until Election Day. Only polls *available* by D are used: a poll counts
as available RELEASE_LAG days after its last field day (historical archive: median field date + 3).
Polls released in the WINDOW days before D are averaged, weighted by sqrt(sample size), recency
(half-life 14 days), half weight for partisan/internal polls, and each pollster counted about once.

A separate model is fitted at each horizon in GRID (h = 1, 7, ..., 56 days), on historical races with
poll averages built the same way at that horizon. A forecast h days out linearly interpolates the
predicted means, state errors and national error between the two neighbouring grid models (h <= 1 uses
the 1-day model, h >= 56 the 56-day model), so the daily series has no artificial jumps.
"""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .. import config
from . import backtest, features, model as m

GRID = [1, 7, 14, 21, 28, 35, 42, 49, 56]
WINDOW = 30
RELEASE_LAG = 2
HALF_LIFE = 14
SPEC = m.SPECS["polls"]


def grid_for(h: int) -> int:
    return next((g for g in GRID if g >= h), GRID[-1])


def bracket(h: int) -> tuple[int, int, float]:
    """Neighbouring grid horizons (lo, hi) and the weight on hi."""
    if h <= GRID[0]:
        return GRID[0], GRID[0], 0.0
    if h >= GRID[-1]:
        return GRID[-1], GRID[-1], 0.0
    hi = grid_for(h)
    lo = GRID[GRID.index(hi) - 1]
    return lo, hi, (h - lo) / (hi - lo)


def simulate_interpolated(models, h, races, n_sims, seed, df_state=5):
    lo, hi, a = bracket(h)
    m_lo, m_hi = models[lo], models[hi]
    mean_lo, w_lo = m.predict_mean(m_lo.params, races, m_lo.spec)
    mean_hi, w_hi = m.predict_mean(m_hi.params, races, m_hi.spec)
    mean = (1 - a) * mean_lo + a * mean_hi
    sd_state = (1 - a) * m_lo.state_sd(races) + a * m_hi.state_sd(races)
    sd_nat = (1 - a) * m_lo.sigma_national + a * m_hi.sigma_national
    rng = np.random.default_rng(seed)
    national = rng.normal(0, sd_nat, size=(n_sims, 1))
    t = rng.standard_t(df_state, size=(n_sims, len(races))) * np.sqrt((df_state - 2) / df_state)
    shares = mean + national + t * sd_state
    out = races.copy()
    out["pred_share"], out["poll_weight"] = mean, (1 - a) * w_lo + a * w_hi
    out["state_sd"], out["p_dem"] = sd_state, (shares > 0.5).mean(axis=0)
    return out, shares, sd_nat


# ---------- averaging (shared by historical and live) ----------

def weighted_average(polls: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """polls needs: keys, pollster, partisan_flag (bool), two_party, samplesize, age (days)."""
    if polls.empty:
        return pd.DataFrame(columns=keys + ["poll_avg", "n_polls", "n_pollsters"])
    p = polls.copy()
    w = np.sqrt(p["samplesize"].fillna(600).clip(100, 3000)) * 0.5 ** (p["age"].clip(lower=0) / HALF_LIFE)
    w = w * np.where(p["partisan_flag"], 0.5, 1.0)
    p["w"] = w / p.groupby(keys + ["pollster"])["two_party"].transform("count")
    p["wy"] = p["w"] * p["two_party"]
    g = p.groupby(keys)
    return pd.DataFrame({"poll_avg": g["wy"].sum() / g["w"].sum(), "n_polls": g.size(),
                         "n_pollsters": g["pollster"].nunique()}).reset_index()


# ---------- historical (FiveThirtyEight archive, 1998-2022) ----------

def _hist(con: sqlite3.Connection, where: str) -> pd.DataFrame:
    p = pd.read_sql(f"SELECT * FROM historical_polls WHERE {where}", con)
    p = p[(p["cand1_party"] == "DEM") & (p["cand2_party"] == "REP") & (p["cycle"] % 2 == 0)].copy()
    p["two_party"] = p["cand1_pct"] / (p["cand1_pct"] + p["cand2_pct"])
    p["partisan_flag"] = p["partisan"].isin(["DEM", "REP", "IND", "LIB"])
    kind = p["race"].str.split("_").str[1]
    p["special"] = (kind == "Sen-GS").astype(int)
    p["year"], p["state_po"] = p["cycle"], p["location"]
    p["days_before_avail"] = p["time_to_election"] - 3
    p = (p.groupby(["year", "state_po", "special", "poll_id", "pollster", "partisan_flag"], as_index=False)
         .agg(two_party=("two_party", "mean"), samplesize=("samplesize", "max"),
              days_before_avail=("days_before_avail", "min"), kind=("race", "first")))
    return p


def historical_polls(con):
    sen = _hist(con, "type_simple = 'Sen-G'")
    sen = sen[sen["kind"].str.split("_").str[1].isin(["Sen-G", "Sen-GS"])]
    gb = _hist(con, "type_simple = 'House-G-US'")
    return sen, gb


def _asof(polls: pd.DataFrame, h: int) -> pd.DataFrame:
    sel = polls[(polls["days_before_avail"] >= h) & (polls["days_before_avail"] - h <= WINDOW)].copy()
    sel["age"] = sel["days_before_avail"] - h
    return sel


def historical_races(con, h: int, base: pd.DataFrame | None = None, polls=None) -> pd.DataFrame:
    races = features.senate_races(con) if base is None else base
    sen, gb = polls or historical_polls(con)
    gb_avg = weighted_average(_asof(gb, h), ["year"]).rename(columns={"poll_avg": "gb_two_party"})
    out = races.merge(gb_avg[["year", "gb_two_party"]], on="year", how="left")
    out["gb_lean"] = out["gb_two_party"] - 0.5
    return out.merge(weighted_average(_asof(sen, h), ["year", "state_po", "special"]),
                     on=["year", "state_po", "special"], how="left")


def backtest_by_horizon(con, years, n_sims=5000) -> dict[str, pd.DataFrame]:
    base, polls = features.senate_races(con), historical_polls(con)
    overall, cycles = [], []
    for h in GRID:
        res = backtest.run(historical_races(con, h, base, polls), SPEC, years, n_sims=n_sims)
        overall.append(res["overall"].assign(days_out=h))
        cycles.append(res["cycles"].assign(days_out=h))
    o = pd.concat(overall, ignore_index=True)
    o = o[["days_out"] + [c for c in o.columns if c not in ("days_out", "model")]]
    return {"overall": o, "cycles": pd.concat(cycles, ignore_index=True)}


def fit_all_horizons(con, train_until: int) -> dict[int, m.FittedModel]:
    base, polls = features.senate_races(con), historical_polls(con)
    models = {}
    for h in GRID:
        data = backtest.eligible(historical_races(con, h, base, polls))
        models[h] = m.fit(data[data["year"] <= train_until], SPEC)
    return models


# ---------- live 2026 ----------

def race_table() -> pd.DataFrame:
    return pd.read_csv(config.DATA_MANUAL / "senate_2026.csv", dtype=str, keep_default_na=False)


def live_polls(con, year: int, table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2026 Senate and generic-ballot polls with availability dates."""
    from ..polls import STATES
    polls = pd.read_sql("SELECT * FROM polls WHERE poll_type IN ('us-senator', 'generic-ballot')", con)
    ans = pd.read_sql("SELECT * FROM poll_answers", con)
    polls["available"] = pd.to_datetime(polls["end_date"]) + pd.Timedelta(days=RELEASE_LAG)
    polls["partisan_flag"] = polls["partisan"].notna() | (polls["internal"] == 1)
    polls["samplesize"] = polls["sample_size"]

    from ..loaders import wiki_gb
    gbp = wiki_gb.polls_for_model(con)
    if not gbp.empty:                                           # fuller source (VoteHub thins out after June 2026)
        gbp["available"] = pd.to_datetime(gbp["end_date"]) + pd.Timedelta(days=RELEASE_LAG)
        gbp = gbp[gbp["available"].dt.year >= year - 1]
    else:
        gbp = polls[(polls["poll_type"] == "generic-ballot") & (polls["subject"] == str(year))]
        ga = ans[ans["poll_id"].isin(gbp["poll_id"])].pivot_table(index="poll_id", columns="choice", values="pct")
        gbp = gbp.merge((ga["Dem"] / (ga["Dem"] + ga["Rep"])).rename("two_party").reset_index(), on="poll_id")

    sp = polls[(polls["poll_type"] == "us-senator") & polls["subject"].str.startswith(str(year))].copy()
    sp["state_name"] = sp["subject"].str[5:]
    sp = sp[sp["state_name"].isin(STATES)]                      # general elections only, not primaries
    sp["state_po"] = sp["state_name"].map(STATES)
    cands = pd.read_sql("SELECT name, state, party FROM candidates WHERE office='S'", con).drop_duplicates(["name", "state"])
    a = ans.drop(columns=["party"]).merge(sp[["poll_id", "state_po"]], on="poll_id")
    a = a.merge(cands, left_on=["choice", "state_po"], right_on=["name", "state"], how="left")
    rows = []
    t = table.set_index("state_po")
    for pid, g in a.groupby("poll_id"):
        st = g["state_po"].iat[0]
        if st not in t.index:
            continue
        challengers = t.at[st, "challengers"].split(";")
        names = set(g["choice"])
        if len(challengers) > 1 and not all(any(c.split()[-1] in n for n in names) for c in challengers):
            continue                                            # multi-challenger race: full-field polls only
        party = g["party"].fillna("")
        rep = g.loc[party == "R", "pct"]
        chal = g.loc[party.isin(t.at[st, "challenger_party"].split(";")), :]
        if rep.empty or chal.empty:
            continue
        top = chal.sort_values("pct").iloc[-1]
        rows.append({"poll_id": pid, "state_po": st, "rep_pct": rep.max(), "chal_pct": top["pct"],
                     "chal_name": top["choice"], "chal_party": top["party"]})
    sen = sp.merge(pd.DataFrame(rows), on=["poll_id", "state_po"])
    sen["two_party"] = sen["chal_pct"] / (sen["chal_pct"] + sen["rep_pct"])
    return sen, gbp


def races_2026(con, year: int, table: pd.DataFrame) -> pd.DataFrame:
    lean = features.presidential_lean(con)
    lean = lean[lean["pres_year"] == ((year - 1) // 4) * 4][["state_po", "pres_lean"]]
    r = table.merge(lean, on="state_po", how="left")
    r["year"], r["special"] = year, r["special"].astype(int)
    r["incumbent"] = r["incumbent"].astype(int)
    r["pres_sign"] = 1 if config.president_party(year) == "D" else -1
    r["midterm"] = int(year % 4 == 2)
    return r


def daily_forecast(con, election_day: date, start: date, models: dict[int, m.FittedModel],
                   n_sims: int = 10_000, seed: int = 2026) -> dict[str, pd.DataFrame]:
    table = race_table()
    year = election_day.year
    base = races_2026(con, year, table)
    sen, gbp = live_polls(con, year, table)
    not_up_r = 53 - int((table["seat_party"] == "R").sum())
    not_up_d = 47 - int((table["seat_party"] == "D").sum())
    race_rows, chamber_rows = [], []
    day = start
    while day <= election_day:
        h = (election_day - day).days
        ts = pd.Timestamp(day)
        def window(df):
            sel = df[(df["available"] <= ts) & (df["available"] > ts - pd.Timedelta(days=WINDOW))].copy()
            sel["age"] = (ts - sel["available"]).dt.days
            assert (sel["available"] <= ts).all() and (pd.to_datetime(sel["end_date"]) < ts).all(), "lookahead"
            return sel
        gb_sel = window(gbp)
        gb = weighted_average(gb_sel.assign(k=1), ["k"])
        races = base.copy()
        races["gb_two_party"] = gb["poll_avg"].iat[0] if len(gb) else np.nan
        races["gb_lean"] = races["gb_two_party"] - 0.5
        s_sel = window(sen)
        avg = weighted_average(s_sel, ["state_po"])
        races = races.merge(avg, on="state_po", how="left")
        leader = (s_sel.groupby(["state_po", "chal_name"])["chal_pct"].mean().reset_index()
                  .sort_values("chal_pct").groupby("state_po").tail(1).set_index("state_po")["chal_name"])
        pred, draws, sd_nat = simulate_interpolated(models, h, races, n_sims, seed)
        wins = draws > 0.5
        is_dem = np.array([cp.split(";")[0] == "D" if ";" not in cp else
                           (leader.get(st, "") == ch.split(";")[1]) for st, cp, ch in
                           zip(races["state_po"], races["challenger_party"], races["challengers"])])
        d_seats = not_up_d + (wins & is_dem).sum(axis=1)
        i_seats = (wins & ~is_dem).sum(axis=1)
        r_seats = not_up_r + (~wins).sum(axis=1)
        chamber_rows.append({
            "date": day.isoformat(), "days_out": h, "national_error_sd": round(sd_nat, 4),
            "generic_ballot_dem_two_party": round(float(races["gb_two_party"].iat[0]), 4),
            "gb_polls_in_window": len(gb_sel),
            "p_rep_control": round(float((r_seats >= 50).mean()), 4),
            "p_dem_control": round(float((d_seats >= 51).mean()), 4),
            "p_independents_decide": round(float(((r_seats < 50) & (d_seats < 51)).mean()), 4),
            "rep_seats_mean": round(float(r_seats.mean()), 2),
            "rep_seats_80pct_low": int(np.percentile(r_seats, 10)),
            "rep_seats_80pct_high": int(np.percentile(r_seats, 90)),
            "dem_seats_mean": round(float(d_seats.mean()), 2),
            "ind_seats_mean": round(float(i_seats.mean()), 2),
        })
        for (_, r), p in zip(races.iterrows(), pred.itertuples()):
            chal = leader.get(r["state_po"], r["challengers"].split(";")[0])
            race_rows.append({
                "date": day.isoformat(), "days_out": h, "state_po": r["state_po"], "special": r["special"],
                "republican": r["rep_candidate"], "challenger": chal,
                "p_republican": round(1 - p.p_dem, 4), "p_challenger": round(p.p_dem, 4),
                "pred_challenger_two_party": round(p.pred_share, 4),
                "poll_avg_challenger_two_party": None if pd.isna(r["poll_avg"]) else round(r["poll_avg"], 4),
                "pollsters_in_window": 0 if pd.isna(r["n_pollsters"]) else int(r["n_pollsters"]),
                "poll_weight": round(p.poll_weight, 3),
            })
        day += timedelta(days=1)
    return {"races": pd.DataFrame(race_rows), "chamber": pd.DataFrame(chamber_rows)}
