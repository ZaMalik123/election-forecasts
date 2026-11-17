"""Dynamic Bayesian Senate model (exact Gaussian inference, NumPy only).

Unknowns for one election cycle, solved jointly:
  mu_s   each race's Democratic (challenger) two-party share on Election Day
  b      a common polling bias shared by every poll in the cycle (the "2020 problem")
  h_p    each pollster's house effect, shared across all the races it polls

Prior (fundamentals):
  mu ~ N(f, S), f from the fundamentals regression (presidential lean, incumbency, generic ballot),
  S = sn^2 * 11' + sr^2 * [same region] + diag(ss_i^2): a national error, a regional error and a state error.
  b ~ N(0, sigma_b^2); h_p ~ N(0, tau^2), wider (tau_partisan) for campaign or party-sponsored pollsters.

Each poll i of race s, fielded t_i days before Election Day:
  y_i = mu_s + b + h_p + d_i + e_i
  d_i: opinion drift between the poll and Election Day, a random walk run backwards from Election Day,
       Cov(d_i, d_j) = q_nat * min(t_i, t_j) + q_state * min(t_i, t_j) * [same race]
       so old polls are less informative, and a national shift moves every state together.
  e_i: sampling error, y(1-y)/n, plus sigma_x^2 for everything else (weighting, mode, wording).

Everything is Gaussian, so the posterior of (mu, b, h) given the polls is exact. Win probabilities and
seat counts come from simulating the posterior of mu, with fat national tails (multivariate t).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, replace
from itertools import product

import numpy as np
import pandas as pd

from . import backtest, features, model as m, timeseries as ts

REGION = {
    **dict.fromkeys(["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"], "Northeast"),
    **dict.fromkeys(["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"], "Midwest"),
    **dict.fromkeys(["DE", "FL", "GA", "MD", "NC", "SC", "VA", "WV", "AL", "KY", "MS", "TN", "AR", "LA", "OK",
                     "TX", "DC"], "South"),
    **dict.fromkeys(["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"], "West"),
}
MAX_DAYS = 61          # the historical archive starts ~61 days before each election
FUND_SPEC = m.SPECS["national"]


@dataclass(frozen=True)
class Params:
    q_state: float = 4e-6      # daily drift variance within a race (sd over 60 days ~1.5 pts)
    q_nat: float = 2e-6        # daily national drift variance
    sigma_b: float = 0.015     # common polling bias
    tau: float = 0.012         # house effect, nonpartisan pollsters
    tau_partisan: float = 0.025
    sigma_x: float = 0.010     # non-sampling error per poll
    sigma_reg: float = 0.010   # regional error in the fundamentals prior
    df: int = 6                # tail heaviness of the national error in simulations


# ---------------- data ----------------

def historical_polls(con: sqlite3.Connection) -> pd.DataFrame:
    p = pd.read_sql("SELECT * FROM historical_polls WHERE type_simple = 'Sen-G'", con)
    kind = p["race"].str.split("_").str[1]
    p = p[kind.isin(["Sen-G", "Sen-GS"]) & (p["cand1_party"] == "DEM") & (p["cand2_party"] == "REP")
          & (p["cycle"] % 2 == 0)].copy()
    p["special"] = (p["race"].str.split("_").str[1] == "Sen-GS").astype(int)
    p["two_party"] = p["cand1_pct"] / (p["cand1_pct"] + p["cand2_pct"])
    p["partisan_flag"] = p["partisan"].isin(["DEM", "REP", "IND", "LIB"])
    p = (p.groupby(["cycle", "location", "special", "poll_id", "pollster", "partisan_flag"], as_index=False)
         .agg(two_party=("two_party", "mean"), samplesize=("samplesize", "max"),
              t_field=("time_to_election", "min")))
    p = p.rename(columns={"cycle": "year", "location": "state_po"})
    p["t_avail"] = p["t_field"] - 3                         # released ~3 days after the median field date
    return p[p["t_field"] <= MAX_DAYS]


def polls_asof(polls: pd.DataFrame, h: int) -> pd.DataFrame:
    return polls[polls["t_avail"] >= h]


# ---------------- inference ----------------

def prior_cov(races: pd.DataFrame, sn: float, state_sd: np.ndarray, sigma_reg: float) -> np.ndarray:
    reg = races["state_po"].map(REGION).fillna("Other").to_numpy()
    same_reg = (reg[:, None] == reg[None, :]).astype(float)
    return sn ** 2 + sigma_reg ** 2 * same_reg + np.diag(state_sd ** 2)


def posterior(f: np.ndarray, S: np.ndarray, polls: pd.DataFrame, race_index: np.ndarray, prm: Params):
    """Posterior mean and covariance of mu (one entry per race). polls rows map to races via race_index."""
    R = len(f)
    if polls.empty:
        return f.copy(), S.copy()
    pollsters, p_idx = np.unique(polls["pollster"].astype(str) + "|" + polls["partisan_flag"].astype(str),
                                 return_inverse=True)
    P = len(pollsters)
    partisan = np.array([k.endswith("|True") for k in pollsters])
    n_theta = R + 1 + P
    m0 = np.concatenate([f, [0.0], np.zeros(P)])
    C0 = np.zeros((n_theta, n_theta))
    C0[:R, :R] = S
    C0[R, R] = prm.sigma_b ** 2
    C0[R + 1:, R + 1:] = np.diag(np.where(partisan, prm.tau_partisan, prm.tau) ** 2)
    N = len(polls)
    A = np.zeros((N, n_theta))
    A[np.arange(N), race_index] = 1.0
    A[:, R] = 1.0
    A[np.arange(N), R + 1 + p_idx] = 1.0
    y = polls["two_party"].to_numpy(float)
    n = polls["samplesize"].fillna(600).clip(100, 5000).to_numpy(float)
    t = polls["t_field"].clip(lower=0).to_numpy(float)
    tmin = np.minimum(t[:, None], t[None, :])
    same = (race_index[:, None] == race_index[None, :]).astype(float)
    Rn = prm.q_nat * tmin + prm.q_state * tmin * same + np.diag(y * (1 - y) / n + prm.sigma_x ** 2)
    G = A @ C0 @ A.T + Rn
    L = np.linalg.cholesky(G + 1e-12 * np.eye(N))
    K = np.linalg.solve(L.T, np.linalg.solve(L, A @ C0)).T          # C0 A' G^-1
    mean = m0 + K @ (y - A @ m0)
    cov = C0 - K @ A @ C0
    return mean[:R], cov[:R, :R]


def fundamentals(train: pd.DataFrame, test: pd.DataFrame):
    fitted = m.fit(train.dropna(subset=FUND_SPEC.required), FUND_SPEC)
    f, _ = m.predict_mean(fitted.params, test, FUND_SPEC)
    return f, fitted.sigma_national, fitted.state_sd(test)


def prepare(train: pd.DataFrame, test: pd.DataFrame, polls: pd.DataFrame):
    """Everything that doesn't depend on the drift/bias settings: fundamentals prior and poll-to-race map."""
    f, sn, ss = fundamentals(train, test)
    pos = {k: i for i, k in enumerate(zip(test["state_po"], test["special"]))}
    idx = np.array([pos.get(k, -1) for k in zip(polls["state_po"], polls["special"])], dtype=int)
    return {"f": f, "sn": sn, "ss": ss, "test": test, "polls": polls[idx >= 0], "idx": idx[idx >= 0]}


def solve(prep, prm: Params):
    S = prior_cov(prep["test"], prep["sn"], prep["ss"], prm.sigma_reg)
    return posterior(prep["f"], S, prep["polls"], prep["idx"], prm) + (prep["f"],)


def forecast(train: pd.DataFrame, test: pd.DataFrame, polls: pd.DataFrame, prm: Params):
    """Posterior mean/cov for the races in `test`, using `polls` (already filtered to what's available)."""
    return solve(prepare(train, test, polls), prm)


def simulate(mean, cov, n_sims=10_000, seed=0, df=6):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov + 1e-10 * np.eye(len(mean)))
    z = rng.standard_normal((n_sims, len(mean))) @ L.T
    if df:
        z *= np.sqrt((df - 2) / rng.chisquare(df, size=(n_sims, 1)))   # same variance, fatter tails
    return mean + z


def log_score(mean, cov, actual, df=6):
    """Student-t log density of the actual shares (same tails as the simulation), so a few
    lopsided races can't dominate the tuning."""
    sd = np.sqrt(np.diag(cov) * (df - 2) / df)
    z = (actual - mean) / sd
    return float(np.sum(-(df + 1) / 2 * np.log1p(z ** 2 / df) - np.log(sd)))


# ---------------- tuning and backtest ----------------

GRID_Q = [1e-6, 4e-6, 1.6e-5, 6.4e-5]
GRID_QN = [5e-7, 2e-6, 8e-6]
GRID_B = [0.01, 0.02]
GRID_X = [0.01, 0.02]


def win_log_loss(mean, cov, won):
    """Log loss of who won, from each race's marginal posterior (normal approximation).
    This is the quantity the forecasts are judged on, so it's what the settings are tuned for."""
    from math import erf, sqrt
    sd = np.sqrt(np.diag(cov))
    p = np.array([0.5 * (1 + erf((m_ - 0.5) / (s_ * sqrt(2)))) for m_, s_ in zip(mean, sd)]).clip(1e-4, 1 - 1e-4)
    return float(-np.sum(won * np.log(p) + (1 - won) * np.log(1 - p)))
TUNE_H = [7, 28, 49]


def _cycle_data(con, horizons):
    base = features.senate_races(con)
    hpolls = ts.historical_polls(con)
    polls = historical_polls(con)
    data = {}
    for h in horizons:
        races = backtest.eligible(ts.historical_races(con, h, base, hpolls))
        races = races.dropna(subset=FUND_SPEC.required)
        data[h] = (races, polls_asof(polls, h))
    return data


def tune(data, train_years, base=Params()):
    """Pick drift, bias and noise settings by out-of-sample log loss of who won, on the training cycles
    only: each training cycle is forecast from the other training cycles."""
    preps = []
    for h in TUNE_H:
        races, polls = data[h]
        for y in train_years:
            tr = races[races["year"].isin([t for t in train_years if t != y])]
            te = races[races["year"] == y]
            if tr["year"].nunique() >= 3 and not te.empty:
                preps.append(prepare(tr, te, polls[polls["year"] == y]))
    best, best_score = base, -np.inf
    for q, qn, sb, sx in product(GRID_Q, GRID_QN, GRID_B, GRID_X):
        prm = replace(base, q_state=q, q_nat=qn, sigma_b=sb, sigma_x=sx)
        score = -sum(win_log_loss(*solve(pr, prm)[:2], pr["test"]["dem_won"].to_numpy()) for pr in preps)
        if score > best_score:
            best, best_score = prm, score
    return best


def run_backtest(con, years, horizons=ts.GRID, n_sims=5000):
    data = _cycle_data(con, sorted(set(horizons) | set(TUNE_H)))
    rows, cycles, chosen = [], [], []
    for year in years:
        train_years = sorted(y for y in data[horizons[0]][0]["year"].unique() if y < year)
        if len(train_years) < 4:
            continue
        prm = tune(data, train_years)
        chosen.append({"year": year, **prm.__dict__})
        for h in horizons:
            races, polls = data[h]
            tr, te = races[races["year"] < year], races[races["year"] == year]
            if te.empty:
                continue
            mean, cov, f = forecast(tr, te, polls[polls["year"] == year], prm)
            draws = simulate(mean, cov, n_sims, seed=year + h, df=prm.df)
            p = (draws > 0.5).mean(axis=0)
            seats = (draws > 0.5).sum(axis=1)
            out = te.assign(pred_share=mean, p_dem=p, sd=np.sqrt(np.diag(cov)), days_out=h)
            rows.append(out)
            lo, hi = np.percentile(seats, [10, 90])
            actual = int(te["dem_won"].sum())
            cycles.append({"days_out": h, "year": year, "races": len(te), "dem_won_actual": actual,
                           "dem_won_expected": round(float(seats.mean()), 1), "seats_80pct_low": int(lo),
                           "seats_80pct_high": int(hi), "in_80pct": int(lo <= actual <= hi),
                           "brier": round(float(((p - te["dem_won"]) ** 2).mean()), 4)})
    return summarize(pd.concat(rows, ignore_index=True), pd.DataFrame(cycles), pd.DataFrame(chosen))


def summarize(preds: pd.DataFrame, cyc: pd.DataFrame, params: pd.DataFrame) -> dict:
    summary = []
    for h, g in preds.groupby("days_out"):
        c = cyc[cyc["days_out"] == h]
        pr = g["p_dem"].clip(1e-4, 1 - 1e-4)
        yv = g["dem_won"]
        summary.append({"days_out": h, "cycles": len(c), "races": len(g),
                        "brier": round(float(((pr - yv) ** 2).mean()), 4),
                        "log_loss": round(float(-(yv * np.log(pr) + (1 - yv) * np.log(1 - pr)).mean()), 4),
                        "pct_called": round(float(((pr > .5).astype(int) == yv).mean()) * 100, 1),
                        "share_mae_pts": round(float((g["pred_share"] - g["dem_share"]).abs().mean() * 100), 2),
                        "share_in_80pct": round(float((np.abs(g["dem_share"] - g["pred_share"]) <= 1.2816 * g["sd"]).mean()), 3),
                        "seats_in_80pct_range": f"{int(c['in_80pct'].sum())}/{len(c)}"})
    return {"overall": pd.DataFrame(summary), "cycles": cyc, "races": preds, "params": params,
            "calibration": backtest.calibration(preds[preds["days_out"] == 1])}


# ---------------- live daily forecast ----------------

def fit_for_live(con, train_until: int):
    """Settings tuned on every historical cycle up to train_until, plus training races per grid horizon."""
    data = _cycle_data(con, sorted(set(ts.GRID) | set(TUNE_H)))
    years = sorted(y for y in data[TUNE_H[0]][0]["year"].unique() if y <= train_until)
    prm = tune(data, years)
    train = {h: data[h][0][data[h][0]["year"] <= train_until] for h in ts.GRID}
    return prm, train


def _live_senate_polls(sen: pd.DataFrame, table: pd.DataFrame, election_day) -> pd.DataFrame:
    p = sen.copy()
    rank = p["population"].fillna("").str.lower().map({"lv": 0, "rv": 1, "v": 1, "a": 2}).fillna(3)
    p = p.assign(rank=rank)
    key = ["state_po", "pollster", "end_date"]
    p = p[p["rank"] == p.groupby(key)["rank"].transform("min")]          # likely voters first
    mid = pd.to_datetime(p["start_date"]) + (pd.to_datetime(p["end_date"]) - pd.to_datetime(p["start_date"])) / 2
    p["t_field"] = (pd.Timestamp(election_day) - mid).dt.days.clip(lower=0)
    p["special"] = p["state_po"].map(table.set_index("state_po")["special"].astype(int))
    return p


def daily_forecast(con, election_day, start, prm: Params, train: dict, n_sims=10_000, seed=2026,
                   poll_lookback=60):
    from datetime import timedelta
    table = ts.race_table()
    year = election_day.year
    base = ts.races_2026(con, year, table)
    sen, gbp = ts.live_polls(con, year, table)
    sen = _live_senate_polls(sen, table, election_day)
    not_up_r = 53 - int((table["seat_party"] == "R").sum())
    not_up_d = 47 - int((table["seat_party"] == "D").sum())
    race_rows, chamber_rows = [], []
    day = start
    while day <= election_day:
        h = (election_day - day).days
        now = pd.Timestamp(day)
        gsel = gbp[(gbp["available"] <= now) & (gbp["available"] > now - pd.Timedelta(days=ts.WINDOW))].copy()
        gsel["age"] = (now - gsel["available"]).dt.days
        gb = ts.weighted_average(gsel.assign(k=1), ["k"])
        races = base.copy()
        races["gb_two_party"] = gb["poll_avg"].iat[0]
        races["gb_lean"] = races["gb_two_party"] - 0.5
        # polls released by this day, fielded within the lookback window
        avail = sen[(sen["available"] <= now) & (sen["t_field"] <= h + poll_lookback)]
        assert (avail["available"] <= now).all() and (pd.to_datetime(avail["end_date"]) < now).all(), "lookahead"
        g = ts.grid_for(h)
        f, sn, ss = fundamentals(train[g], races)
        S = prior_cov(races, sn, ss, prm.sigma_reg)
        pos = {k: i for i, k in enumerate(zip(races["state_po"], races["special"]))}
        idx = np.array([pos[k] for k in zip(avail["state_po"], avail["special"])], dtype=int)
        mean, cov = posterior(f, S, avail, idx, prm)
        draws = simulate(mean, cov, n_sims, seed=seed, df=prm.df)
        wins = draws > 0.5
        # display-only poll average (same recency weighting as before)
        recent = avail[avail["available"] > now - pd.Timedelta(days=ts.WINDOW)].copy()
        recent["age"] = (now - recent["available"]).dt.days
        avg = ts.weighted_average(recent, ["state_po"]).set_index("state_po")
        leader = (recent.groupby(["state_po", "chal_name"])["chal_pct"].mean().reset_index()
                  .sort_values("chal_pct").groupby("state_po").tail(1).set_index("state_po")["chal_name"])
        is_dem = np.array([cp == "D" if ";" not in cp else (leader.get(st, "") == ch.split(";")[1])
                           for st, cp, ch in zip(races["state_po"], races["challenger_party"], races["challengers"])])
        d_seats = not_up_d + (wins & is_dem).sum(axis=1)
        i_seats = (wins & ~is_dem).sum(axis=1)
        r_seats = not_up_r + (~wins).sum(axis=1)
        chamber_rows.append({
            "date": day.isoformat(), "days_out": h,
            "generic_ballot_dem_two_party": round(float(races["gb_two_party"].iat[0]), 4),
            "gb_polls_in_window": len(gsel), "senate_polls_used": len(avail),
            "p_rep_control": round(float((r_seats >= 50).mean()), 4),
            "p_dem_control": round(float((d_seats >= 51).mean()), 4),
            "p_independents_decide": round(float(((r_seats < 50) & (d_seats < 51)).mean()), 4),
            "rep_seats_mean": round(float(r_seats.mean()), 2),
            "rep_seats_80pct_low": int(np.percentile(r_seats, 10)),
            "rep_seats_80pct_high": int(np.percentile(r_seats, 90)),
            "dem_seats_mean": round(float(d_seats.mean()), 2),
            "ind_seats_mean": round(float(i_seats.mean()), 2),
        })
        sd_post, sd_prior = np.sqrt(np.diag(cov)), np.sqrt(np.diag(S))
        p_ch = wins.mean(axis=0)
        for i, r in enumerate(races.itertuples()):
            st = r.state_po
            race_rows.append({
                "date": day.isoformat(), "days_out": h, "state_po": st, "special": r.special,
                "republican": r.rep_candidate, "challenger": leader.get(st, r.challengers.split(";")[0]),
                "p_republican": round(1 - p_ch[i], 4), "p_challenger": round(p_ch[i], 4),
                "pred_challenger_two_party": round(float(mean[i]), 4),
                "pred_sd": round(float(sd_post[i]), 4),
                "fundamentals_challenger_two_party": round(float(f[i]), 4),
                "poll_avg_challenger_two_party": round(float(avg.at[st, "poll_avg"]), 4) if st in avg.index else None,
                "pollsters_in_window": int(avg.at[st, "n_pollsters"]) if st in avg.index else 0,
                "poll_weight": round(float(1 - (sd_post[i] / sd_prior[i]) ** 2), 3),
            })
        day += timedelta(days=1)
    return {"races": pd.DataFrame(race_rows), "chamber": pd.DataFrame(chamber_rows)}
