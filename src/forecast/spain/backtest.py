"""Backtest the Spain model on 2016, April 2019, November 2019 and 2023.

For each election and horizon, the forecast uses only polls published by then and swing
variances estimated from the *other* transitions. Two simple baselines get the same polls:
a 14-day poll average spread to provinces by uniform swing and by proportional swing.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from . import data as D
from . import model as M

HORIZONS = [1, 7, 14, 28, 56]


def poll_average(election: str, asof, days: int = 14) -> np.ndarray:
    m0, _, parties = M.prior(election, M.Params())
    p = M.poll_window(election, asof)
    out = m0.copy()
    for i, k in enumerate(parties):
        for d in (days, 2 * days, 4 * days):
            q = p[(p["group"] == (k,)) & (p["end"] >= pd.Timestamp(asof) - pd.Timedelta(days=d))]
            if len(q):
                out[i] = q["value"].mean()
                break
    return out


def uniform_swing_seats(election: str, nat: np.ndarray) -> np.ndarray:
    spec = D.SPECS[election]
    base = D.baseline(election)
    w = D.province_weights()[D.PREVIOUS[election]].to_numpy()
    w = w / w.sum()
    seats_c = D.seats_by_province(election).to_numpy().astype(int)
    shares = np.zeros(base.shape)
    for i, k in enumerate(spec.parties):
        b = base[k].to_numpy()
        if isinstance(spec.base[k], tuple):
            mask = b > 0
            shares[:, i] = np.where(mask, nat[i] / w[mask].sum(), 0.0)
        else:
            ran = b > 0
            # uniform swing within the provinces where the party stands (regional parties'
            # national share is converted to a share of their own provinces)
            area = w[ran].sum()
            swing = (nat[i] - (w * b).sum()) / area
            shares[:, i] = np.where(ran, np.clip(b + swing, 0, None), 0.0)
    seats = np.zeros(len(spec.parties), dtype=int)
    for c in range(len(seats_c)):
        seats += M.dhondt(shares[c][None, :], seats_c[c])[0]
    return seats


def run(prm: M.Params = M.Params(), horizons=HORIZONS, n_sims: int = 4000, elections=None):
    rows = []
    for e in elections or D.BACKTEST:
        var = M.swing_variances(exclude=e)
        actual, aseats = D.canonical_results(e)
        parties = D.SPECS[e].parties
        truth_s = aseats.loc["Total", parties].to_numpy()
        truth_v = actual.loc["Total", parties].to_numpy()
        total = int(D.seats_by_province(e).sum())
        eday = D.ELECTION_DAY[e]
        for h in horizons:
            asof = eday - timedelta(days=h)
            m, P, _, npolls = M.posterior(e, asof, prm)
            sim = M.simulate(e, m, P, n_sims=n_sims, var=var, seed=7)
            tot = sim["seats"].sum(1)
            avg = poll_average(e, asof)
            us = uniform_swing_seats(e, avg)
            ps = M.simulate(e, avg, np.eye(len(parties)) * 1e-12, n_sims=1, noise=False,
                            var=var)["seats"].sum(1)[0]
            mp = M.simulate(e, m, np.eye(len(parties)) * 1e-12, n_sims=1, noise=False,
                            var=var)["seats"].sum(1)[0]
            for i, k in enumerate(parties):
                x = tot[:, i]
                rows.append({
                    "election": e, "horizon": h, "party": k, "polls": npolls,
                    "actual_share": truth_v[i], "model_share": m[i], "avg_share": avg[i],
                    "actual_seats": truth_s[i], "model_mean": x.mean(),
                    "model_p10": np.percentile(x, 10), "model_p90": np.percentile(x, 90),
                    "model_logp": np.log(max((x == truth_s[i]).mean(), 1.0 / n_sims)),
                    "uniform_seats": us[i], "proportional_seats": ps[i], "model_point": mp[i],
                })
            rows.append({"election": e, "horizon": h, "party": "Others", "polls": npolls,
                         "actual_seats": total - truth_s.sum(), "model_mean": 0.0,
                         "uniform_seats": total - us.sum(), "proportional_seats": total - ps.sum(),
                         "model_point": total - mp.sum()})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["election", "horizon"])
    out = g.apply(lambda d: pd.Series({
        "share_mae_pp": 100 * (d["model_share"] - d["actual_share"]).abs().mean(),
        "avg_share_mae_pp": 100 * (d["avg_share"] - d["actual_share"]).abs().mean(),
        "seats_off_model": 0.5 * (d["model_mean"] - d["actual_seats"]).abs().sum(),
        "seats_off_model_point": 0.5 * (d["model_point"] - d["actual_seats"]).abs().sum(),
        "seat_logscore": d["model_logp"].sum(),
        "seats_off_uniform": 0.5 * (d["uniform_seats"] - d["actual_seats"]).abs().sum(),
        "seats_off_proportional": 0.5 * (d["proportional_seats"] - d["actual_seats"]).abs().sum(),
        "in_80pct": ((d["actual_seats"] >= d["model_p10"]) & (d["actual_seats"] <= d["model_p90"])).sum(),
        "parties": d["model_p10"].notna().sum(),
    }), include_groups=False).reset_index()
    return out
