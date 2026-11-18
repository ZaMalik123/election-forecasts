"""Daily 2026 forecast: one run per day using only polls released by that day.

The forecast freezes on 23 November 2026: Spanish law (LOREG art. 69.7) bans publishing polls
in the five days before the 29 November election, so no forecast after that date is produced.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from . import data as D
from . import model as M

ELECTION = "2026"
START = date(2026, 10, 2)
FREEZE = date(2026, 11, 23)
DAILY_DIR = config.OUTPUTS / "spain_daily"

PARTY_NAMES = {
    "PP": "People's Party", "PSOE": "Socialists (PSOE)", "Vox": "Vox", "FA": "Broad Front",
    "FV": "Compromís–FV", "Podemos": "Podemos", "SALF": "Se Acabó La Fiesta", "ERC": "ERC",
    "Junts": "Junts", "Bildu": "EH Bildu", "PNV": "PNV", "BNG": "BNG", "CC": "Canarian Coalition",
    "AA": "Adelante Andalucía",
}

BLOCS = {
    "PP + Vox": ["PP", "Vox"],
    "PP + Vox + CC": ["PP", "Vox", "CC"],
    "Sánchez's 2023 investiture partners": ["PSOE", "FA", "FV", "Podemos", "ERC", "Junts", "Bildu",
                                            "PNV", "BNG", "CC"],
    "Same, without Junts": ["PSOE", "FA", "FV", "Podemos", "ERC", "Bildu", "PNV", "BNG", "CC"],
    "PSOE + PP": ["PSOE", "PP"],
}


def run_day(day: date, n_sims: int = 4000, prm: M.Params = M.Params(), var=None, keep=False):
    m, P, parties, npolls = M.posterior(ELECTION, day, prm)
    sim = M.simulate(ELECTION, m, P, n_sims=n_sims, seed=int(day.strftime("%Y%m%d")), var=var)
    tot = sim["seats"].sum(1).astype(int)
    ix = {k: i for i, k in enumerate(parties)}
    rows = []
    big = tot.max(1, keepdims=True)
    largest = (tot == big)
    nlargest = largest.sum(1)
    for k, i in ix.items():
        x = tot[:, i]
        v = sim["national"][:, i]
        rows.append({"date": day, "party": k, "name": PARTY_NAMES.get(k, k), "polls": npolls,
                     "share_mean": m[i], "share_sd": float(np.sqrt(P[i, i])),
                     "share_p10": np.percentile(v, 10), "share_p90": np.percentile(v, 90),
                     "seats_mean": x.mean(), "seats_median": np.median(x),
                     "seats_p10": np.percentile(x, 10), "seats_p90": np.percentile(x, 90),
                     "p_largest": (largest[:, i] / nlargest).mean(),
                     "p_any_seat": (x > 0).mean()})
    parties_df = pd.DataFrame(rows)
    brows = []
    for name, ks in BLOCS.items():
        s = tot[:, [ix[k] for k in ks]].sum(1)
        brows.append({"date": day, "bloc": name, "parties": " + ".join(ks), "seats_mean": s.mean(),
                      "seats_p10": np.percentile(s, 10), "seats_p90": np.percentile(s, 90),
                      "p_majority": (s >= M.MAJORITY).mean()})
    blocs_df = pd.DataFrame(brows)
    if keep:
        return parties_df, blocs_df, sim
    return parties_df, blocs_df


def daily(start: date = START, end: date | None = None, n_sims: int = 4000, only_missing=True):
    """Write one CSV pair per day into outputs/spain_daily (resumable in chunks)."""
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    end = min(end or FREEZE, FREEZE)
    var = M.swing_variances(None)
    done = []
    d = start
    while d <= end:
        f = DAILY_DIR / f"{d.isoformat()}_parties.csv"
        if not (only_missing and f.exists()):
            p, b = run_day(d, n_sims=n_sims, var=var)
            p.to_csv(f, index=False)
            b.to_csv(DAILY_DIR / f"{d.isoformat()}_blocs.csv", index=False)
            done.append(d)
        d += timedelta(days=1)
    return done


def combine():
    parts = sorted(DAILY_DIR.glob("*_parties.csv"))
    blocs = sorted(DAILY_DIR.glob("*_blocs.csv"))
    P = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True)
    B = pd.concat([pd.read_csv(f) for f in blocs], ignore_index=True)
    P.to_csv(config.OUTPUTS / "spain_2026_daily_parties.csv", index=False)
    B.to_csv(config.OUTPUTS / "spain_2026_daily_blocs.csv", index=False)
    return P, B


def latest_detail(day: date, n_sims: int = 10000):
    """Full-resolution run for the latest day: seat distributions, provinces, close seats."""
    var = M.swing_variances(None)
    p, b, sim = run_day(day, n_sims=n_sims, var=var, keep=True)
    parties = sim["parties"]
    tot = sim["seats"].sum(1).astype(int)
    dist = []
    for i, k in enumerate(parties):
        vc = pd.Series(tot[:, i]).value_counts(normalize=True).sort_index()
        dist += [{"party": k, "seats": int(s), "prob": float(pr)} for s, pr in vc.items()]
    seats_c = D.seats_by_province(ELECTION)
    prov = []
    close = []
    for c, name in enumerate(D.PROVINCES):
        S = sim["seats"][:, c, :]
        first = np.bincount(sim["provincial"][:, c, :].argmax(1), minlength=len(parties)) / S.shape[0]
        for i, k in enumerate(parties):
            sh = sim["provincial"][:, c, i]
            if sh.mean() < 0.002 and S[:, i].max() == 0:
                continue
            prov.append({"province": name, "community": D.REGION_OF[name], "seats": int(seats_c[name]),
                         "party": k, "share_mean": sh.mean(), "share_p10": np.percentile(sh, 10),
                         "share_p90": np.percentile(sh, 90), "seats_mean": S[:, i].mean(),
                         "p_any_seat": (S[:, i] > 0).mean(), "p_first": first[i]})
        # how often the most common allocation happens; low = seats genuinely in play
        keys = pd.Series([tuple(r) for r in S])
        vc = keys.value_counts(normalize=True)
        top = vc.index[0]
        alt = vc.index[1] if len(vc) > 1 else top
        diff = np.array(alt) - np.array(top)
        gainer = [parties[i] for i in np.where(diff > 0)[0]]
        loser = [parties[i] for i in np.where(diff < 0)[0]]
        close.append({"province": name, "seats": int(seats_c[name]), "p_modal": vc.iloc[0],
                      "modal": ", ".join(f"{parties[i]} {top[i]}" for i in np.argsort(top)[::-1] if top[i] > 0),
                      "next_most_likely": (f"{'/'.join(gainer)} gains from {'/'.join(loser)}"
                                           if gainer else ""),
                      "p_next": vc.iloc[1] if len(vc) > 1 else 0.0,
                      "seat_sd": float(S.std(0).sum())})
    out = config.OUTPUTS
    # the latest day's daily row uses this full 10,000-simulation run, so the pages agree
    p.to_csv(DAILY_DIR / f"{day.isoformat()}_parties.csv", index=False)
    b.to_csv(DAILY_DIR / f"{day.isoformat()}_blocs.csv", index=False)
    p.to_csv(out / "spain_2026_latest_parties.csv", index=False)
    b.to_csv(out / "spain_2026_latest_blocs.csv", index=False)
    pd.DataFrame(dist).to_csv(out / "spain_2026_seat_distribution.csv", index=False)
    pd.DataFrame(prov).to_csv(out / "spain_2026_provinces.csv", index=False)
    pd.DataFrame(close).sort_values("p_modal").to_csv(out / "spain_2026_close_seats.csv", index=False)
    return p, b
