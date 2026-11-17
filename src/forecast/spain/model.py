"""Spanish Congress model: a poll filter for national vote shares + a provincial swing simulation.

1. National vote shares. Unknowns: the Election-Day share vector mu (one entry per canonical
   party; "others" is the remainder), a common polling bias b, and pollster house effects h.
   A poll that reports party group G (one party, or a joint column such as FA+FV) at tau days
   before the election measures 1_G'(mu - W(tau) + b + h_pollster) + sampling error, where W is a
   random walk anchored at Election Day. Every covariance term uses the multinomial shape
   S(p) = diag(p) - p p', so shares stay (approximately) on the simplex and small parties move
   less in absolute terms than big ones:
       Cov(W(t), W(s)) = q min(t, s) S,  Cov(b) = beta S,  Cov(h_j) = eta_j S,
       sampling = S / n_eff + sigma_x^2 I.
   Prior: previous election result, Cov = kappa S (+ extra variance for new parties).
   Everything is Gaussian, so the posterior for mu is exact (generalized least squares).

2. Provinces. Each simulated national vector is spread to provinces by proportional swing from
   the previous result, with a community-level and a province-level shock in log space
   (variances estimated from past election-to-election swings), then raked so the provinces add
   back up to the national draw.

3. Seats: D'Hondt in each constituency among lists with at least 3% of the valid vote.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import data as D

RELEASE_LAG = 2      # a poll counts from two days after fieldwork ends
LOOKBACK = 240       # days of polls used
THRESHOLD = 0.03
MAJORITY = 176

NEW_PRIOR = {"Vox": 0.02, "SALF": 0.02, "AA": 0.005}
PRIOR_SPLIT = {"FA": 0.7, "Podemos": 0.3}   # how the old Sumar vote is split for the prior mean


@dataclass(frozen=True)
class Params:
    q: float = 3e-5          # random-walk variance per day (multinomial shape units)
    beta: float = 1.5e-3     # common polling-industry bias
    eta: float = 1e-3        # house effects
    eta_cis: float = 6e-3    # CIS (state pollster) house effect
    kappa: float = 0.05      # prior variance around the previous result
    new_sd: float = 0.03     # extra prior sd for parties with no previous result
    deff: float = 1.5        # design effect
    n_default: float = 1000
    sigma_x: float = 0.004   # rounding / non-sampling noise per reported figure
    df: int = 6              # tails of the national draw


# ------------------------------------------------------------------------------- national

def shape(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.003, None)
    if p.sum() > 0.98:              # keep an "others" remainder so S stays positive definite
        p = p * 0.98 / p.sum()
    return np.diag(p) - np.outer(p, p)


def prior(election: str, prm: Params) -> tuple[np.ndarray, np.ndarray, list[str]]:
    spec = D.SPECS[election]
    prev = D.PREVIOUS[election]
    w = D.province_weights()[prev].to_numpy()
    base = D.baseline(election)
    nat = (base.to_numpy() * w[:, None]).sum(0) / w.sum()
    m0 = pd.Series(nat, index=spec.parties)
    extra = np.zeros(len(spec.parties))
    for i, k in enumerate(spec.parties):
        rule = spec.base[k]
        if isinstance(rule, tuple):
            m0[k] = NEW_PRIOR.get(k, 0.01)
            extra[i] = prm.new_sd ** 2
        if k in PRIOR_SPLIT:
            m0[k] *= PRIOR_SPLIT[k]
            extra[i] = prm.new_sd ** 2
    m0 = m0.to_numpy()
    if m0.sum() > 0.97:
        m0 *= 0.97 / m0.sum()
    P0 = prm.kappa * shape(m0) + np.diag(extra)
    return m0, P0, spec.parties


def poll_window(election: str, asof: date) -> pd.DataFrame:
    p = D.polls(election)
    asof = pd.Timestamp(asof)
    p = p[(p["end"] + pd.Timedelta(days=RELEASE_LAG) <= asof)
          & (p["end"] >= asof - pd.Timedelta(days=LOOKBACK))]
    return p


def posterior(election: str, asof: date, prm: Params = Params(), passes: int = 2):
    """Posterior mean and covariance of Election-Day national shares, using polls known on asof."""
    m0, P0, parties = prior(election, prm)
    K = len(parties)
    idx = {k: i for i, k in enumerate(parties)}
    p = poll_window(election, asof)
    if p.empty:
        return m0, P0, parties, 0
    eday = pd.Timestamp(D.ELECTION_DAY[election])
    A = np.zeros((len(p), K))
    for r, g in enumerate(p["group"]):
        for k in g:
            A[r, idx[k]] = 1.0
    y = p["value"].to_numpy()
    tau = (eday - p["end"]).dt.days.to_numpy().astype(float)
    n = p["n"].fillna(prm.n_default).clip(upper=3000).to_numpy() / prm.deff
    house = p["house"].to_numpy()
    pid = p["poll_id"].to_numpy()
    same_house = house[:, None] == house[None, :]
    eta = np.where(house == "cis", prm.eta_cis, prm.eta)
    eta_ij = np.sqrt(np.outer(eta, eta)) * same_house
    same_poll = pid[:, None] == pid[None, :]
    samp = same_poll / np.sqrt(np.outer(n, n))
    tmin = np.minimum(tau[:, None], tau[None, :])

    m, P = m0, P0
    pbar = m0
    for _ in range(passes):
        S = shape(pbar)
        G = A @ S @ A.T
        Cy = (prm.q * tmin + prm.beta + eta_ij + samp) * G + prm.sigma_x ** 2 * np.eye(len(y))
        AP = A @ P0
        Syy = AP @ A.T + Cy
        Syy = 0.5 * (Syy + Syy.T) + 1e-9 * np.eye(len(y))
        L = np.linalg.cholesky(Syy)
        Z = np.linalg.solve(L, AP)
        r = np.linalg.solve(L, y - A @ m0)
        m = m0 + Z.T @ r
        P = P0 - Z.T @ Z
        pbar = np.clip(m, 0.003, None)
    return m, P, parties, int(p["poll_id"].nunique())


# ----------------------------------------------------------------------------- provinces

def swing_variances(exclude: str | None = None) -> dict[str, float]:
    """Log-scale variance of proportional-swing residuals, split into community and province parts.

    Uses past transitions (previous election -> election), skipping the one being backtested.
    National and regional parties are estimated separately.
    """
    comm, prov = {"nat": [], "reg": []}, {"nat": [], "reg": []}
    for e in D.BACKTEST:
        if e == exclude:
            continue
        actual, _ = D.canonical_results(e)
        base = D.baseline(e)
        w = D.province_weights()[D.PREVIOUS[e]]
        for k in D.SPECS[e].parties:
            if isinstance(D.SPECS[e].base[k], tuple):
                continue
            a, b = actual.loc[D.PROVINCES, k], base[k]
            ok = (a > 0.01) & (b > 0.01)
            if ok.sum() < 2:
                continue
            nat_a = (a * w).sum() / w[a > 0].sum()
            nat_b = (b * w).sum() / w[b > 0].sum()
            r = np.log(a[ok] / b[ok]) - np.log(nat_a / nat_b)
            r = r - np.average(r, weights=w[ok])
            typ = "nat" if k in D.NATIONAL_PARTIES else "reg"
            df = pd.DataFrame({"r": r, "w": w[ok], "reg": [D.REGION_OF[c] for c in r.index]})
            cm = df.groupby("reg")[["r", "w"]].apply(lambda g: np.average(g["r"], weights=g["w"]))
            comm[typ].extend(cm.to_list())
            prov[typ].extend((df["r"] - df["reg"].map(cm)).to_list())
    out = {}
    for typ in ("nat", "reg"):
        out[f"comm_{typ}"] = float(np.var(comm[typ])) if comm[typ] else 0.04
        out[f"prov_{typ}"] = float(np.var(prov[typ])) if prov[typ] else 0.01
    return out


def dhondt(shares: np.ndarray, seats: int) -> np.ndarray:
    """shares: (N, K) incl. no 'others' column. Returns (N, K) seat counts."""
    N, K = shares.shape
    s = np.where(shares >= THRESHOLD, shares, 0.0)
    q = s[:, :, None] / np.arange(1, seats + 1)[None, None, :]
    q = q.reshape(N, K * seats)
    top = np.argpartition(-q, seats - 1, axis=1)[:, :seats]
    party = top // seats
    out = np.zeros((N, K), dtype=np.int16)
    np.add.at(out, (np.repeat(np.arange(N), seats), party.ravel()), 1)
    return out


def simulate(election: str, mean: np.ndarray, cov: np.ndarray, n_sims: int = 10000,
             seed: int = 2026, var: dict | None = None, noise: bool = True):
    """Simulate national shares, provincial shares and seats. Returns dict of arrays."""
    rng = np.random.default_rng(seed)
    spec = D.SPECS[election]
    parties = spec.parties
    K = len(parties)
    var = var or swing_variances(None)
    prm_df = 6
    # national draws (multivariate t), clipped to the simplex
    L = np.linalg.cholesky(cov + 1e-10 * np.eye(K))
    z = rng.standard_normal((n_sims, K)) @ L.T
    if noise:
        z *= np.sqrt(prm_df / rng.chisquare(prm_df, size=(n_sims, 1)))
    else:
        z *= 0
    nat = np.clip(mean[None, :] + z, 0.0002, None)
    tot = nat.sum(1, keepdims=True)
    nat = np.where(tot > 0.99, nat * 0.99 / tot, nat)
    others = 1 - nat.sum(1)

    base = D.baseline(election).to_numpy()            # (C, K) pattern
    C = base.shape[0]
    w = D.province_weights()[D.PREVIOUS[election]].to_numpy()
    w = w / w.sum()
    known = np.array([not isinstance(spec.base[k], tuple) for k in parties])
    base_oth = np.clip(1 - np.where(known[None, :], base, 0).sum(1), 0.01, None)
    pattern = np.concatenate([base, base_oth[:, None]], axis=1)          # (C, K+1)
    target = np.concatenate([nat, others[:, None]], axis=1)              # (N, K+1)

    if noise:
        region = np.array([D.REGION_OF[c] for c in D.PROVINCES])
        regions, ridx = np.unique(region, return_inverse=True)
        is_nat = np.array([k in D.NATIONAL_PARTIES for k in parties] + [True])
        sd_c = np.where(is_nat, np.sqrt(var["comm_nat"]), np.sqrt(var["comm_reg"]))
        sd_p = np.where(is_nat, np.sqrt(var["prov_nat"]), np.sqrt(var["prov_reg"]))
        u = rng.standard_normal((n_sims, len(regions), K + 1)) * sd_c
        e = rng.standard_normal((n_sims, C, K + 1)) * sd_p
        shock = np.exp(u[:, ridx, :] + e)
    else:
        shock = np.ones((1, C, K + 1))
    s = pattern[None, :, :] * shock                                    # (N, C, K+1)
    for _ in range(8):
        s = s / s.sum(2, keepdims=True)
        natnow = np.einsum("c,nck->nk", w, s)
        f = np.where(natnow > 0, target / np.maximum(natnow, 1e-12), 0.0)
        s = s * f[:, None, :]
    s = s / s.sum(2, keepdims=True)

    seats_c = D.seats_by_province(election).to_numpy().astype(int)
    seats = np.zeros((s.shape[0], C, K), dtype=np.int16)
    for c in range(C):
        seats[:, c, :] = dhondt(s[:, c, :K], seats_c[c])
    return {"parties": parties, "national": nat, "provincial": s[:, :, :K], "seats": seats}


def summarize_seats(sim: dict) -> pd.DataFrame:
    tot = sim["seats"].sum(1)
    rows = []
    for i, k in enumerate(sim["parties"]):
        x = tot[:, i]
        rows.append({"party": k, "mean": x.mean(), "p10": np.percentile(x, 10),
                     "median": np.median(x), "p90": np.percentile(x, 90),
                     "share_mean": sim["national"][:, i].mean(),
                     "share_p10": np.percentile(sim["national"][:, i], 10),
                     "share_p90": np.percentile(sim["national"][:, i], 90)})
    return pd.DataFrame(rows)
