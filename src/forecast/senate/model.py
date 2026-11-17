"""Senate models: fundamentals, national environment, and state polls.

Mean prediction (Democratic two-party share):
  1. Fundamentals regression on `spec.features` (e.g. presidential lean, incumbency, generic ballot).
  2. If `spec.use_polls`, blend with the race's poll average by precision weighting:
        mu = (mu_f / v_f + poll / v_p) / (1 / v_f + 1 / v_p)
     v_f = residual variance of the fundamentals regression,
     v_p = a + b / n_pollsters, fitted on training races by regressing (poll - result)^2 on 1 / n_pollsters.
     Races with no polls use the fundamentals prediction alone.

Uncertainty: residuals split into a national error shared by every race in a year and a state-level
error. With `spec.loyo_sigma`, residuals are leave-one-year-out (each training year predicted by a model
fit on the other training years), so the error scales reflect out-of-sample error rather than in-sample fit.

  State error, per race: sd_i = k * sqrt(v_i), where v_i = v_f for unpolled races and
  1 / (1/v_f + 1/v_p) for polled races (the precision-weighted variance), and k is fitted so the
  standardized within-year residuals have unit variance. Heavily polled races get narrower errors.

  National error: shrunk toward a prior (scaled-inverse-chi-squared) because ~10 cycles are too few to
  estimate it well. Prior: sd NATIONAL_PRIOR_SD of two-party share (about 4 points of margin, the size of
  Senate polling misses in 2014, 2016 and 2020), worth NATIONAL_PRIOR_DF cycles of data.

Simulation: national normal error + Student-t state errors (fat tails), thousands of draws.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NATIONAL_PRIOR_SD = 0.02
NATIONAL_PRIOR_DF = 4

BASE = ["pres_lean", "incumbent", "pres_sign", "midterm_x_sign"]
NATIONAL = ["pres_lean", "incumbent", "gb_lean"]


@dataclass(frozen=True)
class Spec:
    name: str
    features: tuple[str, ...]
    use_polls: bool = False
    loyo_sigma: bool = True       # leave-one-year-out error scales, per-race state error, national prior

    @property
    def required(self) -> list[str]:
        return [f for f in self.features if f != "midterm_x_sign"] + ["dem_share"]


SPECS = {
    "baseline_insample": Spec("Fundamentals (original, in-sample error)", tuple(BASE), loyo_sigma=False),
    "baseline": Spec("Fundamentals", tuple(BASE)),
    "national": Spec("Fundamentals + generic ballot", tuple(NATIONAL)),
    "polls": Spec("Fundamentals + generic ballot + state polls", tuple(NATIONAL), use_polls=True),
}


def _X(df: pd.DataFrame, feats) -> np.ndarray:
    df = df.assign(midterm_x_sign=df["midterm"] * df["pres_sign"])
    return np.column_stack([np.ones(len(df)), df[list(feats)].to_numpy(float)])


@dataclass
class MeanParams:
    coef: np.ndarray
    v_f: float
    a: float = np.nan
    b: float = np.nan


def fit_mean(train: pd.DataFrame, spec: Spec) -> MeanParams:
    X, y = _X(train, spec.features), train["dem_share"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    v_f = float(resid.var(ddof=X.shape[1]))
    params = MeanParams(coef, v_f)
    if spec.use_polls:
        polled = train.dropna(subset=["poll_avg"])
        r2 = (polled["poll_avg"] - polled["dem_share"]).to_numpy() ** 2
        Z = np.column_stack([np.ones(len(polled)), 1.0 / polled["n_pollsters"].to_numpy(float)])
        (a, b), *_ = np.linalg.lstsq(Z, r2, rcond=None)
        params.a, params.b = max(float(a), 1e-5), max(float(b), 0.0)
    return params


def predict_mean(params: MeanParams, df: pd.DataFrame, spec: Spec) -> tuple[np.ndarray, np.ndarray]:
    """Return (blended mean, weight on polls)."""
    mu_f = _X(df, spec.features) @ params.coef
    w_poll = np.zeros(len(df))
    if not spec.use_polls:
        return mu_f, w_poll
    has = df["poll_avg"].notna().to_numpy()
    v_p = params.a + params.b / df["n_pollsters"].to_numpy(float)
    w = (1 / v_p) / (1 / v_p + 1 / params.v_f)
    w_poll[has] = w[has]
    mu = mu_f.copy()
    mu[has] = (1 - w[has]) * mu_f[has] + w[has] * df["poll_avg"].to_numpy(float)[has]
    return mu, w_poll


def race_var(params: MeanParams, df: pd.DataFrame, spec: Spec) -> np.ndarray:
    """Relative error variance of each race's mean prediction (before scaling by k)."""
    v = np.full(len(df), params.v_f)
    if spec.use_polls:
        has = df["poll_avg"].notna().to_numpy()
        v_p = params.a + params.b / df["n_pollsters"].to_numpy(float)
        v[has] = 1.0 / (1.0 / params.v_f + 1.0 / v_p[has])
    return v


@dataclass
class FittedModel:
    spec: Spec
    params: MeanParams
    sigma_national: float
    state_scale: float            # k: state sd_i = k * sqrt(race_var_i)
    n_train: int
    sigma_national_data: float = np.nan
    train_years: list[int] = field(default_factory=list)

    def state_sd(self, df: pd.DataFrame) -> np.ndarray:
        return self.state_scale * np.sqrt(race_var(self.params, df, self.spec))

    @property
    def sigma_state_unpolled(self) -> float:
        return self.state_scale * np.sqrt(self.params.v_f)

    def summary(self) -> pd.Series:
        s = pd.Series(self.params.coef, index=["intercept"] + list(self.spec.features))
        if self.spec.use_polls:
            s["poll_var_floor_a"] = self.params.a
            s["poll_var_per_pollster_b"] = self.params.b
            s["fundamentals_var"] = self.params.v_f
        s["sigma_national_data_only"] = self.sigma_national_data
        s["sigma_national_with_prior"] = self.sigma_national
        s["state_scale_k"] = self.state_scale
        s["sigma_state_unpolled_race"] = self.sigma_state_unpolled
        return s.round(5)


def _decompose(resid: pd.Series, years: pd.Series, var_i: np.ndarray, n_coef: int,
               per_race: bool) -> tuple[float, float]:
    """Return (national sd from data, state scale k)."""
    year_mean = resid.groupby(years).mean()
    within = resid - years.map(year_mean)
    if per_race:
        dof = len(resid) - years.nunique()
        k = float(np.sqrt((within.to_numpy() ** 2 / var_i).sum() / dof))
        state_var = k ** 2 * pd.Series(var_i, index=resid.index)
    else:
        sd = float(within.std(ddof=n_coef))
        k = sd / float(np.sqrt(var_i.mean()))
        state_var = pd.Series(sd ** 2, index=resid.index)
    noise = float((state_var.groupby(years).mean() / years.value_counts()).mean())
    sigma_national = float(np.sqrt(max(year_mean.var(ddof=1) - noise, 1e-6)))
    return sigma_national, k


def fit(train: pd.DataFrame, spec: Spec) -> FittedModel:
    train = train.dropna(subset=spec.required)
    params = fit_mean(train, spec)
    if spec.loyo_sigma:
        resid = pd.Series(np.nan, index=train.index)
        var_i = pd.Series(np.nan, index=train.index)
        for yr in train["year"].unique():
            inner, held = train[train["year"] != yr], train[train["year"] == yr]
            inner_params = fit_mean(inner, spec)
            mu, _ = predict_mean(inner_params, held, spec)
            resid[held.index] = held["dem_share"].to_numpy() - mu
            var_i[held.index] = race_var(inner_params, held, spec)
        sn_data, k = _decompose(resid, train["year"], var_i.to_numpy(), 0, per_race=True)
        n_years = train["year"].nunique()
        sn = float(np.sqrt((NATIONAL_PRIOR_DF * NATIONAL_PRIOR_SD ** 2 + (n_years - 1) * sn_data ** 2)
                           / (NATIONAL_PRIOR_DF + n_years - 1)))
    else:
        mu, _ = predict_mean(params, train, spec)
        resid = pd.Series(train["dem_share"].to_numpy() - mu, index=train.index)
        sn_data, k = _decompose(resid, train["year"], race_var(params, train, spec), len(params.coef),
                                per_race=False)
        sn = sn_data
    return FittedModel(spec, params, sn, k, len(train), sn_data, sorted(train["year"].unique().tolist()))


def simulate(model: FittedModel, races: pd.DataFrame, n_sims: int = 10_000,
             seed: int = 0, df_state: int = 5, return_draws: bool = False):
    """Return per-race predictions and an (n_sims,) array of Democratic seats won."""
    rng = np.random.default_rng(seed)
    mean, w_poll = predict_mean(model.params, races, model.spec)
    national = rng.normal(0, model.sigma_national, size=(n_sims, 1))
    t = rng.standard_t(df_state, size=(n_sims, len(races))) * np.sqrt((df_state - 2) / df_state)
    shares = mean + national + t * model.state_sd(races)
    out = races.copy()
    out["pred_share"] = mean
    out["poll_weight"] = w_poll
    out["state_sd"] = model.state_sd(races)
    out["p_dem"] = (shares > 0.5).mean(axis=0)
    seats = (shares > 0.5).sum(axis=1)
    if return_draws:
        return out, seats, shares
    return out, seats
