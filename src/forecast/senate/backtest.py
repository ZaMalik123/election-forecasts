"""Honest backtest: for each cycle, train only on earlier cycles, then forecast that cycle."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import model as m


def eligible(races: pd.DataFrame) -> pd.DataFrame:
    return races[(races["contested"] == 1) & (races["year"] % 2 == 0)].dropna(subset=["pres_lean", "dem_share"])


def run(races: pd.DataFrame, spec: m.Spec, test_years: list[int], n_sims: int = 10_000,
        min_train_years: int = 4) -> dict[str, pd.DataFrame]:
    data = eligible(races)
    rows, cycles = [], []
    for year in test_years:
        train = data[data["year"] < year].dropna(subset=spec.required)
        test = data[data["year"] == year].dropna(subset=[f for f in spec.required if f != "dem_share"])
        if train["year"].nunique() < min_train_years or test.empty:
            continue
        fitted = m.fit(train, spec)
        pred, seats = m.simulate(fitted, test, n_sims=n_sims, seed=int(year))
        rows.append(pred)
        actual = int(test["dem_won"].sum())
        lo, hi = np.percentile(seats, [10, 90])
        cycles.append({
            "year": year, "races": len(test), "polled": int(test.get("poll_avg", pd.Series(dtype=float)).notna().sum()),
            "dem_won_actual": actual, "dem_won_expected": round(float(seats.mean()), 1),
            "seats_80pct_low": int(lo), "seats_80pct_high": int(hi), "in_80pct": int(lo <= actual <= hi),
            "sigma_national": round(fitted.sigma_national, 4),
            "state_sd_median": round(float(np.median(fitted.state_sd(test))), 4),
            "brier": round(float(((pred["p_dem"] - pred["dem_won"]) ** 2).mean()), 4),
            "share_mae_pts": round(float((pred["pred_share"] - pred["dem_share"]).abs().mean() * 100), 2),
        })
    preds = pd.concat(rows, ignore_index=True)
    cyc = pd.DataFrame(cycles)
    return {"races": preds, "cycles": cyc, "calibration": calibration(preds), "overall": overall(preds, cyc, spec)}


def calibration(preds: pd.DataFrame) -> pd.DataFrame:
    b = pd.cut(preds["p_dem"], np.linspace(0, 1, 11), include_lowest=True)
    return (preds.groupby(b, observed=True)
            .agg(races=("dem_won", "size"), forecast=("p_dem", "mean"), actual=("dem_won", "mean"))
            .round(3).reset_index().rename(columns={"p_dem": "bin"}))


def overall(preds: pd.DataFrame, cycles: pd.DataFrame, spec: m.Spec) -> pd.DataFrame:
    p = preds["p_dem"].clip(1e-4, 1 - 1e-4)
    y = preds["dem_won"]
    tossup = preds[(preds["p_dem"] > 0.2) & (preds["p_dem"] < 0.8)]
    return pd.DataFrame([{
        "model": spec.name, "cycles": len(cycles), "races": len(preds),
        "brier": round(float(((p - y) ** 2).mean()), 4),
        "log_loss": round(float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), 4),
        "pct_called": round(float(((p > 0.5).astype(int) == y).mean()) * 100, 1),
        "share_mae_pts": round(float((preds["pred_share"] - preds["dem_share"]).abs().mean() * 100), 2),
        "seats_in_80pct_range": f"{int(cycles['in_80pct'].sum())}/{len(cycles)}",
        "races_20_80pct": len(tossup),
        "dem_win_rate_in_those": round(float(tossup["dem_won"].mean()), 3) if len(tossup) else np.nan,
        "avg_forecast_in_those": round(float(tossup["p_dem"].mean()), 3) if len(tossup) else np.nan,
    }])
