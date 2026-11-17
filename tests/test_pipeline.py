"""End-to-end test on synthetic data shaped like the real MEDSL and VoteHub files.

Run:  python tests/test_pipeline.py
It uses a temporary database and never touches db/ or data/.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecast import config, db  # noqa: E402
from forecast.loaders import files, medsl, votehub  # noqa: E402
from forecast.senate import backtest, features, model  # noqa: E402

STATES = [f"S{i:02d}" for i in range(50)]


def make_president(rng) -> pd.DataFrame:
    lean = dict(zip(STATES, rng.normal(0, 0.08, 50)))
    rows = []
    for year in range(1976, 2025, 4):
        nat = rng.normal(0.5, 0.03)
        for st in STATES:
            d = float(np.clip(nat + lean[st] + rng.normal(0, 0.01), 0.05, 0.95))
            tot = int(rng.integers(300_000, 5_000_000))
            for cand, party, share in (("DEM CAND", "DEMOCRAT", d * 0.97), ("REP CAND", "REPUBLICAN", (1 - d) * 0.97),
                                       ("OTHER CAND", "OTHER", 0.03)):
                rows.append(dict(year=year, state=st, state_po=st, state_fips=1, state_cen=1, state_ic=1,
                                 office="US PRESIDENT", candidate=cand, party_detailed=party, writein="FALSE",
                                 candidatevotes=int(tot * share), totalvotes=tot, version=1, notes="",
                                 party_simplified=party))
    return pd.DataFrame(rows), lean


def make_senate(rng, lean) -> pd.DataFrame:
    rows, holder = [], {}
    for year in range(1976, 2025, 2):
        pres_d = (year in (1978, 1980) or 1994 <= year <= 2000 or 2010 <= year <= 2016 or year in (2022, 2024))
        swing = rng.normal(0, 0.03) + (-0.02 if pres_d else 0.02) * (year % 4 == 2)
        for i, st in enumerate(STATES):
            if (i + year // 2) % 3:  # about a third of states vote each cycle
                continue
            inc = holder.get(st)
            inc_bonus = 0.0 if inc is None else (0.03 if inc[1] == "DEMOCRAT" else -0.03)
            d = float(np.clip(0.5 + lean[st] * 1.1 + swing + inc_bonus + rng.normal(0, 0.05), 0.05, 0.95))
            tot = int(rng.integers(300_000, 5_000_000))
            dem_name = inc[0] if inc and inc[1] == "DEMOCRAT" else f"D{year}{st} SMITH"
            rep_name = inc[0] if inc and inc[1] == "REPUBLICAN" else f"R{year}{st} JONES"
            for cand, party, share in ((dem_name, "DEMOCRAT", d * 0.98), (rep_name, "REPUBLICAN", (1 - d) * 0.98),
                                       ("LIB CAND", "LIBERTARIAN", 0.02)):
                rows.append(dict(year=year, state=st, state_po=st, state_fips=1, state_cen=1, state_ic=1,
                                 office="US SENATE", district="statewide", stage="gen", special="FALSE",
                                 candidate=cand, party_detailed=party, writein="FALSE", mode="TOTAL",
                                 candidatevotes=int(tot * share), totalvotes=tot, unofficial="FALSE",
                                 version=1, party_simplified=party))
            winner = (dem_name, "DEMOCRAT") if d > 0.5 else (rep_name, "REPUBLICAN")
            holder[st] = winner if rng.random() > 0.2 else None  # some retirements
    return pd.DataFrame(rows)


SAMPLE_POLLS = [
    {"id": "gen202eme3uzn57et", "poll_type": "generic-ballot", "sample_size": "1000", "population": "rv",
     "url": "https://emersoncollegepolling.com/march-2025-national-poll/", "created_at": "2025-03-03",
     "start_date": "2025-03-02", "end_date": "2025-03-03", "pollster": "Emerson College",
     "answers": [{"choice": "Dem", "pct": 44.4}, {"choice": "Rep", "pct": 41.4}],
     "seat_name": "Generic", "sponsors": [], "internal": False, "partisan": None, "subject": "2026"},
]


def main():
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pres, lean = make_president(rng)
        pres.to_csv(tmp / "pres.csv", index=False)
        make_senate(rng, lean).to_csv(tmp / "sen.csv", index=False)

        con = db.connect(tmp / "test.sqlite")
        db.init_db(con)
        print("senate rows:", medsl.load(con, "senate", tmp / "sen.csv"))
        print("president rows:", medsl.load(con, "president", tmp / "pres.csv"))
        print("polls stored:", votehub.store(con, SAMPLE_POLLS))

        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(tmp / "My Data.csv", index=False)
        print("csv import rows:", files.import_file(con, tmp / "My Data.csv"))
        try:
            pd.DataFrame({"c": [1, 2, 3]}).to_excel(tmp / "sheet.xlsx", index=False)
            print("xlsx import rows:", files.import_file(con, tmp / "sheet.xlsx"))
        except ImportError:
            print("openpyxl not installed: skipping Excel import check")

        races = features.senate_races(con)
        assert races["incumbent"].abs().sum() > 0, "incumbent detection found nothing"
        assert races["pres_lean"].notna().mean() > 0.9, "presidential lean missing"
        res = backtest.run(races, model.SPECS['baseline'], list(range(1990, 2025, 2)), n_sims=2000)
        print(res["overall"].to_string(index=False))
        print(res["calibration"].to_string(index=False))
        print(pd.read_sql("SELECT name, license FROM sources", con).to_string(index=False))
        con.close()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
