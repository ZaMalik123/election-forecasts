"""Command-line entry point.

    python run.py init-db                  create the database tables
    python run.py fetch-results            download + load MEDSL Senate and President returns
    python run.py fetch-polls              download + load every VoteHub poll type
    python run.py fetch-econ               download + load FRED economic series
    python run.py import-file PATH [--table NAME] [--sheet NAME]   load any CSV/Excel file
    python run.py backtest                 run the baseline Senate model backtest
    python run.py load-raw                 load files already saved in data/raw (no internet needed)
    python run.py spain-backtest           backtest the Spain model on 2016, 2019 (x2) and 2023
    python run.py spain-forecast [--date YYYY-MM-DD]   daily Spain 2026 forecast up to that date (default today)
    python run.py all                      everything above, in order (except import-file)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402

from forecast import config, db  # noqa: E402
from forecast.loaders import candidates, fred, fte, medsl, votehub, wiki_gb  # noqa: E402
from forecast.loaders.files import import_file  # noqa: E402
from forecast.senate import backtest, features, model, polling, timeseries  # noqa: E402


def cmd_init_db(con, args):
    db.init_db(con)
    print(f"Database ready: {args.db}")


def cmd_fetch_results(con, args):
    db.init_db(con)
    for kind in ("senate", "president"):
        path = medsl.download(kind, force=args.force)
        print(f"  {kind}: {medsl.load(con, kind, path):,} rows loaded")


def cmd_fetch_polls(con, args):
    db.init_db(con)
    for pt, n in votehub.fetch_all(con).items():
        print(f"  {pt}: {n:,} polls")


def cmd_fetch_econ(con, args):
    db.init_db(con)
    for sid, n in fred.load_all(con).items():
        print(f"  {sid}: {n:,} observations")


def cmd_import_file(con, args):
    db.init_db(con)
    sheet = args.sheet if args.sheet is not None else 0
    n = import_file(con, Path(args.path), table=args.table, sheet=sheet)
    print(f"Imported {n:,} rows from {args.path}")


def cmd_backtest(con, args):
    races = polling.add_polls(features.senate_races(con), con)
    years = list(range(args.start, args.end + 1, 2))
    config.OUTPUTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    summary, results = [], {}
    for key, spec in model.SPECS.items():
        res = backtest.run(races, spec, years, n_sims=args.sims)
        results[key] = res
        summary.append(res["overall"])
        for name, df in res.items():
            df.to_csv(config.OUTPUTS / f"senate_backtest_{key}_{name}.csv", index=False)
    print(f"\nSenate backtest {years[0]}-{years[-1]}, each cycle forecast from earlier cycles only")
    print("(election-eve forecast: polls from the final weeks before Election Day)\n")
    print(pd.concat(summary, ignore_index=True).to_string(index=False))
    best = results["polls"]
    print("\nFull model, by cycle:\n" + best["cycles"].to_string(index=False))
    print("\nFull model, calibration:\n" + best["calibration"].to_string(index=False))
    data = backtest.eligible(races)
    full = model.fit(data[data["year"] <= years[-1]], model.SPECS["polls"])
    print("\nFull model fitted on all cycles:\n" + full.summary().to_string())
    print(f"\nCSV files written to {config.OUTPUTS}")


def cmd_backtest_daily(con, args):
    years = list(range(args.start, args.end + 1, 2))
    res = timeseries.backtest_by_horizon(con, years, n_sims=args.sims)
    config.OUTPUTS.mkdir(exist_ok=True)
    for name, df in res.items():
        df.to_csv(config.OUTPUTS / f"senate_backtest_by_horizon_{name}.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\nSenate backtest {years[0]}-{years[-1]} by days before the election (no lookahead)\n")
    print(res["overall"].to_string(index=False))


def cmd_forecast(con, args):
    from datetime import date, timedelta
    eday = date.fromisoformat(args.election_day)
    start = eday - timedelta(days=args.days)
    if args.model == "simple":
        models = timeseries.fit_all_horizons(con, train_until=args.train_until)
        res = timeseries.daily_forecast(con, eday, start, models, n_sims=args.sims)
    else:
        from forecast.senate import dynamic
        prm, train = dynamic.fit_for_live(con, train_until=args.train_until)
        print("Dynamic model settings:", {k: v for k, v in prm.__dict__.items()})
        res = dynamic.daily_forecast(con, eday, start, prm, train, n_sims=args.sims)
    config.OUTPUTS.mkdir(exist_ok=True)
    for name, df in res.items():
        df.to_csv(config.OUTPUTS / f"senate_{eday.year}_daily_{name}.csv", index=False)
    pd.set_option("display.width", 220)
    ch = res["chamber"]
    print("\nChamber, every 7 days:\n" + ch.iloc[::-7][::-1].to_string(index=False))
    last = res["races"][res["races"]["date"] == ch["date"].iat[-1]].sort_values("p_republican")
    print(f"\nRaces on {ch['date'].iat[-1]}:\n" + last.drop(columns=["date", "days_out"]).to_string(index=False))
    print(f"\nCSV files written to {config.OUTPUTS}")


def cmd_spain_backtest(con, args):
    from forecast.spain import backtest as sb
    df = sb.run(n_sims=args.sims)
    df.to_csv(config.OUTPUTS / "spain_backtest_parties.csv", index=False)
    s = sb.summarize(df)
    s.to_csv(config.OUTPUTS / "spain_backtest_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(2).to_string(index=False))


def cmd_spain_forecast(con, args):
    from datetime import date as _date
    from forecast.spain import forecast as sf
    day = min(_date.fromisoformat(args.date) if args.date else _date.today(), sf.FREEZE)
    done = sf.daily(end=day, n_sims=args.sims)
    print(f"Ran {len(done)} new day(s) through {day}")
    sf.latest_detail(day)
    sf.combine()
    print("Latest forecast written to outputs/spain_2026_*.csv")


def cmd_build_site(con, args):
    from forecast.site import build
    out = build.build(con, preview=args.preview)
    n = len(list(out.rglob("*.html")))
    print(f"Built {n} pages in {out}")


def cmd_load_raw(con, args):
    db.init_db(con)
    raw = config.DATA_RAW
    for kind in ("senate", "president"):
        f = raw / config.MEDSL_DATASETS[kind]["local_name"]
        if f.exists():
            print(f"  {kind}: {medsl.load(con, kind, f):,} rows")
        else:
            print(f"  {kind}: {f.name} not found")
    for f in sorted(raw.glob("votehub*.json")):
        for pt, n in votehub.load_file(con, f).items():
            print(f"  votehub {pt}: {n:,} polls")
    f = raw / config.FTE_RAW_POLLS["local_name"]
    if f.exists():
        print(f"  FiveThirtyEight historical polls: {fte.load(con, f):,} rows")
    if candidates.PATH.exists():
        print(f"  candidates: {candidates.load(con):,} rows")
    f = raw / wiki_gb.FILE
    if f.exists():
        print(f"  Wikipedia generic-ballot polls: {wiki_gb.load(con, f):,} rows")
    for f in sorted(raw.glob("fred_*.csv")):
        sid, n = fred.load_file(con, f)
        print(f"  FRED {sid}: {n:,} observations")


def cmd_all(con, args):
    for f in (cmd_init_db, cmd_fetch_results, cmd_fetch_polls, cmd_fetch_econ, cmd_backtest):
        print(f"\n== {f.__name__[4:].replace('_', '-')} ==")
        f(con, args)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Machine Learning Winds election forecasts")
    ap.add_argument("--db", type=Path, default=config.DB_PATH, help="SQLite database path")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db")
    for name in ("fetch-results", "all"):
        p = sub.add_parser(name)
        p.add_argument("--force", action="store_true", help="re-download files that already exist")
        if name == "all":
            p.add_argument("--start", type=int, default=2006)
            p.add_argument("--end", type=int, default=2022)
            p.add_argument("--sims", type=int, default=10_000)
    sub.add_parser("fetch-polls")
    sub.add_parser("fetch-econ")
    sub.add_parser("load-raw")
    p = sub.add_parser("spain-backtest", help="backtest the Spain model")
    p.add_argument("--sims", type=int, default=4000)
    p = sub.add_parser("spain-forecast", help="daily Spain 2026 forecast")
    p.add_argument("--date", default=None)
    p.add_argument("--sims", type=int, default=4000)
    p = sub.add_parser("build-site", help="generate the static website in site/dist")
    p.add_argument("--preview", action="store_true", help="explicit index.html links for opening from disk")
    p = sub.add_parser("backtest-daily", help="backtest the daily model at 1-56 days before elections")
    p.add_argument("--start", type=int, default=2006)
    p.add_argument("--end", type=int, default=2022)
    p.add_argument("--sims", type=int, default=5000)
    p = sub.add_parser("forecast", help="daily Senate forecast for a cycle, using only data available each day")
    p.add_argument("--election-day", default="2026-11-03")
    p.add_argument("--days", type=int, default=61, help="start this many days before Election Day")
    p.add_argument("--train-until", type=int, default=2022, help="last historical cycle to train on")
    p.add_argument("--sims", type=int, default=10000)
    p.add_argument("--model", choices=["dynamic", "simple"], default="dynamic")
    p = sub.add_parser("import-file")
    p.add_argument("path")
    p.add_argument("--table")
    p.add_argument("--sheet")
    p = sub.add_parser("backtest")
    p.add_argument("--start", type=int, default=2006, help="first cycle to test")
    p.add_argument("--end", type=int, default=2022, help="last cycle to test (polls archive ends 2022)")
    p.add_argument("--sims", type=int, default=10_000, help="simulations per cycle")
    args = ap.parse_args(argv)

    con = db.connect(args.db)
    try:
        globals()["cmd_" + args.cmd.replace("-", "_")](con, args)
    finally:
        con.close()


if __name__ == "__main__":
    main()
