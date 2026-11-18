# Election Forecasts — Machine Learning Winds

Stage 1 of the project plan: a SQLite database of election data and a baseline,
fundamentals-only Senate model with an honest backtest.

## Setup (Windows, Anaconda Prompt)

```
cd %USERPROFILE%\Documents\election-forecasts
pip install -r requirements.txt
python run.py all
```

`all` creates the database, downloads Senate and presidential results (MIT Election Lab),
every VoteHub poll type and FRED economic series, then runs the backtest.
You can also run each step on its own:

| Command | What it does |
|---|---|
| `python run.py init-db` | Create the tables in `db/forecasts.sqlite` |
| `python run.py fetch-results` | Download and load MEDSL Senate + President returns |
| `python run.py fetch-polls` | Download and load all VoteHub polls |
| `python run.py fetch-econ` | Download and load FRED series |
| `python run.py load-raw` | Load files already saved in `data/raw/` (MEDSL CSVs, `votehub*.json`, `fred_*.csv`) |
| `python run.py import-file PATH [--table NAME] [--sheet NAME]` | Load any CSV or Excel file into its own table |
| `python run.py backtest [--start 1990] [--sims 10000]` | Run the Senate backtest; results go to `outputs/` |
| `python tests/test_pipeline.py` | End-to-end check on synthetic data (no internet needed) |

If a download fails, open the dataset page (see `SOURCES.md`), download the CSV yourself,
save it in `data/raw/` as `medsl_senate.csv` or `medsl_president.csv`, and rerun.

## Layout

```
run.py                 command-line entry point
src/forecast/
  config.py            paths, source URLs, president's party by year
  schema.sql, db.py    SQLite tables and helpers (incl. a sources/licenses table)
  http.py              polite downloads, standard library only
  loaders/             medsl.py, votehub.py, fred.py, files.py (CSV/Excel importer)
  senate/              features.py, model.py, backtest.py
data/raw/              downloaded files (not committed)
data/manual/           hand-entered data you want to keep in git
db/                    the SQLite database (not committed)
outputs/               backtest CSVs (not committed)
tests/                 synthetic end-to-end test
```

## The model

`python run.py backtest` compares four versions on 2006–2022 (each cycle forecast from earlier cycles only):

1. **Fundamentals (original)**: presidential lean, incumbency, president's party, midterm penalty;
   error scales estimated in-sample.
2. **Fundamentals**: same, with error scales estimated leave-one-year-out (honest out-of-sample).
3. **+ generic ballot**: national environment from the generic congressional ballot polling average
   replaces the president's-party terms.
4. **+ state polls** (full model): each race's poll average blended with the fundamentals prediction by
   precision weighting. Poll error variance is `a + b / number of pollsters`, fitted on training races,
   so a race polled by one firm leans on fundamentals and a heavily polled race leans on polls.

Poll averages (`src/forecast/senate/polling.py`): weight = sqrt(sample size) x 0.5^(days out / 14),
halved for partisan polls, divided by the pollster's poll count in the race (each pollster counts ~once).

Errors are split into a national error shared by every race in a year and a state error (Student-t, fat
tails):

- **State error is per race**: sd = k x sqrt(v), where v is the fundamentals variance for unpolled races
  and the precision-weighted variance for polled ones, so heavily polled races get narrower ranges.
- **National error is shrunk toward a prior** of 2 points of two-party share (about 4 points of margin,
  the size of Senate polling misses in 2014, 2016 and 2020), worth 4 cycles of data
  (`NATIONAL_PRIOR_SD`, `NATIONAL_PRIOR_DF` in `model.py`).

The simulation draws 10,000 elections for win probabilities and seat distributions.

**Backtest data**: FiveThirtyEight's archive covers polls from the final weeks before each election,
1998–2022, so the backtest scores the *election-eve* forecast. 2024 isn't in the archive yet.

### Known simplifications

- The national-error prior was chosen after seeing the backtest, so the backtest is slightly optimistic
  about it. Judge it on 2024 and 2026 once those are scored.
- Races given 10-30% are still a bit too generous to the underdog (0 of 17 won).
- Incumbents detected by last name + party; appointed senators are missed.
- Races where independents or third parties took over 25% (e.g. Maine 2018) are excluded, and polls with
  an independent as a top-two candidate (Montana and Nebraska in 2026) aren't used yet.
- Runoffs (Georgia) use only the general-election round.

## Dynamic model (default since Nov 14, 2026)

`src/forecast/senate/dynamic.py`: exact Gaussian inference, NumPy only. Unknowns per cycle: each race's
Election Day share, a common polling bias, and each pollster's house effect. Fundamentals set the prior
(national + regional + state errors). Polls are noisy readings with sampling error, extra non-sampling error,
and drift between the field date and Election Day (national + state random walks). Drift, bias and noise
settings are tuned by out-of-sample log loss of who won, on earlier cycles only.

Backtest (2006-2022, `outputs/senate_dynamic_backtest_*.csv`): beats the simple model at every horizon up
to 42 days (e.g. 21 days out: Brier 0.046 vs 0.052) and ties at 49-56 days. Three tuning objectives were
tried with the test cycles visible, so treat the gain as indicative; 2024 and 2026 are the clean tests.

Run the old model with `python run.py forecast --model simple`.

## Daily forecasts (2026 Senate)

```
python run.py backtest-daily      # accuracy at 1-56 days before the election, 2006-2022
python run.py forecast            # 2026: one forecast per day from Sept 3 to Nov 3
```

Outputs `outputs/senate_2026_daily_races.csv` (each race, each day: both candidates' odds, poll average,
pollsters in window, poll weight) and `outputs/senate_2026_daily_chamber.csv` (each day: chances of
Republican control, Democratic control, or independents holding the balance; seat ranges).

**No lookahead.** A forecast dated D uses only polls released by D: a poll counts as released 2 days after
its last field day (VoteHub's "date added" is often weeks late because of backfilling, so it isn't used).
The code asserts this for every day. Fundamentals use 2024 presidential results and the candidate list.
Nominees come from `data/manual/senate_2026.csv`. Every primary was decided before Sept 3 except
DE, NH and RI, where the incumbent ran or the seat was open, so incumbency was already known.

**Horizon models.** Separate models are fitted at 1, 7, ..., 56 days out on 1998-2022, with poll averages
built the same way. Daily forecasts interpolate between neighbouring horizons.

**Generic ballot.** VoteHub's coverage stops in mid-2026, so 2026 uses Wikipedia's table
(`data/raw/wiki_generic_ballot_2026.tsv`); likely-voter results preferred, then registered voters, then adults.

**Special cases.** Nebraska, Idaho and South Dakota have independent challengers and no Democrat; the
challenger is modeled like a Democrat. Montana is three-way (Alme R, Bodnar I, Bankhead D): only full-field
polls are used, and the challenger is whichever non-Republican leads the poll average.

**Caveats.** At 50+ days out the historical archive has few polls, so the backtest there is weak (seat range
right in 5 of 9 cycles at 56 days). 2026 has not been scored yet: add certified results to the MEDSL file to do it.

## Website

`python run.py build-site` writes the static site to `site/dist/` (add `--preview` to open it from disk).
Code: `src/forecast/site/` (pages and server-side SVG charts) and `site/static/` (CSS, chart hover script).
Hosting: Cloudflare Pages, connected to this repository, build command empty, output directory `site/dist`,
custom domain forecasts.machinelearningwinds.com (CNAME at the domain registrar).

## Git

Data and the database are in `.gitignore`. Create the repository yourself:

```
git init
git add .
git commit -m "Stage 1: data pipeline and baseline Senate model"
```

## Spain 2026 (Congress of Deputies)

Code in `src/forecast/spain/`. Data: `data/raw/spain_data.json` (polls, regional totals, 2026 seats) and
`data/raw/spain_constituency_grid.json` (constituency results 2015–2023), both from English Wikipedia.

    python run.py spain-backtest                     # 2016, 2019A, 2019N, 2023 vs uniform/proportional swing
    python run.py spain-forecast --date 2026-11-15   # adds any missing days, then a 10,000-sim run for that day
    python run.py build-site

Daily routine until the Nov 23 freeze: refresh the polls (re-download `spain_data.json`), run
`spain-forecast`, `build-site`, commit and push. `spain-forecast` never goes past Nov 23.

Model: exact Gaussian poll filter on national shares (multinomial-shaped drift, common bias and house
effects; CIS gets a larger house effect; joint columns like FA+FV are read as sums), proportional swing
to provinces with community and province shocks estimated from 2015–2023, raking back to the national
draw, D'Hondt with a 3% constituency threshold. Party mapping per election is in `spain/data.py` (SPECS).

### Automatic daily updates (GitHub Actions)

`.github/workflows/spain-daily.yml` runs every day at 14:37 UTC (and on demand from the Actions tab):
`fetch-spain-polls` → `spain-forecast` → `build-site` → commit and push, which makes Cloudflare redeploy.
`fetch-spain-polls` refuses to overwrite the poll file if the parse looks broken (fewer than 20 polls,
missing PP/PSOE/Vox columns, implausible values); the run then fails and GitHub emails you.
After Nov 23 both steps do nothing, so the frozen forecast stays as published.

Because the Action commits to `main`, click **Fetch origin / Pull** in GitHub Desktop before making
your own changes. The Spain inputs, the daily history in `outputs/spain_daily/` and the files the site
build reads are committed for this reason (see `.gitignore`); everything else in `data/raw` and
`outputs` stays local.
