"""Static site generator: python run.py build-site [--preview]

Writes site/dist/. Production links are directory-style (/us/2026/senate/); --preview writes explicit
index.html links so the folder can be opened from disk or a file preview.
"""
from __future__ import annotations

import html
import shutil
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

from .. import config
from ..senate import timeseries as ts
from . import spain as spain_pages
from . import svg

SITE = config.ROOT / "site"
DIST = SITE / "dist"
SITE_NAME = "Machine Learning Winds Forecasts"
BASE_URL = "https://forecasts.machinelearningwinds.com"
PARTY = {"D": ("--dem", "Democrat"), "R": ("--rep", "Republican"), "I": ("--ind", "Independent")}
E = html.escape


class Site:
    def __init__(self, preview: bool):
        self.preview = preview

    def href(self, path: str, depth: int) -> str:
        """path like 'us/2026/senate/' relative to the site root."""
        prefix = "../" * depth
        if path in ("", "/"):
            path = ""
        if self.preview and (path == "" or path.endswith("/")):
            path += "index.html"
        return prefix + path if prefix or path else "./"

    def page(self, rel_dir: str, title: str, description: str, body: str, nav: str = "") -> None:
        depth = len([p for p in rel_dir.split("/") if p])
        h = lambda p: self.href(p, depth)
        cur = lambda key: ' aria-current="page"' if key == nav else ""
        doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{E(title)}</title>
<meta name="description" content="{E(description)}">
<link rel="canonical" href="{BASE_URL}/{rel_dir}">
<meta property="og:title" content="{E(title)}">
<meta property="og:description" content="{E(description)}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="{h('static/style.css')}">
<link rel="icon" href="{h('static/favicon.svg')}" type="image/svg+xml">
</head>
<body>
<header class="site-head"><div class="wrap">
  <a class="mark" href="{h('')}"><svg viewBox="0 0 30 20" aria-hidden="true"><path d="M2 5h17a3 3 0 1 0-3-3"/><path d="M2 10h24a3 3 0 1 1-3 3"/><path d="M2 15h11"/></svg>Machine Learning Winds <span>Forecasts</span></a>
  <nav class="site-nav" aria-label="Main">
    <a href="{h('spain/2026/')}"{cur('home')}>Spain 2026</a>
    <a href="{h('us/2026/senate/')}"{cur('us')}>US Senate 2026</a>
    <a href="{h('methodology/')}"{cur('method')}>Methodology</a>
    <a href="{h('about/')}"{cur('about')}>About</a>
  </nav>
</div></header>
<main><div class="wrap">
{body}
</div></main>
<footer class="site-foot"><div class="wrap">
  <span>© {date.today().year} Machine Learning Winds LLC. Forecasts are probabilities, not predictions.</span>
  <nav aria-label="Footer"><a href="{h('methodology/')}">Methodology and sources</a><a href="{h('privacy/')}">Privacy</a><a href="{h('terms/')}">Terms</a></nav>
</div></footer>
<script src="{h('static/charts.js')}" defer></script>
</body>
</html>
"""
        out = DIST / rel_dir / "index.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(doc, encoding="utf-8")


def pct(p: float) -> str:
    if p >= 0.995:
        return ">99%"
    if p <= 0.005:
        return "<1%"
    return f"{p * 100:.0f}%"


def last_name(name: str) -> str:
    return name.split()[-1] if name else ""


def chal_party(row, leader: str) -> str:
    parties, names = row["challenger_party"].split(";"), row["challengers"].split(";")
    for p, n in zip(parties, names):
        if last_name(n) == last_name(leader):
            return p
    return parties[0]


# ---------------- pages ----------------

def build_home(site: Site, chamber: pd.DataFrame, es: dict) -> None:
    dates = chamber["date"].tolist()
    mini = svg.timeline(dates, [{"name": "Democrats", "values": chamber["p_dem_control"].tolist(), "color": "--dem",
                                  "end_label": f"{pct(chamber['p_dem_control'].iat[-1])}"}],
                        width=640, height=220, shade=("--dem-wash", "--rep-wash"),
                        label="Chance Democrats control the Senate, Sept 3 to Nov 3, 2026")
    status = (f"Final forecast, frozen on {svg.nice_date(es['asof'])} under Spain's pre-election poll ban." if es["frozen"] else
              f"Updated {svg.nice_date(es['asof'], True)}. Updates daily until November 23, when Spain's pre-election poll ban begins.")
    lead = "The PP" if es["lead"] == "PP" else "The PSOE"
    body = f"""
<section class="feature">
  <div>
    <h1>Spain votes on November 29</h1>
    <p class="lede">{lead} has a {pct(es['lead_p'])} chance of winning the most seats in the Congress of Deputies. The bigger question is whether any bloc reaches the 176 seats needed for a majority.</p>
    <p class="when">{status}</p>
    <p><a class="btn" href="{site.href('spain/2026/', 0)}">See the Spain forecast</a></p>
  </div>
  <div class="topline">
    <div class="chance" style="--c: var(--pp)"><div class="n">{pct(es['right'])}</div><div class="l">chance PP and Vox win 176 seats or more</div></div>
    <div class="chance" style="--c: var(--psoe)"><div class="n">{pct(es['left'])}</div><div class="l">chance the parties that backed Sánchez in 2023 do</div></div>
  </div>
</section>
<section style="padding-bottom: 32px; border-bottom: 1px solid var(--rule)">{spain_pages.seat_bar(es['latest'])}<p class="small">Average seats in 10,000 simulations, parties ordered from left to right. The line marks 176.</p></section>
<section class="archive">
  <div>
    <h2 style="margin-top:0">US Senate 2026, backtested</h2>
    <p>We rebuilt the 2026 Senate forecast for every day from September 3 to Election Day, using only the polls that were public on each date. The final forecast gave Democrats a {pct(chamber['p_dem_control'].iat[-1])} chance of control.</p>
    <p><a class="btn" href="{site.href('us/2026/senate/', 0)}">See the Senate backtest</a></p>
  </div>
  {svg.figure(mini, legend=None, caption="Chance Democrats control the Senate. Blue shading means Democrats were favored; red means Republicans were.")}
</section>"""
    site.page("", "Election forecasts | Machine Learning Winds",
              "Probabilistic election forecasts, starting with Spain's November 29, 2026 general election.", body, "home")


def build_senate(site: Site, chamber: pd.DataFrame, races: pd.DataFrame, table: pd.DataFrame) -> None:
    dates = chamber["date"].tolist()
    last = chamber.iloc[-1]
    hero = svg.timeline(dates, [{"name": "Democratic control", "values": chamber["p_dem_control"].tolist(), "color": "--dem",
                                  "end_label": f"Democrats {pct(last.p_dem_control)}"},
                                 {"name": "Republican control", "values": chamber["p_rep_control"].tolist(), "color": "--rep",
                                  "end_label": f"Republicans {pct(last.p_rep_control)}"}],
                        label="Chance each party controls the Senate, Sept 3 to Nov 3, 2026")
    rows = "".join(f"<tr><td>{svg.nice_date(r.date)}</td><td class='num'>{pct(r.p_dem_control)}</td><td class='num'>{pct(r.p_rep_control)}</td>"
                   f"<td class='num'>{pct(r.p_independents_decide)}</td><td class='num'>{r.rep_seats_mean:.1f}</td></tr>" for r in chamber.itertuples())
    num_table = ("<table><thead><tr><th>Date</th><th class='num'>Democratic control</th><th class='num'>Republican control</th>"
                 "<th class='num'>Independents decide</th><th class='num'>Expected Republican seats</th></tr></thead><tbody>" + rows + "</tbody></table>")

    fin = races[races["date"] == races["date"].max()].copy()
    fin["closeness"] = (fin["p_challenger"] - 0.5).abs()
    t = table.set_index("state_po")
    trs = []
    for r in fin.sort_values("closeness").itertuples():
        row = t.loc[r.state_po]
        cp = chal_party(row, r.challenger)
        ccol, cname = PARTY[cp]
        fav_rep = r.p_republican >= r.p_challenger
        fav = f"{E(r.republican)} {pct(r.p_republican)}" if fav_rep else f"{E(r.challenger)} {pct(r.p_challenger)}"
        name = row["state"] + (" (special)" if int(row["special"]) else "")
        trs.append(
            f'<tr class="race"><td><a href="{site.href(f"us/2026/senate/{r.state_po.lower()}/", 3)}">{E(name)}</a></td>'
            f'<td><span class="who">{E(r.republican)} <small>R</small></span><span class="who">{E(r.challenger)} <small>{cp}</small></span></td>'
            f'<td class="hide-sm">{svg.split_bar([(r.p_republican, "--rep"), (r.p_challenger, ccol)], "racebar")}</td>'
            f"<td>{fav}</td></tr>")
    map_races = {}
    for r in fin.itertuples():
        row = t.loc[r.state_po]
        cp = chal_party(row, r.challenger)
        sname = row["state"] + (" (special)" if int(row["special"]) else "")
        first = [(r.challenger, cp, r.p_challenger), (r.republican, "R", r.p_republican)]
        first.sort(key=lambda x: -x[2])
        lines = "".join(f'<span style="--c: var({PARTY[p][0]})"><em>{E(n)} ({p})</em>{pct(v)}</span>' for n, p, v in first)
        map_races[r.state_po] = {
            "p_rep": r.p_republican, "href": site.href(f"us/2026/senate/{r.state_po.lower()}/", 3),
            "tip": f"<b>{E(sname)}</b>{lines}",
            "aria": f"{sname}: {first[0][0]} {pct(first[0][2])} to win",
        }
    body = f"""
<div class="notice"><strong>This is a backtest, not a live forecast.</strong>We built it after Election Day. Each day's numbers use only polls that had been released by that date, so they show what this model would have said at the time. We have not scored it against the results yet.</div>
<h1>2026 US Senate forecast</h1>
<p class="lede">On Election Day the model gave Democrats a {pct(last.p_dem_control)} chance of controlling the Senate and Republicans {pct(last.p_rep_control)}. Independents held the balance in {pct(last.p_independents_decide)} of simulations.</p>
<div class="topline">
  <div class="chance" style="--c: var(--dem)"><div class="n">{pct(last.p_dem_control)}</div><div class="l">chance of Democratic control</div></div>
  <div class="chance" style="--c: var(--rep)"><div class="n">{pct(last.p_rep_control)}</div><div class="l">chance of Republican control</div></div>
  <div class="chance" style="--c: var(--ink)"><div class="n">{last.dem_seats_mean:.0f}–{last.rep_seats_mean:.0f}</div><div class="l">expected seats, Democrats to Republicans (Republican range {last.rep_seats_80pct_low}–{last.rep_seats_80pct_high} in 8 of 10 simulations)</div></div>
</div>
<h2>The map on Election Day</h2>
<p class="prose">Each state is shaded by the chance its winner is a Republican. Deeper blue means a surer non-Republican win and deeper red a surer Republican win; pale states were close. Select a state for its forecast.</p>
<figure class="chart map" data-map>{svg.us_map(map_races, label="Map of each state's chance of electing a Republican senator on Election Day 2026")}<div class="tip" aria-hidden="true"></div>
{svg.map_key()}
<figcaption>In Idaho, Montana, Nebraska and South Dakota, the main challenger is an independent, so blue there means the independent winning. Florida and Ohio are special elections.</figcaption></figure>
<h2>How the odds moved</h2>
{svg.figure(hero, legend=[("Democratic control", "--dem"), ("Republican control", "--rep")],
            caption="Democrats need 51 seats; Republicans keep control with 50 because the vice president breaks ties. The two lines don't add to 100% when independents could decide control.",
            table_html=num_table)}
<h2>Every race</h2>
<p class="small">Closest races first. Chances are from the Election Day forecast. Independents running without a Democratic opponent are shown as the challenger.</p>
<div class="tablewrap"><table><thead><tr><th>State</th><th>Candidates</th><th class="hide-sm">Chance of winning</th><th>Favored</th></tr></thead><tbody>{''.join(trs)}</tbody></table></div>
<h2>Download the data</h2>
<p><a href="{site.href('data/senate_2026_daily_chamber.csv', 3)}">Daily chamber forecast (CSV)</a><br><a href="{site.href('data/senate_2026_daily_races.csv', 3)}">Daily race forecasts (CSV)</a></p>
"""
    site.page("us/2026/senate/", "2026 US Senate forecast (backtest) | Machine Learning Winds",
              "Daily 2026 US Senate forecast from September 3 to Election Day, built from polls public on each date.", body, "us")


def build_race_pages(site: Site, races: pd.DataFrame, table: pd.DataFrame, polls: pd.DataFrame) -> None:
    t = table.set_index("state_po")
    for st, g in races.groupby("state_po"):
        g = g.sort_values("date")
        row = t.loc[st]
        fin = g.iloc[-1]
        cp = chal_party(row, fin.challenger)
        ccol, cname = PARTY[cp]
        name = row["state"] + (" special election" if int(row["special"]) else "")
        dates = g["date"].tolist()
        odds = svg.timeline(dates, [{"name": fin.republican, "values": g["p_republican"].tolist(), "color": "--rep",
                                      "end_label": f"{last_name(fin.republican)} {pct(fin.p_republican)}"},
                                     {"name": fin.challenger, "values": g["p_challenger"].tolist(), "color": ccol,
                                      "end_label": f"{last_name(fin.challenger)} {pct(fin.p_challenger)}"}],
                            label=f"Chance of winning, {name}")
        share_series = [{"name": "Forecast", "values": g["pred_challenger_two_party"].tolist(), "color": ccol,
                         "end_label": f"Forecast {fin.pred_challenger_two_party * 100:.1f}%"}]
        has_polls = g["poll_avg_challenger_two_party"].notna().all()
        if has_polls:
            share_series.append({"name": "Poll average", "values": g["poll_avg_challenger_two_party"].tolist(), "color": "--muted",
                                 "end_label": f"Polls {fin.poll_avg_challenger_two_party * 100:.1f}%"})
        lo = min(min(s["values"]) for s in share_series)
        hi = max(max(s["values"]) for s in share_series)
        y_min, y_max = min(0.40, round(lo - 0.03, 2)), max(0.60, round(hi + 0.03, 2))
        ticks = [v / 100 for v in range(int(y_min * 100 + 4) // 5 * 5, int(y_max * 100) + 1, 5)]
        share = svg.timeline(dates, share_series, y_min=y_min, y_max=y_max, ticks=ticks,
                             fmt=lambda v: f"{v * 100:.0f}%", label=f"{fin.challenger}'s share of the two-candidate vote")
        p = polls[polls["state_po"] == st].sort_values("end_date", ascending=False)
        if len(p):
            prow = "".join(
                f"<tr><td>{'<a href=' + chr(34) + E(r.url) + chr(34) + ' rel=' + chr(34) + 'nofollow noopener' + chr(34) + '>' + E(r.pollster) + '</a>' if isinstance(r.url, str) and r.url.startswith('http') else E(r.pollster)}"
                f"{' <small>partisan</small>' if r.partisan_flag else ''}</td>"
                f"<td>{svg.date_range(r.start_date, r.end_date)}</td>"
                f"<td class='num'>{'' if pd.isna(r.sample_size) else f'{int(r.sample_size):,}'} {E(str(r.population or '').upper())}</td>"
                f"<td class='num'>{r.rep_pct:g}%</td><td class='num'>{r.chal_pct:g}% <small>{E(last_name(r.chal_name))}</small></td></tr>"
                for r in p.itertuples())
            poll_html = (f"<div class='tablewrap'><table><thead><tr><th>Pollster</th><th>Dates</th><th class='num'>Sample</th>"
                         f"<th class='num'>{E(last_name(fin.republican))}</th><th class='num'>Challenger</th></tr></thead><tbody>{prow}</tbody></table></div>"
                         "<p class='small'>Poll data from VoteHub (CC BY 4.0) with links to each original release. Partisan polls count half.</p>")
        else:
            poll_html = "<p>No public polls of this race were released between August and Election Day, so the forecast rests on the state's partisan lean, incumbency and the national environment.</p>"
        inc = int(row["incumbent"])
        inc_txt = (f"{E(row['incumbent_name'])} is the incumbent." if inc else "Open seat: no incumbent on the ballot.")
        notes = f"<p class='small'>{E(row['notes'])}</p>" if row["notes"] else ""
        body = f"""
<p class="small"><a href="{site.href('us/2026/senate/', 4)}">US Senate 2026 backtest</a></p>
<h1>{E(name)}</h1>
<p class="lede">{E(fin.republican)} (Republican) vs {E(fin.challenger)} ({cname}). {inc_txt}</p>
<div class="topline">
  <div class="chance" style="--c: var(--rep)"><div class="n">{pct(fin.p_republican)}</div><div class="l">chance {E(fin.republican)} wins</div></div>
  <div class="chance" style="--c: var({ccol})"><div class="n">{pct(fin.p_challenger)}</div><div class="l">chance {E(fin.challenger)} wins</div></div>
</div>
{notes}
<h2>Chance of winning over time</h2>
{svg.figure(odds, legend=[(fin.republican, "--rep"), (fin.challenger, ccol)])}
<h2>Forecast vote share and the polls</h2>
{svg.figure(share, legend=[(s["name"], s["color"]) for s in share_series],
            caption=f"{E(fin.challenger)}'s share of the vote between the top two candidates. The forecast starts from the state's fundamentals and updates on every poll, adjusted for each pollster's lean. By Election Day the polls had removed {max(fin.poll_weight, 0) * 100:.0f}% of the starting uncertainty.")}
<h2>Polls</h2>
{poll_html}
"""
        site.page(f"us/2026/senate/{st.lower()}/", f"{name} Senate forecast 2026 (backtest) | Machine Learning Winds",
                  f"Daily forecast for the 2026 {name} Senate race: {fin.republican} vs {fin.challenger}.", body, "us")


def spain_backtest_table() -> str:
    bt = pd.read_csv(config.OUTPUTS / "spain_backtest_summary.csv")
    g = bt.groupby("horizon").agg(model=("seats_off_model_point", "mean"), prop=("seats_off_proportional", "mean"),
                                  unif=("seats_off_uniform", "mean"), share=("share_mae_pp", "mean"),
                                  inr=("in_80pct", "sum"), n=("parties", "sum"))
    rows = "".join(f"<tr><td class='num'>{h}</td><td class='num'>{r.model:.0f}</td><td class='num'>{r.prop:.0f}</td>"
                   f"<td class='num'>{r.unif:.0f}</td><td class='num'>{r.share:.1f}</td><td class='num'>{r.inr:.0f} of {r.n:.0f}</td></tr>"
                   for h, r in g.iterrows())
    return ("<div class='tablewrap' style='max-width: 760px'><table><thead><tr><th class='num'>Days before election</th>"
            "<th class='num'>Seats misallocated: our model</th><th class='num'>Poll average, proportional swing</th>"
            "<th class='num'>Poll average, uniform swing</th><th class='num'>Vote-share error (points)</th>"
            "<th class='num'>Seat counts inside 80% range</th></tr></thead><tbody>" + rows + "</tbody></table></div>"
            "<p class='small'>Seats misallocated is half the sum, over all parties, of the gap between forecast and actual seats, "
            "averaged over the four elections. Our model's figure uses the most likely seat count given its central vote estimate.</p>")


def build_methodology(site: Site) -> None:
    spain_bt = spain_backtest_table()
    hz = pd.read_csv(config.OUTPUTS / "senate_dynamic_backtest_overall.csv")
    old = pd.read_csv(config.OUTPUTS / "senate_backtest_by_horizon_overall.csv").set_index("days_out")
    rows = "".join(f"<tr><td class='num'>{r.days_out}</td><td class='num'>{r.brier:.3f}</td><td class='num'>{old.at[r.days_out, 'brier']:.3f}</td>"
                   f"<td class='num'>{r.pct_called:.0f}%</td><td class='num'>{r.share_mae_pts:.1f}</td><td class='num'>{r.seats_in_80pct_range}</td></tr>"
                   for r in hz.itertuples())
    body = f"""
<div class="prose">
<h1>How the forecasts work</h1>
<p class="lede">Each forecast is a simulation. We draw thousands of possible elections from a model of how far polls and fundamentals usually miss, then count how often each candidate or party wins.</p>
<h2>US Senate</h2>
<p>A number of factors influence our models: the state's lean in the last presidential election, whether an incumbent is running, and the national environment measured by generic-ballot polls. This gives our starting estimates.</p>
<p>We use the polls to update our races. The model treats each poll as a noisy reading of where the race will finish, and it accounts for four things a simple average misses:</p>
<ul>
<li><strong>Pollster leans.</strong> Some firms consistently lean toward one party. The model estimates each pollster's lean from all the races it polls in a year and corrects for it. Campaign and party polls are allowed a larger lean.</li>
<li><strong>Time.</strong> Opinion moves before Election Day, so a poll from six weeks ago is less important than one from last week. </li>
<li><strong>Shared errors.</strong> Polls can miss in the same direction everywhere, as they did in 2020. Our model treats polls as correlated pieces of evidence to account for this error.</li>
<li><strong>Similar states.</strong> States share a national swing and a regional one, so a poll in one state also nudges its neighbors. This is another aspect of that aforementioned correlation in polling.</li>
</ul>
<p>As we use a linear-Gaussian model, our posterior estimation admits a closed-form solution. We then draw 10,000 simulated elections from it to get each candidate's chances and the range of Senate seats. A poll counts as public two days after its last day of fieldwork.</p>
<h3>How accurate it has been</h3>
<p>We test the model on every Senate election from 2006 to 2022, forecasting each year using only earlier years, including the settings for drift and pollster leans. The Brier score measures probability accuracy: 0 is perfect and 0.25 is a coin flip. The table also shows our earlier, simpler model, which blended a weighted poll average with fundamentals.</p>
</div>
<div class="tablewrap" style="max-width: 760px"><table><thead><tr><th class="num">Days before election</th><th class="num">Brier score</th><th class="num">Earlier model</th><th class="num">Races called</th><th class="num">Vote-share error (points)</th><th class="num">Seat count inside 80% range</th></tr></thead><tbody>{rows}</tbody></table></div>
<div class="prose">
<h2 id="spain">Spain</h2>
<p>The Spanish model works in three steps.</p>
<p><strong>National vote shares.</strong> Every published poll is treated as a noisy reading of each party's vote on election day. As in the US model, the polls are combined with pollster leans, drift over time and an error shared by all pollsters. Shares of different parties move together the way a fixed electorate splits between them: when one party gains, the others lose in proportion to their size. A joint figure, such as Broad Front and Compromís–FV reported together, is used as an estimate of their combined share. The state pollster CIS has historically tilted towards PSOE far more than private pollsters, so the model accounts for this explicitly. The starting point is the previous election's result, with a wide range around it.</p>
<p><strong>Constituencies.</strong> Each simulated national result is spread across the 52 constituencies by proportional swing: a party that is up 10% nationally is assumed up about 10% everywhere it stands. Each simulation adds random shifts for every autonomous community and every province. Their sizes come from how far real results strayed from proportional swing between 2015 and 2023. The provinces are then adjusted so they add back up to the national result.</p>
<p><strong>Seats.</strong> Seats are allocated with the D'Hondt method in each constituency, among lists with at least 3% of the valid vote there (blank ballots count as valid). Ceuta and Melilla elect one deputy each. We draw 10,000 simulations.</p>
<p>Each day's forecast uses only polls published by that day. A poll counts as public two days after its fieldwork ends. Spanish law bans publishing polls in the five days before the vote, so the forecast freezes on November 23.</p>
<h3>How accurate it has been</h3>
<p>We tested the model on the elections of 2016, April 2019, November 2019 and 2023. Each test used only the polls available at the time, plus constituency shifts estimated from the other elections. Our model performs roughly as well as if we just trusted the polls head-on (either allocating seats by a proportional or uniform swing).</p>
</div>
{spain_bt}
<div class="prose">
<p>Our (slightly more) sophisticated model is roughly comparable to predicting just by polls alone. Our hope is to treat this as a learning experience and improve modeling/forecasting results as we study more elections moving forward. A benefit of our model is that we can provide honest uncertainty regarding our prediction. Across the previous four elections, about 9 in 10 parties' seat totals landed inside its 80% range, so if anything the ranges are slightly too wide.</p>
<p>We set the model's drift, pollster-lean and shared-error settings in advance rather than fitting them to these four elections. Trying a range of alternatives changed the backtest very little.</p>
<h2>Sources</h2>
<ul>
<li>Election results: MIT Election Data and Science Lab, U.S. Senate and President statewide returns (CC0).</li>
<li>2026 polls: VoteHub Polling API (CC BY 4.0), with links to every original release.</li>
<li>Historical polls: FiveThirtyEight pollster-ratings archive (CC BY 4.0).</li>
<li>2026 generic-ballot polls: Wikipedia, "2026 United States elections polling" (CC BY-SA 4.0).</li>
<li>Candidates: FEC candidate data and Ballotpedia.</li>
<li>Spain: constituency results for 2015 to 2023, voting-intention polls and the 2026 seat allocation, from English Wikipedia (CC BY-SA 4.0): the "Results of the ... Spanish Congress election" and "Opinion polling for the ... Spanish general election" pages. The underlying figures come from Spain's Ministry of the Interior and the pollsters.</li>
</ul>
<h2>Corrections</h2>
<p>When we find a mistake in our data or code, we fix it, rerun the forecast, and note the change on this page.</p>
<h3>Changes</h3>
<ul><li>November 18, 2026: Spain forecast: daily tracking polls and rolling panels now count once per non-overlapping fieldwork period, instead of every daily release counting as a new poll. The whole daily history was recomputed this way; the forecast barely moved.</li><li>November 14, 2026: the Spain forecast launched.</li><li>November 14, 2026: the US Senate model now estimates pollster leans, opinion drift and shared polling errors jointly. It replaced a model that blended a weighted poll average with fundamentals.</li></ul>
</div>"""
    site.page("methodology/", "Methodology | Machine Learning Winds", "How Machine Learning Winds builds and tests its election forecasts.", body, "method")


def build_static_pages(site: Site) -> None:
    about = """<div class="prose"><h1>About</h1>
<p class="lede">Machine Learning Winds builds probabilistic election forecasts, starting with the United States and expanding to elections that get little data coverage in English.</p>
<p>Forecasts here are probabilities, not predictions. A candidate with a 70% chance should lose about three times in ten, and a forecast that never misses is a forecast that is too timid.</p>
<p>We publish our methods and test them against past elections before trusting them with a live race. When a forecast is a backtest built after the fact, we say so at the top of the page.</p>
<p>Machine Learning Winds LLC is based in Colorado.</p>
<h2>Contact</h2>
<p>Questions, corrections and press inquiries: <a href="mailto:research.fellow@machinelearningwinds.com">research.fellow@machinelearningwinds.com</a></p></div>"""
    privacy = """<div class="prose"><h1>Privacy</h1>
<p class="lede">Draft policy, to be reviewed before launch.</p>
<p>This site does not use cookies, accounts or tracking scripts. Our hosting provider keeps standard server logs, such as IP addresses and pages requested, to run and protect the site.</p>
<p>If we add analytics or advertising, we will update this page first and ask for consent where the law requires it.</p></div>"""
    terms = """<div class="prose"><h1>Terms</h1>
<p class="lede">Draft terms, to be reviewed before launch.</p>
<p>Forecasts are provided for information only. They are estimates with real uncertainty and are not advice of any kind, including financial or betting advice.</p>
<p>You may share and quote our forecasts with credit to Machine Learning Winds and a link to this site. Poll data belongs to the pollsters and sources credited on each page.</p></div>"""
    site.page("about/", "About | Machine Learning Winds", "Who builds these forecasts and how we work.", about, "about")
    site.page("privacy/", "Privacy | Machine Learning Winds", "Privacy policy.", privacy)
    site.page("terms/", "Terms | Machine Learning Winds", "Terms of use.", terms)
    notfound = """<h1>Page not found</h1><p class="lede">That page doesn't exist. It may have moved when a forecast was updated.</p>"""
    site.page("404/", "Page not found | Machine Learning Winds", "Page not found.", notfound)
    shutil.move(str(DIST / "404" / "index.html"), str(DIST / "404.html"))


def build(con: sqlite3.Connection, preview: bool = False) -> Path:
    DIST.mkdir(parents=True, exist_ok=True)
    site = Site(preview)
    static = DIST / "static"
    static.mkdir(exist_ok=True)
    for f in (SITE / "static").iterdir():
        shutil.copyfile(f, static / f.name)
    (static / "favicon.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30 20"><g fill="none" stroke="#2f62c9" stroke-width="2.4" stroke-linecap="round">'
        '<path d="M2 5h17a3 3 0 1 0-3-3"/><path d="M2 10h24a3 3 0 1 1-3 3"/><path d="M2 15h11"/></g></svg>', encoding="utf-8")
    chamber = pd.read_csv(config.OUTPUTS / "senate_2026_daily_chamber.csv")
    races = pd.read_csv(config.OUTPUTS / "senate_2026_daily_races.csv")
    table = ts.race_table()
    # The race pages list each race's polls. Those come from the local database; a snapshot is saved
    # so the site can be rebuilt without it (for example by the daily GitHub Action).
    snap = config.OUTPUTS / "senate_2026_site_polls.csv"
    try:
        sen, _ = ts.live_polls(con, 2026, table)
        sen = sen[(sen["end_date"] >= "2026-08-04") & (sen["end_date"] <= "2026-11-03")]
        sen.to_csv(snap, index=False)
    except Exception:
        if not snap.exists():
            raise
        sen = pd.read_csv(snap, dtype={"poll_id": str, "start_date": str, "end_date": str},
                          parse_dates=["available"])
    (DIST / "data").mkdir(exist_ok=True)
    for n in ("senate_2026_daily_chamber.csv", "senate_2026_daily_races.csv"):
        shutil.copyfile(config.OUTPUTS / n, DIST / "data" / n)
    es = spain_pages.build(site)
    for n in ("spain_2026_daily_parties.csv", "spain_2026_daily_blocs.csv", "spain_2026_provinces.csv",
              "spain_2026_seat_distribution.csv"):
        shutil.copyfile(config.OUTPUTS / n, DIST / "data" / n)
    build_home(site, chamber, es)
    build_senate(site, chamber, races, table)
    build_race_pages(site, races, table, sen)
    build_methodology(site)
    build_static_pages(site)
    pages = [p for p in DIST.rglob("index.html")]
    urls = "".join(f"<url><loc>{BASE_URL}/{p.parent.relative_to(DIST).as_posix() + '/' if p.parent != DIST else ''}</loc></url>" for p in pages)
    (DIST / "sitemap.xml").write_text(f'<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>', encoding="utf-8")
    (DIST / "robots.txt").write_text(f"User-agent: *\nAllow: /\nSitemap: {BASE_URL}/sitemap.xml\n", encoding="utf-8")
    return DIST
