"""Spain 2026 pages."""
from __future__ import annotations

import html
from datetime import date

import pandas as pd

from .. import config
from ..spain import data as D
from ..spain import forecast as F
from . import svg

E = html.escape
COLOR = {"PP": "--pp", "PSOE": "--psoe", "Vox": "--vox", "FA": "--fa", "FV": "--fv", "Podemos": "--podemos",
         "SALF": "--salf", "ERC": "--erc", "Junts": "--junts", "Bildu": "--bildu", "PNV": "--pnv", "BNG": "--bng",
         "CC": "--cc", "AA": "--aa"}
# left-to-right order used for the seat bar
SPECTRUM = ["Bildu", "BNG", "ERC", "Podemos", "FA", "FV", "AA", "PSOE", "PNV", "Junts", "CC", "PP", "Vox", "SALF"]
SHORT = {"FA": "Broad Front", "FV": "Compromís–FV", "SALF": "SALF", "Bildu": "EH Bildu", "CC": "CC",
         "AA": "Adelante Andalucía"}


def pct(p: float) -> str:
    if p >= 0.995:
        return ">99%"
    if p <= 0.005:
        return "<1%"
    return f"{p * 100:.0f}%"


def share(v: float) -> str:
    return f"{v * 100:.1f}%"


def name(k: str) -> str:
    return SHORT.get(k, k)


def load():
    out = config.OUTPUTS
    return (pd.read_csv(out / "spain_2026_daily_parties.csv"), pd.read_csv(out / "spain_2026_daily_blocs.csv"),
            pd.read_csv(out / "spain_2026_latest_parties.csv"), pd.read_csv(out / "spain_2026_latest_blocs.csv"),
            pd.read_csv(out / "spain_2026_provinces.csv"), pd.read_csv(out / "spain_2026_close_seats.csv"))


def seat_bar(latest: pd.DataFrame) -> str:
    m = latest.set_index("party")["seats_mean"]
    segs, key = [], []
    for k in SPECTRUM:
        if k in m and m[k] >= 0.5:
            segs.append(f'<div style="flex: {m[k]:.2f} 1 0; background: var({COLOR[k]})" title="{E(name(k))}: {m[k]:.0f} seats"></div>')
            key.append(f'<li style="--c: var({COLOR[k]})"><i></i>{E(name(k))} <b>{m[k]:.0f}</b></li>')
    return (f'<div class="seatbar" role="img" aria-label="Average simulated seats by party, ordered from left to right">'
            f'<div class="maj"><span>176</span></div><div class="bar">{"".join(segs)}</div></div>'
            f'<ul class="seatkey">{"".join(key)}</ul>')


def latest_polls(n: int = 15) -> str:
    p = D.polls("2026")
    p = p[p["end"] <= pd.Timestamp(date.today())]
    cols = ["PP", "PSOE", "Vox", "FA", "FV", "Podemos", "SALF"]
    rows = []
    for pid, g in sorted(p.groupby("poll_id"), key=lambda x: (-x[1]["end"].iloc[0].value, x[0]))[:n]:
        vals = {"+".join(k): v for k, v in zip(g["group"], g["value"])}
        cells = "".join(f"<td class='num'>{vals[c] * 100:.1f}</td>" if c in vals else
                        (f"<td class='num' title='Broad Front and Compromís–FV together'>{vals['FA+FV'] * 100:.1f}*</td>" if c == "FA" and "FA+FV" in vals else
                         "<td class='num'>–</td>")
                        for c in cols)
        n_ = g["n"].iloc[0]
        rows.append(f"<tr><td>{E(g['pollster'].iloc[0])}</td><td>{svg.nice_date(g['end'].iloc[0].date().isoformat())}</td>"
                    f"<td class='num hide-sm'>{'' if pd.isna(n_) else f'{n_:,.0f}'}</td>{cells}</tr>")
    head = "".join(f"<th class='num'>{E(name(c))}</th>" for c in cols)
    return (f"<div class='tablewrap'><table><thead><tr><th>Pollster</th><th>Fieldwork ends</th><th class='num hide-sm'>Sample</th>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
            "<p class='small'>Vote shares in percent. * Broad Front and Compromís–FV reported together. Source: Wikipedia, "
            "“Opinion polling for the 2026 Spanish general election” (CC BY-SA 4.0), which links each poll.</p>")


def build(site) -> dict:
    daily, dblocs, latest, blocs, prov, close = load()
    asof = daily["date"].max()
    frozen = date.fromisoformat(asof) >= F.FREEZE
    dates = sorted(daily["date"].unique())
    bl = blocs.set_index("bloc")
    right, left = bl.loc["PP + Vox"], bl.loc["Sánchez's 2023 investiture partners"]
    lp = latest.set_index("party")
    pp_largest = lp.at["PP", "p_largest"]

    notice = (f"<div class='notice'><strong>Final forecast.</strong> Spanish law bars publishing polls in the five days before the "
              f"November 29 vote, so this forecast stopped updating on {svg.nice_date(asof, True)}.</div>" if frozen else
              f"<div class='notice'><strong>Updated {svg.nice_date(asof, True)}.</strong> The forecast updates every day until "
              f"November 23. After that, Spanish law bars publishing new polls before the November 29 vote, and the last "
              f"forecast stays up.</div>")

    # charts
    piv = dblocs.pivot(index="date", columns="bloc", values="p_majority").loc[dates]
    maj = svg.timeline(dates, [
        {"name": "PP + Vox", "values": piv["PP + Vox"].tolist(), "color": "--pp",
         "end_label": f"PP + Vox {pct(piv['PP + Vox'].iat[-1])}"},
        {"name": "2023 investiture partners", "values": piv["Sánchez's 2023 investiture partners"].tolist(), "color": "--psoe",
         "end_label": f"Investiture partners {pct(piv['Sánchez' + chr(39) + 's 2023 investiture partners'].iat[-1])}"}],
        ref=0.5, label="Chance of reaching 176 seats, by bloc")
    seats = daily.pivot(index="date", columns="party", values="seats_mean").loc[dates]
    sch = svg.timeline(dates, [{"name": name(k), "values": seats[k].tolist(), "color": COLOR[k],
                                "end_label": f"{name(k)} {seats[k].iat[-1]:.0f}"} for k in ["PP", "PSOE", "Vox", "FA"]],
                       y_min=0, y_max=160, ticks=(0, 40, 80, 120, 160), ref=None, fmt=lambda v: f"{v:.0f}",
                       label="Average simulated seats for the four largest national parties")
    seat_tbl = ("<table><thead><tr><th>Date</th>" + "".join(f"<th class='num'>{name(k)}</th>" for k in ["PP", "PSOE", "Vox", "FA"]) +
                "<th class='num'>PP + Vox majority</th></tr></thead><tbody>" +
                "".join(f"<tr><td>{svg.nice_date(d)}</td>" + "".join(f"<td class='num'>{seats.at[d, k]:.0f}</td>" for k in ["PP", "PSOE", "Vox", "FA"]) +
                        f"<td class='num'>{pct(piv.at[d, 'PP + Vox'])}</td></tr>" for d in reversed(dates)) + "</tbody></table>")

    # party table
    rows = []
    for r in latest.sort_values("seats_mean", ascending=False).itertuples():
        rows.append(f"<tr><td><span class='dot' style='--c: var({COLOR[r.party]})'></span>{E(r.name)}</td>"
                    f"<td class='num'>{share(r.share_mean)} <span class='range'>{r.share_p10 * 100:.1f}–{r.share_p90 * 100:.1f}</span></td>"
                    f"<td class='num'><b>{r.seats_median:.0f}</b> <span class='range'>{r.seats_p10:.0f}–{r.seats_p90:.0f}</span></td>"
                    f"<td class='num hide-sm'>{pct(r.p_any_seat)}</td><td class='num'>{pct(r.p_largest)}</td></tr>")
    party_tbl = ("<div class='tablewrap'><table><thead><tr><th>Party</th><th class='num'>Vote share <span class='range'>80% range</span></th>"
                 "<th class='num'>Seats <span class='range'>80% range</span></th><th class='num hide-sm'>Wins any seat</th>"
                 "<th class='num'>Largest party</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")

    brows = "".join(f"<tr><td>{E(r.bloc)}<br><span class='range'>{E(' + '.join(name(p) for p in r.parties.split(' + ')))}</span></td>"
                    f"<td class='num'><b>{r.seats_mean:.0f}</b> <span class='range'>{r.seats_p10:.0f}–{r.seats_p90:.0f}</span></td>"
                    f"<td class='num'>{pct(r.p_majority)}</td></tr>" for r in blocs.itertuples())
    bloc_tbl = ("<div class='tablewrap'><table><thead><tr><th>Combination</th><th class='num'>Seats <span class='range'>80% range</span></th>"
                "<th class='num'>176 or more</th></tr></thead><tbody>" + brows + "</tbody></table></div>")

    # provinces
    close = close.sort_values("seat_sd", ascending=False)
    crow = "".join(f"<tr><td>{E(r.province)}</td><td class='num'>{r.seats}</td><td>{E(r.modal)}</td>"
                   f"<td class='hide-sm'>{E(r.next_most_likely)}</td><td class='num'>{r.seat_sd:.1f}</td></tr>"
                   for r in close.head(12).itertuples())
    close_tbl = ("<div class='tablewrap'><table><thead><tr><th>Constituency</th><th class='num'>Seats</th><th>Most likely result</th>"
                 "<th class='hide-sm'>Most likely change</th><th class='num'>Seats in play</th></tr></thead><tbody>" + crow + "</tbody></table></div>")
    prow = []
    for (comm, provn), g in prov.groupby(["community", "province"], sort=False):
        g = g[g["seats_mean"] >= 0.05].sort_values("seats_mean", ascending=False)
        pills = "".join(f"<span><span class='dot' style='--c: var({COLOR[r.party]})'></span>{E(name(r.party))} {r.seats_mean:.1f}</span>" for r in g.itertuples())
        prow.append(f"<tr><td>{E(provn)}<br><span class='range'>{E(comm)}</span></td><td class='num'>{int(g['seats'].iat[0]) if len(g) else ''}</td>"
                    f"<td><div class='pills'>{pills}</div></td></tr>")
    prov_tbl = ("<div class='tablewrap'><table><thead><tr><th>Constituency</th><th class='num'>Seats</th><th>Average seats by party</th></tr></thead><tbody>"
                + "".join(prow) + "</tbody></table></div>")

    lead = "PP" if pp_largest >= 0.5 else "PSOE"
    body = f"""
<h1>Spain general election, November 29, 2026</h1>
<p class="lede">Forecast for the 350-seat Congress of Deputies. A government needs 176 votes to win an investiture on the first ballot, or more yes than no votes on the second.</p>
{notice}
<div class="topline">
  <div class="chance" style="--c: var(--pp)"><div class="n">{pct(right.p_majority)}</div><div class="l">chance PP and Vox together win 176 seats or more</div></div>
  <div class="chance" style="--c: var(--psoe)"><div class="n">{pct(left.p_majority)}</div><div class="l">chance the parties that backed Sánchez in 2023 win 176 or more</div></div>
  <div class="chance" style="--c: var({COLOR[lead]})"><div class="n">{pct(lp.at[lead, 'p_largest'])}</div><div class="l">chance {'the PP' if lead == 'PP' else 'the PSOE'} wins the most seats</div></div>
</div>
<h2>Average seats in our simulations</h2>
<p class="prose">Each simulation draws a national result from the polls, spreads it across the 52 constituencies, and allocates seats with the D'Hondt method. Parties run from left to right.</p>
{seat_bar(latest)}
<h2>Who leads where</h2>
{map_figure(prov)}
<h2>Parties</h2>
{party_tbl}
<p class="small">Ranges cover the middle 80% of 10,000 simulations. Compromís–FV runs only in the Valencian Community and includes Podemos there; elsewhere Podemos runs alone. UPN runs inside the PP list in Navarre.</p>
<h2>Can anyone reach 176?</h2>
{bloc_tbl}
<p class="small">These are seat counts, not predictions of who would vote for whom. Junts and PNV backed Sánchez's investiture in 2023 but have since voted against the government on major bills.</p>
{svg.figure(maj, caption="Chance of reaching 176 seats, by day. Each day's forecast uses only polls published by then.", table_html=seat_tbl)}
<h2>Seats over time</h2>
{svg.figure(sch, caption="Average simulated seats for the four largest parties that run nationwide.")}
<h2>Where seats are in play</h2>
<p class="prose">Seats in play adds up how much each party's seat count varies across simulations in that constituency. Large constituencies have more seats in play simply because they have more seats.</p>
{close_tbl}
<details><summary>All 52 constituencies</summary>{prov_tbl}</details>
<h2>Latest polls</h2>
{latest_polls()}
<p><a href="{site.href('methodology/', 2)}#spain">How the Spain model works and how it did on past elections</a></p>
"""
    site.page("spain/2026/", "Spain 2026 election forecast | Machine Learning Winds",
              f"Daily forecast of Spain's November 29, 2026 general election: seats for every party and the chance of a 176-seat majority.",
              body, "home")
    return {"asof": asof, "right": right.p_majority, "left": left.p_majority, "lead": lead,
            "lead_p": lp.at[lead, "p_largest"], "latest": latest, "frozen": frozen}


# ------------------------------------------------------------------------------------- map

_ES = None


def lead_fill(party: str, p: float) -> str:
    """Party color, paler when the lead is less certain (three steps keep it readable)."""
    strength = 100 if p >= 0.85 else 62 if p >= 0.6 else 30
    return f"color-mix(in oklab, var({COLOR[party]}) {strength}%, var(--map-none))"


def province_map(prov: pd.DataFrame) -> tuple[str, list[str]]:
    """Choropleth: each province in the color of the party most likely to get the most votes there."""
    global _ES
    if _ES is None:
        from pathlib import Path
        import json
        _ES = json.loads(Path(__file__).with_name("es_provinces.json").read_text(encoding="utf-8"))
    W, H = _ES["width"], _ES["height"]
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Map of Spain\'s 52 constituencies, colored by the party most likely to win the most votes in each">']
    leaders = []
    dots = []
    for provn, g in prov.groupby("province", sort=False):
        lead = g.loc[g["p_first"].idxmax()]
        leaders.append(lead.party)
        seats = int(g["seats"].iat[0])
        top = g[g["seats_mean"] >= 0.3].sort_values("seats_mean", ascending=False)
        lines = "".join(f'<span style="--c: var({COLOR[r.party]})"><em>{E(name(r.party))}</em>{r.seats_mean:.1f}</span>'
                        for r in top.itertuples())
        tip = (f"<b>{E(provn)} · {seats} seat{'s' if seats != 1 else ''}</b>"
               f"<small>{E(name(lead.party))} wins the most votes in {pct(lead.p_first)} of simulations. Average seats:</small>{lines}")
        aria = f"{provn}, {seats} seats: {name(lead.party)} most votes, {pct(lead.p_first)} chance"
        fill = lead_fill(lead.party, lead.p_first)
        geo = _ES["provinces"].get(provn)
        if geo is None:
            continue
        attrs = f'tabindex="0" data-tip="{E(tip)}" aria-label="{E(aria)}"'
        if provn in ("Ceuta", "Melilla"):
            # far too small to see at this scale: draw a marked square next to the city
            x, y = geo["cx"], geo["cy"]
            dots.append(f'<g class="city" {attrs}><rect class="st" x="{x - 9:.1f}" y="{y - 9:.1f}" width="18" height="18" rx="3" style="fill: {fill}"/>'
                        f'<text x="{x + 14:.1f}" y="{y + 5:.1f}">{provn}</text></g>')
        else:
            out.append(f'<path class="st" {attrs} d="{geo["d"]}" style="fill: {fill}"/>')
    out.append(f'<path class="inset" d="{_ES["inset"]}"/>')
    out += dots
    out.append("</svg>")
    return "".join(out), list(dict.fromkeys(leaders))


def map_figure(prov: pd.DataFrame) -> str:
    svg_, leaders = province_map(prov)
    order = [k for k in SPECTRUM if k in leaders]
    parties = "".join(f'<li style="--c: var({COLOR[k]})"><i></i>{E(name(k))}</li>' for k in order)
    k0 = "PP" if "PP" in order else order[0]
    steps = "".join(f'<span class="step" style="background: {lead_fill(k0, p)}"></span>{lab}'
                    for p, lab in [(0.5, "under 60%"), (0.7, "60–85%"), (0.9, "85% or more")])
    return (f'<figure class="chart map esmap" data-map>{svg_}<div class="tip" aria-hidden="true"></div>'
            f'<ul class="seatkey">{parties}</ul>'
            f'<p class="small certainty">Chance of winning the most votes: {steps}</p>'
            '<figcaption>Each constituency is colored by the party most likely to win the most votes there; paler shades mean '
            'that lead is less certain. Select a constituency for its average seats by party. The Canary Islands are shown '
            'in the inset; Ceuta and Melilla are marked with squares. Boundaries: Natural Earth (public domain).</figcaption></figure>')
