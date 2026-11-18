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
<h2>The forecast, seat by seat</h2>
<p class="prose">Each simulation draws a national result from the polls, spreads it across the 52 constituencies, and allocates seats with the D'Hondt method, as in the real count.</p>
{map_figure(prov, latest)}
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
            "lead_p": lp.at[lead, "p_largest"], "latest": latest, "frozen": frozen,
            "map": map_figure(prov, latest)}


# ------------------------------------------------------------------------------------- map

_ES = None

# Constituencies whose dots don't fit inside them get a labeled box with a leader line:
# (box x, box y, columns). Coordinates are in the map's own units (760 x 560 frame).
CALLOUTS = {"Madrid": (30, 150, 10), "Barcelona": (735, 55, 8), "Biscay": (395, -48, 4), "Gipuzkoa": (480, -48, 3)}
PITCH, R = 7.0, 2.9            # dots drawn inside provinces
BPITCH, BR = 9.0, 3.6          # dots in callout boxes
TOP, RIGHT, BAR_H = 70, 130, 92  # extra room above (Basque callouts), right (Barcelona), and below (seat bar)


def _geo():
    global _ES
    if _ES is None:
        import json
        from pathlib import Path
        _ES = json.loads(Path(__file__).with_name("es_provinces.json").read_text(encoding="utf-8"))
    return _ES


def seat_allocation(prov: pd.DataFrame) -> pd.DataFrame:
    """Whole seats per constituency and party, from average simulated seats.

    Each party gets the integer part of its average seats in each constituency; the constituency's
    remaining seats go to the largest remainders, preferring parties still short of their rounded
    national total, so the dots add up to each constituency's seats and (almost always) to each
    party's rounded national average.
    """
    m = prov.pivot_table(index="province", columns="party", values="seats_mean", aggfunc="sum").fillna(0.0)
    seats = prov.groupby("province")["seats"].first().reindex(m.index)
    nat = m.sum()
    target = nat.apply(int)
    left = int(seats.sum() - target.sum())
    for k in (nat - target).sort_values(ascending=False).index[:left]:
        target[k] += 1
    x = m.apply(lambda col: col.apply(int))
    rem = m - x
    need_row = seats - x.sum(1)
    need_col = target - x.sum()
    cand = sorted(((rem.at[c, k], c, k) for c in m.index for k in m.columns if rem.at[c, k] > 0), reverse=True)
    for _, c, k in cand:
        if need_row[c] > 0 and need_col[k] > 0:
            x.at[c, k] += 1
            need_row[c] -= 1
            need_col[k] -= 1
    for c in m.index:                       # rare dead ends: fill by remainder alone
        while need_row[c] > 0:
            k = (rem.loc[c] - x.loc[c] + m.loc[c].apply(int)).idxmax()
            x.at[c, k] += 1
            rem.at[c, k] = -1
            need_row[c] -= 1
    return x.astype(int)


def dominance_fill(party: str, frac: float) -> str:
    """Province color: the party with the most seats there, deeper the larger its share of the seats."""
    strength = 100 if frac >= 0.65 else 75 if frac >= 0.5 else 50 if frac >= 0.35 else 28
    return f"color-mix(in oklab, var({COLOR[party]}) {strength}%, var(--map-none))"


def _dots(parties: list[str], x0: float, y0: float, cols: int, pitch: float, r: float, center=True) -> str:
    n = len(parties)
    cols = max(1, min(cols, n))
    rows = -(-n // cols)
    out = []
    for i, k in enumerate(parties):
        row, col = divmod(i, cols)
        in_row = cols if row < rows - 1 else n - cols * (rows - 1)
        off = (cols - in_row) * pitch / 2 if center else 0
        if center:
            x = x0 - (cols - 1) * pitch / 2 + col * pitch + off
            y = y0 - (rows - 1) * pitch / 2 + row * pitch
        else:
            x, y = x0 + col * pitch + off, y0 + row * pitch
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" style="fill: var({COLOR[k]})"/>')
    return "".join(out)


def seat_map(prov: pd.DataFrame, latest: pd.DataFrame) -> str:
    G = _geo()
    W, H = G["width"], G["height"]
    alloc = seat_allocation(prov)
    by = {p: g for p, g in prov.groupby("province", sort=False)}
    vb = f"0 {-TOP} {W + RIGHT} {H + TOP + BAR_H}"
    out = [f'<svg viewBox="{vb}" role="img" aria-label="Map of Spain\'s 52 constituencies with one dot per seat, colored by party, and national seat totals">']
    shapes, dotlayer, boxes = [], [], []
    for provn, geo in G["provinces"].items():
        if provn not in alloc.index:
            continue
        a = alloc.loc[provn]
        g = by[provn]
        M = int(g["seats"].iat[0])
        lead = a.idxmax() if a.max() > 0 else g.loc[g["seats_mean"].idxmax(), "party"]
        frac = g.set_index("party")["seats_mean"].get(lead, 0) / M
        top = g[g["seats_mean"] >= 0.3].sort_values("seats_mean", ascending=False)
        lines = "".join(f'<span style="--c: var({COLOR[r.party]})"><em>{E(name(r.party))}</em>{r.seats_mean:.1f}</span>'
                        for r in top.itertuples())
        tip = f"<b>{E(provn)} · {M} seat{'s' if M != 1 else ''}</b><small>Average seats in our simulations:</small>{lines}"
        aria = f"{provn}, {M} seats: " + ", ".join(f"{name(k)} {v}" for k, v in a[a > 0].sort_values(ascending=False).items())
        attrs = f'tabindex="0" data-tip="{E(tip)}" aria-label="{E(aria)}"'
        fill = dominance_fill(lead, frac)
        seat_parties = [k for k in SPECTRUM for _ in range(int(a.get(k, 0)))]
        if provn in ("Ceuta", "Melilla"):
            x, y = geo["cx"], geo["cy"]
            dotlayer.append(f'<g {attrs} class="city"><circle cx="{x}" cy="{y}" r="5" class="ring" style="fill: var({COLOR[seat_parties[0]]})"/>'
                            f'<text x="{x + 9}" y="{y + 4}">{provn}</text></g>')
            continue
        shapes.append(f'<path class="st" {attrs} d="{geo["d"]}" style="fill: {fill}"/>')
        if provn in CALLOUTS:
            bx, by_, cols = CALLOUTS[provn]
            rows = -(-M // cols)
            bw, bh = cols * BPITCH + 10, rows * BPITCH + 26
            # leader line from the province to the nearest edge of the box
            ex = min(max(geo["cx"], bx), bx + bw)
            ey = min(max(geo["cy"], by_), by_ + bh)
            boxes.append(f'<g class="callout" {attrs}><line x1="{geo["cx"]}" y1="{geo["cy"]}" x2="{ex:.1f}" y2="{ey:.1f}"/>'
                         f'<rect x="{bx}" y="{by_}" width="{bw:.0f}" height="{bh:.0f}" rx="4"/>'
                         f'<text x="{bx + 6}" y="{by_ + 14}">{E(provn)}</text>'
                         + _dots(seat_parties, bx + 5 + BPITCH / 2, by_ + 20 + BPITCH / 2, cols, BPITCH, BR, center=False) + "</g>")
        else:
            cols = {1: 1, 2: 2, 3: 3, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4}.get(M, 4 if M <= 16 else 5)
            dotlayer.append(_dots(seat_parties, geo["cx"], geo["cy"], cols, PITCH, R))
    out += shapes
    out.append(f'<path class="inset" d="{G["inset"]}"/>')
    out.append('<g class="dots">' + "".join(dotlayer) + "</g>")
    out += boxes

    # national seat bar, built from the same dots so the image adds up
    tot = alloc.sum()
    x0, x1, y = 20, W + RIGHT - 20, H + 34
    scale = (x1 - x0) / tot.sum()
    x = x0
    out.append(f'<g class="nbar"><text class="ttl" x="{x0}" y="{H + 18}">Seats in the map, all 350</text>')
    for k in SPECTRUM:
        n = int(tot.get(k, 0))
        if n == 0:
            continue
        w = n * scale
        out.append(f'<rect x="{x:.1f}" y="{y}" width="{max(w - 2, 1):.1f}" height="26" style="fill: var({COLOR[k]})"><title>{E(name(k))}: {n}</title></rect>')
        if w >= 26:
            out.append(f'<text class="n" x="{x + w / 2 - 1:.1f}" y="{y + 18}">{n}</text>')
        x += w
    mx = x0 + 176 * scale - 1
    out.append(f'<line class="maj" x1="{mx:.1f}" x2="{mx:.1f}" y1="{y - 8}" y2="{y + 34}"/>'
               f'<text class="majl" x="{mx:.1f}" y="{y + 48}">176 for a majority</text></g>')
    out.append("</svg>")
    return "".join(out), tot


def map_figure(prov: pd.DataFrame, latest: pd.DataFrame) -> str:
    svg_, tot = seat_map(prov, latest)
    key = "".join(f'<li style="--c: var({COLOR[k]})"><i></i>{E(name(k))} <b>{int(tot[k])}</b></li>'
                  for k in SPECTRUM if tot.get(k, 0) > 0)
    steps = "".join(f'<span class="step" style="background: {dominance_fill("PP", f)}"></span>{lab}'
                    for f, lab in [(0.3, "under 35%"), (0.4, "35–50%"), (0.55, "50–65%"), (0.7, "65% or more")])
    return (f'<figure class="chart map esmap" data-map>{svg_}<div class="tip" aria-hidden="true"></div>'
            f'<ul class="seatkey">{key}</ul>'
            f'<p class="small certainty">Province shading, the leading party\'s share of the seats: {steps}</p>'
            '<figcaption>One dot per seat. Each constituency\'s seats are colored by party, rounding each party\'s average '
            'seats there in 10,000 simulations, so the map adds up to the totals in the bar. Provinces are shaded in the '
            'color of the party with the most seats, deeper where it takes a larger share. Madrid, Barcelona, Biscay and '
            'Gipuzkoa are drawn in boxes because their dots don\'t fit; the Canary Islands are in the inset. Select a '
            'constituency for its average seats by party. Boundaries: Natural Earth (public domain).</figcaption></figure>')
