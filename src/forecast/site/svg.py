"""Server-side SVG charts (no JavaScript needed to see them; charts.js adds the hover layer)."""
from __future__ import annotations

import html
import json
from datetime import date
from itertools import count

_ids = count(1)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def nice_date(iso: str, year: bool = False) -> str:
    d = date.fromisoformat(iso)
    return f"{MONTHS[d.month - 1]} {d.day}" + (f", {d.year}" if year else "")


def timeline(dates, series, *, width: int = 900, height: int = 320, **kw):
    """Wide and narrow (phone) versions of the same chart; CSS shows one, charts.js drives whichever is visible."""
    wide = _timeline(dates, series, width=width, height=height, **kw)
    narrow = _timeline(dates, series, width=420, height=max(260, int(height * 0.9)), narrow=True, **kw)
    return [("wide", *wide), ("narrow", *narrow)]


def _timeline(dates: list[str], series: list[dict], *, width: int = 900, height: int = 320, narrow: bool = False,
             y_min: float = 0.0, y_max: float = 1.0, ref: float | None = 0.5, shade: tuple | None = None,
             fmt=lambda v: f"{v * 100:.0f}%", ticks=(0, .25, .5, .75, 1), label: str = "") -> str:
    """series: [{"name", "values", "color" (CSS var), "end_label" (optional)}].
    shade=(above_wash, below_wash) fills between the first series and `ref`."""
    cid = next(_ids)
    x0, x1, y0, y1 = 40, width - (112 if narrow else 130), 16, height - 30
    n = len(dates)
    X = lambda i: x0 + (x1 - x0) * (i / (n - 1) if n > 1 else 0)
    Y = lambda v: y1 - (y1 - y0) * ((v - y_min) / (y_max - y_min))
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(label)}">']
    out.append('<g class="grid">' + "".join(f'<line x1="{x0}" x2="{x1}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>' for t in ticks) + "</g>")
    out.append('<g class="axis">' + "".join(
        f'<text x="{x0 - 8}" y="{Y(t) + 4:.1f}" text-anchor="end">{fmt(t)}</text>' for t in ticks))
    for i, d in enumerate(dates):
        day = date.fromisoformat(d).day
        if i == n - 1 or ((day == 1 or (day == 15 and not narrow)) and X(n - 1) - X(i) > 60):
            anchor = "end" if i == n - 1 else "middle"
            out.append(f'<text x="{X(i):.1f}" y="{height - 8}" text-anchor="{anchor}">{nice_date(d)}</text>')
    out.append("</g>")
    if shade and series:
        v = series[0]["values"]
        pts = " ".join(f"{X(i):.1f},{Y(val):.1f}" for i, val in enumerate(v))
        area = f"{X(0):.1f},{Y(ref):.1f} {pts} {X(n - 1):.1f},{Y(ref):.1f}"
        out.append(f'<defs><clipPath id="a{cid}"><rect x="{x0}" y="{y0 - 20}" width="{x1 - x0}" height="{Y(ref) - y0 + 20:.1f}"/></clipPath>'
                   f'<clipPath id="b{cid}"><rect x="{x0}" y="{Y(ref):.1f}" width="{x1 - x0}" height="{y1 - Y(ref) + 20:.1f}"/></clipPath></defs>')
        out.append(f'<polygon points="{area}" fill="var({shade[0]})" clip-path="url(#a{cid})"/>')
        out.append(f'<polygon points="{area}" fill="var({shade[1]})" clip-path="url(#b{cid})"/>')
    if ref is not None:
        out.append(f'<g class="ref"><line x1="{x0}" x2="{x1}" y1="{Y(ref):.1f}" y2="{Y(ref):.1f}"/></g>')
    out.append('<g class="series">')
    for s in series:
        d = "M" + " L".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(s["values"]))
        out.append(f'<path d="{d}" style="stroke: var({s["color"]})"/>')
    out.append("</g>")
    # End labels, nudged apart so they never collide.
    labs = sorted(({"y": Y(s["values"][-1]), "s": s} for s in series), key=lambda r: r["y"])
    for a, b in zip(labs, labs[1:]):
        if b["y"] - a["y"] < 18:
            b["y"] = a["y"] + 18
    for r in labs:
        s = r["s"]
        text = s.get("end_label") or f'{s["name"]} {fmt(s["values"][-1])}'
        out.append(f'<circle cx="{X(n - 1):.1f}" cy="{Y(s["values"][-1]):.1f}" r="4" fill="var({s["color"]})" stroke="var(--page)" stroke-width="2"/>')
        out.append(f'<text class="endlab" x="{X(n - 1) + 10:.1f}" y="{r["y"] + 4:.1f}" style="fill: var(--ink)">{html.escape(text)}</text>')
    out.append(f'<line class="crosshair" x1="0" x2="0" y1="{y0}" y2="{y1}" style="display:none"/>')
    out.append("</svg>")
    data = {"width": width, "box": {"x0": x0, "x1": x1, "y0": y0}, "dates": dates,
            "labels": [nice_date(d, True) for d in dates],
            "series": [{"name": s["name"], "color": f'var({s["color"]})', "fmt": [fmt(v) for v in s["values"]]} for s in series]}
    return "".join(out), json.dumps(data, separators=(",", ":"))


def figure(variants, caption: str = "", legend: list[tuple[str, str]] | None = None, table_html: str = "") -> str:
    leg = ""
    if legend:
        leg = '<ul class="legend">' + "".join(f'<li style="--c: var({c})"><i></i>{html.escape(n)}</li>' for n, c in legend) + "</ul>"
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    tbl = f"<details><summary>Show the numbers</summary><div class=\"tablewrap\">{table_html}</div></details>" if table_html else ""
    body = "".join(f'<div class="v v-{name}">{svg}<script type="application/json">{data}</script></div>'
                   for name, svg, data in variants)
    return f'<figure class="chart" data-timeline>{leg}{body}<div class="tip" aria-hidden="true"></div>{cap}</figure>{tbl}'



def split_bar(parts: list[tuple[float, str]], cls: str = "split") -> str:
    """Probability bar: parts = [(share, css var)], 2px gaps between segments."""
    segs = "".join(f'<div style="flex: {max(p, 0.0001):.4f} 1 0; background: var({c})"></div>' for p, c in parts if p > 0.0005)
    return f'<div class="{cls}" aria-hidden="true">{segs}</div>'


def date_range(start: str, end: str) -> str:
    a, b = date.fromisoformat(start), date.fromisoformat(end)
    if a == b:
        return nice_date(end)
    if a.month == b.month:
        return f"{MONTHS[a.month - 1]} {a.day}–{b.day}"
    return f"{nice_date(start)}–{nice_date(end)}"
