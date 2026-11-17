"""Parse data/raw/spain_data.json (English Wikipedia, CC BY-SA 4.0) into tidy tables.

Everything is expressed in *canonical parties* per election, so that each election's polls, its
results and the previous election's results (the baseline for the swing model) line up.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import config

RAW = config.ROOT / "data" / "raw" / "spain_data.json"

ELECTION_DAY = {
    "2015": date(2015, 12, 20), "2016": date(2016, 6, 26), "2019A": date(2019, 4, 28),
    "2019N": date(2019, 11, 10), "2023": date(2023, 7, 23), "2026": date(2026, 11, 29),
}
PREVIOUS = {"2016": "2015", "2019A": "2016", "2019N": "2019A", "2023": "2019N", "2026": "2023"}
BACKTEST = ["2016", "2019A", "2019N", "2023"]

COMMUNITY = {
    "Andalusia": ["Almería", "Cádiz", "Córdoba", "Granada", "Huelva", "Jaén", "Málaga", "Seville"],
    "Aragon": ["Huesca", "Teruel", "Zaragoza"],
    "Asturias": ["Asturias"],
    "Balearic Islands": ["Balearic Islands"],
    "Basque Country": ["Álava", "Biscay", "Gipuzkoa"],
    "Canary Islands": ["Las Palmas", "Santa Cruz de Tenerife"],
    "Cantabria": ["Cantabria"],
    "Castile and León": ["Ávila", "Burgos", "León", "Palencia", "Salamanca", "Segovia", "Soria",
                         "Valladolid", "Zamora"],
    "Castilla–La Mancha": ["Albacete", "Ciudad Real", "Cuenca", "Guadalajara", "Toledo"],
    "Catalonia": ["Barcelona", "Girona", "Lleida", "Tarragona"],
    "Extremadura": ["Badajoz", "Cáceres"],
    "Galicia": ["A Coruña", "Lugo", "Ourense", "Pontevedra"],
    "La Rioja": ["La Rioja"],
    "Madrid": ["Madrid"],
    "Murcia": ["Murcia"],
    "Navarre": ["Navarre"],
    "Valencian Community": ["Alicante", "Castellón", "Valencia"],
    "Ceuta": ["Ceuta"],
    "Melilla": ["Melilla"],
}
PROVINCES = [p for ps in COMMUNITY.values() for p in ps]
REGION_OF = {p: r for r, ps in COMMUNITY.items() for p in ps}
VALENCIA = COMMUNITY["Valencian Community"]

# ---------------------------------------------------------------------------------------------
# Party mapping. RESULT maps a results-table column to a canonical party (None = "others").
# BASE gives each canonical party its baseline in the previous election: a list of
# (previous column, factor), or ("uniform", provinces) for a party with no previous result.
# POLL maps poll-table columns to canonical parties ("FA|FV" = a sum of two parties).
# ---------------------------------------------------------------------------------------------


@dataclass
class Spec:
    parties: list[str]
    result: dict[str, str | None]
    base: dict[str, object]
    poll: dict[str, str | None]
    only_in: dict[str, list[str]] = field(default_factory=dict)  # baseline restricted to provinces
    not_in: dict[str, list[str]] = field(default_factory=dict)


SPECS = {
    "2016": Spec(
        ["PP", "PSOE", "UP", "Cs", "ERC", "CDC", "PNV", "Bildu", "CC"],
        {"PP": "PP", "PSOE": "PSOE", "UP": "UP", "C's": "Cs", "ERC": "ERC", "CDC": "CDC",
         "PNV": "PNV", "EH Bildu": "Bildu", "CC–PNC": "CC"},
        {"PP": [("PP", 1)], "PSOE": [("PSOE", 1)], "UP": [("Podemos", 1), ("IU–UPeC", 1)],
         "Cs": [("C's", 1)], "ERC": [("ERC", 1)], "CDC": [("DiL", 1)], "PNV": [("PNV", 1)],
         "Bildu": [("EH Bildu", 1)], "CC": [("CC–PNC", 1)]},
        {"PP": "PP", "PSOE": "PSOE", "Podemos": "UP", "IU–UPeC": "UP", "Unidos Podemos": "UP",
         "C's": "Cs", "ERC–CatSí": "ERC", "DiL": "CDC", "PNV": "PNV", "EH Bildu": "Bildu", "CC": "CC"},
    ),
    "2019A": Spec(
        ["PP", "PSOE", "UP", "Cs", "Vox", "ERC", "Junts", "PNV", "Bildu", "CC"],
        {"PSOE": "PSOE", "PP": "PP", "NA+": "PP", "Cs": "Cs", "UP": "UP", "Compr.": "UP", "Vox": "Vox",
         "ERC–Sob": "ERC", "Junts": "Junts", "PNV": "PNV", "EH Bildu": "Bildu", "CC–PNC": "CC"},
        {"PP": [("PP", 1)], "PSOE": [("PSOE", 1)], "UP": [("UP", 1)], "Cs": [("C's", 1)],
         "Vox": ("uniform", None), "ERC": [("ERC", 1)], "Junts": [("CDC", 1)], "PNV": [("PNV", 1)],
         "Bildu": [("EH Bildu", 1)], "CC": [("CC–PNC", 1)]},
        {"PP": "PP", "PSOE": "PSOE", "Unidas Podemos": "UP", "Unidos Podemos": "UP", "Compromís": "UP",
         "Cs": "Cs", "ERC–Sobiranistes": "ERC", "PDeCAT": "Junts", "JxCat": "Junts", "PNV": "PNV",
         "EH Bildu": "Bildu", "CCa": "CC", "Vox": "Vox"},
    ),
    "2019N": Spec(
        ["PSOE", "PP", "Vox", "UP", "Cs", "ERC", "Junts", "PNV", "Bildu", "CC", "PRC"],
        {"PSOE": "PSOE", "PP": "PP", "NA+": "PP", "Vox": "Vox", "UP": "UP", "Más País": "UP", "Cs": "Cs",
         "Junts": "Junts", "ERC–Sob": "ERC", "PNV": "PNV", "EH Bildu": "Bildu", "CCa–NC": "CC",
         "PRC": "PRC"},
        {"PP": [("PP", 1), ("NA+", 1)], "PSOE": [("PSOE", 1)], "UP": [("UP", 1), ("Compr.", 1)],
         "Cs": [("Cs", 1)], "Vox": [("Vox", 1)], "ERC": [("ERC–Sob", 1)], "Junts": [("Junts", 1)],
         "PNV": [("PNV", 1)], "Bildu": [("EH Bildu", 1)], "CC": [("CC–PNC", 1)], "PRC": [("PRC", 1)]},
        {"PSOE": "PSOE", "PP": "PP", "Cs": "Cs", "Unidas Podemos": "UP", "Vox": "Vox",
         "ERC–Sobiranistes": "ERC", "JxCat": "Junts", "PNV": "PNV", "EH Bildu": "Bildu",
         "Compromís": "UP", "Más País": "UP", "CC–NCa": "CC", "Regionalist Party of Cantabria": "PRC"},
    ),
    "2023": Spec(
        ["PP", "PSOE", "Vox", "Sumar", "ERC", "Junts", "Bildu", "PNV", "BNG", "CC", "UPN"],
        {"PP": "PP", "PSOE": "PSOE", "Vox": "Vox", "Sumar": "Sumar", "BNG": "BNG", "ERC": "ERC",
         "Junts": "Junts", "EH Bildu": "Bildu", "PNV": "PNV", "CCa": "CC", "UPN": "UPN"},
        {"PP": [("PP", 1), ("NA+", 0.5)], "PSOE": [("PSOE", 1)], "Vox": [("Vox", 1)],
         "Sumar": [("UP", 1), ("Más País", 1)], "ERC": [("ERC–Sob", 1)], "Junts": [("Junts", 1)],
         "Bildu": [("EH Bildu", 1)], "PNV": [("PNV", 1)], "BNG": [("BNG", 1)], "CC": [("CCa–NC", 1)],
         "UPN": [("NA+", 0.5)]},
        {"PSOE": "PSOE", "PP": "PP", "Vox": "Vox", "Sumar": "Sumar", "Podemos": "Sumar",
         "Unidas Podemos": "Sumar", "Más País": "Sumar", "ERC": "ERC", "ERC–Sobiranistes": "ERC",
         "Junts": "Junts", "PNV": "PNV", "EH Bildu": "Bildu", "CCa": "CC", "CC–NCa": "CC", "BNG": "BNG",
         "UPN": "UPN"},
    ),
    # 2026: UPN runs inside the PP list; Compromís–FV (with Podemos) is the Valencian left list;
    # Broad Front (FA) is the rest of the old Sumar; Podemos runs alone outside Valencia.
    # Aliança Catalana and the CUP are not standing.
    "2026": Spec(
        ["PP", "PSOE", "Vox", "FA", "FV", "Podemos", "SALF", "ERC", "Junts", "Bildu", "PNV", "BNG",
         "CC", "AA"],
        {},
        {"PP": [("PP", 1), ("UPN", 1)], "PSOE": [("PSOE", 1)], "Vox": [("Vox", 1)],
         "FA": [("Sumar", 1)], "FV": [("Sumar", 1)], "Podemos": [("Sumar", 1)],
         "SALF": ("uniform", None), "ERC": [("ERC", 1)], "Junts": [("Junts", 1)],
         "Bildu": [("EH Bildu", 1)], "PNV": [("PNV", 1)], "BNG": [("BNG", 1)], "CC": [("CCa", 1)],
         "AA": ("uniform", COMMUNITY["Andalusia"])},
        {"PP": "PP", "UPN": "PP", "PSOE": "PSOE", "Vox": "Vox", "FA": "FA", "FV": "FV",
         "Sumar": "FA|FV", "Podemos": "Podemos", "Se Acabó La Fiesta": "SALF", "ERC": "ERC",
         "Junts": "Junts", "EH Bildu": "Bildu", "PNV": "PNV", "BNG": "BNG", "CCa": "CC",
         "Adelante Andalucía (2021)": "AA"},
        only_in={"FV": VALENCIA},
        not_in={"FA": VALENCIA, "Podemos": VALENCIA},
    ),
}

NATIONAL_PARTIES = {"PP", "PSOE", "UP", "Cs", "Vox", "Sumar", "FA", "Podemos", "SALF"}


def _num(s) -> float:
    s = str(s).replace(",", "").replace("−", "-").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


@lru_cache(maxsize=1)
def raw() -> dict:
    return json.loads(RAW.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------- results

GRID = config.ROOT / "data" / "raw" / "spain_constituency_grid.json"


@lru_cache(maxsize=1)
def results_long() -> pd.DataFrame:
    """election, province, column (as in Wikipedia), pct (of valid votes), seats.

    Read from the colspan-expanded constituency tables (spain_constituency_grid.json).
    """
    grids = json.loads(GRID.read_text(encoding="utf-8"))["grids"]
    out = []
    for e, g in grids.items():
        rows = g["grid"]
        head = rows[0]
        sub = next(r for r in rows if len(r) > 2 and r[1] == "%")
        for r in rows:
            if not r or (r[0] not in PROVINCES and r[0] != "Total"):
                continue
            for j in range(1, len(head)):
                if j >= len(r) or sub[j] != "%":
                    continue
                pct = r[j]
                seats = r[j + 1] if j + 1 < len(r) else ""
                if pct.startswith("§") or pct == "":
                    continue
                out.append((e, r[0], head[j], pct, seats.lstrip("§")))
    df = pd.DataFrame(out, columns=["election", "province", "column", "pct", "seats"])
    df["pct"] = df["pct"].map(_num).fillna(0.0) / 100.0
    df["seats"] = df["seats"].map(_num).fillna(0).astype(int)
    return df


def results_wide(election: str) -> pd.DataFrame:
    """Province x Wikipedia column share matrix (fractions), plus 'Total' row."""
    df = results_long()
    df = df[df["election"] == election]
    return df.pivot_table(index="province", columns="column", values="pct", aggfunc="sum").fillna(0.0)


def seats_by_province(election: str) -> pd.Series:
    if election == "2026":
        return pd.Series(raw()["seats2026"]).reindex(PROVINCES)
    df = results_long()
    df = df[(df["election"] == election) & (df["province"] != "Total")]
    return df.groupby("province")["seats"].sum().reindex(PROVINCES)


def canonical_results(election: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Actual canonical-party shares and seats per province (others = unmapped columns)."""
    spec = SPECS[election]
    df = results_long()
    df = df[df["election"] == election].copy()
    df["party"] = df["column"].map(spec.result)
    shares = df.pivot_table(index="province", columns="party", values="pct", aggfunc="sum")
    seats = df.pivot_table(index="province", columns="party", values="seats", aggfunc="sum")
    shares = shares.reindex(index=PROVINCES + ["Total"], columns=spec.parties).fillna(0.0)
    seats = seats.reindex(index=PROVINCES + ["Total"], columns=spec.parties).fillna(0).astype(int)
    return shares, seats


def baseline(election: str) -> pd.DataFrame:
    """Previous-election provincial shares re-expressed in this election's canonical parties.

    Uniform baselines (new parties) get a nominal 1.0 in their provinces; only the *relative*
    pattern across provinces matters, because shares are raked to the national level later.
    """
    spec = SPECS[election]
    prev = results_wide(PREVIOUS[election]).reindex(PROVINCES).fillna(0.0)
    out = pd.DataFrame(0.0, index=PROVINCES, columns=spec.parties)
    for k, rule in spec.base.items():
        if isinstance(rule, tuple) and rule[0] == "uniform":
            provs = rule[1] or PROVINCES
            out.loc[provs, k] = 1.0
        else:
            for col, f in rule:
                if col in prev.columns:
                    out[k] += f * prev[col]
        if k in spec.only_in:
            out.loc[~out.index.isin(spec.only_in[k]), k] = 0.0
        if k in spec.not_in:
            out.loc[out.index.isin(spec.not_in[k]), k] = 0.0
    return out


# --------------------------------------------------------------------- province weights

def _region_tables(election: str) -> dict[str, dict]:
    out = {}
    for t in raw()["regional"][election]:
        m = re.search(r"results in (?:the )?(.+?)\s*→?$", t["cap"].replace("→", "").strip())
        if not m:
            continue
        region = m.group(1).strip()
        valid, parties = np.nan, {}
        for row in t["rows"]:
            cells = [c for c in row if c != ""]
            if not cells:
                continue
            if cells[0].startswith("Valid votes"):
                valid = _num(cells[1])
            elif len(cells) >= 3 and "(" in cells[0]:
                abbr = re.search(r"\(([^()]*)\)\s*\d*\s*$", cells[0])
                if abbr:
                    parties.setdefault(abbr.group(1), _num(cells[1]))
        out[region] = {"valid": valid, "parties": parties}
    return out


_MATCH = {"PP": r"^PP$|^PP–|–PP$", "PSOE": r"PSOE|^PSC", "Cs": r"^C'?s$", "C's": r"^C'?s$",
          "Vox": r"^Vox$"}


@lru_cache(maxsize=1)
def province_weights() -> pd.DataFrame:
    """Valid votes per province and election.

    Wikipedia gives each community's valid votes and party votes, and each province's party
    percentages. A province's share of its community's valid votes is very stable, so it is
    estimated once per province by non-negative least squares pooled over the five elections
    (community party votes = sum over provinces of weight x province share), and rescaled per
    election to the community's valid-vote total.
    """
    from scipy.optimize import nnls

    elections = ["2015", "2016", "2019A", "2019N", "2023"]
    tables = {e: _region_tables(e) for e in elections}
    res = {e: results_wide(e) for e in elections}
    out = {}
    for region, provs in COMMUNITY.items():
        rows, rhs = [], []
        for e in elections:
            reg = tables[e].get(region)
            if reg is None or not np.isfinite(reg["valid"]):
                continue
            V = reg["valid"]
            rows.append(np.ones(len(provs)))  # the shares sum to one
            rhs.append(1.0)
            for col, pat in _MATCH.items():
                if col not in res[e].columns:
                    continue
                hits = [v for a, v in reg["parties"].items() if re.search(pat, a)]
                if not hits:
                    continue
                x = res[e][col].reindex(provs).fillna(0.0).to_numpy()
                if x.sum() == 0:
                    continue
                rows.append(x)
                rhs.append(hits[0] / V)
        A, b = np.array(rows), np.array(rhs)
        w, _ = nnls(A * 1.0, b)
        w = w / w.sum() if w.sum() > 0 else np.full(len(provs), 1 / len(provs))
        for e in elections:
            reg = tables[e].get(region)
            V = reg["valid"] if reg else np.nan
            for p, wp in zip(provs, w):
                out[(e, p)] = wp * V
    s = pd.Series(out).unstack(0).reindex(PROVINCES)
    s["2026"] = s["2023"]
    return s


# ---------------------------------------------------------------------------------- polls

MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
_DATE = re.compile(r"(\d{1,2})\s+([A-Z][a-z]{2})[a-z]*(?:\s+(\d{4}))?")


def _end_date(text: str):
    found = _DATE.findall(text)
    if not found:
        return None
    d, mon, y = found[-1]
    if mon not in MONTHS:
        return None
    return int(d), MONTHS[mon], (int(y) if y else None)


def pollster_key(name: str) -> str:
    return re.split(r"[/(]", name)[0].strip().lower()


@lru_cache(maxsize=None)
def polls(election: str) -> pd.DataFrame:
    """One row per poll x observed party group: date, pollster, n, group (tuple), value."""
    spec = SPECS[election]
    eday = ELECTION_DAY[election]
    prev_day = ELECTION_DAY[PREVIOUS[election]]
    rows = [r for r in raw()["polls"] if r[0] == election]
    polls_, order = {}, []
    for e, t, pol, dt, n, party, v in rows:
        key = (t, pol, dt, n)
        if key not in polls_:
            polls_[key] = []
            order.append(key)
        polls_[key].append((party, v))

    out = []
    year, last_month = eday.year, eday.month
    for idx, key in enumerate(order):
        t, pol, dt, n = key
        ed = _end_date(dt)
        if ed is None:
            continue
        d, m, y = ed
        if y is None:
            if m > last_month:
                year -= 1
            y = year
        else:
            year = y
        last_month = m
        try:
            end = date(y, m, d)
        except ValueError:
            continue
        if end >= eday or end <= prev_day or "election" in pol.lower():
            continue
        size = _num(n)
        # groups: union-find over canonical parties that share a cell
        groups: list[tuple[set, float, bool]] = []
        for party, v in polls_[key]:
            canon = set()
            for comp in party.split("+"):
                c = spec.poll.get(comp.strip())
                if c:
                    canon.update(c.split("|"))
            if not canon:
                continue
            val = _num(v)
            ok = np.isfinite(val)
            merged = [g for g in groups if g[0] & canon]
            for g in merged:
                groups.remove(g)
                canon |= g[0]
                val = (val if ok else 0) + g[1]
                ok = ok and g[2]
            groups.append((canon, val, ok))
        for canon, val, ok in groups:
            if ok and val > 0:
                out.append({"end": pd.Timestamp(end), "pollster": pol, "house": pollster_key(pol),
                            "n": size, "group": tuple(sorted(canon)), "value": val / 100.0,
                            "poll_id": idx})
    return pd.DataFrame(out)
