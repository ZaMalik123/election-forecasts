"""Project paths and data source settings."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_MANUAL = ROOT / "data" / "manual"
DB_PATH = ROOT / "db" / "forecasts.sqlite"
OUTPUTS = ROOT / "outputs"

USER_AGENT = "MachineLearningWinds-ElectionForecasts/0.1 (research; contact via machinelearningwinds.com)"

# MIT Election Data + Science Lab datasets on Harvard Dataverse (CC0 1.0).
# Files are looked up by DOI at download time, so new dataset versions are picked up automatically.
MEDSL_DATASETS = {
    "senate": {
        "doi": "doi:10.7910/DVN/PEJ5QU",
        "title": "U.S. Senate statewide 1976-2024",
        "local_name": "medsl_senate.csv",
        "license": "CC0 1.0",
    },
    "president": {
        "doi": "doi:10.7910/DVN/42MVDX",
        "title": "U.S. President 1976-2024",
        "local_name": "medsl_president.csv",
        "license": "CC0 1.0",
    },
}
DATAVERSE_BASE = "https://dataverse.harvard.edu"

VOTEHUB_BASE = "https://api.votehub.com"
VOTEHUB_LICENSE = "CC BY 4.0 - credit VoteHub (votehub.com) and cite each poll's original source"

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
FRED_SERIES = {
    "UNRATE": "Unemployment rate (%)",
    "CPIAUCSL": "Consumer price index, all urban consumers",
    "A191RL1Q225SBEA": "Real GDP growth, annualized quarterly (%)",
    "UMCSENT": "University of Michigan consumer sentiment",
}

# FiveThirtyEight pollster-ratings archive: polls from the final weeks of each race, 1998-2022,
# with actual results attached (CC BY 4.0). Used for backtesting the poll-based model.
FTE_RAW_POLLS = {
    "url": "https://raw.githubusercontent.com/fivethirtyeight/data/master/pollster-ratings/raw_polls.csv",
    "local_name": "fte_raw_polls.csv",
    "license": "CC BY 4.0 - credit FiveThirtyEight (github.com/fivethirtyeight/data)",
}

# Party of the sitting president in each even election year (stable historical fact).
PRESIDENT_PARTY = {
    1976: "R", 1978: "D", 1980: "D", 1982: "R", 1984: "R", 1986: "R", 1988: "R",
    1990: "R", 1992: "R", 1994: "D", 1996: "D", 1998: "D", 2000: "D", 2002: "R",
    2004: "R", 2006: "R", 2008: "R", 2010: "D", 2012: "D", 2014: "D", 2016: "D",
    2018: "R", 2020: "R", 2022: "D", 2024: "D", 2026: "R", 2028: "R",
}


def president_party(year: int) -> str:
    """Party of the president during the given election year (odd years use the following even year)."""
    return PRESIDENT_PARTY.get(year, PRESIDENT_PARTY.get(year + 1, "R"))
