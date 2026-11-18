# Data sources and licenses

Every dataset loaded is also recorded in the `sources` table of the database.
Before publishing anything from a new source, add it here with its license.

| Source | What | License | Credit required |
|---|---|---|---|
| MIT Election Data + Science Lab, *U.S. Senate statewide 1976–2024* (doi:10.7910/DVN/PEJ5QU) | Senate results by state | CC0 1.0 | No (courtesy citation recommended) |
| MIT Election Data + Science Lab, *U.S. President 1976–2024* (doi:10.7910/DVN/42MVDX) | Presidential results by state | CC0 1.0 | No (courtesy citation recommended) |
| VoteHub Polling API (api.votehub.com) | Polls: races, generic ballot, approval | CC BY 4.0 | Yes: credit VoteHub and cite each poll's original source URL |
| FRED, Federal Reserve Bank of St. Louis | Economic series | Public; some series carry third-party copyright | Check each series' notes |
| FiveThirtyEight data repository, `pollster-ratings/raw_polls.csv` (github.com/fivethirtyeight/data) | Historical polls 1998–2022 (final weeks of each race) with results | CC BY 4.0 | Yes: credit FiveThirtyEight |
| FEC candidate data (api.open.fec.gov) | Candidate names and parties (`data/manual/candidates.csv`) | Public domain | No |
| Wikipedia, *2026 United States elections polling* (generic ballot tables) | 2026 generic-ballot polls (`data/raw/wiki_generic_ballot_2026.tsv`) | CC BY-SA 4.0 (poll numbers themselves are facts) | Yes: credit Wikipedia and the pollsters |
| Ballotpedia, 2026 U.S. Senate election pages | 2026 nominees and incumbents (`data/manual/senate_2026.csv`, names and parties only) | Facts; Ballotpedia text is not copied | Courtesy citation |

Avoid for commercial use: scraping aggregator websites (e.g. RealClearPolitics) whose terms forbid it.
Wikipedia tables are CC BY-SA: allowed, but derived tables must carry the same license.

## Spain
- English Wikipedia, CC BY-SA 4.0: "Results of the {2015, 2016, April 2019, November 2019, 2023} Spanish Congress election",
  "Opinion polling for the {2016 … 2026} Spanish general election", "2026 Spanish general election" (seat allocation).
  Page revision IDs are stored inside the JSON files. Underlying results: Spanish Ministry of the Interior.

## Maps
- Spain constituency boundaries: Natural Earth 1:10m admin-1 states and provinces (public domain, naturalearthdata.com),
  projected with the Canary Islands inset from d3-composite-projections (BSD-3-Clause). Saved as `src/forecast/site/es_provinces.json`.
- US state boundaries: us-atlas 3.0.1 (ISC licence), derived from US Census Bureau cartographic boundary files (public domain).
  Saved as `src/forecast/site/us_states.json`.
Both are free for commercial use, including on an ad-supported site.
