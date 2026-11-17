-- Every dataset we load is recorded here, so licenses and attribution are never lost.
CREATE TABLE IF NOT EXISTS sources (
    name          TEXT PRIMARY KEY,
    url           TEXT,
    license       TEXT,
    retrieved_at  TEXT,
    notes         TEXT
);

-- One row per candidate per race (MEDSL statewide returns).
CREATE TABLE IF NOT EXISTS senate_results (
    year             INTEGER NOT NULL,
    state            TEXT,
    state_po         TEXT NOT NULL,
    special          INTEGER NOT NULL DEFAULT 0,
    stage            TEXT,
    candidate        TEXT,
    party_detailed   TEXT,
    party_simplified TEXT,
    writein          INTEGER,
    candidatevotes   INTEGER,
    totalvotes       INTEGER
);
CREATE INDEX IF NOT EXISTS ix_senate_year_state ON senate_results (year, state_po);

CREATE TABLE IF NOT EXISTS president_results (
    year             INTEGER NOT NULL,
    state            TEXT,
    state_po         TEXT NOT NULL,
    candidate        TEXT,
    party_detailed   TEXT,
    party_simplified TEXT,
    writein          INTEGER,
    candidatevotes   INTEGER,
    totalvotes       INTEGER
);
CREATE INDEX IF NOT EXISTS ix_president_year_state ON president_results (year, state_po);

-- Polls (VoteHub API or polls you collect yourself).
CREATE TABLE IF NOT EXISTS polls (
    poll_id      TEXT PRIMARY KEY,
    source       TEXT NOT NULL,           -- 'votehub', 'manual', ...
    poll_type    TEXT,
    subject      TEXT,
    seat_name    TEXT,
    pollster     TEXT,
    sponsors     TEXT,                    -- JSON list
    start_date   TEXT,
    end_date     TEXT,
    created_at   TEXT,
    sample_size  INTEGER,
    population   TEXT,                    -- lv, rv, a, v
    internal     INTEGER,
    partisan     TEXT,
    url          TEXT,                    -- original source: always keep it
    fetched_at   TEXT
);
CREATE TABLE IF NOT EXISTS poll_answers (
    poll_id  TEXT NOT NULL REFERENCES polls (poll_id),
    choice   TEXT NOT NULL,
    party    TEXT,
    pct      REAL,
    PRIMARY KEY (poll_id, choice)
);

-- Economic time series (FRED).
CREATE TABLE IF NOT EXISTS econ_series (
    series_id  TEXT NOT NULL,
    date       TEXT NOT NULL,
    value      REAL,
    PRIMARY KEY (series_id, date)
);

-- Candidate -> party lookup used to read race polls (built from FEC data plus manual fixes).
CREATE TABLE IF NOT EXISTS candidates (
    name               TEXT NOT NULL,
    office             TEXT NOT NULL,         -- S, H, G (governor)
    state              TEXT NOT NULL,
    district           INTEGER,
    party              TEXT,                  -- D, R, I, L, G, O; blank = needs review
    source_party_code  TEXT,
    source_id          TEXT,
    source             TEXT,
    notes              TEXT
);

-- FiveThirtyEight historical polls (final weeks of each race), for backtesting.
CREATE TABLE IF NOT EXISTS historical_polls (
    poll_id INTEGER, question_id INTEGER, race_id INTEGER, cycle INTEGER, location TEXT,
    type_simple TEXT, race TEXT, pollster TEXT, methodology TEXT, partisan TEXT,
    polldate TEXT, electiondate TEXT, time_to_election INTEGER, samplesize REAL,
    cand1_name TEXT, cand1_party TEXT, cand1_pct REAL, cand1_actual REAL,
    cand2_name TEXT, cand2_party TEXT, cand2_pct REAL, cand2_actual REAL,
    margin_poll REAL, margin_actual REAL
);
CREATE INDEX IF NOT EXISTS ix_hist_polls_race ON historical_polls (race);

-- Generic-ballot polls from other sources (e.g. Wikipedia tables), one row per poll question.
CREATE TABLE IF NOT EXISTS generic_ballot_polls (
    source      TEXT,
    pollster    TEXT,
    start_date  TEXT,
    end_date    TEXT,
    sample_size INTEGER,
    population  TEXT,
    partisan    INTEGER,
    dem         REAL,
    rep         REAL
);
