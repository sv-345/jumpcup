"""Persistent state so the daemon is idempotent across restarts.

Tracks, per market, the last forecast we submitted, when, the prediction_id the
API gave us (needed for updates), and the status. SQLite keeps it durable and
crash-safe; a single file under data/.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

from . import config


@dataclass
class MarketState:
    market_id: str
    lobby_id: str
    last_prob: Optional[float]      # last submitted probability in [0,1]
    last_ts: Optional[float]        # epoch seconds of last submit
    prediction_id: Optional[str]    # API id, for updates
    status: str                     # 'new' | 'submitted' | 'updated' | 'failed' | 'closed'
    question: str
    match_name: str
    close_ts: Optional[float]
    attempts: int


_SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    market_id TEXT PRIMARY KEY,
    lobby_id TEXT NOT NULL,
    last_prob REAL,
    last_ts REAL,
    prediction_id TEXT,
    status TEXT NOT NULL DEFAULT 'new',
    question TEXT,
    match_name TEXT,
    close_ts REAL,
    attempts INTEGER NOT NULL DEFAULT 0,
    match_id TEXT
);
CREATE TABLE IF NOT EXISTS events (
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    market_id TEXT,
    detail TEXT
);
CREATE TABLE IF NOT EXISTS results (
    market_id TEXT PRIMARY KEY,
    question TEXT,
    kind TEXT,
    prob REAL,        -- our submitted probability (0-1)
    outcome INTEGER,  -- recovered 0/1 outcome
    brier REAL,       -- (prob - outcome)^2
    settled_ts REAL
);
CREATE TABLE IF NOT EXISTS crowd_results (
    market_id TEXT PRIMARY KEY,
    match_tag TEXT,   -- crowd_stats tag e.g. BRA_JPN_KO
    question TEXT,
    kind TEXT,
    our_prob REAL,    -- our submitted probability (0-1)
    crowd_prob REAL,  -- field consensus prediction_average (0-1)
    outcome INTEGER,  -- 0/1
    our_brier REAL,   -- (our_prob - outcome)^2
    crowd_brier REAL, -- (crowd_prob - outcome)^2
    market_rbp REAL,  -- (crowd_brier - our_brier)*100 = relative brier score vs field
    captured_ts REAL
);
"""


_migrated = False


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    global _migrated
    conn = sqlite3.connect(config.STATE_DB, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        if not _migrated:  # pre-2026-07-01 DBs lack match_id (additive, safe)
            try:
                conn.execute("ALTER TABLE markets ADD COLUMN match_id TEXT")
            except sqlite3.OperationalError:
                pass  # column already exists
            _migrated = True
        yield conn
        conn.commit()
    finally:
        conn.close()


def get(market_id: str) -> Optional[MarketState]:
    with _conn() as c:
        row = c.execute("SELECT * FROM markets WHERE market_id=?", (market_id,)).fetchone()
        return _row_to_state(row) if row else None


def all_states() -> Dict[str, MarketState]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM markets").fetchall()
        return {r["market_id"]: _row_to_state(r) for r in rows}


def upsert_market(market_id: str, lobby_id: str, question: str,
                  match_name: str, close_ts: Optional[float],
                  match_id: Optional[str] = None) -> None:
    """Register a market we've seen, without changing forecast fields."""
    with _conn() as c:
        c.execute(
            """INSERT INTO markets (market_id, lobby_id, question, match_name, close_ts, match_id)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(market_id) DO UPDATE SET
                 question=excluded.question,
                 match_name=excluded.match_name,
                 close_ts=excluded.close_ts,
                 match_id=COALESCE(excluded.match_id, markets.match_id)""",
            (market_id, lobby_id, question, match_name, close_ts, match_id),
        )


def record_submission(market_id: str, prob: float, prediction_id: Optional[str],
                      status: str) -> None:
    with _conn() as c:
        c.execute(
            """UPDATE markets
               SET last_prob=?, last_ts=?, prediction_id=COALESCE(?, prediction_id),
                   status=?, attempts=attempts+1
               WHERE market_id=?""",
            (prob, time.time(), prediction_id, status, market_id),
        )


def mark_failed(market_id: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE markets SET status='failed', attempts=attempts+1 WHERE market_id=?",
            (market_id,),
        )


def log_event(kind: str, market_id: Optional[str] = None, detail: str = "") -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO events (ts, kind, market_id, detail) VALUES (?,?,?,?)",
            (time.time(), kind, market_id, detail),
        )


def have_result(market_id: str) -> bool:
    with _conn() as c:
        return c.execute("SELECT 1 FROM results WHERE market_id=?", (market_id,)).fetchone() is not None


def record_result(market_id: str, question: str, kind: str, prob: float,
                  outcome: int, brier: float) -> None:
    with _conn() as c:
        c.execute(
            """INSERT OR REPLACE INTO results
               (market_id, question, kind, prob, outcome, brier, settled_ts)
               VALUES (?,?,?,?,?,?,?)""",
            (market_id, question, kind, prob, outcome, brier, time.time()),
        )


def calibration_summary() -> Dict[str, dict]:
    """Realized calibration per market kind: count, mean Brier, predicted vs actual."""
    with _conn() as c:
        rows = c.execute(
            """SELECT kind, COUNT(*) n, AVG(brier) mb, AVG(prob) mp, AVG(outcome) mo
               FROM results GROUP BY kind ORDER BY n DESC"""
        ).fetchall()
        return {r["kind"]: {"n": r["n"], "mean_brier": r["mb"],
                            "mean_pred": r["mp"], "mean_outcome": r["mo"]} for r in rows}


def have_crowd_result(market_id: str) -> bool:
    with _conn() as c:
        return c.execute("SELECT 1 FROM crowd_results WHERE market_id=?",
                         (market_id,)).fetchone() is not None


def crowd_match_captured(match_tag: str) -> int:
    """How many markets we've already captured for a crowd match tag."""
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM crowd_results WHERE match_tag=?",
                         (match_tag,)).fetchone()[0]


def record_crowd_result(market_id: str, match_tag: str, question: str, kind: str,
                        our_prob: float, crowd_prob: float, outcome: int,
                        our_brier: float, crowd_brier: float, market_rbp: float) -> None:
    with _conn() as c:
        c.execute(
            """INSERT OR REPLACE INTO crowd_results
               (market_id, match_tag, question, kind, our_prob, crowd_prob, outcome,
                our_brier, crowd_brier, market_rbp, captured_ts)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (market_id, match_tag, question, kind, our_prob, crowd_prob, outcome,
             our_brier, crowd_brier, market_rbp, time.time()),
        )


def crowd_rbp_summary() -> Dict[str, dict]:
    """Realized RBP vs the field per market kind: where do we actually beat the crowd?

    rbp = (crowd_brier - our_brier)*100, summed/averaged per kind. Positive = we beat
    the field; negative = the field beats us (we should anchor toward it there).
    """
    with _conn() as c:
        rows = c.execute(
            """SELECT kind, COUNT(*) n, AVG(our_brier) ob, AVG(crowd_brier) cb,
                      AVG(market_rbp) rbp, SUM(market_rbp) total_rbp,
                      AVG(outcome) mo, AVG(our_prob) mop, AVG(crowd_prob) mcp
               FROM crowd_results GROUP BY kind ORDER BY total_rbp ASC"""
        ).fetchall()
        return {r["kind"]: {"n": r["n"], "our_brier": r["ob"], "crowd_brier": r["cb"],
                            "rbp": r["rbp"], "total_rbp": r["total_rbp"],
                            "mean_outcome": r["mo"], "our_pred": r["mop"],
                            "crowd_pred": r["mcp"]} for r in rows}


def crowd_base_rate(kind: str) -> Optional[dict]:
    """Field/empirical base-rate anchor for a no-edge kind: the crowd's mean prediction
    and the realized frequency, with sample size. None if no settled crowd data yet."""
    with _conn() as c:
        r = c.execute(
            """SELECT COUNT(*) n, AVG(crowd_prob) cp, AVG(outcome) mo
               FROM crowd_results WHERE kind=?""", (kind,)).fetchone()
        if not r or r["n"] == 0:
            return None
        return {"n": r["n"], "crowd_pred": r["cp"], "mean_outcome": r["mo"]}


def _row_to_state(row: sqlite3.Row) -> MarketState:
    return MarketState(
        market_id=row["market_id"],
        lobby_id=row["lobby_id"],
        last_prob=row["last_prob"],
        last_ts=row["last_ts"],
        prediction_id=row["prediction_id"],
        status=row["status"],
        question=row["question"],
        match_name=row["match_name"],
        close_ts=row["close_ts"],
        attempts=row["attempts"],
    )
