# Jump Trading Probability Cup — autonomous floor bot

A self-contained Python daemon that submits **calibrated, de-vigged probability
forecasts** to the SportsPredict Probability Cup (weighted-Brier scored) and runs
unattended. It floors the board with honest market-derived numbers and captures
closing-line value with pre-kickoff resubmissions.

## Strategy (why this scores)

Scoring is `RBP = (crowd_brier − your_brier) × 100`, cumulative across settled
markets. Submitting a probability closer to the truth than the crowd gives an
expected Brier **provably ≤** the crowd's (by `(crowd−truth)²`). So the winning
play is: **(1) be better-calibrated than the casual crowd, (2) cover every
market** (cumulative = sum). Eligibility needs ≥1 settled prediction; group
stage weights 1×, knockout 2×, final 3×.

Pricing, in confidence order:
- **market** — sharp 1X2 de-vigged with **Shin (1992)** → match-winner.
- **model_anchored** — a Poisson/Dixon-Coles goals model **fit to the de-vigged
  line** (supremacy + total) prices every goal-derived market: totals, BTTS,
  team-to-score, half goals, HT tie/lead, first-goal, half-goal comparisons.
- **base_rate** — calibrated empirical means (offsides, shots-on-target, corners,
  cards, fouls, penalties, player props), tilted by team strength where sensible.

Lower-confidence tiers get a light shrink toward 0.5 to hedge model error (never
enough to flatten a good number toward the crowd). All numbers are clamped to a
safe band and bounded-move-capped between ticks so one bad data point can't wipe
a good line; the cap lifts inside the pre-kickoff lock window.

## Architecture

Pure-Python deterministic engine + a direct SportsPredict REST client (no MCP
dependency — runs fully standalone).

| module | role |
|---|---|
| `submission.py` | SportsPredict REST client (auth, rate-limit, batch/PATCH, 409→PATCH) |
| `odds.py` | the-odds-api client: fetch + de-vig 1X2/totals, budget-aware caching |
| `devig.py` | Shin (3-way) + power (2-way) margin removal, self-checked |
| `model.py` | Poisson/Dixon-Coles goals model + count base-rate pricers |
| `mapping.py` | classify question text → market kind + params; team normalization |
| `stats.py` | live enrichment (API-Football): rolling per-team rates (EB-shrunk) + lineup/squad-aware player props; rate-limited, cached, fallback-safe |
| `ratings.py` | bundled Elo prior (no-network fallback for lambdas) |
| `engine.py` | price every market, confidence tiers, sanity gates, bounded moves |
| `state.py` | SQLite persistence → idempotent resume |
| `daemon.py` | the loop: fetch → price → gate → submit/update → sleep |

Data: **the-odds-api** (sharp odds, 500/mo free) + **API-Football** /
**football-data.org** (free) + keyless SofaScore/FBref. Secrets in `.env`.

## Run / stop / inspect

```bash
# one-off dry run (price the board, no submission) — writes data/dry_run.txt
.venv/bin/python -m jumpbot.daemon dry-run

# one live cycle (submit/update once)
.venv/bin/python -m jumpbot.daemon once --live

# the unattended loop (what runs for ~12h)
caffeinate -i .venv/bin/python -m jumpbot.daemon run

# current standing (server-side counts + settled Brier)
.venv/bin/python -m jumpbot.daemon status
```

Convenience wrappers: `./run.sh` (launches under tmux+caffeinate), `./stop.sh`.

### Stop it
- **Graceful kill switch:** `touch ./STOP` (the loop checks each tick and exits).
- Or end the session: `tmux kill-session -t jumpbot`.
- Restart is safe and idempotent — it recovers existing predictions via
  `GET /predictions` and only PATCHes genuine moves.

## Cadence & CLV
Base loop every ~12 min; it tightens automatically as a kickoff approaches and
forces a fresh odds pull inside the 60-min pre-kickoff window to capture
closing-line value. **Lock = kickoff** (`opening_time`); predictions must be in
before then.

## Files
- `logs/jumpbot.log` — full structured audit (every fetch, de-vig, submit, skip).
- `logs/daemon.out` — stdout/stderr of the running loop.
- `data/state.db` — SQLite state.
- `data/dry_run.txt` — latest decision table.
- `data/decisions.json`, `data/submit_responses.json` — latest cycle detail.

## Tuning (`jumpbot/config.py`)
`POLL_INTERVAL_MIN`, `LINEUP_WINDOW_MIN`, `LOCK_WINDOW_MIN`, `MOVE_CAP_PP`,
`MIN_MOVE_TO_RESUBMIT_PP`, `PROB_FLOOR/CEIL`, `CONFIDENCE_SHRINK`.

## Web-session tooling (crowd data, rivals)

- `sp_login.py request` / `sp_login.py verify <code>` — refresh the 7-day web JWT via
  email OTP (code lands in Gmail; no browser). `crowd_stats.py` reads the token
  per-request, so the running daemon picks up a refresh without a restart.
- `rival_tracker.py` — per-match RBP vs named rivals (winston weng et al.) + live
  leaderboard rank/gap, appended to `data/rival_log.json`. Run after locks/settlements.
