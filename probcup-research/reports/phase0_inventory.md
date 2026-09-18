# Phase 0 — Inventory (operator interview, answered empirically)

The operator was away for this window, so the four interview questions were resolved from the
environment and code rather than a live chat. Assumptions are flagged for confirmation on return.

## 1. Historical data for backtesting
- **No sharp odds+results history was stored locally** (the live bot's `data/cache/` holds only two
  *live* WC snapshots: `oddsapi_wc.json` 28 matches, `oddsapi_wc_winner.json` outrights).
- **Fallback used (as the brief permits): football-data.co.uk free CSVs.** Downloaded **44,251
  matches**, 16 leagues × 8 seasons (2018/19–2025/26), with closing **and** opening odds incl.
  **Pinnacle** (PSC*), consensus (Avg*), Bet365, Max; totals lines; and crucially **shots-on-target
  (HST/AST, 44,183 rows)** and cards — which let candidate 2's SoT/discipline props be backtested.
- `.env` also holds **unused** `FOOTBALLDATA_KEY` + `APIFOOTBALL_KEY` (loaded but wired into no live
  module) — available for richer/international/player data without contending with the live bot.
- **Caveat carried through every result:** calibration is on **club** football; the contest is
  **international**. Methods transfer; specific constants should be re-checked on international data.
  WC group-stage mismatches are *more* extreme than club leagues, so the SoT edge should be *larger*.

## 2. Compute / time budget
- Local machine, `.venv` with numpy 2.5 / scipy 1.18 / pandas 3.0. All backtests run in seconds–
  minutes single-process. Candidate-3 Monte Carlo ran at **N=50,000** sims comfortably. No GPU/Spark
  needed. Time budget (~12 h, operator away) spent on backtests + open-market pricing + shadow
  scaffold, per the brief's guidance that live resolutions yield little in-window.

## 3. Contest market types (the big surprise)
- **278 open markets across 28 matches, ALL individual-match props. ZERO futures** (no group-
  advancement / group-winner / outright / top-scorer markets on the board).
- Mix: ~19 win-match (1X2), ~22 totals/BTTS, ~34 half-goals, **~50 shots-on-target** (incl. 24
  "more SoT than opponent"), 21 offsides, ~55 discipline (fouls/cards/pen-or-red), ~44 player props
  (scorer / 1+ SoT), 21 corners. **Props dominate; only one clean 1X2 per match.**
- Consequence: the brief's candidate list was written for 1X2/totals/exact-score/outrights. Real
  board is **prop-heavy** → candidate weighting shifted to the SoT prop model (candidate 2) and
  candidate 3 (tournament MC) was parked (no market to price).

## 4. Read-only access
- Confirmed working **read-only**: `list_events`, `list_lobbies`, `list_matches`, `list_markets`,
  `list_predictions`, `list_results`. The shared account is already a member of lobby
  `8df8038c-…` (the live floor joined), so **no join action and no bot slot were consumed by me.**
- The floor bot has **submitted all 278 markets** (readable via `list_predictions`) — used read-only
  as the shadow baseline. **No settled results yet** (`list_results` == []) — nothing has resolved.
- **No write tool was ever called.** Avoided both write paths (MCP `submit_*`/`update_*`/`join_lobby`
  and the REST `api.sportspredict.com/.../predictions*`).
