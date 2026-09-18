# probcup-research — offline strategy study (research & shadow only)

Isolated research alongside the live floor bot. **Never writes to the contest.** Read-only on the
SportsPredict API; every candidate forecast is logged and scored locally.

## Start here
**`reports/DECISION_REPORT.md`** — ranked promote / keep-watching / discard calls.

## Layout
- `lib/` — `devig.py` (Shin/power/log/additive/multiplicative), `scoring.py` (Brier/logloss/bootstrap),
  `props.py` (team SoT model), `price.py` (live-odds parsing, λ recovery, prop pricers)
- `scripts/` — backtests (`bt_candidate*`), open-market pricer (`price_open_markets.py`),
  shadow scorer (`shadow_log.py`, `shadow_score.py`), data downloader (`download_hist.py`)
- `data/hist/` — 44,251 historical matches (football-data.co.uk); `data/open_markets.json` — 278 open
- `reports/` — per-candidate results + `open_market_divergences.{md,csv,json}` + DECISION_REPORT
- `shadow/` — `ledger.jsonl` (locked forecasts), `scoreboard.md`, re-run loop for the rest of the contest

## Reproduce
```
PY=../.venv/bin/python
$PY scripts/download_hist.py                 # historical panel (already downloaded)
$PY scripts/bt_candidate1_devig.py           # + bt_candidate1_paired.py
$PY scripts/bt_candidate2_sot_vs_floor.py    # headline SoT edge
$PY scripts/bt_candidate2_goals_disc.py      # totals-control + discipline
$PY scripts/bt_candidate5_clv.py             # CLV timing
$PY scripts/price_open_markets.py            # divergence log vs floor-actual
$PY scripts/shadow_log.py                    # lock forecasts
# resolutions loop: agent dumps list_results -> shadow/results_dump.json ; then:
$PY scripts/shadow_score.py                  # score candidates vs floor's realized Brier
```

## Headline
Board is **prop-heavy** (no futures). Biggest edge: a **team-specific shots-on-target model** beats
the floor's clamped base-rate SoT pricing by ~0.004–0.0075 Brier/market out of sample (p≈0) — the
dominant market type. De-vig source (Pinnacle-closing) is a small free win; late-submission (CLV) is
validated; goals-on-lined-markets can't beat the market (keep anchoring); tournament MC parked.
