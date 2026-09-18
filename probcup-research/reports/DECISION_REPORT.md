# Probability Cup — Strategy Decision Report
**Prepared offline while operator away · 2026-06-23 · research & shadow only · nothing live was touched**

---

## 0. Guardrail attestation
- **No writes to the live contest.** No `submit_prediction` / `submit_predictions_batch` /
  `update_prediction` / `join_lobby` (MCP) and no POST/PATCH to `api.sportspredict.com/.../predictions*`
  were ever called. Only `list_*` reads were used.
- **No bot slot consumed.** The shared account was already a lobby member (the live floor joined); I
  read markets/predictions/results without joining anything.
- **No interference with the live bot.** Worked entirely in `probcup-research/`. `jumpbot/`, `data/`,
  `state/`, and the running process were read-only; nothing modified or restarted.
- **Promotion is your call.** This report ends at recommendations + a running shadow scorer.

## 1. TL;DR — ranked candidates

| # | Candidate | Verdict | Out-of-sample edge vs floor | Sample | Mechanism |
|---|---|---|---|---|---|
| **2a** | **SoT prop model** | **PROMOTE** | **−0.004 to −0.0075 Brier/market** (p≈0) | 37,810 team-games | team-specific NegBin, *unclamped*, beats floor's clamped-Poisson base rate; dominant market type |
| 5 | CLV / late submission | **CONFIRMED — keep** | −0.0022 Brier/market (1X2, p≈0) | 40,822 | closing line sharper than opening; value concentrates in line-move games |
| 1 | De-vig + source | **PROMOTE (low-risk)** | −0.00024 Brier/market (1X2, p=0.001) | 35,545 | Pinnacle-closing sharper than consensus-median; power≈shin; never multiplicative |
| 2b | Goals model on lined markets | **DISCARD as replacement** | **+0.012 Brier (worse)** (p=1.0) | 13,324 | market is unbeatable on totals → **keep** floor's market-anchoring |
| 2c | Offsides / discipline / player props | **KEEP-WATCHING** | not backtestable here | — | needs international + lineup data; shadow scorer will adjudicate |
| 3 | Tournament Monte Carlo | **PARK** | n/a — **no futures market open** | — | market outright sharper than Elo-prior sim; reactivate only if futures appear |
| 4 | Situational (lineup latency) | **PROMOTE to sized bot** | research flag | — | floor's prop prices ingest no lineup news; props move on confirmed XIs |

**Headline:** the contest board is **prop-heavy** (278 markets, 28 matches, **all individual-match
props, zero futures**). The floor de-vigs 1X2/totals well but prices the no-line props
(shots-on-target, offsides, scorer) from **base rates with a clamped strength multiplier**. The
single largest, most defensible edge is a **team-specific shots-on-target model** — exactly the
market type that fills the board and where the casual field is weakest.

## 2. Why the board reframes the brief
The brief anticipated 1X2/totals/exact-score/outrights. The live board is instead ~50 SoT markets,
~55 discipline, ~44 player props, 21 offsides, 21 corners, ~22 totals/BTTS, ~34 half-goals, and only
~19 clean 1X2. Two structural facts about the floor (from reading `jumpbot/`):
- **Lined markets** (1X2, totals, BTTS, half-goals): priced from the **de-vigged market** (1X2) or a
  **market-anchored** goals model. Backtests below show the market is **efficient** here — you cannot
  beat the closing line with a model. The floor is already doing the right thing; only de-vig
  source/method is tunable (candidate 1).
- **No-line props** (SoT, offsides, scorer, pen-or-red): priced from a **static base table**
  (`BASE`) × a goals-attack multiplier **clamped to [0.6, 1.6]**, Poisson, then shrunk toward 0.5.
  This is the weak seam — and it is most of the board.

## 3. Candidate detail

### Candidate 2a — Shots-on-target model → **PROMOTE** (the edge)
- **Faithful test:** both my model and the floor anchor to the **same de-vigged market goal λ**
  (closing 1X2 + O/U2.5). Floor = `4.3 × clamp(λ/avg, 0.6, 1.6)`, Poisson. Mine = calibrated
  `E[SoT] = 1.28 + 2.20·λ`, **NegBin (r≈34)**, **unclamped**. Map + dispersion fit on 2018–2022,
  tested on 2022–2026.
- **Result (out of sample, 37,810 team-games):**
  | Market | mine Brier | floor Brier | Δ | p |
  |---|---|---|---|---|
  | SoT ≥ 3 | 0.1659 | 0.1701 | −0.0042 | 0.000 |
  | SoT ≥ 4 | 0.2135 | 0.2207 | −0.0073 | 0.000 |
  | SoT ≥ 5 | 0.2107 | 0.2182 | −0.0075 | 0.000 |
- The edge holds on the 89% of games where the clamp **doesn't** bind (NegBin + better map), and is
  *larger* on the 11% where it does. **WC group stage has more blowouts than club leagues** → expect
  the realized edge to be bigger, not smaller. (`reports/candidate2_sot_vs_floor.md`)
- **Action:** replace the floor's `team_sot`/`team_sot_half`/`half_sot_compare`/`both_teams_sot`
  pricing with the NegBin map (`lib/price.py:sot_lambda`, `p_sot_ge`, `p_a_more_sot_than_b`). Remove
  the [0.6,1.6] clamp on SoT and the 0.70 base-rate shrink (the shrink *hurt* informative props in
  test). This is the recommendation to action first.

### Candidate 1 — De-vig method + consensus source → **PROMOTE (low-risk, small)**
- 1X2 closing, 41k matches: best **source** is **Pinnacle-closing** (Brier 0.5889) < Max (0.5897) <
  consensus-Avg (0.5899) < Bet365 (0.5903). Among **transforms**, multiplicative (the floor's
  fallback) is reliably worst; power/additive/shin/log differ by only ~0.0001, power marginally best.
- Clean paired test (35,545 common matches): live **Avg+shin 0.58975** → **Pinnacle+power 0.58951**
  (Δ −0.00024, p=0.001). Source effect −0.00017 (p=0.015), method effect −0.00007 (p=0.009).
  Using multiplicative throws the gain away (p=0.26). On **totals**, all methods/sources are within
  0.0001 — de-vig choice is irrelevant there. (`reports/candidate1_devig.md`)
- **Action:** anchor 1X2 to **Pinnacle closing** (or Pinnacle-weighted), keep Shin or switch to
  power, and **never fall through to multiplicative** for 1X2. Free, zero model risk. **Caveat:** the
  effect is tiny and measured on club data; the live median-consensus already *includes* Pinnacle, so
  the realized WC gain may be smaller still. Promote as hygiene, not a difference-maker.

### Candidate 2b — Goals model on lined markets → **DISCARD as a replacement (keep market-anchoring)**
- Dixon-Coles goals model vs de-vigged closing total on O/U2.5 (13,324 test matches): model Brier
  **0.2520 vs market 0.2397** — the model is **0.012 worse** (p=1.0); a 50/50 blend also loses. **The
  closing line is unbeatable on totals.** Useful negative result: do **not** replace the floor's
  market-anchored totals/BTTS/half-goal pricing with a pure model. (`reports/candidate2_goals_disc.json`)

### Candidate 2c — Offsides / discipline / player props → **KEEP-WATCHING**
- **Offsides** not in the free data → unbacktestable here; the floor's static 1.9 is un-team-adjusted.
  Provided a team-strength heuristic (LOW confidence) only.
- **Discipline:** empirical P(≥1 red)=0.174 vs floor-implied 0.229 → floor red-rate runs **slightly
  high** (club data); pen-or-red ≈0.34 is plausible. Minor; not an edge.
- **Player props (44 markets):** need lineup/role data (`APIFOOTBALL_KEY` available, unused by floor).
  Provided team-scaled heuristics; flagged LOW. The shadow scorer will tell us if these matter.

### Candidate 3 — Tournament Monte Carlo → **PARK (no market to trade)**
- **There is no group-advancement / group-winner / outright market open** — all 278 markets are
  match props. Built anyway (N=50,000, real 2026 draw, Elo prior): every big divergence vs the market
  outright traces to the **static Elo prior** (over-weights Brazil/Argentina) — the **market is the
  sharper object**. Reactivate only if futures markets appear, and then shrink hard to the market
  (use the sim for *shape* among longshots, the market for *level*). (`reports/candidate3_tournament.md`)

### Candidate 5 — CLV / late submission → **CONFIRMED (keep the latency engineering)**
- Closing vs opening (Pinnacle): 1X2 closing Brier **0.5890 vs 0.5913** (Δ −0.0022, p≈0); O/U
  −0.0012 (p≈0). The edge **concentrates in big-line-move games**: top-5% most-moved → 0.015 Brier
  gain vs 0.0002 for quiet games. **The floor's 12-min lock-window resubmit earns its keep.** This is
  a validation of existing design, not a new candidate — keep doing it; pull odds close to KO.
  (`reports/candidate5_clv.json`)

## 4. Top situational flags (research; see `candidate4_situational.md`)
1. **Lineup-news latency on props** — floor prop prices ingest **no** team-news; props move on
   confirmed XIs (~T-60). Highest-value flag; wire into the sized bot.
2. **Favourite-blowout SoT** — most-lopsided open games (Tunisia–Netherlands, Curaçao–Ivory Coast,
   Portugal–Uzbekistan, Morocco–Haiti, England–Ghana, Jordan–Argentina…) are where the floor's clamp
   is most wrong → already captured by candidate 2a; these top the divergence log.
3. **Dead-rubber/rotation** — parked; trigger = **group matchday 3** (none open yet).
4. **2026 venue (heat/altitude/travel)** — floor has none; additive, low-urgency, needs a venue feed.

## 5. Keep accruing evidence — the shadow scorer
Forecasts for 133 priced markets are **locked** in `shadow/ledger.jsonl` (time-stamped, pre-
resolution). Nothing has resolved yet (`list_results` == []), so there is **no in-window scoreboard**
— as the brief warned. To accrue over the rest of the contest, re-run this loop (read-only):
1. An agent calls `list_results` (read-only) and saves the JSON array to `shadow/results_dump.json`.
2. `python scripts/shadow_score.py` — recovers each market's outcome from the floor's settled
   (probability, brier_score), scores **each candidate's Brier vs the floor's realized Brier**, and
   appends to `shadow/scoreboard.md` + `shadow/history.csv`. Negative `cand_minus_floor` = candidate
   beat the live floor. This is where the SoT-model edge gets confirmed on *contest* data.

## 6. Honest limitations
- **Club → international transfer:** all backtests calibrate on football-data.co.uk **club** leagues.
  Methods transfer; constants should be re-checked on international data (`APIFOOTBALL_KEY`/
  `FOOTBALLDATA_KEY` are available and unused by the floor). Direction of the SoT edge should
  *strengthen* at the WC (more blowouts); the de-vig edge could shrink.
- **Effect sizes are small in absolute Brier** (0.59 scale) but **consistent and significant**;
  they compound across hundreds of markets in a Brier-scored contest.
- **Offsides/player/corner models are heuristic** (LOW tier) pending lineup/stat feeds.
- **12-hour window:** zero resolutions → the value here is the backtests + open-market divergences +
  a scorer that will pay off over the rest of the contest, exactly as the brief framed it.

## 7. File index
- `reports/phase0_inventory.md` — data/markets/access/compute inventory
- `reports/candidate1_devig.md`, `_paired` (console) — de-vig backtest
- `reports/candidate2_sot_vs_floor.md` — **SoT vs faithful floor (headline)**
- `reports/candidate2_props.md`, `candidate2_goals_disc.json` — props / totals-control / discipline
- `reports/candidate3_tournament.md` — parked tournament MC
- `reports/candidate4_situational.md` — situational flags
- `reports/candidate5_clv.json` — CLV timing
- `reports/open_market_divergences.{md,csv,json}` — every open market: candidate vs floor-actual vs line
- `shadow/ledger.jsonl`, `scripts/shadow_log.py`, `scripts/shadow_score.py`, `shadow/scoreboard.md`
- `lib/` (devig, scoring, props, price), `scripts/` (backtests + pricer) — reproducible
