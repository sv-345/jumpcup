# Candidate 4 — Situational edge flags (research, not backtests)

These are mechanism flags on the currently-open board, ranked by actionability **in this
window**. Per the brief these are research signals, not backtested edges.

## Window context
We are in **group-stage matchdays 1–2** (28 open matches, KO 2026-06-23 → 06-28). That timing
gates which situational edges are live now versus parked.

## Flag 1 — Lineup-news latency on PROP markets (HIGHEST value, live now)
**Mechanism.** Confirmed XIs post ~60 min pre-KO. They move exactly the markets that dominate
this board: player SoT/scorer, team SoT, offsides. The live floor **polls odds every 12 min and
resubmits in a 12-min lock window**, so its *goal-derived/market* prices do track late moves — but
its **prop prices come from base rates that ingest no lineup data at all** (`engine.py` prices
SoT/offsides/scorer from `BASE` × a clamped goals-attack factor; team-news never enters). So on a
rested-star or B-team teamsheet, the floor's player-scorer and team-SoT prices **do not react**.
- **Action:** the operator's second (sized) bot should pull confirmed lineups (~T-60) for player
  and team-SoT props and shade accordingly. This is a structural floor gap, not a tuning issue.
- Backtest support: candidate 5 (CLV) shows closing-line value concentrates in big-move games
  (top-5% most-moved: 0.015 Brier vs 0.0002 for quiet games) — lineup news is a prime mover.

## Flag 2 — Favourite-blowout SoT/scorer mispricing (live now)
The floor's SoT attack multiplier is **clamped to [0.6, 1.6]**, so it cannot fully express the
lopsided matchups this group stage is full of. The most extreme open mismatches (model goal λ):

| Match | λ_fav / λ_dog | Where the floor is most wrong |
|---|---|---|
| Tunisia–Netherlands | 0.43 / 2.99 | NED high-SoT over-discounted by clamp; TUN SoT over-stated |
| Curaçao–Ivory Coast | 0.45 / 2.84 | same shape |
| Portugal–Uzbekistan | 2.81 / 0.50 | UZB "4+ SoT" floor 0.28 vs model 0.22; POR high-SoT capped |
| Morocco–Haiti, England–Ghana, Jordan–Argentina, NZ–Belgium, Panama–England, Senegal–Iraq | ~2.6 / ~0.5 | favourite SoT capped at 4.3×1.6; underdog floored at 4.3×0.6 |

These are the same markets topping the divergence log (`open_market_divergences.md`) and are the
**concrete per-question watch-list** for candidate 2.

## Flag 3 — Dead-rubber / rotation (PARKED — trigger = group matchday 3)
Not live: matchdays 1–2 teams play full strength. **Activate on the final group matchday**, when
already-qualified or already-eliminated sides rest starters — which shifts win prob, totals, team
SoT, and player props sharply. None of those fixtures are on the board yet; the floor has **no
rotation logic**, so this is a clean future edge. Operator action: when matchday-3 fixtures open,
flag any team mathematically qualified/eliminated and fade its starters' player props + team SoT.

## Flag 4 — 2026 venue effects (UNBUILT — additive, low-urgency)
The floor has **zero** venue adjustment (no heat/altitude/travel/rest — confirmed by code grep).
2026 is hosted across US/Mexico/Canada: afternoon heat (Dallas/Houston/Monterrey) depresses tempo
→ fewer goals/SoT/corners; altitude (Mexico City ~2240 m, Guadalajara ~1560 m) lifts scoring and
fatigue; congested travel shortens legs late. **Not built this run** (no per-match venue/KO-temp
feed wired). Recommendation: ingest venue + local KO temperature; apply a small totals/SoT/corner
multiplier. Additive to the floor, but second-order vs Flags 1–2.

## Bottom line
The one situational edge worth wiring **now** is **Flag 1 (lineup latency on props)** for the sized
bot; **Flag 2** is already captured by candidate 2's SoT model; **Flags 3–4** are parked with clear
triggers. No live action taken — research only.
