# Candidate 3 — Tournament Monte Carlo (2026 FIFA World Cup)

> **STATUS: RESEARCH / PARKED — NOT ACTIONABLE IN-WINDOW.**
> The contest board currently exposes **only individual-match props**. There are
> **no outright (tournament-winner), group-advancement, or group-winner futures
> markets open**. This tournament Monte Carlo therefore prices nothing the
> contest is currently asking us to price. It is parked research, to be
> reactivated **only if and when** futures markets appear later in the contest.
> Everything below is illustrative, not a live edge.

This is candidate 3 of the strategy study for the "Jump Trading Probability Cup."
It compares a **market-implied** P(win tournament) (de-vigged from a bookmaker
outright snapshot) against a **simulated** P(win tournament) from an Elo-driven
Monte Carlo over the real group draw.

---

## Headline conclusion

- **The outright market is the sharper object.** It is a real-money, multi-book
  consensus that incorporates squad news, form, fitness, and draw difficulty.
  Our sim's only input is a **static Elo prior**, so where the two disagree the
  market is almost certainly closer to truth.
- **The disagreements are systematic and explainable, not noise.** The bundled
  Elo table ranks **Brazil (2080) and Argentina (2070) as the two strongest
  teams in the world**; the betting market instead makes **France, Spain, and
  England** the favourites. So the sim over-weights the South American pair and
  under-weights the European top tier. This is a prior-vs-market gap, not a
  pricing inefficiency we could exploit.
- **If futures markets open**, this engine would be worth activating **only**
  after (a) refreshing Elo to a live source and (b) shrinking the sim hard
  toward the de-vigged market — i.e. use the sim for *shape* (R32 / group-winner
  distributions among casual-field longshots) and the market for *level*. Raw
  sim outrights should not be traded against a 4-book outright consensus.

---

## Top 15 teams by market-implied P(win tournament)

De-vig: **power** method on the full 45-team William Hill outright vector
(booksum = 1.203, i.e. **~20.3% overround**). Sim: N = 50,000, `numpy
default_rng(0)`.

| Team | market_implied_pct | sim_pct | divergence_pp |
|---|---:|---:|---:|
| France | 17.79 | 14.98 | -2.81 |
| Spain | 14.63 | 12.22 | -2.41 |
| England | 12.40 | 5.79 | -6.61 |
| Argentina | 11.52 | 16.06 | +4.55 |
| Portugal | 6.38 | 6.86 | +0.47 |
| Brazil | 5.47 | 17.87 | +12.39 |
| Germany | 5.47 | 6.19 | +0.72 |
| Netherlands | 5.47 | 5.11 | -0.36 |
| Morocco | 2.70 | 0.61 | -2.09 |
| Norway | 2.70 | 0.53 | -2.17 |
| USA | 2.28 | 0.32 | -1.96 |
| Japan | 1.86 | 0.37 | -1.49 |
| Mexico | 1.47 | 0.49 | -0.98 |
| Belgium | 1.47 | 3.64 | +2.17 |
| Colombia | 1.47 | 1.07 | -0.40 |

`divergence_pp = sim_pct − market_implied_pct` (positive = sim more bullish than
market). Both columns sum to 100% across the full 48-team field.

---

## 5 biggest positive divergences (sim > market)

| Team | market % | sim % | div pp | Plausibility note |
|---|---:|---:|---:|---|
| Brazil | 5.47 | 17.87 | +12.39 | Sim artifact: Elo ranks Brazil #1 in the world (2080). Market has cooled hard on Brazil — the sim, not the market, is the outlier here. |
| Argentina | 11.52 | 16.06 | +4.55 | Holders, Elo #2 (2070). Sim and market both bullish; sim just leans further on the high prior. Market is the safer number. |
| Belgium | 1.47 | 3.64 | +2.17 | Elo 1940 (10th) but market prices Belgium as a ~67:1 longshot. Plausible casual-field underrate of a top-10-Elo side — the most *interesting* gap if futures open. |
| Uruguay | 0.34 | 1.89 | +1.55 | Elo 1890 (11th) yet a market longshot. Drew a soft group (H, behind Spain). Could be genuine casual mispricing of a deep South American squad. |
| Croatia | 0.56 | 1.97 | +1.41 | Elo 1900 (10th-ish). Perennial knockout over-performers historically; market discounts an ageing core. Sim's structural edge is plausible but soft. |

## 5 biggest negative divergences (market > sim)

| Team | market % | sim % | div pp | Plausibility note |
|---|---:|---:|---:|---|
| England | 12.40 | 5.79 | -6.61 | Market 3rd-favourite; Elo only ranks ENG 7th (1975), so sim discounts heavily. Market reflects squad depth the static Elo can't see — **market right.** |
| France | 17.79 | 14.98 | -2.81 | Clear market favourite; Elo has FRA 3rd. Sim still bullish, just less so. Market level looks correct. |
| Spain | 14.63 | 12.22 | -2.41 | Market 2nd-favourite, Elo 4th. Same direction as France — Elo prior slightly underrates the European elite. |
| Norway | 2.70 | 0.53 | -2.17 | Market buys the Haaland-led attack to ~37:1; Elo (1820) and a tough Group I (France, Senegal) crush the sim path. Market may be over-romantic, but small stakes. |
| Morocco | 2.70 | 0.61 | -2.09 | 2022 semifinalists; market respects the run. Elo 1830 + drawn into Brazil's group (C) tanks the sim. Market likely sharper on intangibles. |

---

## Methods

- **Team strengths:** bundled Elo prior from `jumpbot/ratings.py` (`ELO` dict).
  A static, approximate June-2026 men's-international snapshot. Unknown teams →
  default 1700 (all 48 field teams were present in the table).
- **Group draw:** the **real** 2026 FIFA World Cup final draw (Dec 5, 2025),
  taken from the Wikipedia draw page and cross-checked against ESPN / Sky group
  pages for the playoff-resolved teams (CZE, BIH, TUR, SWE, IRQ, COD). All 12
  groups of 4 used as drawn — *not* an Elo re-seed.
- **Market de-vig:** `lib/devig.py`, **power** method (primary) on the full
  45-outcome William Hill outright vector; multiplicative also stored in the
  JSON for reference. Both are appropriate for a long n-way vector.
- **Group match model:** logistic expected score `E = 1/(1+10^(−Δelo/400))`,
  neutral venue (no home advantage). Split into W/D/L with a draw band of base
  0.27 that decays toward 0 as the mismatch grows; points 3/1/0. Group rank =
  points, tie-broken by Elo + small Gaussian noise (a proxy for goal difference
  / head-to-head, which we do not simulate explicitly).
- **Qualification:** top 2 of each group + the 8 best third-placed teams
  (ranked by points, same Elo-noise tiebreak) → Round of 32.
- **Knockout:** single elimination, same Elo logistic win prob, **no draws**
  (the logistic *is* the coin weighting). The **field of 32 is exact**; the
  **bracket pairings are randomized per sim** (the real bracket is fixed by
  group position, which we do not model).
- **Monte Carlo:** N = 50,000 sims, `numpy.random.default_rng(0)` (deterministic).

## Limitations (read before trusting any number)

1. **Elo is a static prior.** No injuries, form, lineup, or fatigue. The
   Brazil/Argentina inflation above is the direct fingerprint of this.
2. **The market is sharper for outrights.** A 4-book, real-money consensus beats
   a single static rating table. Treat the market column as ground truth and the
   sim column as a prior to be shrunk toward it.
3. **Bracket pairings are randomized, not the official bracket.** This smears the
   path-difficulty structure (who can meet whom, and when) that the real
   group-position bracket fixes. R32 / group-winner numbers are robust to this;
   deep-run and outright numbers are softer.
4. **No goal model.** Group tiebreaks use an Elo-noise proxy for GD/H2H rather
   than simulated scorelines, which can mis-order near-tied groups.
5. **Illustrative only.** Per the header, no futures market is open — none of
   this is currently actionable in the contest.

## Artifacts

- Sim + market probabilities (all 48 teams, R32 / group-win / outright):
  `candidates/candidate3_sim_probs.json`
