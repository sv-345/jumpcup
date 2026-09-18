# Candidate 2 (faithful): team SoT model vs LIVE FLOOR (clamped market-anchored)
out-of-sample test team-games = 37810 (seasons ['2223', '2324', '2425', '2526'])
my map SoT_mean=1.284+2.197*lam ; NegBin r=33.78; floor=4.3*clamp(lam/avg,0.6,1.6) Poisson

## SoT >= 3  (emp 0.757)
  mine  Brier=0.16587 logloss=0.50316
  floor Brier=0.17005 logloss=0.51482
  mine-floor dBrier=-0.00418  p_mine_not_better=0.000
  clamp binds on 11.0% of games; edge there=+0.00528 vs unclamped=+0.00404

## SoT >= 4  (emp 0.586)
  mine  Brier=0.21346 logloss=0.61367
  floor Brier=0.22070 logloss=0.63070
  mine-floor dBrier=-0.00725  p_mine_not_better=0.000
  clamp binds on 11.0% of games; edge there=+0.00761 vs unclamped=+0.00720

## SoT >= 5  (emp 0.417)
  mine  Brier=0.21071 logloss=0.60905
  floor Brier=0.21823 logloss=0.62711
  mine-floor dBrier=-0.00753  p_mine_not_better=0.000
  clamp binds on 11.0% of games; edge there=+0.00532 vs unclamped=+0.00780
