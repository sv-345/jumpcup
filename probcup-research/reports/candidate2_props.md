# Candidate 2: team SoT model vs static base rate (out-of-sample ['2324', '2425', '2526'])
test team-games = 33066, NegBin r = 27.52

## SoT >= 3  (empirical rate 0.762, n=33066)
model                        brier   logloss   dVsFloor  p_notbetter
team_nbinom                0.17712   0.53566   -0.00664        0.000
team_poisson               0.17727   0.53843   -0.00649        0.000
floor_static4.3            0.18285   0.55337   -0.00091        0.017
floor_static_shrunk0.7     0.18376   0.55489    0.00000        1.000

## SoT >= 4  (empirical rate 0.592, n=33066)
model                        brier   logloss   dVsFloor  p_notbetter
team_nbinom                0.23410   0.66088   -0.00741        0.000
team_poisson               0.23513   0.66470   -0.00638        0.000
floor_static_shrunk0.7     0.24151   0.67607    0.00000        1.000
floor_static4.3            0.24240   0.67795    0.00089        1.000

## SoT >= 5  (empirical rate 0.423, n=33066)
model                        brier   logloss   dVsFloor  p_notbetter
team_nbinom                0.23419   0.66335   -0.01065        0.000
team_poisson               0.23568   0.66780   -0.00915        0.000
floor_static4.3            0.24411   0.68133   -0.00072        0.000
floor_static_shrunk0.7     0.24484   0.68280    0.00000        1.000
