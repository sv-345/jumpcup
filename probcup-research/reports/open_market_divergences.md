# Open-market divergences (candidate vs live floor's ACTUAL submitted price)
278 open markets priced. Full table: open_market_divergences.csv
div = candidate price - floor's actual submitted price (from read-only list_predictions)

## Top HIGH-confidence divergences (backtested candidates: C1 win-devig, C2 SoT)

match              kind             cand floorA     div  question
CRO vs GHA         team_sot        0.389  0.540  -0.151  Will Croatia have 6 or more shots on target?
ALG vs AUT         team_sot        0.381  0.500  -0.119  Will Austria have 5 or more shots on target?
PAN vs CRO         team_sot        0.599  0.480  +0.119  Will Panama have 3 or more shots on target?
EGY vs IRN         team_sot        0.719  0.830  -0.111  Will Egypt have 3 or more shots on target?
RSA vs KOR         sot_compare     0.594  0.680  -0.086  Will South Korea have more shots on target than South Afri
COL vs COD         sot_compare     0.634  0.720  -0.086  Will Colombia have more shots on target than DR Congo in t
POR vs UZB         sot_compare     0.822  0.750  +0.072  In the second half, will Portugal have more shots on targe
MAR vs Haiti       sot_compare     0.820  0.750  +0.070  Will Morocco have more shots on target than Haiti in the s
COL vs POR         team_sot        0.650  0.720  -0.070  Will Portugal have 4 or more shots on target?
Curacao vs CIV     sot_compare     0.072  0.140  -0.068  Will Curaçao have more shots on target than Ivory Coast in
ALG vs AUT         team_sot        0.665  0.730  -0.065  Will Algeria have 3 or more shots on target?
RSA vs KOR         team_sot        0.574  0.510  +0.064  Will South Africa have 3 or more shots on target?
POR vs UZB         team_sot        0.221  0.280  -0.059  Will Uzbekistan have 4 or more shots on target?
SCO vs BRA         sot_compare     0.683  0.740  -0.057  Will Brazil have more shots on target than Scotland in the
JOR vs ARG         sot_compare     0.799  0.750  +0.049  Will Argentina have more shots on target than Jordan in th
ECU vs GER         sot_compare     0.522  0.570  -0.048  Will Germany have more shots on target than Ecuador in the
ENG vs GHA         sot_compare     0.095  0.140  -0.045  Will Ghana have more shots on target than England in the s
NOR vs FRA         sot_compare     0.586  0.630  -0.044  Will France have more shots on target than Norway in the s
ECU vs GER         both_teams_sot  0.963  0.920  +0.043  At halftime, will both teams have at least 1 shot on targe
JPN vs SWE         sot_compare     0.537  0.570  -0.033  In the second half, will Japan have more shots on target t
JPN vs SWE         sot_compare     0.537  0.570  -0.033  Will Japan have more shots on target than Sweden in the se
New Zealand vs BEL both_teams_sot  0.910  0.880  +0.030  At halftime, will both teams have at least 1 shot on targe
New Zealand vs BEL team_sot        0.900  0.870  +0.030  Will Belgium have 4 or more shots on target?
CZE vs MEX         sot_compare     0.298  0.270  +0.028  Will Czechia have more shots on target than Mexico in the 
ALG vs AUT         sot_compare     0.452  0.480  -0.028  Will Austria have more shots on target than Algeria in the
SEN vs IRQ         sot_compare     0.777  0.750  +0.027  Will Senegal have more shots on target than Iraq in the se
PAN vs CRO         sot_compare     0.196  0.170  +0.026  Will Panama have more shots on target than Croatia in the 
COD vs UZB         sot_compare     0.326  0.300  +0.026  Will Uzbekistan have more shots on target than DR Congo in
ENG vs GHA         both_teams_sot  0.905  0.880  +0.025  At halftime, will both teams have at least 1 shot on targe
TUN vs NED         both_teams_sot  0.666  0.690  -0.024  Will both teams have at least 1 shot on target in the seco
URU vs ESP         sot_compare     0.182  0.160  +0.022  In the second half, will Uruguay have more shots on target
SUI vs CAN         sot_compare     0.458  0.480  -0.022  Will Switzerland have more shots on target than Canada in 
SEN vs IRQ         win_match       0.801  0.780  +0.021  Will Senegal win the match?
ALG vs AUT         win_match       0.242  0.260  -0.018  Will Algeria win the match?
COL vs POR         sot_compare     0.307  0.290  +0.018  In the second half, will Colombia have more shots on targe
MAR vs Haiti       win_match       0.847  0.830  +0.017  Will Morocco win the match?
JOR vs ARG         team_sot        0.686  0.670  +0.017  Will Argentina have 6 or more shots on target?
New Zealand vs BEL team_sot        0.374  0.390  -0.016  Will New Zealand have 2 or more shots on target in the sec
EGY vs IRN         sot_compare     0.332  0.320  +0.012  Will Iran have more shots on target than Egypt in the seco
CZE vs MEX         win_match       0.261  0.250  +0.011  Will Czechia win the match?
PAR vs AUS         win_match       0.321  0.330  -0.009  Will Paraguay win the match?
BIH vs QAT         sot_compare     0.168  0.160  +0.008  Will Qatar have more shots on target than Bosnia and Herze
COD vs UZB         win_match       0.463  0.470  -0.007  Will DR Congo win the match?
CPV vs KSA         both_teams_sot  0.767  0.760  +0.007  Will both teams have at least 1 shot on target in the seco
COL vs COD         win_match       0.637  0.630  +0.007  Will Colombia win the match?

tier counts: {'LOW': 141, 'HIGH': 62, 'MED': 75}
kind counts: {'team_offsides': 21, 'sot_compare': 24, 'pen_or_red': 14, 'player_sot_half': 5, 'btts_and_total': 12, 'win_match': 19, 'half_total_over': 5, 'team_sot': 12, 'player_scorer': 5, 'both_teams_sot': 7, 'fouls_compare': 22, 'player_involve': 9, 'player_sot': 25, 'corners': 21, 'first_goal': 6, 'cards': 17, 'team_score': 10, 'half_goals_compare': 5, 'ht_lead': 2, 'total_under': 7, 'pen': 2, 'total_sot': 6, 'team_score_half': 12, 'total_over': 3, 'ht_tie': 4, 'half_vs_half_goals': 3}