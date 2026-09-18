
## 1X2 closing-line de-vig (benchmark: Avg_close+shin = live-bot style)
baseline = ['Avg_close', 'shin']  (lower Brier = better; d_vs_base<0 means better than baseline)
source           method              n     brier   logloss    dBrier p_notbetter
Pinnacle_close   power           41016   0.58888   0.98714       nan         nan
Pinnacle_close   additive        41016   0.58890   0.98716       nan         nan
Pinnacle_close   shin            41016   0.58893   0.98721       nan         nan
Pinnacle_close   log             41016   0.58893   0.98723       nan         nan
Pinnacle_close   multiplicative  41016   0.58908   0.98750       nan         nan
Max_close        power           38723   0.58974   0.98835       nan         nan
Max_close        shin            38723   0.58976   0.98837       nan         nan
Max_close        log             38723   0.58976   0.98839       nan         nan
Max_close        additive        38723   0.58980   0.98844       nan         nan
Avg_close        power           38724   0.58980   0.98844  -0.00007       0.012
Max_close        multiplicative  38723   0.58981   0.98848       nan         nan
Avg_close        additive        38724   0.58982   0.98843  -0.00005       0.005
Avg_close        shin            38724   0.58987   0.98855   0.00000       1.000
Avg_close        log             38724   0.58989   0.98862   0.00002       0.946
Avg_close        multiplicative  38724   0.59019   0.98917   0.00032       1.000
Bet365_close     power           38717   0.59028   0.98934       nan         nan
Bet365_close     additive        38717   0.59030   0.98939       nan         nan
Bet365_close     log             38717   0.59032   0.98944       nan         nan
Bet365_close     shin            38717   0.59033   0.98944       nan         nan
Bet365_close     multiplicative  38717   0.59060   0.98996       nan         nan
Pinnacle_open    power           40830   0.59119   0.99063       nan         nan
Pinnacle_open    additive        40830   0.59121   0.99067       nan         nan
Pinnacle_open    log             40830   0.59123   0.99071       nan         nan
Pinnacle_open    shin            40830   0.59123   0.99071       nan         nan
Pinnacle_open    multiplicative  40830   0.59139   0.99101       nan         nan
Avg_open         power           38716   0.59228   0.99210       nan         nan
Avg_open         additive        38716   0.59230   0.99211       nan         nan
Avg_open         shin            38716   0.59234   0.99222       nan         nan
Avg_open         log             38716   0.59236   0.99227       nan         nan
Avg_open         multiplicative  38716   0.59266   0.99281       nan         nan

## Over/Under 2.5 closing-line de-vig (benchmark: Avg_close+power)
baseline = ['Avg_close', 'power']  (lower Brier = better; d_vs_base<0 means better than baseline)
source           method              n     brier   logloss    dBrier p_notbetter
Pinnacle_close   power           35472   0.23940   0.67146       nan         nan
Pinnacle_close   log             35472   0.23940   0.67147       nan         nan
Pinnacle_close   shin            35472   0.23940   0.67148       nan         nan
Pinnacle_close   additive        35472   0.23940   0.67148       nan         nan
Pinnacle_close   multiplicative  35472   0.23943   0.67155       nan         nan
Bet365_close     log             38713   0.23946   0.67157       nan         nan
Bet365_close     power           38713   0.23946   0.67157       nan         nan
Bet365_close     shin            38713   0.23946   0.67157       nan         nan
Bet365_close     additive        38713   0.23946   0.67158       nan         nan
Avg_close        log             38723   0.23947   0.67161  -0.00000       0.510
Avg_close        power           38723   0.23947   0.67160   0.00000       1.000
Avg_close        shin            38723   0.23947   0.67161   0.00000       0.533
Avg_close        additive        38723   0.23947   0.67161   0.00000       0.534
Bet365_close     multiplicative  38713   0.23950   0.67169       nan         nan
Avg_close        multiplicative  38723   0.23953   0.67173   0.00005       0.902
Pinnacle_open    power           35342   0.24059   0.67397       nan         nan
Pinnacle_open    log             35342   0.24060   0.67398       nan         nan
Pinnacle_open    shin            35342   0.24060   0.67398       nan         nan
Pinnacle_open    additive        35342   0.24060   0.67398       nan         nan
Pinnacle_open    multiplicative  35342   0.24063   0.67406       nan         nan