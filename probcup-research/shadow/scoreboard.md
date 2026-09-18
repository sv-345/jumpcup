# Shadow scoreboard (updated 2026-06-23T06:31:52.786416+00:00)
settled markets in dump: 0 | matched to ledger: 0
Scores each candidate's Brier vs the floor's REALIZED Brier (lower = better).
Negative cand_minus_floor => candidate beat the live floor.

No settled markets matched yet. Re-run after resolutions accrue:
  1) an agent calls list_results (read-only) -> save JSON to shadow/results_dump.json
  2) python scripts/shadow_score.py