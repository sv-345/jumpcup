"""Rapid override applier: python apply_overrides.py '<question fragment>' <prob> [...]
Resolves fragments to market ids, writes data/overrides.json (expires at lock),
then forces one live daemon cycle so the new prices submit immediately."""
import json, sqlite3, subprocess, sys

pairs = list(zip(sys.argv[1::2], [float(x) for x in sys.argv[2::2]]))
c = sqlite3.connect("data/state.db")
c.row_factory = sqlite3.Row
try:
    ov = json.load(open("data/overrides.json"))
except Exception:
    ov = {}
for frag, prob in pairs:
    r = c.execute("SELECT market_id, close_ts, question FROM markets "
                  "WHERE question LIKE ? AND close_ts > strftime('%s','now')",
                  (f"%{frag}%",)).fetchall()
    assert len(r) == 1, f"fragment '{frag}' matched {len(r)} open markets"
    ov[r[0]["market_id"]] = {"prob": prob, "note": f"conviction: {frag}",
                             "expires_ts": r[0]["close_ts"]}
    print(f"override: {prob:.2f}  {r[0]['question'][:60]}")
json.dump(ov, open("data/overrides.json", "w"), indent=1)
print("submitting live...")
subprocess.run([".venv/bin/python", "-m", "jumpbot.daemon", "once", "--live"],
               timeout=240, check=False)
