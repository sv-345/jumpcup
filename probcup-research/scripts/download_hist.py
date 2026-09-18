#!/usr/bin/env python3
"""Download football-data.co.uk free historical CSVs (closing+opening odds,
shots-on-target, cards, totals) for de-vig / prop / CLV backtests.

Read-only external fetch from a free public source. No contest API touched.
Writes to data/hist/{DIV}_{SEASON}.csv and a manifest.json.
"""
import json, os, sys, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "data", "hist")
os.makedirs(OUT, exist_ok=True)

# Big-5 + secondary leagues with good odds+stats coverage
DIVS = ["E0","E1","SP1","SP2","I1","I2","D1","D2","F1","F2","N1","P1","B1","T1","G1","SC0"]
# Seasons: Pinnacle-closing (PSC*) reliably present from ~2019/20 onward.
SEASONS = ["1819","1920","2021","2122","2223","2324","2425","2526"]
BASE = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"

manifest = []
for season in SEASONS:
    for div in DIVS:
        url = BASE.format(season=season, div=div)
        dest = os.path.join(OUT, f"{div}_{season}.csv")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 research"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            if len(data) < 200 or b"," not in data[:200]:
                manifest.append({"div": div, "season": season, "status": "empty", "bytes": len(data)})
                continue
            with open(dest, "wb") as f:
                f.write(data)
            nrows = data.count(b"\n")
            manifest.append({"div": div, "season": season, "status": "ok", "bytes": len(data), "rows": nrows})
            print(f"OK   {div} {season}  {len(data):>7} bytes  ~{nrows} rows", flush=True)
        except urllib.error.HTTPError as e:
            manifest.append({"div": div, "season": season, "status": f"http_{e.code}"})
            print(f"MISS {div} {season}  HTTP {e.code}", flush=True)
        except Exception as e:
            manifest.append({"div": div, "season": season, "status": f"err_{type(e).__name__}"})
            print(f"ERR  {div} {season}  {e}", flush=True)
        time.sleep(0.4)  # gentle

ok = [m for m in manifest if m["status"] == "ok"]
total_rows = sum(m.get("rows", 0) for m in ok)
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
print(f"\nDONE: {len(ok)}/{len(manifest)} files, ~{total_rows} total rows")
