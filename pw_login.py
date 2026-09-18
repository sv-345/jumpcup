"""Launch a visible Chrome (persistent profile) and capture the SportsPredict
web UI's JSON API traffic, so we can find the endpoint that serves crowd
consensus per market. The user logs in + clicks a match in the window; every
sportspredict JSON response (and the session auth header) is saved to
pw-capture/ for inspection. Stops on a pw-capture/STOP sentinel or after 15 min.

Credentials are typed by the user into Chrome directly — never seen here. The
captured auth token stays local and is only used to read the user's own contest
data.
"""
import hashlib
import json
import os
import pathlib
import time

from playwright.sync_api import sync_playwright

CAP = pathlib.Path("pw-capture")
CAP.mkdir(exist_ok=True)
reqlog = (CAP / "requests.log").open("a")
PROFILE = os.path.abspath("pw-profile")


def on_response(resp):
    try:
        url = resp.url
        if "sportspredict" not in url.lower():
            return
        if "json" not in resp.headers.get("content-type", ""):
            return
        reqlog.write(f"{resp.status}\t{resp.request.method}\t{url}\n")
        reqlog.flush()
        auth = resp.request.headers.get("authorization")
        if auth:
            (CAP / "auth.txt").write_text(auth)
        rec = {"url": url, "status": resp.status, "method": resp.request.method}
        try:
            rec["body"] = resp.json()
        except Exception:
            rec["body"] = None
        h = hashlib.md5(url.encode()).hexdigest()[:10]
        (CAP / f"resp_{h}.json").write_text(json.dumps(rec, default=str)[:300000])
    except Exception:
        pass


with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(
        PROFILE, channel="chrome", headless=False,
        viewport={"width": 1400, "height": 900})
    ctx.on("response", on_response)
    pg = ctx.pages[0] if ctx.pages else ctx.new_page()
    try:
        pg.goto("https://sportspredict.com", timeout=20000)
    except Exception:
        pass
    print("BROWSER OPEN — log in + click into a match. Capturing up to 15 min.")
    for _ in range(180):
        if (CAP / "STOP").exists():
            break
        time.sleep(5)
    ctx.close()
print("capture window ended")
