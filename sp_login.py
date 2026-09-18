"""SportsPredict web login via email OTP — no browser needed.

The web JWT (data/sp_web_token.txt) lasts ~7 days and powers crowd_stats /
crowd_guard / rival_tracker. When it expires (HTTP 401s):

    .venv/bin/python sp_login.py request          # emails a 6-digit code
    .venv/bin/python sp_login.py verify 123456    # saves fresh token + cookie

The code arrives at the account email (from noreply@sportspredict.com,
subject "Email verification code") and expires in 5 minutes; if you request
twice, only the LATEST code is valid. /api/auth/refresh is NOT usable once the
server session lapses ("This session does not exist") — OTP is the reliable path.
Discovered/validated 2026-07-03 (curl flow).
"""
import json
import os
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
EMAIL = "velaga.sourabh@gmail.com"
BASE = "https://api.sportspredict.com/api/auth"
HEADERS = {
    "content-type": "application/json",
    "accept": "application/json",
    "origin": "https://play.sportspredict.com",
    "referer": "https://play.sportspredict.com/",
}


def _post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 method="POST", headers=HEADERS)
    resp = urllib.request.urlopen(req, timeout=30)
    raw = resp.read()
    cookie = None
    for h, v in resp.headers.items():
        if h.lower() == "set-cookie" and v.startswith("jwt="):
            cookie = v.split(";")[0][4:]
    return (json.loads(raw) if raw.strip() else {}), cookie


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("request", "verify"):
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == "request":
        _post("/sign-in/otp/request", {"email": EMAIL})
        print(f"OTP sent to {EMAIL} (expires in 5 min). Next:")
        print("  .venv/bin/python sp_login.py verify <code>")
        return
    if len(sys.argv) != 3 or not sys.argv[2].isdigit():
        print("usage: sp_login.py verify <6-digit-code>")
        sys.exit(2)
    try:
        body, cookie = _post("/sign-in/otp/verify", {"email": EMAIL, "code": sys.argv[2]})
    except urllib.error.HTTPError as e:
        print(f"verify failed: HTTP {e.code}: {e.read().decode()[:200]}")
        sys.exit(1)
    tok = body.get("accessToken")
    if not tok:
        print("no accessToken in response:", json.dumps(body)[:200])
        sys.exit(1)
    open(os.path.join(ROOT, "data", "sp_web_token.txt"), "w").write(tok)
    print("token saved (7-day bearer)")
    if cookie:
        open(os.path.join(ROOT, "data", "sp_web_cookie.txt"), "w").write(cookie)
        print("cookie saved")
    else:
        print("WARNING: no jwt set-cookie seen; kept the old cookie file")


if __name__ == "__main__":
    main()
