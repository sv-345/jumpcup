"""Pull SportsPredict's per-market CROWD consensus (our prediction vs the field
average) — the signal the public bot API does NOT expose, only the web UI does.

Uses the user's captured web session token (data/sp_web_token.txt, a ~7-day JWT
distinct from the sp_live_ bot key). Refresh it by copying any authenticated
api.sportspredict.com request as cURL from the logged-in browser's DevTools and
pasting the bearer + jwt cookie into data/sp_web_token.txt / sp_web_cookie.txt.

  POST /api/probability/match-stats  {matchId, lobbyId, targetUserId}
    -> brier_user, brier_average (field), relative_brier_score (our RBP),
       markets[]: {question, prediction (ours), prediction_average (crowd),
                   relative_brier_score, potential_rbp_if_(not_)happening}
  POST /api/probability/leaderboard  {eventId, lobbyId, skip, limit, userId,
                                      leaderboardType: daily|global}

Refresh MATCHES via the MCP list_matches tool (web matchId == bot match id).
Run:  .venv/bin/python crowd_stats.py
"""
import json, os, sys, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.abspath(__file__))
def _bearer():
    """Read per call so long-running consumers (daemon crowd_guard/capture) pick up
    a refreshed token without a restart (see sp_login.py for the OTP refresh flow)."""
    return open(os.path.join(ROOT, "data", "sp_web_token.txt")).read().strip()


def _cookie():
    return open(os.path.join(ROOT, "data", "sp_web_cookie.txt")).read().strip()
LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
US = "2aa07fec-27dd-4d79-b604-e7eac8ca9a57"  # our bot (claudius/neptune), confirmed

# Refresh via MCP list_matches(lobby_id=LOBBY). "_KO" = knockout (15-market).
MATCHES = [
    ("NZL_BEL", "790fcb5f-e9ae-4b31-9608-17977ef89a1a"),
    ("EGY_IRN", "f24da99f-ccaf-451f-ae8b-f402047ee832"),
    ("PAN_ENG", "3d8d1b96-0cca-4fd5-8f21-ad44f457d689"),
    ("CRO_GHA", "780369fb-f757-4166-998b-c2db986f87f4"),
    ("COL_POR", "d55a394e-9ba6-4a4a-8112-994724f4b464"),
    ("COD_UZB", "03d5b4fb-f094-43e8-ae66-b99987ea03e7"),
    ("JOR_ARG", "03995b66-4456-4acf-bc95-cf9f1645be0c"),
    ("ALG_AUT", "c5b16a05-c823-48c9-ac79-8923c44df945"),
    ("RSA_CAN_KO", "c03909dc-6898-4224-851b-0250b179e749"),
    ("BRA_JPN_KO", "f2212dc4-88e2-4170-b629-07a9ac003d2d"),
    ("GER_PAR_KO", "fd54f839-84ea-41be-9e7d-36858f54277f"),
    ("NED_MAR_KO", "9b65e63b-e296-4f3a-900f-75ae9a493e16"),
    ("USA_BIH_KO", "b2c86c3b-8060-4173-8321-fb68699210f4"),
    ("ENG_COD_KO", "56218a5e-0156-4c5c-bba9-b7c5da02bf10"),
    ("BEL_SEN_KO", "373353ee-5a7b-4a24-afe1-cd9336f03ca3"),
    ("ESP_AUT_KO", "11bcf2a2-6564-4ade-9cf7-43ee2ecbedd1"),
    ("POR_CRO_KO", "b6287c24-0362-4169-ad1d-3aa4116a8bce"),
    ("SUI_ALG_KO", "66a5b1e2-aff3-435f-835b-2ca80de5e571"),
    ("AUS_EGY_KO", "0bc25d75-319a-493e-86b2-96f84ea9ad55"),
    ("ARG_CPV_KO", "9abe28c5-f74a-492f-a6f9-96d9bc03ecec"),
    ("COL_GHA_KO", "f4efe654-74c0-4f84-9aa0-6f34dacdd45f"),
    ("CAN_MAR_KO", "1b5f4970-9c13-4fec-90e0-092d731e8278"),
    ("PAR_FRA_KO", "e51aee45-86d1-4b0a-aa36-2f0b533b2f26"),
    ("BRA_NOR_KO", "ecb51716-f69a-48a3-8154-707b4245af44"),
    ("MEX_ENG_KO", "653824b9-0a4d-41df-ad1d-d7661043f805"),
    ("POR_ESP_KO", "466a94ea-c68c-4a1d-beb8-beca6760dc4a"),
    ("USA_BEL_KO", "afab1d11-0a40-434a-8d3e-15b7d1a3a1e1"),
    ("ARG_EGY_KO", "9cdeea1d-c4bc-44d8-af43-bc2a2410bdbc"),
    ("SUI_COL_KO", "68875a62-5a1e-4099-afe3-3d2a68316b5f"),
    ("NOR_ENG_KO", "8fd166b3-1b8b-4834-a761-ba7caf26ed93"),
    ("FRA_MAR_KO", "b8499206-a6a8-4951-9bfd-5f60ccab361c"),
    ("ARG_SUI_KO", "f444b793-6a20-48a1-862c-e748781b5f91"),
    ("ESP_BEL_KO", "5da5b95b-4b52-4b16-8b8d-dd9547f749ca"),
    ("FRA_ESP_KO", "de7b7b6f-f3a8-4948-af5e-f0f4565728d0"),
    ("ENG_ARG_KO", "4dc764d8-9ed2-416a-8e9f-02159518e853"),
]


def _post(path, body):
    req = urllib.request.Request(
        "https://api.sportspredict.com/api/probability/" + path,
        data=json.dumps(body).encode(), method="POST",
        headers={"authorization": "Bearer " + _bearer(), "cookie": "jwt=" + _cookie(),
                 "content-type": "application/json", "accept": "application/json",
                 "origin": "https://play.sportspredict.com",
                 "referer": "https://play.sportspredict.com/"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def match_stats(match_id):
    return _post("match-stats", {"matchId": match_id, "lobbyId": LOBBY,
                                 "targetUserId": US})


def main():
    rows = []
    for name, mid in MATCHES:
        try:
            d = match_stats(mid)
        except urllib.error.HTTPError as e:
            print(f"{name}: HTTP {e.code} (token expired? refresh data/sp_web_token.txt)")
            continue
        bu, ba = d.get("brier_user"), d.get("brier_average")
        tag = f"  brier ours={bu} field={ba} RBP={d.get('relative_brier_score')}" if bu else ""
        for m in d.get("markets") or []:
            o, c = m.get("prediction"), m.get("prediction_average")
            if o is None or c is None:
                continue
            rows.append({"match": name, "our": o, "crowd": round(c, 1),
                         "gap": round(o - c, 1), "status": m.get("status"),
                         "q": m.get("question"), "mrbp": m.get("relative_brier_score")})
        print(f"{name:<12} {len(d.get('markets') or [])} markets{tag}")
    out = os.path.join(ROOT, "data", "crowd_rows.json")
    json.dump(rows, open(out, "w"), indent=0)
    big = sorted(rows, key=lambda r: -abs(r["gap"]))[:25]
    print(f"\n{len(rows)} markets with ours+crowd -> {out}")
    print("\nTOP DIVERGENCES (our edge bets):")
    print(f"{'match':<11}{'our':>4}{'crowd':>7}{'gap':>7}  question")
    for r in big:
        print(f"{r['match']:<11}{r['our']:>4}{r['crowd']:>7.0f}{r['gap']:>+7.0f}  {r['q'][:48]}")


if __name__ == "__main__":
    main()
