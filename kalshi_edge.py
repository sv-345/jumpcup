#!/usr/bin/env python3
"""Kalshi World Cup edge scanner.

Read-only. Finds +EV bets by comparing Kalshi's public order books to the
jumpbot model's de-vigged fair values. Two things make this correct where a
naive comparison is not:

1. SETTLEMENT SCOPE.  Kalshi WC knockout markets settle on "regulation,
   stoppage AND any extra time".  Our model prices regulation 90'+stoppage
   only (the SportsPredict Cup basis).  So every over/score market must be
   lifted onto Kalshi's incl-ET basis before comparing, or the extra-time
   corners/goals masquerade as a fake "NO is cheap" edge.
   - ADVANCE markets are already scope-clean (both settle on who advances,
     incl. ET/pens) -> compared directly, highest confidence.
   - count markets (corners, totals) -> Poisson lift by P(ET)=P(draw).
   - binary score markets (player/team/btts) -> hazard lift by P(ET).

2. EXACT MATCHING.  team / player / threshold / goal-count, never game+kind.

Output: ranked edges net of Kalshi fees, with available size, tagged
CLEAN (scope matches) vs ET-EST (depends on the extra-time adjustment).

Run from the project root:  python3 kalshi_edge.py
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict

# Reuse the bot's own de-vigged 1X2 (24-book consensus) for P(ET) + advance.
from jumpbot import odds as jodds

KBASE = "https://api.elections.kalshi.com/trade-api/v2"
DECISIONS = "data/decisions.json"
OUT = "data/kalshi_edges.json"

# NB: KXWCFIRSTGOAL is *player* first-goalscorer (no clean model) -> excluded.
WC_SERIES = ["KXWCADVANCE", "KXWCCORNERS", "KXWCTOTAL", "KXWCGOAL", "KXWCBTTS"]

MIN_EDGE = 0.03       # flag only edges >= 3 cents net of fee
MIN_SIZE = 20         # contracts available at the quoted ask
ET_MINUTES = 30.0     # extra time length
REG_MINUTES = 90.0


# ----------------------------------------------------------------------------
# Kalshi client (public, no auth)
# ----------------------------------------------------------------------------
def _get(url):
    for _ in range(4):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return json.load(r)
        except Exception:
            time.sleep(0.4)
    return None


def list_open_markets(series):
    out, cur = [], ""
    for _ in range(6):
        u = f"{KBASE}/markets?series_ticker={series}&status=open&limit=200"
        if cur:
            u += "&cursor=" + urllib.parse.quote(cur)
        d = _get(u)
        if not d:
            break
        ms = d.get("markets", [])
        out.extend(ms)
        cur = d.get("cursor", "")
        if not cur or not ms:
            break
    return out


def best_ask(ticker):
    """Return prices/sizes a taker actually gets.

    Kalshi orderbook_fp: yes_dollars = resting YES bids, no_dollars = resting
    NO bids.  To BUY YES you lift the best NO bid -> yes_ask = 1 - best_no_bid
    with that level's size.  Symmetrically for NO.
    """
    d = _get(f"{KBASE}/markets/{ticker}/orderbook?depth=20")
    if not d:
        return None
    ob = d.get("orderbook_fp") or {}
    yb = [(float(p), float(s)) for p, s in (ob.get("yes_dollars") or [])]
    nb = [(float(p), float(s)) for p, s in (ob.get("no_dollars") or [])]

    def ask_from(opposite):
        # opposite = resting bids on the other side; ask = 1 - best opposite bid
        if not opposite:
            return (None, 0.0, 0.0)
        opposite = sorted(opposite, reverse=True)  # highest bid first
        best_p, best_s = opposite[0]
        ask = round(1.0 - best_p, 4)
        # size within 2c of the best ask (cumulative depth a taker can hit)
        depth2 = sum(s for p, s in opposite if (1.0 - p) <= ask + 0.02 + 1e-9)
        return (ask, best_s, depth2)

    yes_ask, yes_sz, yes_d2 = ask_from(nb)
    no_ask, no_sz, no_d2 = ask_from(yb)
    return dict(yes_ask=yes_ask, yes_sz=yes_sz, yes_d2=yes_d2,
                no_ask=no_ask, no_sz=no_sz, no_d2=no_d2)


# ----------------------------------------------------------------------------
# Kalshi fee
# ----------------------------------------------------------------------------
def fee(price):
    if price is None:
        return 0.0
    return math.ceil(0.07 * price * (1.0 - price) * 100.0) / 100.0


# ----------------------------------------------------------------------------
# Extra-time lifts (regulation -> Kalshi incl-ET basis)
# ----------------------------------------------------------------------------
def _solve_poisson_mean(p_ge_k, k):
    """Find Poisson mean m with P(X>=k)=p_ge_k."""
    def p_ge(m):
        # P(X>=k) = 1 - sum_{i<k} e^-m m^i/i!
        cdf = sum(math.exp(-m) * m ** i / math.factorial(i) for i in range(k))
        return 1.0 - cdf
    lo, hi = 1e-6, 60.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if p_ge(mid) < p_ge_k:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def lift_count(reg_p_ge_k, k, p_et):
    """Lift P(>=k in 90') to P(>=k incl ET) for a Poisson count (corners/goals).

    Extra time adds events at the same per-minute rate, but only for the
    P(ET) fraction of games that reach it."""
    if reg_p_ge_k is None or reg_p_ge_k <= 0 or reg_p_ge_k >= 1:
        return reg_p_ge_k
    m90 = _solve_poisson_mean(reg_p_ge_k, k)
    m_et = m90 * ET_MINUTES / REG_MINUTES
    extra = 0.0
    for j in range(k):
        p_j = math.exp(-m90) * m90 ** j / math.factorial(j)         # P(=j in 90)
        need = k - j
        p_reach = 1.0 - sum(math.exp(-m_et) * m_et ** i / math.factorial(i)
                            for i in range(need))                    # P(>=need in ET)
        extra += p_j * p_et * p_reach
    return min(0.999, reg_p_ge_k + extra)


def lift_binary(reg_p, p_et):
    """Lift P(event in 90') to incl-ET for a 'happens at least once' event
    (player scores, team scores, BTTS side).  Hazard model."""
    if reg_p is None or reg_p <= 0 or reg_p >= 1:
        return reg_p
    lam90 = -math.log(1.0 - reg_p)            # reg hazard
    lam_et = lam90 * ET_MINUTES / REG_MINUTES
    p_et_event = 1.0 - math.exp(-lam_et)      # P(event in ET | reached ET)
    return min(0.999, reg_p + (1.0 - reg_p) * p_et * p_et_event)


# ----------------------------------------------------------------------------
# Model side
# ----------------------------------------------------------------------------
def load_decisions():
    dec = json.load(open(DECISIONS))["decisions"]
    idx = defaultdict(list)
    for x in dec:
        teams = tuple(sorted(t for t in x.get("match_name", "").replace(" vs ", " ").split()))
        idx[teams].append(x)
    return idx


def norm(s):
    return re.sub(r"[^a-z]", "", (s or "").lower())


# 1X2 cache so we hit the odds API once per game
_MO = {}


def match_odds(h, a):
    key = tuple(sorted((h, a)))
    if key not in _MO:
        try:
            _MO[key] = jodds.match_odds_for(h, a)
        except Exception:
            _MO[key] = None
    return _MO[key]


# ----------------------------------------------------------------------------
# Ticker parsing
# ----------------------------------------------------------------------------
def parse_game(ticker):
    m = re.match(r'[A-Z]+-\d{2}[A-Z]{3}\d{2}([A-Z]{3})([A-Z]{3})', ticker)
    if not m:
        return None
    return (m.group(1), m.group(2))  # (home3, away3) in Kalshi's order


def teams_key(ticker):
    g = parse_game(ticker)
    return tuple(sorted(g)) if g else None


# ----------------------------------------------------------------------------
# Build edges
# ----------------------------------------------------------------------------
def find_dec(idx, ticker, kinds, thr=None, player=None, side_team=None):
    tk = teams_key(ticker)
    if not tk:
        return None
    for x in idx.get(tk, []):
        if x.get("kind") not in kinds:
            continue
        q = x.get("question", "").lower()
        if thr is not None and f"{thr} or more" not in q and f"{thr}+" not in q:
            continue
        if player is not None:
            pl = x.get("inputs", {}).get("player", "")
            if not pl or (norm(player) not in norm(pl) and norm(pl) not in norm(player)):
                continue
        if side_team is not None and side_team.lower() not in q:
            continue
        return x
    return None


def main():
    idx = load_decisions()
    edges = []
    seen_titles = {}

    for series in WC_SERIES:
        mkts = list_open_markets(series)
        for m in mkts:
            tk = m["ticker"]
            title = m.get("title", "") or ""
            sub = m.get("yes_sub_title", "") or ""
            seen_titles[tk] = title
            g = parse_game(tk)
            if not g:
                continue
            home3, away3 = g
            detail = tk.rsplit("-", 1)[-1]
            mo = match_odds(home3, away3)
            p_et = mo.p_draw if mo else None      # P(reach extra time) = P(reg draw)

            fair = None
            reg_fair = None
            scope = "ET-EST"
            human = f"{home3}/{away3} {title}".strip()
            src = ""
            meta_tier = ""   # tier/source of the underlying model value
            meta_src = ""

            if series == "KXWCADVANCE":
                if not mo:
                    continue
                team = detail
                is_home = (team == home3)
                pw = mo.p_home if is_home else mo.p_away
                pl = mo.p_away if is_home else mo.p_home
                et_share = pw / (pw + pl) if (pw + pl) > 1e-9 else 0.5
                fair = pw + mo.p_draw * et_share
                reg_fair = fair
                scope = "CLEAN"
                human = f"{team} advances ({home3}/{away3})"
                src = f"advance from 1X2 ({mo.n_books}bk)"
                meta_tier, meta_src = "market", f"1X2 {mo.n_books}bk"

            elif series == "KXWCCORNERS" and detail.isdigit():
                k = int(detail)
                d = find_dec(idx, tk, {"total_corners"}, thr=k)
                if d and p_et is not None:
                    reg_fair = d["prob"]
                    fair = lift_count(reg_fair, k, p_et)
                    src = f"corners>={k} (P_ET={p_et:.2f})"
                    meta_tier, meta_src = d.get("tier", ""), d.get("source", "")

            elif series == "KXWCTOTAL" and detail.isdigit():
                k = int(detail)
                d = find_dec(idx, tk, {"total_over"}, thr=k)
                if d and p_et is not None:
                    reg_fair = d["prob"]
                    fair = lift_count(reg_fair, k, p_et)
                    src = f"total>={k} (P_ET={p_et:.2f})"
                    meta_tier, meta_src = d.get("tier", ""), d.get("source", "")

            elif series == "KXWCGOAL":
                # title like "Pedri: 1+ goals"; detail suffix = goal count
                mnum = re.search(r'(\d+)\+', title)
                ngoals = int(mnum.group(1)) if mnum else (int(detail) if detail.isdigit() else 1)
                pname = title.split(":")[0].strip() if ":" in title else None
                if ngoals == 1 and pname:
                    d = find_dec(idx, tk, {"player_scorer"}, player=pname)
                    if d and p_et is not None:
                        reg_fair = d["prob"]
                        fair = lift_binary(reg_fair, p_et)
                        src = f"{pname} 1+ (P_ET={p_et:.2f})"
                        meta_tier, meta_src = d.get("tier", ""), d.get("source", "")
                # ngoals>=2: model-only, skip (low confidence)

            elif series == "KXWCBTTS":
                d = find_dec(idx, tk, {"btts"})
                if d and p_et is not None:
                    reg_fair = d["prob"]
                    fair = lift_binary(reg_fair, p_et)
                    src = f"btts (P_ET={p_et:.2f})"
                    meta_tier, meta_src = d.get("tier", ""), d.get("source", "")

            if fair is None:
                continue

            # CONFIDENCE: is the fair value grounded in a SHARP book line, or just
            # our model? A model-vs-liquid-market divergence is the thin-data fade
            # our own findings say LOSES — so flag it, don't trust the big number.
            sharp = (meta_tier == "market" and "0bk" not in meta_src)
            confidence = "SHARP" if sharp else "MODEL"

            bk = best_ask(tk)
            if not bk:
                continue

            # YES side
            cands = []
            if bk["yes_ask"] is not None and bk["yes_d2"] >= MIN_SIZE:
                e = fair - bk["yes_ask"] - fee(bk["yes_ask"])
                cands.append(("BUY YES", e, bk["yes_ask"], bk["yes_d2"]))
            if bk["no_ask"] is not None and bk["no_d2"] >= MIN_SIZE:
                e = (1 - fair) - bk["no_ask"] - fee(bk["no_ask"])
                cands.append(("BUY NO", e, bk["no_ask"], bk["no_d2"]))
            if not cands:
                continue
            side, edge, px, sz = max(cands, key=lambda c: c[1])
            if edge < MIN_EDGE:
                continue
            # regulation-only edge for the SAME side (shows how much is the ET lift)
            rf = reg_fair if reg_fair is not None else fair
            reg_edge = (rf - px - fee(px)) if side == "BUY YES" else ((1 - rf) - px - fee(px))
            edges.append(dict(ticker=tk, human=human, side=side, kalshi_px=round(px, 3),
                              size=int(sz), fair=round(fair, 3), reg_fair=round(rf, 3),
                              edge=round(edge, 4), reg_edge=round(reg_edge, 4),
                              scope=scope, confidence=confidence, source=src,
                              model_src=meta_src, model_tier=meta_tier))

    edges.sort(key=lambda e: (-{"SHARP": 1, "MODEL": 0}[e["confidence"]], -e["edge"]))
    json.dump({"generated_ts": time.time(), "edges": edges}, open(OUT, "w"), indent=1)

    # report
    print(f"\n=== Kalshi WC edges (net of fee, size>= {MIN_SIZE} within 2c) ===\n")
    for grp, label in [
        ("SHARP", "SHARP-GROUNDED — fair value comes from a real book line (24-book 1X2 / market tier). These are the only edges worth trusting."),
        ("MODEL", "MODEL vs LIQUID MARKET — fair value is our base_rate/0-book model disagreeing with a real-money market. Per our own findings (thin-data fades LOSE) treat these as our model being wrong, NOT free edge.")]:
        rows = [e for e in edges if e["confidence"] == grp]
        print(f"-- {label} --")
        if not rows:
            print("   (none)\n")
            continue
        print(f"   {'edge':>6} {'reg':>6} {'side':8} {'px':>5} {'reg→ET':>9} {'size':>7}  market  [model src]")
        for e in rows:
            print(f"   {e['edge']*100:5.1f}c {e['reg_edge']*100:5.1f}c {e['side']:8} {e['kalshi_px']:5.2f} "
                  f"{e['reg_fair']:.2f}→{e['fair']:.2f} {e['size']:7d}  {e['human'][:30]}  [{e['model_src'][:22]}]")
        print()
    print(f"wrote {OUT}  ({len(edges)} edges)")


if __name__ == "__main__":
    main()
