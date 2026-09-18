"""Classify a SportsPredict question into a priceable market.

The live board has ~44 distinct question templates. We normalize team names to
FIFA codes, then pattern-match each question into a market `kind` + parameters
(side, threshold, half, player). Unmapped questions return an UNMAPPED spec with
a reason and the engine skips them — we never submit a number we don't
understand. Rules are ordered specific -> general; the first match wins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

# --- Team normalization ----------------------------------------------------
_TEAMS = {
    "POR": ["portugal"], "UZB": ["uzbekistan"], "ENG": ["england"],
    "GHA": ["ghana"], "PAN": ["panama"], "CRO": ["croatia"],
    "COL": ["colombia"], "COD": ["dr congo", "congo dr", "democratic republic of the congo", "congo"],
    "BIH": ["bosnia and herzegovina", "bosnia", "bosnia & herzegovina"],
    "QAT": ["qatar"], "SUI": ["switzerland"], "CAN": ["canada"],
    "MAR": ["morocco"], "HAI": ["haiti"], "SCO": ["scotland"],
    "BRA": ["brazil"], "RSA": ["south africa"], "KOR": ["south korea", "korea republic", "korea"],
    "CZE": ["czech republic", "czechia"], "MEX": ["mexico"],
    "CUW": ["curacao", "curaçao"], "CIV": ["ivory coast", "cote d'ivoire", "côte d'ivoire"],
    "ECU": ["ecuador"], "GER": ["germany"], "JPN": ["japan"], "SWE": ["sweden"],
    "TUN": ["tunisia"], "NED": ["netherlands", "holland"], "PAR": ["paraguay"],
    "AUS": ["australia"], "TUR": ["turkey", "türkiye", "turkiye"],
    "USA": ["united states", "usa", "united states of america"],
    "SEN": ["senegal"], "IRQ": ["iraq"], "NOR": ["norway"], "FRA": ["france"],
    "URU": ["uruguay"], "ESP": ["spain"], "CPV": ["cape verde", "cabo verde"],
    "KSA": ["saudi arabia"], "NZL": ["new zealand"], "BEL": ["belgium"],
    "EGY": ["egypt"], "IRN": ["iran"], "ALG": ["algeria"], "AUT": ["austria"],
    "JOR": ["jordan"], "ARG": ["argentina"],
}
_ALIAS = {}
for _code, _names in _TEAMS.items():
    _ALIAS[_code.lower()] = _code
    for _n in _names:
        _ALIAS[_n] = _code


def resolve_team(token: str) -> Optional[str]:
    if not token:
        return None
    t = token.strip().lower()
    if t in _ALIAS:
        return _ALIAS[t]
    for alias, code in sorted(_ALIAS.items(), key=lambda kv: -len(kv[0])):
        if len(alias) > 3 and alias in t:
            return code
    return None


def parse_match_name(name: str) -> Tuple[Optional[str], Optional[str]]:
    if not name:
        return None, None
    parts = re.split(r"\s+vs\.?\s+", name, flags=re.IGNORECASE)
    if len(parts) != 2:
        return None, None
    return resolve_team(parts[0]), resolve_team(parts[1])


# --- Classification --------------------------------------------------------
@dataclass
class Classification:
    kind: str
    side: Optional[str] = None       # 'home' | 'away'
    side2: Optional[str] = None      # second side, for compound markets
    threshold: Optional[int] = None
    half: Optional[int] = None       # 1 or 2
    player: Optional[str] = None
    player2: Optional[str] = None    # second player, for player-vs-player duels
    event: Optional[str] = None      # hydration-break event: goal/card/offside/corner
    window: Optional[str] = None     # hydration window: before_first / after_second / ...
    mapped: bool = True
    reason: str = ""

    @classmethod
    def unmapped(cls, reason: str) -> "Classification":
        return cls(kind="unmapped", mapped=False, reason=reason)


KINDS = {
    # goal-derived (model-anchored, high edge)
    "win_match", "total_over", "total_under", "half_total_over", "btts",
    "btts_and_total", "team_score", "team_score_half", "half_goals_compare",
    "half_vs_half_goals", "ht_tie", "ht_lead", "first_goal_half", "first_goal_game",
    "compound_first_2h",
    # count / prop (base-rate)
    "team_offsides", "team_sot", "team_sot_half", "total_sot", "total_sot_half",
    "both_teams_sot", "half_sot_compare", "player_scorer", "player_involve",
    "player_sot", "player_sot_half", "team_corners", "total_corners",
    "corners_compare", "team_cards", "total_cards", "cards_compare",
    "fouls_compare", "pen_or_red", "pen_only",
    # novel knockout-stage markets
    "match_tie", "team_advance", "hydration_event", "any_brace", "sub_scorer",
    "any_player_2sot", "team_score_both_halves", "stoppage_goal", "red_card",
    "both_teams_card", "sub_before_ht", "win_margin", "total_offsides",
    "any_card_half", "total_shots", "shootout", "hold_lead", "outside_box_goal",
    "extra_time", "goal_each_half",
    # finals-week novelties (2026-07-16): bronze final + final slates
    "two_goals_same_half", "player_sot_compare", "distinct_shooters",
    "first_goal_assist",
}

_NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
              "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _num(token: str) -> Optional[int]:
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    return _NUM_WORDS.get(token)


def _threshold(low: str) -> Optional[int]:
    for pat in (r"at least (\w+)", r"(\d+) or more", r"(\d+) or fewer", r"(\d+)\+"):
        m = re.search(pat, low)
        if m:
            return _num(m.group(1))
    return None


def _half_of(low: str) -> Optional[int]:
    if "second half" in low or "2nd half" in low:
        return 2
    if "first half" in low or "1st half" in low or "at halftime" in low or "halftime" in low:
        return 1
    return None


def _find_team_in(text: str, home: Optional[str], away: Optional[str]) -> Optional[str]:
    """Side of the first match-team named in text."""
    low = text.lower()
    best_pos, best_side = None, None
    for code in (home, away):
        if not code:
            continue
        for alias in _TEAMS.get(code, []) + [code.lower()]:
            pos = low.find(alias)
            if pos >= 0 and (best_pos is None or pos < best_pos):
                best_pos, best_side = pos, "home" if code == home else "away"
    return best_side


def _subject_side(question: str, home, away) -> Optional[str]:
    """The team that is the grammatical subject (first team after 'will ')."""
    return _find_team_in(question, home, away)


def _extract_player(question: str, anchors) -> Optional[str]:
    low = question.lower()
    idx = -1
    for a in anchors:
        idx = low.find(a)
        if idx >= 0:
            break
    if idx < 0:
        return None
    prefix = question[:idx]
    m = re.search(r"will\s+(.+?)\s*$", prefix, flags=re.IGNORECASE)
    if not m:
        return None
    name = re.sub(r"\s*\([^)]*\)", "", m.group(1)).strip()  # drop "(Country)" tag
    name = re.sub(r"\s+(have|to|score)$", "", name, flags=re.IGNORECASE).strip()
    return name or None


def classify(question: str, match_name: str) -> Classification:  # noqa: C901 - a dispatch table
    q = question.strip()
    low = q.lower()
    # Knockout questions carry a boilerplate "(90 minutes + stoppage time)" suffix.
    # Strip it so its "stoppage time"/"minutes" tokens can't pollute rule matching
    # (e.g. "win by 2 goals (… + stoppage time)" must NOT look like a stoppage goal).
    low = low.replace("(90 minutes + stoppage time)", "")
    home, away = parse_match_name(match_name)
    thr = _threshold(low)
    half = _half_of(low)

    def subj() -> Optional[str]:
        return _subject_side(q, home, away)

    # --- compound: "A score the first goal of the game and B score in 2H" ---
    if "first goal of the game and" in low and "second half" in low:
        m = re.search(r"will\s+(.+?)\s+score the first goal.*?and\s+(.+?)\s+score", low)
        if m:
            sa = _find_team_in(m.group(1), home, away)
            sb = _find_team_in(m.group(2), home, away)
            if sa and sb:
                return Classification(kind="compound_first_2h", side=sa, side2=sb)

    # --- novel knockout markets (must precede the generic metric rules) ----
    # Hydration-break time-segmented events: "<event> before/after the
    # first/second hydration break". Keep ahead of the goal/card/offside/corner
    # rules, which would otherwise swallow them.
    if "hydration break" in low:
        ev = ("goal" if "goal" in low else "card" if "card" in low
              else "offside" if "offside" in low else "corner" if "corner" in low else None)
        if ev:
            ordn = "second" if "second hydration" in low else "first"
            pos = "after" if "after" in low else "before"
            # threshold matters: "2 or more corner kicks before the break" is a
            # COUNT question (P(>=2)), not an occurrence question (P(>=1)).
            return Classification("hydration_event", event=ev, window=f"{pos}_{ordn}",
                                  threshold=thr,
                                  reason="incl_et" if "extra time" in low else "")
    # "Will <Team> keep a clean sheet?" — the opponent scores 0. (New R16 kind,
    # 2026-07-01: was unmapped -> never submitted -> forfeited RBP at 2x weight.)
    if "clean sheet" in low:
        side = subj()
        return (Classification("clean_sheet", side=side) if side
                else Classification.unmapped("clean sheet no team"))
    # "Will a header goal be scored?" — a headed goal by anyone. (New R16 kind.)
    if "header" in low and "goal" in low:
        return Classification("header_goal")
    # "Will a substitute score a goal?" — any sub, not a named player.
    if "substitute" in low and "score" in low:
        return Classification("sub_scorer")
    # "Will any player score more than 1 goal?" — a brace by anyone.
    if (("more than 1 goal" in low or "2 or more goals" in low or "two or more goals" in low
            or "multiple goals" in low) and ("any player" in low or "a player" in low)):
        return Classification("any_brace")
    # "Will any player record 2+ shots on target?" — anyone, not a named player.
    # QF variant scopes it to one team ("any Portugal player ..."): without the
    # side match it fell through to team_sot and priced P(TEAM 2+ SoT) ~0.88
    # for a P(some single player 2+ SoT) ~0.6 question (caught 2026-07-05).
    if (("2 or more shots on target" in low or "more than 1 shot on target" in low
            or "two or more shots on target" in low)
            and ("any player" in low or "a player" in low or
                 re.search(r"any\s+\S.*?\s+player", low))):
        m = re.search(r"any\s+(.+?)\s+player", low)
        side = _find_team_in(m.group(1), home, away) if m else None
        return Classification("any_player_2sot", side=side)
    # "Will a goal be scored in first/second-half stoppage time?" (also matches
    # the "stoppage (added) time" phrasing variant the KO slate uses).
    if re.search(r"stoppage\b.*\btime", low) and "goal" in low:
        return Classification("stoppage_goal", half=2 if "second" in low else 1)
    # "Will a substitution be made before halftime?" (a sub event, not a sub scoring)
    if "substitution" in low and ("before halftime" in low or "before half-time" in low
            or "in the first half" in low or "before the first" in low):
        return Classification("sub_before_ht")
    # "Will the match go to extra time?" == level after 90' (QF novelty; the
    # hydration rule above already owns "... including any extra time" phrasings).
    if "go to extra time" in low or "goes to extra time" in low:
        return Classification("extra_time")
    # "Will at least one goal be scored in each half?" — no team subject.
    if "goal" in low and "each half" in low:
        return Classification("goal_each_half")
    # "Will there be N+ total shots (on and off target)?" — distinct from total SoT.
    if "total shots" in low and "on target" not in low and thr:
        return Classification("total_shots", threshold=thr)
    # "Will N or more different <Team> players attempt a shot?" (final novelty)
    if "different" in low and "player" in low and "shot" in low:
        m = re.search(r"different\s+(.+?)\s+players", low)
        side = _find_team_in(m.group(1), home, away) if m else None
        return (Classification("distinct_shooters", side=side, threshold=thr or 5)
                if side else Classification.unmapped("distinct shooters no team"))

    # --- comparison markets: "more <metric> than" --------------------------
    if "more" in low and "than" in low:
        side = subj()
        if "foul" in low:
            return Classification("fouls_compare", side=side) if side else Classification.unmapped("fouls cmp team")
        if "card" in low:
            return Classification("cards_compare", side=side) if side else Classification.unmapped("cards cmp team")
        if "corner" in low:
            return Classification("corners_compare", side=side, half=half) if side else Classification.unmapped("corner cmp team")
        if "shots on target" in low:
            # Player-vs-player SoT duel (final novelty): "Will <Player> (<Country>,
            # #N) record more shots on target than <Player2> (<Country>, #N)?" —
            # the parenthetical country resolves to a match side, which would
            # misroute a PLAYER duel into the TEAM compare. A "(...)" tag on both
            # subjects marks players; side carries player A's team.
            pm = re.search(r"will\s+(.+?)\s*\(([^),]+)[^)]*\)\s+record more shots on target "
                           r"than\s+(.+?)\s*\(", q, flags=re.IGNORECASE)
            if pm:
                pside = _find_team_in(pm.group(2), home, away)
                return Classification("player_sot_compare", player=pm.group(1).strip(),
                                      player2=pm.group(3).strip(), side=pside)
            return Classification("half_sot_compare", side=side, half=half or 2) if side else Classification.unmapped("sot cmp team")
        if "goal" in low and "first half" in low and ("second half" in low or "more total goals than the first" in low):
            return Classification("half_vs_half_goals")
        if "goal" in low and side:
            return Classification("half_goals_compare", side=side, half=half or 2)
        if "goals than the first half" in low or "more goals than the first half" in low:
            return Classification("half_vs_half_goals")

    # --- both teams shots on target (before team/player SoT) ---------------
    # "each team record N+ shots on target" is the bronze-final phrasing.
    # "(both teams combined)" is a TOTAL, not an each-team floor — leave it to
    # the total_sot rules below.
    if (("both teams" in low or "each team" in low) and "combined" not in low
            and ("shot on target" in low or "shots on target" in low)):
        return Classification("both_teams_sot", threshold=thr or 1, half=half)

    # --- both teams score (+ optional total) -------------------------------
    if "both teams" in low and "score" in low:
        m = re.search(r"(\d+)\s+or\s+more\s+total\s+goals", low)
        if m:
            return Classification("btts_and_total", threshold=int(m.group(1)))
        return Classification("btts")

    # --- penalty / red card ------------------------------------------------
    # Three question families contain the word "penalty" but are NOT penalty
    # markets — route them first or the catch-all below prices them all at the
    # pen-awarded base rate (found 2026-07-05: an "outside the penalty area"
    # goal market and a "hold a lead (excluding a penalty shootout)" market
    # both went out at ~0.15-0.18 via the pen path).
    if "outside the penalty area" in low:
        return Classification("outside_box_goal")
    # hold-lead MUST precede shootout: its questions end in "(excluding a
    # penalty shootout)" and would match a bare "shootout" test.
    if "hold a lead" in low or "held a lead" in low:
        side = subj()
        return (Classification("hold_lead", side=side) if side
                else Classification.unmapped("hold-lead no team"))
    if "shootout" in low and ("decided by" in low or "excluding" not in low):
        return Classification("shootout")
    if "penalty" in low and "red card" in low:
        return Classification("pen_or_red")
    if "penalty" in low and "red card" not in low:
        return Classification("pen_only")
    # standalone red card ("will a red card be shown") — combos handled above
    if "red card" in low:
        return Classification("red_card")

    # --- offsides ----------------------------------------------------------
    if "offside" in low:
        side = subj()
        if side and thr:
            return Classification("team_offsides", side=side, threshold=thr)
        if thr:  # "N+ offside calls" with no team named -> match total
            return Classification("total_offsides", threshold=thr)
        return Classification.unmapped("offside no team/thr")

    # --- shots on target counts (team / player / total) --------------------
    if "shots on target" in low or "shot on target" in low:
        if "total shots on target" in low or re.search(r"there be \d+ or more.*shots on target", low):
            if thr:
                return Classification("total_sot_half" if half == 2 else "total_sot", threshold=thr)
        side = _find_team_in(q, home, away)
        # team vs player: if the subject resolves to a match team -> team SoT.
        # Player SoT props are tagged "Will <Player> (<Country>) have N+ shots on
        # target"; resolve_team() substring-matches the parenthetical country, so a
        # "(...)" between the subject and "have" marks a PLAYER, not a team (else a
        # player's "3+ SoT" gets mispriced as a *team* total ~0.9).
        player_tagged = re.search(r"will\s+.+?\([^)]+\)\s+have\b", low)
        subj_token = re.search(r"will\s+(.+?)\s+have", low)
        subj_is_team = (subj_token and not player_tagged
                        and resolve_team(subj_token.group(1)) is not None)
        if subj_is_team and thr is not None:
            return Classification("team_sot_half" if half == 2 else "team_sot", side=side, threshold=thr)
        player = _extract_player(q, ["have at least", "have "])
        if player:
            return Classification("player_sot_half" if half == 2 else "player_sot",
                                  player=player, threshold=thr or 1, half=half)
        if side and thr is not None:
            return Classification("team_sot_half" if half == 2 else "team_sot", side=side, threshold=thr)
        return Classification.unmapped("SoT unresolved")

    # --- corners -----------------------------------------------------------
    if "corner" in low:
        if "total corner" in low or re.search(r"there be \d+ or more.*corner", low):
            if thr:
                return Classification("total_corners", threshold=thr, half=half)
        side = subj()
        if side and thr is not None:
            return Classification("team_corners", side=side, threshold=thr, half=half)
        return Classification.unmapped("corner unresolved")

    # --- cards -------------------------------------------------------------
    if "card" in low:
        if "both teams" in low:  # "both teams receive at least one card"
            return Classification("both_teams_card")
        if "total cards" in low or re.search(r"there be \d+ or more.*cards", low):
            if thr:
                return Classification("total_cards", threshold=thr, half=half)
        side = subj()
        if side and thr is not None:
            return Classification("team_cards", side=side, threshold=thr, half=half)
        if half and thr is None:  # "will a card be shown in the first/second half"
            return Classification("any_card_half", half=half)
        return Classification.unmapped("card unresolved")

    # --- player goal / involvement -----------------------------------------
    if "score or assist" in low:
        return Classification("player_involve", player=_extract_player(q, ["score or assist"]))
    if "score a goal" in low:
        player = _extract_player(q, ["score a goal"])
        if player:
            return Classification("player_scorer", player=player)

    # --- half-time state ---------------------------------------------------
    if "at halftime" in low or "halftime" in low or "half-time" in low:
        if "tied" in low:
            return Classification("ht_tie")
        if "winning" in low or "be ahead" in low or "leading" in low:
            side = subj()
            if side:
                return Classification("ht_lead", side=side)

    # "Will the first goal of the match be credited with an assist? Settles No
    # if the match is 0-0." (final novelty) — must precede the first-goal block.
    if "credited with an assist" in low:
        return Classification("first_goal_assist")

    # --- first goal of half / game -----------------------------------------
    if "first goal" in low:
        side = subj()
        if side:
            if "second half" in low:
                return Classification("first_goal_half", side=side, half=2)
            if "first half" in low:
                return Classification("first_goal_half", side=side, half=1)
            return Classification("first_goal_game", side=side)

    # "Will <team> score in both halves?" (before the generic score rules)
    if "both halves" in low and "score" in low:
        side = subj()
        if side:
            return Classification("team_score_both_halves", side=side)

    # "Will either team score 2 or more goals in the same half ...?" — a one-team
    # within-half burst (final novelty); must NOT fall through to the match-goals
    # totals regex, which priced it as P(total >= 2) ~0.67 vs true ~0.36.
    if "same half" in low and "goal" in low:
        return Classification("two_goals_same_half", threshold=thr or 2)

    # --- team scores (match / half) ----------------------------------------
    if "score" in low and ("at least" in low or "or more" in low) and "goal" in low:
        side = subj()
        if side:
            return Classification("team_score", side=side, threshold=thr or 1)
    if "score in the second half" in low or "score in the first half" in low:
        side = subj()
        if side:
            return Classification("team_score_half", side=side, half=2 if "second" in low else 1)

    # winning margin BEFORE totals — "win by 2 or more goals" must not read as a
    # goals-total ("2 or more goals" matches the totals regex).
    if "win by" in low and thr:
        side = subj()
        if side:
            return Classification("win_margin", side=side, threshold=thr)

    # --- totals: over / under / half ---------------------------------------
    if "total goals" in low or re.search(r"\d+\s+or\s+(more|fewer)\s+goals", low) or "total goals" in low:
        if "or fewer" in low:
            m = re.search(r"(\d+)\s+or\s+fewer", low)
            if m:
                return Classification("total_under", threshold=int(m.group(1)))
        if "second half" in low or "first half" in low:
            m = re.search(r"(\d+)\s+or\s+more", low)
            if m:
                return Classification("half_total_over", threshold=int(m.group(1)), half=2 if "second" in low else 1)
        m = re.search(r"(\d+)\s+or\s+more", low)
        if m:
            return Classification("total_over", threshold=int(m.group(1)))
    if ("second half" in low or "first half" in low) and "goal" in low and ("or more" in low):
        m = re.search(r"(\d+)\s+or\s+more", low)
        if m:
            return Classification("half_total_over", threshold=int(m.group(1)), half=2 if "second" in low else 1)

    # --- advancement / full-time tie (knockouts) ---------------------------
    if "advance" in low or "qualify" in low or ("reach" in low and "round of" in low):
        side = subj()
        if side:
            return Classification("team_advance", side=side)
    if ("end in a tie" in low or "end in a draw" in low or "be a draw" in low
            or "end in a stalemate" in low):
        return Classification("match_tie")
    # "Will <Team> win the World Cup / the third-place match (Bronze Final)?" —
    # the trophy/bronze is decided including ET+pens (there is no draw outcome),
    # so it's an advance-style price, NOT win-in-regulation. Questions qualified
    # "in regulation" keep routing to win_match below.
    if re.search(r"win the (world cup|third[- ]place|bronze)", low) and "regulation" not in low:
        side = subj()
        if side:
            return Classification("team_advance", side=side)

    # --- match winner ------------------------------------------------------
    if "win the match" in low or re.search(r"\bwin\b", low):
        side = subj()
        if side:
            return Classification("win_match", side=side)
        return Classification.unmapped("win no team")

    return Classification.unmapped(f"no rule: {q[:70]!r}")
