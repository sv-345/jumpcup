"""Bundled national-team strength ratings (Elo-like) + lambda construction.

This is the always-available, no-network prior so the bot can price every match
even with zero API keys and no reachable stats feed. When live odds are present
they override this entirely; when team stats are fetched they refine it. Ratings
are an approximate June-2026 men's international snapshot — good enough as a
prior for flooring, and shrunk hard by the engine's confidence tiers.
"""

from __future__ import annotations

from typing import Optional, Tuple

from . import model

# Approximate Elo-style ratings for the live field. Ordering matters more than
# absolute level; the engine shrinks anything derived from these.
ELO = {
    "BRA": 2080, "ARG": 2070, "FRA": 2060, "ESP": 2040, "POR": 1990,
    "GER": 1980, "ENG": 1975, "NED": 1970, "BEL": 1940, "CRO": 1900,
    "URU": 1890, "COL": 1860, "MAR": 1830, "SUI": 1820, "NOR": 1820,
    "MEX": 1820, "TUR": 1820, "SEN": 1820, "JPN": 1810, "USA": 1800,
    "ECU": 1780, "AUT": 1780, "CZE": 1780, "IRN": 1760, "CIV": 1760,
    "KOR": 1760, "SWE": 1760, "CAN": 1760, "ALG": 1750, "SCO": 1740,
    "PAR": 1730, "TUN": 1730, "GHA": 1720, "BIH": 1720, "AUS": 1720,
    "RSA": 1700, "COD": 1700, "EGY": 1770, "PAN": 1680, "QAT": 1680,
    "UZB": 1680, "KSA": 1670, "IRQ": 1640, "CPV": 1640, "JOR": 1620,
    "NZL": 1620, "CUW": 1560, "HAI": 1560,
}
_DEFAULT_ELO = 1700  # unknown team -> mid-low prior


def elo(code: Optional[str]) -> float:
    return ELO.get(code or "", _DEFAULT_ELO)


def lambdas_for_match(home_code: Optional[str], away_code: Optional[str]
                      ) -> Tuple[float, float, str]:
    """Expected goals for a fixture from bundled ratings (model_elo tier)."""
    return model.lambdas_from_elo(elo(home_code), elo(away_code))
