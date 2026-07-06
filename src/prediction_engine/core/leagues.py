"""League -> sport mapping and market-shape constants (ADR-026 / ADR-027).

Model artifacts, feature registries, and synthetic generators are keyed by
sport; games arrive keyed by league. This module owns the mapping so the
predictor, training pipeline, and registry all agree on it.
"""

LEAGUE_TO_SPORT: dict[str, str] = {
    "NFL": "FOOTBALL",
    "NCAA_FB": "FOOTBALL",
    "NBA": "BASKETBALL",
    "NCAA_BB": "BASKETBALL",
    "MLB": "BASEBALL",
    "NCAA_BSB": "BASEBALL",
    "FIFA_WC": "SOCCER",
    "EPL": "SOCCER",
    "NHL": "HOCKEY",
    "NCAA_HKY": "HOCKEY",
}

# Sports whose MONEYLINE market is three-way (HOME/DRAW/AWAY, ADR-027).
# NHL regulation-time three-way markets are deferred to the hockey wave.
THREE_WAY_MONEYLINE_SPORTS: frozenset[str] = frozenset({"SOCCER"})


def sport_for_league(league: str) -> str:
    """Return the sport for a league, raising a clear error for unknown leagues."""
    try:
        return LEAGUE_TO_SPORT[league]
    except KeyError:
        raise ValueError(f"unknown league {league!r}; no sport mapping registered") from None
