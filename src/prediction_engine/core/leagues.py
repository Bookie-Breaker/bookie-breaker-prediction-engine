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
# Hockey stays two-way: NHL moneylines settle on the final including
# overtime/shootout, and regulation-time three-way markets remain deferred
# (ADR-027 hockey note). Football stays two-way as well: rare NFL ties
# grade the moneyline as PUSH rather than adding a third outcome.
THREE_WAY_MONEYLINE_SPORTS: frozenset[str] = frozenset({"SOCCER"})

# Prediction-model / feature-registry key overrides per league. Sports pool
# their leagues into one model by default (ADR-026: BASEBALL pools MLB +
# NCAA_BSB, FOOTBALL pools NFL + NCAA_FB with a league one-hot). NCAA_BB is
# the exception: the college game's scoring environment, pace, and the
# CBBD adjusted-efficiency block differ enough from the order-locked NBA
# tuple that it trains its own single-league model, keyed by the league
# name. NOTE: model_versions.sport is sport_enum in the shared schema, so
# registering an NCAA_BB bootstrap in the database requires infra-ops to
# add the enum value before the league enables at season start (November);
# the league ships dormant this wave and nothing bootstraps it until then.
_MODEL_KEY_OVERRIDES: dict[str, str] = {"NCAA_BB": "NCAA_BB"}


def sport_for_league(league: str) -> str:
    """Return the sport for a league, raising a clear error for unknown leagues."""
    try:
        return LEAGUE_TO_SPORT[league]
    except KeyError:
        raise ValueError(f"unknown league {league!r}; no sport mapping registered") from None


def model_key_for_league(league: str) -> str:
    """Return the model / feature-registry key for a league.

    This is the league's sport for every pooled sport, and the league name
    itself for leagues that train their own model (NCAA_BB).
    """
    return _MODEL_KEY_OVERRIDES.get(league, sport_for_league(league))
