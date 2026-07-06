"""Rest and schedule features from statistics-service game lists."""

from datetime import date, datetime

from prediction_engine.clients.statistics import Game
from prediction_engine.core.features.registry import FeatureMap


def _parse_date(value: str) -> date | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


# A normal football week is 6 full days off (Sunday to Sunday); more than
# 10 means the team sat out a week -- the bye (a Sunday-to-Sunday bye is 13).
BYE_WEEK_REST_THRESHOLD = 10


def rest_features(game_date: date, recent_games: list[Game], prefix: str) -> FeatureMap:
    """Rest days and back-to-back indicator from a team's completed games.

    rest_days = full days off between games (played yesterday -> 0, i.e. a
    back-to-back). None when the team has no completed games (season start).
    """
    played_dates = sorted(
        (d for g in recent_games if (d := _parse_date(g.scheduled_start)) is not None and d < game_date),
        reverse=True,
    )
    if not played_dates:
        return {f"{prefix}_rest_days": None, f"{prefix}_back_to_back": None}
    rest_days = max((game_date - played_dates[0]).days - 1, 0)
    return {f"{prefix}_rest_days": float(rest_days), f"{prefix}_back_to_back": 1.0 if rest_days == 0 else 0.0}


def football_rest_features(game_date: date, recent_games: list[Game], prefix: str) -> FeatureMap:
    """Rest days plus the coming-off-a-bye flag for weekly-schedule football.

    Back-to-backs cannot happen in football, so that flag is replaced by a
    bye-week indicator: more than BYE_WEEK_REST_THRESHOLD full days off.
    None-propagation matches rest_features (no completed games -> both None).
    """
    rest = rest_features(game_date, recent_games, prefix)[f"{prefix}_rest_days"]
    bye = None if rest is None else (1.0 if rest > BYE_WEEK_REST_THRESHOLD else 0.0)
    return {f"{prefix}_rest_days": rest, f"{prefix}_bye_week": bye}
