"""Injury impact proxy.

The statistics-service injury report has status only (no WAR or impact
score), so impact is approximated as points-per-game at risk weighted by
the probability the player misses the game given their status and their
minutes share. Returned negative: more injured production = more negative.
"""

from prediction_engine.clients.statistics import InjuryReport, PlayerDetail

# P(misses the game) by injury status
MISS_PROBABILITY: dict[str, float] = {
    "OUT": 0.95,
    "INJURED": 0.60,  # day-to-day
    "SUSPENDED": 1.00,
    "INACTIVE": 0.90,
}
DEFAULT_MISS_PROBABILITY = 0.25
MAX_PLAYERS_CONSIDERED = 8
REGULATION_MINUTES = 48.0


def injury_impact(injuries: list[InjuryReport], players: dict[str, PlayerDetail]) -> float:
    """Negative expected points-per-game lost to injuries."""
    impact = 0.0
    for report in injuries[:MAX_PLAYERS_CONSIDERED]:
        player = players.get(report.player_id)
        if player is None or player.season_stats is None:
            continue
        miss_prob = MISS_PROBABILITY.get(report.status.upper(), DEFAULT_MISS_PROBABILITY)
        minutes_share = min(player.season_stats.minutes_per_game / REGULATION_MINUTES, 1.0)
        impact += miss_prob * player.season_stats.points_per_game * minutes_share
    return -impact
