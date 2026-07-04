"""Market signal features from lines-service data."""

import statistics as stats_module

from prediction_engine.clients.lines import LineMovement, LineSnapshot
from prediction_engine.core.features.registry import FeatureMap


def market_features(movements: list[LineMovement], current_spread_lines: list[LineSnapshot]) -> FeatureMap:
    """Line movement magnitude and cross-book consensus for the spread market."""
    movement: float | None = None
    for entry in movements:
        if entry.total_movement is not None:
            movement = entry.total_movement
            break

    home_lines = [s.line_value for s in current_spread_lines if s.side == "HOME" and s.line_value is not None]
    books = {s.sportsbook_key for s in current_spread_lines if s.sportsbook_key}

    consensus_std: float | None = None
    if len(home_lines) >= 2:
        consensus_std = float(stats_module.pstdev(home_lines))
    elif home_lines:
        consensus_std = 0.0

    return {
        "line_movement": movement,
        "n_books_reporting": float(len(books)) if books else None,
        "line_consensus_std": consensus_std,
    }
