"""Pure outcome settlement for prediction rows (Phase 7 Wave 4).

Mirrors the bookie-emulator's grading semantics (core/grading.py there;
local copy per ADR-009, no shared package) reduced to what retraining and
experiment evaluation need: a binary outcome per prediction row, with None
for push/void rows that carry no gradeable signal.

Sign conventions match this service's own rows: SPREAD lines are stored in
the selection relative to the selected side (negative favorites), TOTAL
lines are the over/under number, and soccer moneylines grade three-way
(ADR-027, ties win DRAW with no push path). Soccer settles on the
90-minute regulation score when present (ADR-027), which
scores_for_settlement handles via the statistics-service GameResult.
"""

from prediction_engine.clients.statistics import Game
from prediction_engine.core.leagues import THREE_WAY_MONEYLINE_SPORTS, sport_for_league

# Default side for rows persisted before Phase 6 (side NULL; the single row
# per market was always the HOME/OVER perspective).
DEFAULT_SIDE_BY_MARKET: dict[str, str] = {"SPREAD": "HOME", "MONEYLINE": "HOME", "TOTAL": "OVER"}


def settle_outcome(
    market_type: str,
    side: str,
    line: float | None,
    home_score: int,
    away_score: int,
    three_way_moneyline: bool = False,
) -> float | None:
    """Grade one prediction row: 1.0 win, 0.0 loss, None push/void.

    ``line`` is relative to the selected side for SPREAD (a home favorite's
    HOME row carries a negative line) and is the total number for TOTAL;
    MONEYLINE ignores it. A tied two-way moneyline is a void (None), a tied
    three-way moneyline wins DRAW and loses HOME/AWAY.
    """
    if side == "DRAW" and not (three_way_moneyline and market_type == "MONEYLINE"):
        raise ValueError(f"side DRAW is only gradeable on a three-way MONEYLINE, not {market_type}")
    margin = home_score - away_score
    total = home_score + away_score
    selected_margin = margin if side == "HOME" else -margin

    if market_type == "MONEYLINE":
        if three_way_moneyline:
            winner = "HOME" if margin > 0 else "AWAY" if margin < 0 else "DRAW"
            return 1.0 if side == winner else 0.0
        if margin == 0:
            return None
        return 1.0 if selected_margin > 0 else 0.0

    if market_type == "SPREAD":
        if line is None:
            raise ValueError("SPREAD settlement requires a line")
        result = selected_margin + line
        if result == 0:
            return None
        return 1.0 if result > 0 else 0.0

    if market_type == "TOTAL":
        if line is None:
            raise ValueError("TOTAL settlement requires a line")
        over_result = total - line
        if over_result == 0:
            return None
        won_over = over_result > 0
        return 1.0 if (won_over if side == "OVER" else not won_over) else 0.0

    raise ValueError(f"cannot settle market type {market_type}")


def line_from_selection(market_type: str, selection: str) -> float | None:
    """Recover the target line from this service's own selection strings.

    Prediction rows do not store a line column for game markets; the line
    is embedded in the selection this service formats itself ("Los Angeles
    Lakers -3.5", "Over 220.5"), so parsing the trailing token is exact by
    construction. MONEYLINE selections ("... ML", "Draw") carry no line.
    """
    if market_type not in ("SPREAD", "TOTAL"):
        return None
    token = selection.rsplit(maxsplit=1)[-1]
    try:
        return float(token)
    except ValueError:
        raise ValueError(f"selection {selection!r} has no trailing {market_type} line") from None


def is_three_way_moneyline_league(league: str) -> bool:
    """True when the league's moneylines grade three-way (ADR-027)."""
    return sport_for_league(league) in THREE_WAY_MONEYLINE_SPORTS


def scores_for_settlement(game: Game) -> tuple[int, int] | None:
    """Settlement-relevant (home, away) scores for a fetched game, or None.

    None while the game is not FINAL or scores are missing. Soccer settles
    every standard market on the regulation score when the game went past
    90 minutes (ADR-027); other sports settle on the final score.
    """
    if game.status != "FINAL":
        return None
    result = game.result
    if (
        sport_for_league(game.league) == "SOCCER"
        and result is not None
        and result.regulation_home_score is not None
        and result.regulation_away_score is not None
    ):
        return result.regulation_home_score, result.regulation_away_score
    if result is not None:
        return result.home_score, result.away_score
    if game.home_score is not None and game.away_score is not None:
        return game.home_score, game.away_score
    return None
