"""settle_outcome matrix and settlement helpers (Phase 7 Wave 4)."""

import pytest

from prediction_engine.clients.statistics import Game, GameResult, TeamRef
from prediction_engine.core.settlement import (
    DEFAULT_SIDE_BY_MARKET,
    is_three_way_moneyline_league,
    line_from_selection,
    scores_for_settlement,
    settle_outcome,
)


def make_game(
    league: str = "NBA",
    status: str = "FINAL",
    result: GameResult | None = None,
    home_score: int | None = None,
    away_score: int | None = None,
) -> Game:
    return Game(
        id="game-1",
        league=league,
        status=status,
        home_team=TeamRef(id="t-home", name="Home"),
        away_team=TeamRef(id="t-away", name="Away"),
        scheduled_start="2026-03-01T00:00:00Z",
        result=result,
        home_score=home_score,
        away_score=away_score,
    )


class TestSpreadSettlement:
    def test_home_favorite_covers(self) -> None:
        assert settle_outcome("SPREAD", "HOME", -3.5, 110, 100) == 1.0

    def test_home_favorite_fails_to_cover(self) -> None:
        assert settle_outcome("SPREAD", "HOME", -3.5, 102, 100) == 0.0

    def test_away_side_uses_selected_margin(self) -> None:
        # away line +3.5, away loses by 2 -> covers
        assert settle_outcome("SPREAD", "AWAY", 3.5, 102, 100) == 1.0
        assert settle_outcome("SPREAD", "AWAY", 3.5, 104, 100) == 0.0

    def test_whole_number_line_pushes(self) -> None:
        assert settle_outcome("SPREAD", "HOME", -3.0, 103, 100) is None

    def test_spread_requires_line(self) -> None:
        with pytest.raises(ValueError, match="requires a line"):
            settle_outcome("SPREAD", "HOME", None, 103, 100)


class TestTotalSettlement:
    def test_over_wins_and_loses(self) -> None:
        assert settle_outcome("TOTAL", "OVER", 220.5, 115, 110) == 1.0
        assert settle_outcome("TOTAL", "OVER", 230.5, 115, 110) == 0.0

    def test_under_is_complement(self) -> None:
        assert settle_outcome("TOTAL", "UNDER", 230.5, 115, 110) == 1.0
        assert settle_outcome("TOTAL", "UNDER", 220.5, 115, 110) == 0.0

    def test_whole_number_total_pushes(self) -> None:
        assert settle_outcome("TOTAL", "OVER", 225.0, 115, 110) is None
        assert settle_outcome("TOTAL", "UNDER", 225.0, 115, 110) is None


class TestMoneylineSettlement:
    def test_two_way_home_and_away(self) -> None:
        assert settle_outcome("MONEYLINE", "HOME", None, 3, 1) == 1.0
        assert settle_outcome("MONEYLINE", "AWAY", None, 3, 1) == 0.0
        assert settle_outcome("MONEYLINE", "AWAY", None, 1, 3) == 1.0

    def test_two_way_tie_is_void(self) -> None:
        assert settle_outcome("MONEYLINE", "HOME", None, 20, 20) is None
        assert settle_outcome("MONEYLINE", "AWAY", None, 20, 20) is None

    def test_three_way_tie_wins_draw_no_push(self) -> None:
        assert settle_outcome("MONEYLINE", "DRAW", None, 1, 1, three_way_moneyline=True) == 1.0
        assert settle_outcome("MONEYLINE", "HOME", None, 1, 1, three_way_moneyline=True) == 0.0
        assert settle_outcome("MONEYLINE", "AWAY", None, 1, 1, three_way_moneyline=True) == 0.0

    def test_three_way_decisive_results(self) -> None:
        assert settle_outcome("MONEYLINE", "HOME", None, 2, 0, three_way_moneyline=True) == 1.0
        assert settle_outcome("MONEYLINE", "DRAW", None, 2, 0, three_way_moneyline=True) == 0.0
        assert settle_outcome("MONEYLINE", "AWAY", None, 0, 2, three_way_moneyline=True) == 1.0

    def test_draw_side_invalid_on_two_way(self) -> None:
        with pytest.raises(ValueError, match="three-way MONEYLINE"):
            settle_outcome("MONEYLINE", "DRAW", None, 1, 1)
        with pytest.raises(ValueError, match="three-way MONEYLINE"):
            settle_outcome("SPREAD", "DRAW", -1.5, 2, 1, three_way_moneyline=True)

    def test_unknown_market_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot settle market type PLAYER_PROP"):
            settle_outcome("PLAYER_PROP", "OVER", 25.5, 1, 0)


class TestLineFromSelection:
    def test_spread_selection_trailing_line(self) -> None:
        assert line_from_selection("SPREAD", "Los Angeles Lakers -3.5") == -3.5
        assert line_from_selection("SPREAD", "Boston Celtics +7.5") == 7.5

    def test_total_selection(self) -> None:
        assert line_from_selection("TOTAL", "Over 220.5") == 220.5

    def test_moneyline_has_no_line(self) -> None:
        assert line_from_selection("MONEYLINE", "Los Angeles Lakers ML") is None
        assert line_from_selection("MONEYLINE", "Draw") is None

    def test_unparseable_selection_raises(self) -> None:
        with pytest.raises(ValueError, match="no trailing SPREAD line"):
            line_from_selection("SPREAD", "Los Angeles Lakers ML")


class TestScoresForSettlement:
    def test_not_final_returns_none(self) -> None:
        assert scores_for_settlement(make_game(status="SCHEDULED", home_score=1, away_score=0)) is None

    def test_final_uses_result_scores(self) -> None:
        game = make_game(result=GameResult(home_score=110, away_score=104))
        assert scores_for_settlement(game) == (110, 104)

    def test_final_falls_back_to_top_level_scores(self) -> None:
        assert scores_for_settlement(make_game(home_score=99, away_score=98)) == (99, 98)

    def test_final_without_scores_returns_none(self) -> None:
        assert scores_for_settlement(make_game()) is None

    def test_soccer_prefers_regulation_scores(self) -> None:
        result = GameResult(home_score=2, away_score=1, overtime=True, regulation_home_score=1, regulation_away_score=1)
        assert scores_for_settlement(make_game(league="FIFA_WC", result=result)) == (1, 1)

    def test_soccer_without_regulation_block_uses_final(self) -> None:
        game = make_game(league="EPL", result=GameResult(home_score=3, away_score=0))
        assert scores_for_settlement(game) == (3, 0)

    def test_non_soccer_ignores_regulation_scores(self) -> None:
        result = GameResult(home_score=5, away_score=4, overtime=True, regulation_home_score=4, regulation_away_score=4)
        assert scores_for_settlement(make_game(league="NHL", result=result)) == (5, 4)


class TestLeagueHelpers:
    def test_three_way_leagues(self) -> None:
        assert is_three_way_moneyline_league("FIFA_WC")
        assert is_three_way_moneyline_league("EPL")
        assert not is_three_way_moneyline_league("NBA")
        assert not is_three_way_moneyline_league("NHL")

    def test_default_sides(self) -> None:
        assert DEFAULT_SIDE_BY_MARKET == {"SPREAD": "HOME", "MONEYLINE": "HOME", "TOTAL": "OVER"}
