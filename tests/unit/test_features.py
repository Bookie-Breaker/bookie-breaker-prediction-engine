"""Feature computation unit tests (pure functions, no HTTP)."""

from datetime import date

import pytest

from prediction_engine.clients.lines import LineMovement, LineSnapshot
from prediction_engine.clients.statistics import Game, InjuryReport, PlayerDetail, PlayerSeasonStats, TeamRef
from prediction_engine.core.features.injuries import injury_impact
from prediction_engine.core.features.market import market_features
from prediction_engine.core.features.registry import NBA_FEATURES
from prediction_engine.core.features.situational import rest_features


def make_game(game_id: str, start: str) -> Game:
    return Game(
        id=game_id,
        league="NBA",
        status="FINAL",
        home_team=TeamRef(id="h"),
        away_team=TeamRef(id="a"),
        scheduled_start=start,
    )


class TestRegistry:
    def test_no_duplicates_and_stable_order(self) -> None:
        assert len(NBA_FEATURES) == len(set(NBA_FEATURES))
        assert NBA_FEATURES[0] == "sim_probability"


class TestRestFeatures:
    game_date = date(2026, 1, 15)

    def test_two_days_rest(self) -> None:
        games = [make_game("g1", "2026-01-12T19:00:00Z")]
        features = rest_features(self.game_date, games, "home")
        assert features["home_rest_days"] == 2.0
        assert features["home_back_to_back"] == 0.0

    def test_back_to_back(self) -> None:
        games = [make_game("g1", "2026-01-14T19:00:00Z")]
        features = rest_features(self.game_date, games, "away")
        assert features["away_rest_days"] == 0.0
        assert features["away_back_to_back"] == 1.0

    def test_no_games_yields_none(self) -> None:
        features = rest_features(self.game_date, [], "home")
        assert features["home_rest_days"] is None

    def test_same_day_and_future_games_ignored(self) -> None:
        games = [make_game("g1", "2026-01-15T12:00:00Z"), make_game("g2", "2026-01-10T19:00:00Z")]
        features = rest_features(self.game_date, games, "home")
        assert features["home_rest_days"] == 4.0


class TestInjuryImpact:
    def test_weighted_by_status_and_minutes(self) -> None:
        injuries = [
            InjuryReport(player_id="p1", status="OUT"),
            InjuryReport(player_id="p2", status="INJURED"),
        ]
        players = {
            "p1": PlayerDetail(id="p1", season_stats=PlayerSeasonStats(points_per_game=24.0, minutes_per_game=36.0)),
            "p2": PlayerDetail(id="p2", season_stats=PlayerSeasonStats(points_per_game=12.0, minutes_per_game=24.0)),
        }
        impact = injury_impact(injuries, players)
        expected = -(0.95 * 24.0 * (36 / 48) + 0.60 * 12.0 * (24 / 48))
        assert impact == pytest.approx(expected)

    def test_missing_player_details_skipped(self) -> None:
        injuries = [InjuryReport(player_id="p1", status="OUT")]
        assert injury_impact(injuries, {}) == 0.0

    def test_healthy_team_zero(self) -> None:
        assert injury_impact([], {}) == 0.0


class TestMarketFeatures:
    def test_movement_and_consensus(self) -> None:
        movements = [LineMovement(total_movement=1.5)]
        snapshots = [
            LineSnapshot(id="1", game_id="g", sportsbook_key="dk", side="HOME", line_value=-3.5),
            LineSnapshot(id="2", game_id="g", sportsbook_key="fd", side="HOME", line_value=-4.0),
            LineSnapshot(id="3", game_id="g", sportsbook_key="mgm", side="AWAY", line_value=3.5),
        ]
        features = market_features(movements, snapshots)
        assert features["line_movement"] == 1.5
        assert features["n_books_reporting"] == 3.0
        assert features["line_consensus_std"] == pytest.approx(0.25)

    def test_empty_inputs_yield_none(self) -> None:
        features = market_features([], [])
        assert features["line_movement"] is None
        assert features["n_books_reporting"] is None
        assert features["line_consensus_std"] is None
