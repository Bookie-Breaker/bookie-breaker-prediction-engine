"""FeatureBuilder NCAA_BB path tests with stubbed clients (Phase 6 Wave 5).

NCAA_BB is a BASKETBALL-sport league but trains its own single-league model:
the NBA feature approach minus injuries (no reliable college source) plus a
CBBD adjusted-efficiency margin per side. The NBA path must stay unchanged.
"""

from typing import Any

import pytest

from prediction_engine.clients.lines import LineMovement, LineSnapshot
from prediction_engine.clients.statistics import (
    AdvancedStats,
    DefensiveStats,
    Game,
    HomeAwaySplit,
    HomeAwaySplits,
    OffensiveStats,
    StatBlocks,
    TeamRef,
    TeamStats,
)
from prediction_engine.core.features.builder import FeatureBuilder
from prediction_engine.core.features.registry import NCAA_BB_FEATURES

GAME_START = "2026-12-05T00:00:00Z"


def make_game(league: str = "NCAA_BB") -> Game:
    return Game(
        id="game-cbb-1",
        league=league,
        status="SCHEDULED",
        home_team=TeamRef(id="t-home", name="Duke Blue Devils"),
        away_team=TeamRef(id="t-away", name="Kansas Jayhawks"),
        scheduled_start=GAME_START,
        season=2026,
        season_type="REGULAR",
    )


def cbb_stats(team_id: str, off_rating: float, aem: float, with_splits: bool = False) -> TeamStats:
    return TeamStats(
        team_id=team_id,
        season=2026,
        games_played=12,
        stats=StatBlocks(
            offensive=OffensiveStats(
                points_per_game=78.0, three_point_pct=0.35, offensive_rating=off_rating, pace=68.0
            ),
            defensive=DefensiveStats(points_allowed_per_game=68.0, defensive_rating=95.0),
            advanced=AdvancedStats(net_rating=off_rating - 95.0, adjusted_efficiency_margin=aem),
        ),
        home_away_splits=(
            HomeAwaySplits(
                home=HomeAwaySplit(points_per_game=82.0, points_allowed_per_game=66.0),
                away=HomeAwaySplit(points_per_game=74.0, points_allowed_per_game=70.0),
            )
            if with_splits
            else None
        ),
    )


def previous_game(game_id: str, start: str) -> Game:
    return Game(
        id=game_id,
        league="NCAA_BB",
        status="FINAL",
        home_team=TeamRef(id="t-home"),
        away_team=TeamRef(id="x"),
        scheduled_start=start,
    )


class FakeStatistics:
    def __init__(self, stats_by_team: dict[str, TeamStats], recent_by_team: dict[str, list[Game]]) -> None:
        self._stats = stats_by_team
        self._recent = recent_by_team
        self.injury_calls: list[str] = []

    async def get_team_stats(self, team_id: str, rolling_window: int | None = None) -> TeamStats:
        return self._stats[team_id]

    async def list_recent_games(self, team_id: str, date_to: str, limit: int = 10) -> list[Game]:
        return self._recent.get(team_id, [])

    async def get_injuries(self, team_id: str, league: str) -> list[Any]:
        self.injury_calls.append(team_id)
        return []

    async def get_player(self, player_id: str) -> Any:
        raise AssertionError("player lookups are not part of the NCAA_BB path")

    async def get_meta_timestamp(self, path: str) -> str | None:
        return "2026-12-05T00:00:00Z"


class FakeLines:
    async def movement(self, game_external_id: str, market_type: str | None = None) -> list[LineMovement]:
        return [LineMovement(total_movement=0.5)]

    async def current_lines(self, **kwargs: Any) -> list[LineSnapshot]:
        return [
            LineSnapshot(id="s1", game_id="ext-1", sportsbook_key="dk", side="HOME", line_value=-6.5),
            LineSnapshot(id="s2", game_id="ext-1", sportsbook_key="fd", side="HOME", line_value=-6.5),
        ]


class FakeReconciler:
    def __init__(self, resolved: str | None = "ext-1") -> None:
        self._resolved = resolved

    async def resolve(self, game: Game) -> str | None:
        return self._resolved


def make_builder(stats: FakeStatistics) -> FeatureBuilder:
    return FeatureBuilder(stats, FakeLines(), FakeReconciler())  # type: ignore[arg-type]


def default_stats() -> FakeStatistics:
    return FakeStatistics(
        stats_by_team={
            "t-home": cbb_stats("t-home", off_rating=118.0, aem=24.5, with_splits=True),
            "t-away": cbb_stats("t-away", off_rating=112.0, aem=15.0),
        },
        recent_by_team={
            "t-home": [previous_game("prev-h", "2026-12-03T00:00:00Z")],  # 1 rest day
            "t-away": [previous_game("prev-a", "2026-12-04T00:00:00Z")],  # played yesterday: back-to-back
        },
    )


class TestNcaaBbBuilderPath:
    async def test_adjusted_efficiency_margin_per_side(self) -> None:
        bundle = await make_builder(default_stats()).build(make_game())
        assert bundle.features["home_adjusted_efficiency_margin"] == 24.5
        assert bundle.features["away_adjusted_efficiency_margin"] == 15.0

    async def test_nba_style_features_present(self) -> None:
        bundle = await make_builder(default_stats()).build(make_game())
        assert bundle.features["home_offensive_rating"] == 118.0
        assert bundle.features["home_pace"] == 68.0
        assert bundle.features["net_rating_diff"] == pytest.approx((118.0 - 95.0) - (112.0 - 95.0))
        assert bundle.features["pace_differential"] == pytest.approx(0.0)
        assert bundle.features["home_away_split_diff"] is not None
        assert bundle.lines_game_external_id == "ext-1"

    async def test_rest_and_back_to_back(self) -> None:
        bundle = await make_builder(default_stats()).build(make_game())
        assert bundle.features["home_rest_days"] == 1.0
        assert bundle.features["away_rest_days"] == 0.0
        assert bundle.features["away_back_to_back"] == 1.0
        assert bundle.features["rest_advantage"] == pytest.approx(1.0)

    async def test_no_injury_features_or_lookups(self) -> None:
        stats = default_stats()
        bundle = await make_builder(stats).build(make_game())
        assert stats.injury_calls == []
        assert not any("injury" in name for name in bundle.features)

    async def test_no_league_one_hot(self) -> None:
        bundle = await make_builder(default_stats()).build(make_game())
        assert not any("league_is" in name for name in bundle.features)

    async def test_missing_advanced_block_yields_none_aem(self) -> None:
        stats = FakeStatistics(
            stats_by_team={
                "t-home": TeamStats(team_id="t-home", games_played=0),
                "t-away": TeamStats(team_id="t-away", games_played=0),
            },
            recent_by_team={},
        )
        bundle = await make_builder(stats).build(make_game())
        assert bundle.features["home_adjusted_efficiency_margin"] is None
        assert bundle.features["home_offensive_rating"] is None
        assert bundle.features["n_books_reporting"] == 2.0

    async def test_all_non_row_registry_features_are_emitted(self) -> None:
        bundle = await make_builder(default_stats()).build(make_game())
        row_level = {
            "sim_probability",
            "sim_margin_mean",
            "sim_total_mean",
            "sim_converged",
            "market_is_spread",
            "market_is_total",
            "market_is_moneyline",
        }
        assert set(NCAA_BB_FEATURES) - row_level == set(bundle.features)


class TestNbaParityRegression:
    async def test_nba_path_still_has_injuries_and_no_aem(self) -> None:
        stats = FakeStatistics(
            stats_by_team={
                "t-home": TeamStats(team_id="t-home", games_played=60),
                "t-away": TeamStats(team_id="t-away", games_played=60),
            },
            recent_by_team={"t-home": [previous_game("prev-h", "2026-12-03T00:00:00Z")]},
        )
        game = Game(
            id="game-nba",
            league="NBA",
            status="SCHEDULED",
            home_team=TeamRef(id="t-home", name="Los Angeles Lakers"),
            away_team=TeamRef(id="t-away", name="Boston Celtics"),
            scheduled_start=GAME_START,
        )
        bundle = await make_builder(stats).build(game)
        # NBA keeps its injury features and does NOT gain the college AEM column
        assert "home_injury_impact" in bundle.features
        assert "injury_impact_diff" in bundle.features
        assert stats.injury_calls == ["t-home", "t-away"]
        assert "home_adjusted_efficiency_margin" not in bundle.features
