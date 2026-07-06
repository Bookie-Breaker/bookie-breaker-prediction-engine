"""FeatureBuilder hockey path tests with stubbed clients (Phase 6 Wave 4).

Single-league NHL model: goals/shots per game, power play / penalty kill,
team save pct, plus rest and back-to-back (dense schedule). No draws
(OT/SO resolves), no injuries, no starting-goaltender feed.
"""

from typing import Any

from prediction_engine.clients.lines import LineMovement, LineSnapshot
from prediction_engine.clients.statistics import (
    Game,
    HockeyStats,
    SoccerStats,
    StatBlocks,
    TeamRef,
    TeamStats,
)
from prediction_engine.core.features.builder import FeatureBuilder
from prediction_engine.core.features.registry import HOCKEY_FEATURES

GAME_START = "2026-11-15T00:00:00Z"


def make_game(league: str = "NHL") -> Game:
    return Game(
        id="game-nhl-1",
        league=league,
        status="SCHEDULED",
        home_team=TeamRef(id="t-home", name="Colorado Avalanche"),
        away_team=TeamRef(id="t-away", name="Vegas Golden Knights"),
        scheduled_start=GAME_START,
        season=2026,
        season_type="REGULAR",
    )


def hockey_stats(team_id: str, gf: float, save_pct: float) -> TeamStats:
    return TeamStats(
        team_id=team_id,
        season=2026,
        games_played=18,
        stats=StatBlocks(
            hockey=HockeyStats(
                goals_for_per_game=gf,
                goals_against_per_game=2.8,
                shots_for_per_game=31.5,
                shots_against_per_game=29.0,
                power_play_pct=0.235,
                penalty_kill_pct=0.815,
                team_save_pct=save_pct,
            )
        ),
    )


def previous_game(game_id: str, start: str) -> Game:
    return Game(
        id=game_id,
        league="NHL",
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
        raise AssertionError("player lookups are not part of the hockey path")

    async def get_meta_timestamp(self, path: str) -> str | None:
        return "2026-11-15T00:00:00Z"


class FakeLines:
    async def movement(self, game_external_id: str, market_type: str | None = None) -> list[LineMovement]:
        return [LineMovement(total_movement=0.5)]

    async def current_lines(self, **kwargs: Any) -> list[LineSnapshot]:
        return [
            LineSnapshot(id="s1", game_id="ext-1", sportsbook_key="dk", side="HOME", line_value=-1.5),
            LineSnapshot(id="s2", game_id="ext-1", sportsbook_key="fd", side="HOME", line_value=-1.5),
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
            "t-home": hockey_stats("t-home", gf=3.4, save_pct=0.915),
            "t-away": hockey_stats("t-away", gf=3.0, save_pct=0.902),
        },
        recent_by_team={
            "t-home": [previous_game("prev-h", "2026-11-14T00:00:00Z")],  # played yesterday: back-to-back
            "t-away": [previous_game("prev-a", "2026-11-12T00:00:00Z")],  # 2 full rest days
        },
    )


class TestHockeyBuilderPath:
    async def test_hockey_block_team_features(self) -> None:
        bundle = await make_builder(default_stats()).build(make_game())

        assert bundle.features["home_goals_for_per_game"] == 3.4
        assert bundle.features["away_goals_for_per_game"] == 3.0
        assert bundle.features["home_goals_against_per_game"] == 2.8
        assert bundle.features["home_shots_for_per_game"] == 31.5
        assert bundle.features["home_power_play_pct"] == 0.235
        assert bundle.features["home_penalty_kill_pct"] == 0.815
        assert bundle.features["home_team_save_pct"] == 0.915
        assert bundle.features["away_team_save_pct"] == 0.902
        assert bundle.lines_game_external_id == "ext-1"

    async def test_rest_and_back_to_back_from_schedule(self) -> None:
        bundle = await make_builder(default_stats()).build(make_game())
        assert bundle.features["home_rest_days"] == 0.0  # played yesterday
        assert bundle.features["home_back_to_back"] == 1.0
        assert bundle.features["away_rest_days"] == 2.0
        assert bundle.features["away_back_to_back"] == 0.0

    async def test_no_injury_lookups_for_hockey(self) -> None:
        stats = default_stats()
        bundle = await make_builder(stats).build(make_game())
        assert stats.injury_calls == []
        assert not any("injury" in name for name in bundle.features)

    async def test_no_draw_or_league_features(self) -> None:
        bundle = await make_builder(default_stats()).build(make_game())
        assert not any("draw" in name for name in bundle.features)
        assert "league_is_nhl" not in bundle.features

    async def test_missing_hockey_block_yields_none(self) -> None:
        stats = FakeStatistics(
            stats_by_team={
                "t-home": TeamStats(team_id="t-home", games_played=0),
                "t-away": TeamStats(team_id="t-away", games_played=0),
            },
            recent_by_team={},
        )
        bundle = await make_builder(stats).build(make_game())
        assert bundle.features["home_goals_for_per_game"] is None
        assert bundle.features["home_team_save_pct"] is None
        assert bundle.features["home_rest_days"] is None
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
        assert set(HOCKEY_FEATURES) - row_level == set(bundle.features)


class TestOtherSportParityRegressions:
    async def test_nba_path_unchanged(self) -> None:
        stats = FakeStatistics(
            stats_by_team={
                "t-home": TeamStats(team_id="t-home", games_played=60),
                "t-away": TeamStats(team_id="t-away", games_played=60),
            },
            recent_by_team={"t-home": [previous_game("prev-h", "2026-11-13T00:00:00Z")]},
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
        for name in ("home_offensive_rating", "net_rating_diff", "home_injury_impact"):
            assert name in bundle.features
        assert stats.injury_calls == ["t-home", "t-away"]
        assert not any("save_pct" in name for name in bundle.features)
        assert not any("penalty_kill" in name for name in bundle.features)

    async def test_soccer_path_unchanged(self) -> None:
        stats = FakeStatistics(
            stats_by_team={
                "t-home": TeamStats(
                    team_id="t-home",
                    games_played=5,
                    stats=StatBlocks(soccer=SoccerStats(attack_strength=1.3, form_points_last5=11)),
                ),
                "t-away": TeamStats(
                    team_id="t-away",
                    games_played=5,
                    stats=StatBlocks(soccer=SoccerStats(attack_strength=1.1, form_points_last5=8)),
                ),
            },
            recent_by_team={},
        )
        game = Game(
            id="game-epl",
            league="EPL",
            status="SCHEDULED",
            home_team=TeamRef(id="t-home", name="Arsenal"),
            away_team=TeamRef(id="t-away", name="Chelsea"),
            scheduled_start=GAME_START,
            season_type="REGULAR",
        )
        bundle = await make_builder(stats).build(game)
        assert bundle.features["home_attack_strength"] == 1.3
        assert not any("save_pct" in name for name in bundle.features)
