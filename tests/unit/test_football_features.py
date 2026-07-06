"""FeatureBuilder football path tests with stubbed clients (Phase 6 Wave 3).

Football pools NFL + NCAA_FB into one model with a league one-hot. EPA is
NFL-only, SP+ is NCAA_FB-only (league-gated to None), bye-week flags come
from rest days, and the moneyline stays two-way (a tie is a PUSH).
"""

from typing import Any

import pytest

from prediction_engine.clients.lines import LineMovement, LineSnapshot
from prediction_engine.clients.statistics import (
    FootballStats,
    Game,
    SoccerStats,
    StatBlocks,
    TeamRef,
    TeamStats,
)
from prediction_engine.core.features.builder import FeatureBuilder
from prediction_engine.core.features.registry import FOOTBALL_FEATURES

GAME_START = "2026-11-15T18:00:00Z"


def make_game(league: str = "NFL", scheduled_start: str = GAME_START) -> Game:
    return Game(
        id="game-nfl-1",
        league=league,
        status="SCHEDULED",
        home_team=TeamRef(id="t-home", name="Kansas City Chiefs"),
        away_team=TeamRef(id="t-away", name="Buffalo Bills"),
        scheduled_start=scheduled_start,
        season=2026,
        season_type="REGULAR",
    )


def nfl_stats(team_id: str, ppg: float, epa_off: float) -> TeamStats:
    return TeamStats(
        team_id=team_id,
        season=2026,
        games_played=10,
        stats=StatBlocks(
            football=FootballStats(
                points_per_game=ppg,
                points_allowed_per_game=20.5,
                drives_per_game=11.0,
                points_per_drive_off=2.1,
                points_per_drive_def=1.8,
                epa_per_play_off=epa_off,
                epa_per_play_def=-0.02,
                turnover_margin_per_game=0.4,
                # sp_plus_rating left at 0.0 default (absent for the NFL)
            )
        ),
    )


def college_stats(team_id: str, ppg: float, sp_plus: float) -> TeamStats:
    return TeamStats(
        team_id=team_id,
        season=2026,
        games_played=10,
        stats=StatBlocks(
            football=FootballStats(
                points_per_game=ppg,
                points_allowed_per_game=24.0,
                drives_per_game=12.0,
                points_per_drive_off=2.4,
                points_per_drive_def=2.0,
                turnover_margin_per_game=0.6,
                sp_plus_rating=sp_plus,
                # epa_* left at 0.0 default (absent for college)
            )
        ),
    )


def previous_game(game_id: str, start: str) -> Game:
    return Game(
        id=game_id,
        league="NFL",
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
        raise AssertionError("player lookups are not part of the football path")

    async def get_meta_timestamp(self, path: str) -> str | None:
        return "2026-11-15T00:00:00Z"


class FakeLines:
    async def movement(self, game_external_id: str, market_type: str | None = None) -> list[LineMovement]:
        return [LineMovement(total_movement=0.5)]

    async def current_lines(self, **kwargs: Any) -> list[LineSnapshot]:
        return [
            LineSnapshot(id="s1", game_id="ext-1", sportsbook_key="dk", side="HOME", line_value=-3.5),
            LineSnapshot(id="s2", game_id="ext-1", sportsbook_key="fd", side="HOME", line_value=-3.5),
        ]


class FakeReconciler:
    def __init__(self, resolved: str | None = "ext-1") -> None:
        self._resolved = resolved

    async def resolve(self, game: Game) -> str | None:
        return self._resolved


def make_builder(stats: FakeStatistics) -> FeatureBuilder:
    return FeatureBuilder(stats, FakeLines(), FakeReconciler())  # type: ignore[arg-type]


def nfl_default_stats() -> FakeStatistics:
    return FakeStatistics(
        stats_by_team={
            "t-home": nfl_stats("t-home", ppg=27.5, epa_off=0.12),
            "t-away": nfl_stats("t-away", ppg=24.0, epa_off=0.05),
        },
        recent_by_team={
            # played 6 days ago (normal week -> not a bye)
            "t-home": [previous_game("prev-h", "2026-11-09T18:00:00Z")],
            # played 14 days ago (> 10 full days off -> coming off a bye)
            "t-away": [previous_game("prev-a", "2026-11-01T18:00:00Z")],
        },
    )


class TestFootballBuilderPath:
    async def test_football_block_team_features(self) -> None:
        bundle = await make_builder(nfl_default_stats()).build(make_game())

        assert bundle.features["home_points_per_game"] == 27.5
        assert bundle.features["away_points_per_game"] == 24.0
        assert bundle.features["home_points_allowed_per_game"] == 20.5
        assert bundle.features["home_points_per_drive_off"] == 2.1
        assert bundle.features["home_turnover_margin_per_game"] == pytest.approx(0.4)
        assert bundle.lines_game_external_id == "ext-1"

    async def test_epa_populated_for_nfl_and_sp_plus_null(self) -> None:
        bundle = await make_builder(nfl_default_stats()).build(make_game(league="NFL"))
        assert bundle.features["home_epa_per_play_off"] == pytest.approx(0.12)
        assert bundle.features["home_epa_per_play_def"] == pytest.approx(-0.02)
        # SP+ has no NFL source: gated to None, not the 0.0 placeholder
        assert bundle.features["home_sp_plus_rating"] is None
        assert bundle.features["away_sp_plus_rating"] is None

    async def test_sp_plus_populated_for_college_and_epa_null(self) -> None:
        stats = FakeStatistics(
            stats_by_team={
                "t-home": college_stats("t-home", ppg=34.0, sp_plus=18.5),
                "t-away": college_stats("t-away", ppg=28.0, sp_plus=6.0),
            },
            recent_by_team={},
        )
        bundle = await make_builder(stats).build(make_game(league="NCAA_FB"))
        assert bundle.features["home_sp_plus_rating"] == 18.5
        assert bundle.features["away_sp_plus_rating"] == 6.0
        # EPA has no college source: gated to None, not the 0.0 placeholder
        assert bundle.features["home_epa_per_play_off"] is None
        assert bundle.features["away_epa_per_play_def"] is None

    async def test_bye_week_flags_from_rest_days(self) -> None:
        bundle = await make_builder(nfl_default_stats()).build(make_game())
        # home played 6 days ago -> 5 rest days -> not a bye
        assert bundle.features["home_rest_days"] == 5.0
        assert bundle.features["home_bye_week"] == 0.0
        # away played 14 days ago -> 13 rest days (> 10) -> coming off a bye
        assert bundle.features["away_rest_days"] == 13.0
        assert bundle.features["away_bye_week"] == 1.0
        # football is weekly: no back-to-back flags
        assert not any("back_to_back" in name for name in bundle.features)

    async def test_league_one_hot(self) -> None:
        builder = make_builder(nfl_default_stats())
        nfl = await builder.build(make_game(league="NFL"))
        assert nfl.features["league_is_nfl"] == 1.0
        college = FakeStatistics(
            stats_by_team={
                "t-home": college_stats("t-home", ppg=34.0, sp_plus=18.5),
                "t-away": college_stats("t-away", ppg=28.0, sp_plus=6.0),
            },
            recent_by_team={},
        )
        ncaa = await make_builder(college).build(make_game(league="NCAA_FB"))
        assert ncaa.features["league_is_nfl"] == 0.0

    async def test_no_injury_lookups_for_football(self) -> None:
        stats = nfl_default_stats()
        bundle = await make_builder(stats).build(make_game())
        assert stats.injury_calls == []
        assert not any("injury" in name for name in bundle.features)

    async def test_missing_football_block_yields_none(self) -> None:
        stats = FakeStatistics(
            stats_by_team={
                "t-home": TeamStats(team_id="t-home", games_played=0),
                "t-away": TeamStats(team_id="t-away", games_played=0),
            },
            recent_by_team={},
        )
        bundle = await make_builder(stats).build(make_game())
        assert bundle.features["home_points_per_game"] is None
        assert bundle.features["home_epa_per_play_off"] is None
        assert bundle.features["home_rest_days"] is None
        assert bundle.features["home_bye_week"] is None
        # game-level and market features still populate
        assert bundle.features["league_is_nfl"] == 1.0
        assert bundle.features["n_books_reporting"] == 2.0

    async def test_all_non_row_registry_features_are_emitted(self) -> None:
        bundle = await make_builder(nfl_default_stats()).build(make_game())
        row_level = {
            "sim_probability",
            "sim_margin_mean",
            "sim_total_mean",
            "sim_draw_probability",
            "sim_converged",
            "market_is_spread",
            "market_is_total",
            "market_is_moneyline",
        }
        assert set(FOOTBALL_FEATURES) - row_level == set(bundle.features)


class TestOtherSportParityRegressions:
    async def test_nba_path_unchanged(self) -> None:
        stats = FakeStatistics(
            stats_by_team={
                "t-home": TeamStats(team_id="t-home", games_played=60),
                "t-away": TeamStats(team_id="t-away", games_played=60),
            },
            recent_by_team={"t-home": [previous_game("prev-h", "2026-11-14T18:00:00Z")]},
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

        for name in ("home_offensive_rating", "net_rating_diff", "home_injury_impact", "home_back_to_back"):
            assert name in bundle.features
        assert stats.injury_calls == ["t-home", "t-away"]
        # no football features leak into the basketball path
        assert not any("epa" in name for name in bundle.features)
        assert "league_is_nfl" not in bundle.features
        assert "home_bye_week" not in bundle.features

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
        assert bundle.features["competition_is_fifa_wc"] == 0.0
        assert not any("epa" in name for name in bundle.features)
        assert "league_is_nfl" not in bundle.features


@pytest.mark.parametrize("league", ["NFL", "NCAA_FB"])
async def test_both_football_leagues_take_the_football_path(league: str) -> None:
    stats = FakeStatistics(
        stats_by_team={
            "t-home": nfl_stats("t-home", ppg=27.5, epa_off=0.12)
            if league == "NFL"
            else college_stats("t-home", 34.0, 18.5),
            "t-away": nfl_stats("t-away", ppg=24.0, epa_off=0.05)
            if league == "NFL"
            else college_stats("t-away", 28.0, 6.0),
        },
        recent_by_team={},
    )
    bundle = await make_builder(stats).build(make_game(league=league))
    assert "home_points_per_game" in bundle.features
    assert "home_bye_week" in bundle.features
    assert "home_offensive_rating" not in bundle.features
    assert "home_attack_strength" not in bundle.features
