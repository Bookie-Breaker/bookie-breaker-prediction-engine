"""FeatureBuilder baseball path tests with stubbed clients (Phase 6 Wave 2)."""

from typing import Any

import pytest

from prediction_engine.clients.lines import LineMovement, LineSnapshot
from prediction_engine.clients.statistics import (
    BaseballStats,
    Game,
    ProbablePitcher,
    SoccerStats,
    StatBlocks,
    TeamRef,
    TeamStats,
)
from prediction_engine.core.features.builder import FeatureBuilder
from prediction_engine.core.features.registry import BASEBALL_FEATURES

GAME_START = "2026-07-10T19:10:00Z"


def home_pitcher() -> ProbablePitcher:
    return ProbablePitcher(name="Ace Home", external_id="p-home", throws="R", era=2.95, fip=3.10, k_bb_pct=0.21)


def away_pitcher() -> ProbablePitcher:
    return ProbablePitcher(name="Lefty Away", external_id="p-away", throws="L", era=4.40, fip=4.25, k_bb_pct=0.11)


def make_game(
    league: str = "MLB",
    home_probable: ProbablePitcher | None = None,
    away_probable: ProbablePitcher | None = None,
) -> Game:
    return Game(
        id="game-mlb-1",
        league=league,
        status="SCHEDULED",
        home_team=TeamRef(id="t-home", name="New York Yankees"),
        away_team=TeamRef(id="t-away", name="Boston Red Sox"),
        scheduled_start=GAME_START,
        season=2026,
        season_type="REGULAR",
        home_probable_pitcher=home_probable,
        away_probable_pitcher=away_probable,
    )


def baseball_stats(team_id: str, runs_scored: float, woba: float) -> TeamStats:
    return TeamStats(
        team_id=team_id,
        season=2026,
        games_played=88,
        stats=StatBlocks(
            baseball=BaseballStats(
                runs_scored_per_game=runs_scored,
                runs_allowed_per_game=4.1,
                team_woba=woba,
                team_obp=0.330,
                team_slg=0.415,
                batting_strikeout_pct=0.22,
                batting_walk_pct=0.085,
                team_era=3.90,
                team_fip=3.95,
                bullpen_era=3.60,
            )
        ),
    )


def previous_game(game_id: str, start: str) -> Game:
    return Game(
        id=game_id,
        league="MLB",
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
        raise AssertionError("player lookups are not part of the baseball path")

    async def get_meta_timestamp(self, path: str) -> str | None:
        return "2026-07-10T00:00:00Z"


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
            "t-home": baseball_stats("t-home", runs_scored=5.1, woba=0.334),
            "t-away": baseball_stats("t-away", runs_scored=4.4, woba=0.318),
        },
        recent_by_team={
            "t-home": [previous_game("prev-h", "2026-07-09T19:10:00Z")],  # played yesterday: 0 rest days
            "t-away": [previous_game("prev-a", "2026-07-08T19:10:00Z")],  # 1 full rest day
        },
    )


class TestBaseballBuilderPath:
    async def test_baseball_block_team_features(self) -> None:
        bundle = await make_builder(default_stats()).build(make_game(home_probable=home_pitcher()))

        assert bundle.features["home_runs_scored_per_game"] == 5.1
        assert bundle.features["away_runs_scored_per_game"] == 4.4
        assert bundle.features["home_runs_allowed_per_game"] == 4.1
        assert bundle.features["home_team_woba"] == 0.334
        assert bundle.features["away_team_woba"] == 0.318
        assert bundle.features["home_team_fip"] == 3.95
        assert bundle.features["home_bullpen_era"] == 3.6
        assert bundle.lines_game_external_id == "ext-1"

    async def test_announced_starters_populate_stats_flags_and_diff(self) -> None:
        game = make_game(home_probable=home_pitcher(), away_probable=away_pitcher())
        bundle = await make_builder(default_stats()).build(game)

        assert bundle.features["home_starter_fip"] == 3.10
        assert bundle.features["home_starter_era"] == 2.95
        assert bundle.features["home_starter_kbb"] == 0.21
        assert bundle.features["home_starter_announced"] == 1.0
        assert bundle.features["away_starter_fip"] == 4.25
        assert bundle.features["away_starter_announced"] == 1.0
        assert bundle.features["starter_fip_diff"] == pytest.approx(3.10 - 4.25)

    async def test_unannounced_starters_are_null_safe(self) -> None:
        bundle = await make_builder(default_stats()).build(make_game())

        for prefix in ("home", "away"):
            assert bundle.features[f"{prefix}_starter_fip"] is None
            assert bundle.features[f"{prefix}_starter_era"] is None
            assert bundle.features[f"{prefix}_starter_kbb"] is None
            assert bundle.features[f"{prefix}_starter_announced"] == 0.0
        assert bundle.features["starter_fip_diff"] is None

    async def test_partially_announced_starters(self) -> None:
        bundle = await make_builder(default_stats()).build(make_game(home_probable=home_pitcher()))

        assert bundle.features["home_starter_announced"] == 1.0
        assert bundle.features["home_starter_fip"] == 3.10
        assert bundle.features["away_starter_announced"] == 0.0
        assert bundle.features["away_starter_fip"] is None
        assert bundle.features["starter_fip_diff"] is None  # needs both starters

    async def test_announced_starter_without_season_stats_keeps_flag(self) -> None:
        debut = ProbablePitcher(name="Callup Kid", external_id="p-debut", throws="R")
        bundle = await make_builder(default_stats()).build(make_game(home_probable=debut))

        assert bundle.features["home_starter_announced"] == 1.0
        assert bundle.features["home_starter_fip"] is None
        assert bundle.features["starter_fip_diff"] is None

    async def test_rest_days_from_schedule(self) -> None:
        bundle = await make_builder(default_stats()).build(make_game())
        assert bundle.features["home_rest_days"] == 0.0  # played yesterday
        assert bundle.features["away_rest_days"] == 1.0
        assert not any("back_to_back" in name for name in bundle.features)

    async def test_league_one_hot(self) -> None:
        builder = make_builder(default_stats())
        mlb = await builder.build(make_game(league="MLB"))
        ncaa = await builder.build(make_game(league="NCAA_BSB"))
        assert mlb.features["league_is_mlb"] == 1.0
        assert ncaa.features["league_is_mlb"] == 0.0

    async def test_no_injury_lookups_for_baseball(self) -> None:
        # Injuries are null-documented for Wave 2: the NBA impact proxy is
        # minutes-based and does not transfer; starters carry the signal.
        stats = default_stats()
        bundle = await make_builder(stats).build(make_game())
        assert stats.injury_calls == []
        assert not any("injury" in name for name in bundle.features)

    async def test_missing_baseball_block_yields_none(self) -> None:
        stats = FakeStatistics(
            stats_by_team={
                "t-home": TeamStats(team_id="t-home", games_played=0),
                "t-away": TeamStats(team_id="t-away", games_played=0),
            },
            recent_by_team={},
        )
        bundle = await make_builder(stats).build(make_game())
        assert bundle.features["home_runs_scored_per_game"] is None
        assert bundle.features["home_team_woba"] is None
        assert bundle.features["home_rest_days"] is None
        # game-level and market features still populate
        assert bundle.features["league_is_mlb"] == 1.0
        assert bundle.features["n_books_reporting"] == 2.0

    async def test_all_non_row_registry_features_are_emitted(self) -> None:
        bundle = await make_builder(default_stats()).build(make_game(home_probable=home_pitcher()))
        # per-row features are stamped by the predictor, not the builder
        row_level = {
            "sim_probability",
            "sim_margin_mean",
            "sim_total_mean",
            "sim_converged",
            "market_is_spread",
            "market_is_total",
            "market_is_moneyline",
        }
        assert set(BASEBALL_FEATURES) - row_level == set(bundle.features)


class TestOtherSportParityRegressions:
    async def test_nba_path_unchanged(self) -> None:
        stats = FakeStatistics(
            stats_by_team={
                "t-home": TeamStats(team_id="t-home", games_played=60),
                "t-away": TeamStats(team_id="t-away", games_played=60),
            },
            recent_by_team={"t-home": [previous_game("prev-h", "2026-07-08T19:10:00Z")]},
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
        # no baseball features leak into the basketball path
        assert not any("starter" in name for name in bundle.features)
        assert "league_is_mlb" not in bundle.features

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
        assert stats.injury_calls == []
        # no baseball features leak into the soccer path
        assert not any("starter" in name for name in bundle.features)
        assert "league_is_mlb" not in bundle.features


@pytest.mark.parametrize("league", ["MLB", "NCAA_BSB"])
async def test_both_baseball_leagues_take_the_baseball_path(league: str) -> None:
    bundle = await make_builder(default_stats()).build(make_game(league=league))
    assert "home_team_woba" in bundle.features
    assert "home_offensive_rating" not in bundle.features
    assert "home_attack_strength" not in bundle.features
