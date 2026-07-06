"""FeatureBuilder soccer path tests with stubbed clients (Phase 6 Wave 1)."""

from typing import Any

import pytest

from prediction_engine.clients.lines import LineMovement, LineSnapshot
from prediction_engine.clients.statistics import Game, SoccerStats, StatBlocks, TeamRef, TeamStats
from prediction_engine.core.features.builder import FeatureBuilder
from prediction_engine.core.features.registry import SOCCER_FEATURES

GAME_START = "2026-07-10T20:00:00Z"


def make_game(league: str = "FIFA_WC", season_type: str = "POSTSEASON") -> Game:
    return Game(
        id="game-soccer-1",
        league=league,
        status="SCHEDULED",
        home_team=TeamRef(id="t-home", name="Argentina"),
        away_team=TeamRef(id="t-away", name="France"),
        scheduled_start=GAME_START,
        season=2026,
        season_type=season_type,
    )


def soccer_stats(team_id: str, attack: float, defense: float, form: int, games_played: int = 5) -> TeamStats:
    return TeamStats(
        team_id=team_id,
        season=2026,
        games_played=games_played,
        stats=StatBlocks(
            soccer=SoccerStats(
                goals_for_per_match=2.2,
                goals_against_per_match=0.6,
                attack_strength=attack,
                defense_strength=defense,
                draws=1,
                form_points_last5=form,
            )
        ),
    )


def previous_game(game_id: str, start: str) -> Game:
    return Game(
        id=game_id,
        league="FIFA_WC",
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
        raise AssertionError("player lookups are not part of the soccer path")

    async def get_meta_timestamp(self, path: str) -> str | None:
        return "2026-07-10T00:00:00Z"


class FakeLines:
    async def movement(self, game_external_id: str, market_type: str | None = None) -> list[LineMovement]:
        return [LineMovement(total_movement=0.5)]

    async def current_lines(self, **kwargs: Any) -> list[LineSnapshot]:
        return [
            LineSnapshot(id="s1", game_id="ext-1", sportsbook_key="dk", side="HOME", line_value=-0.5),
            LineSnapshot(id="s2", game_id="ext-1", sportsbook_key="fd", side="HOME", line_value=-0.5),
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
            "t-home": soccer_stats("t-home", attack=1.35, defense=0.7, form=13),
            "t-away": soccer_stats("t-away", attack=1.2, defense=0.8, form=10, games_played=6),
        },
        recent_by_team={
            "t-home": [previous_game("prev-h", "2026-07-05T20:00:00Z")],  # 4 full rest days
            "t-away": [previous_game("prev-a", "2026-07-07T20:00:00Z")],  # 2 full rest days
        },
    )


class TestSoccerBuilderPath:
    async def test_soccer_block_strengths_form_and_sample_size(self) -> None:
        bundle = await make_builder(default_stats()).build(make_game())

        assert bundle.features["home_attack_strength"] == 1.35
        assert bundle.features["home_defense_strength"] == 0.7
        assert bundle.features["away_attack_strength"] == 1.2
        assert bundle.features["away_defense_strength"] == 0.8
        assert bundle.features["home_goals_for_per_match"] == 2.2
        assert bundle.features["away_goals_against_per_match"] == 0.6
        assert bundle.features["home_form_points_last5"] == 13.0
        assert bundle.features["away_form_points_last5"] == 10.0
        assert bundle.features["home_matches_played"] == 5.0
        assert bundle.features["away_matches_played"] == 6.0
        assert bundle.lines_game_external_id == "ext-1"

    async def test_rest_days_from_schedule(self) -> None:
        bundle = await make_builder(default_stats()).build(make_game())
        assert bundle.features["home_rest_days"] == 4.0
        assert bundle.features["away_rest_days"] == 2.0

    async def test_is_knockout_from_season_type(self) -> None:
        builder = make_builder(default_stats())
        knockout = await builder.build(make_game(season_type="POSTSEASON"))
        group_stage = await builder.build(make_game(season_type="REGULAR"))
        assert knockout.features["is_knockout"] == 1.0
        assert group_stage.features["is_knockout"] == 0.0

    async def test_competition_one_hot(self) -> None:
        builder = make_builder(default_stats())
        fifa = await builder.build(make_game(league="FIFA_WC"))
        epl = await builder.build(make_game(league="EPL", season_type="REGULAR"))
        assert fifa.features["competition_is_fifa_wc"] == 1.0
        assert epl.features["competition_is_fifa_wc"] == 0.0

    async def test_no_injury_lookups_for_soccer(self) -> None:
        stats = default_stats()
        bundle = await make_builder(stats).build(make_game())
        assert stats.injury_calls == []
        assert not any("injury" in name for name in bundle.features)
        assert not any("back_to_back" in name for name in bundle.features)

    async def test_missing_soccer_block_yields_none(self) -> None:
        stats = FakeStatistics(
            stats_by_team={
                "t-home": TeamStats(team_id="t-home", games_played=0),
                "t-away": TeamStats(team_id="t-away", games_played=0),
            },
            recent_by_team={},
        )
        bundle = await make_builder(stats).build(make_game())
        assert bundle.features["home_attack_strength"] is None
        assert bundle.features["home_goals_for_per_match"] is None
        assert bundle.features["home_matches_played"] is None
        assert bundle.features["home_rest_days"] is None
        # game-level and market features still populate
        assert bundle.features["is_knockout"] == 1.0
        assert bundle.features["n_books_reporting"] == 2.0

    async def test_all_non_row_registry_features_are_emitted(self) -> None:
        bundle = await make_builder(default_stats()).build(make_game())
        # per-row features are stamped by the predictor, not the builder
        row_level = {
            "sim_probability",
            "sim_margin_mean",
            "sim_total_mean",
            "sim_draw_probability",
            "sim_converged",
            "market_is_spread",
            "market_is_total",
            "market_is_moneyline",
            "selection_is_draw",
        }
        assert set(SOCCER_FEATURES) - row_level == set(bundle.features)


class TestBasketballParityRegression:
    async def test_nba_path_unchanged(self) -> None:
        nba_stats = TeamStats(team_id="t-home", games_played=60)
        stats = FakeStatistics(
            stats_by_team={"t-home": nba_stats, "t-away": TeamStats(team_id="t-away", games_played=60)},
            recent_by_team={"t-home": [previous_game("prev-h", "2026-07-08T20:00:00Z")]},
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

        # NBA feature surface is intact, including derived diffs and injuries
        for name in ("home_offensive_rating", "net_rating_diff", "home_injury_impact", "home_back_to_back"):
            assert name in bundle.features
        assert bundle.features["home_rest_days"] == 1.0
        assert stats.injury_calls == ["t-home", "t-away"]
        # and no soccer features leak into the basketball path
        assert not any(name in bundle.features for name in ("is_knockout", "competition_is_fifa_wc"))
        assert not any("attack_strength" in name for name in bundle.features)

    async def test_unknown_league_defaults_to_basketball_path(self) -> None:
        stats = FakeStatistics(
            stats_by_team={
                "t-home": TeamStats(team_id="t-home", games_played=1),
                "t-away": TeamStats(team_id="t-away", games_played=1),
            },
            recent_by_team={},
        )
        game = make_game(league="MYSTERY_LEAGUE")
        bundle = await make_builder(stats).build(game)
        assert "home_offensive_rating" in bundle.features


@pytest.mark.parametrize("league", ["FIFA_WC", "EPL"])
async def test_both_soccer_leagues_take_the_soccer_path(league: str) -> None:
    bundle = await make_builder(default_stats()).build(make_game(league=league, season_type="REGULAR"))
    assert "home_attack_strength" in bundle.features
    assert "home_offensive_rating" not in bundle.features
