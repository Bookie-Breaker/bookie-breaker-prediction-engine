"""Training assembly tests with stubbed repository and statistics client (Wave 4)."""

import uuid
from datetime import UTC, datetime
from typing import Any

from prediction_engine.api.errors import NotFoundError
from prediction_engine.clients.statistics import Game, GameResult, TeamRef
from prediction_engine.core.training.assemble import assemble_training_rows, leagues_for_model_key
from prediction_engine.db.repository import PredictionRecord

MODEL_VERSION = uuid.uuid4()


def make_record(
    game_id: str,
    market_type: str,
    selection: str,
    side: str | None,
    sim_probability: float | None = 0.6,
) -> PredictionRecord:
    return PredictionRecord(
        id=uuid.uuid4(),
        game_external_id=game_id,
        model_version_id=MODEL_VERSION,
        league="NBA",
        market_type=market_type,
        side=side,
        selection=selection,
        predicted_probability=0.62,
        simulation_probability=sim_probability,
        implied_probability=None,
        edge=None,
        confidence_lower=0.55,
        confidence_upper=0.70,
        feature_importance={},
        created_at=datetime.now(tz=UTC),
    )


def make_game(game_id: str, status: str = "FINAL", season: int = 2026, scheduled_start: str = "") -> Game:
    return Game(
        id=game_id,
        league="NBA",
        status=status,
        home_team=TeamRef(id="t-home", name="Los Angeles Lakers"),
        away_team=TeamRef(id="t-away", name="Boston Celtics"),
        scheduled_start=scheduled_start,
        season=season,
        result=GameResult(home_score=110, away_score=104) if status == "FINAL" else None,
    )


class FakeRepo:
    def __init__(self, rows: list[tuple[PredictionRecord, dict[str, Any]]]) -> None:
        self.rows = rows
        self.calls: list[tuple[list[str], list[str]]] = []

    async def training_rows(
        self, leagues: list[str], market_types: list[str]
    ) -> list[tuple[PredictionRecord, dict[str, Any]]]:
        self.calls.append((leagues, market_types))
        return self.rows


class FakeStatistics:
    def __init__(self, games: dict[str, Game]) -> None:
        self._games = games

    async def get_game(self, game_id: str) -> Game:
        if game_id not in self._games:
            raise NotFoundError(f"game {game_id} not found")
        return self._games[game_id]


FEATURES = {"net_rating_diff": 4.0, "home_rest_days": None}
MARKETS = ["SPREAD", "TOTAL", "MONEYLINE"]


class TestLeaguesForModelKey:
    def test_pooled_and_single_league_keys(self) -> None:
        assert leagues_for_model_key("BASKETBALL") == ["NBA"]
        assert leagues_for_model_key("NCAA_BB") == ["NCAA_BB"]
        assert set(leagues_for_model_key("SOCCER")) == {"FIFA_WC", "EPL"}
        assert set(leagues_for_model_key("BASEBALL")) == {"MLB", "NCAA_BSB"}


class TestAssembleTrainingRows:
    async def test_settles_rows_and_derives_season(self) -> None:
        # game landed 110-104 (margin 6, total 214)
        rows = [
            (make_record("g1", "SPREAD", "Los Angeles Lakers -3.5", "HOME"), FEATURES),  # win
            (make_record("g1", "TOTAL", "Over 220.5", "OVER"), FEATURES),  # loss
            (make_record("g1", "MONEYLINE", "Los Angeles Lakers ML", "HOME"), FEATURES),  # win
        ]
        repo = FakeRepo(rows)
        assembled = await assemble_training_rows("BASKETBALL", MARKETS, repo, FakeStatistics({"g1": make_game("g1")}))
        assert repo.calls == [(["NBA"], MARKETS)]
        assert assembled.rows == 3
        assert assembled.dataset.outcomes.tolist() == [1.0, 0.0, 1.0]
        assert assembled.dataset.seasons.tolist() == [2026, 2026, 2026]
        assert assembled.dataset.sim_probs.tolist() == [0.6, 0.6, 0.6]
        # stored feature vector values (incl. explicit nulls) survive the join
        assert assembled.dataset.features[0] == {"net_rating_diff": 4.0, "home_rest_days": None}

    async def test_season_falls_back_to_scheduled_start_year(self) -> None:
        game = make_game("g1", season=0, scheduled_start="2025-11-02T00:00:00Z")
        rows = [(make_record("g1", "MONEYLINE", "Los Angeles Lakers ML", "HOME"), FEATURES)]
        assembled = await assemble_training_rows("BASKETBALL", MARKETS, FakeRepo(rows), FakeStatistics({"g1": game}))
        assert assembled.dataset.seasons.tolist() == [2025]

    async def test_push_rows_are_dropped(self) -> None:
        rows = [
            (make_record("g1", "SPREAD", "Los Angeles Lakers -6.0", "HOME"), FEATURES),  # margin 6 -> push
            (make_record("g1", "TOTAL", "Over 214.0", "OVER"), FEATURES),  # total 214 -> push
            (make_record("g1", "MONEYLINE", "Los Angeles Lakers ML", "HOME"), FEATURES),
        ]
        assembled = await assemble_training_rows(
            "BASKETBALL", MARKETS, FakeRepo(rows), FakeStatistics({"g1": make_game("g1")})
        )
        assert assembled.rows == 1
        assert assembled.rows_pushed == 2

    async def test_unfinished_and_missing_games_are_skipped(self) -> None:
        rows = [
            (make_record("g-final", "MONEYLINE", "Los Angeles Lakers ML", "HOME"), FEATURES),
            (make_record("g-live", "MONEYLINE", "Los Angeles Lakers ML", "HOME"), FEATURES),
            (make_record("g-unknown", "MONEYLINE", "Los Angeles Lakers ML", "HOME"), FEATURES),
        ]
        stats = FakeStatistics({"g-final": make_game("g-final"), "g-live": make_game("g-live", status="LIVE")})
        assembled = await assemble_training_rows("BASKETBALL", MARKETS, FakeRepo(rows), stats)
        assert assembled.rows == 1
        assert assembled.games_graded == 1
        assert assembled.games_pending == 2
        assert assembled.rows_skipped == 2

    async def test_rows_without_sim_probability_are_skipped(self) -> None:
        rows = [
            (make_record("g1", "MONEYLINE", "Los Angeles Lakers ML", "HOME", sim_probability=None), FEATURES),
            (make_record("g1", "MONEYLINE", "Los Angeles Lakers ML", "HOME"), FEATURES),
        ]
        assembled = await assemble_training_rows(
            "BASKETBALL", MARKETS, FakeRepo(rows), FakeStatistics({"g1": make_game("g1")})
        )
        assert assembled.rows == 1
        assert assembled.rows_skipped == 1

    async def test_legacy_null_sides_use_default_perspective(self) -> None:
        rows = [
            (make_record("g1", "SPREAD", "Los Angeles Lakers -3.5", None), FEATURES),  # HOME default
            (make_record("g1", "TOTAL", "Over 210.5", None), FEATURES),  # OVER default
        ]
        assembled = await assemble_training_rows(
            "BASKETBALL", MARKETS, FakeRepo(rows), FakeStatistics({"g1": make_game("g1")})
        )
        assert assembled.dataset.outcomes.tolist() == [1.0, 1.0]

    async def test_empty_repo_yields_empty_dataset(self) -> None:
        assembled = await assemble_training_rows("BASKETBALL", MARKETS, FakeRepo([]), FakeStatistics({}))
        assert assembled.rows == 0
        assert len(assembled.dataset) == 0
