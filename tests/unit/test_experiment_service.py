"""ExperimentService grading and promotion branches (services/experiments.py)."""

import uuid
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from prediction_engine.api.errors import NotFoundError, UnprocessableError
from prediction_engine.clients.statistics import Game, GameResult, TeamRef
from prediction_engine.core.model.registry import MARKET_TYPES
from prediction_engine.db.repository import ModelVersionRecord, PredictionRecord
from prediction_engine.services.experiments import ExperimentService


def make_model(role: str) -> ModelVersionRecord:
    return ModelVersionRecord(
        id=uuid.uuid4(),
        sport="BASKETBALL",
        model_type="MONEYLINE",
        version=f"v_{role}",
        algorithm="xgboost",
        trained_at=datetime.now(tz=UTC),
        training_samples=100,
        evaluation_metrics={},
        feature_names=["sim_probability"],
        is_active=True,
        artifact_path="/tmp/model",
        notes=None,
        role=role,
    )


def make_row(
    model_id: uuid.UUID,
    game_id: str,
    probability: float,
    market_type: str = "MONEYLINE",
    side: str | None = "HOME",
    selection: str = "Los Angeles Lakers ML",
) -> PredictionRecord:
    return PredictionRecord(
        id=uuid.uuid4(),
        game_external_id=game_id,
        model_version_id=model_id,
        league="NBA",
        market_type=market_type,
        side=side,
        selection=selection,
        predicted_probability=probability,
        simulation_probability=None,
        implied_probability=None,
        edge=None,
        confidence_lower=None,
        confidence_upper=None,
        feature_importance={},
        created_at=datetime.now(tz=UTC),
    )


def make_game(game_id: str, home: int | None = None, away: int | None = None, status: str = "FINAL") -> Game:
    return Game(
        id=game_id,
        league="NBA",
        status=status,
        home_team=TeamRef(id="t-home", name="Los Angeles Lakers"),
        away_team=TeamRef(id="t-away", name="Boston Celtics"),
        result=GameResult(home_score=home, away_score=away) if home is not None and away is not None else None,
    )


class FakeModelRepo:
    def __init__(self, champion: ModelVersionRecord | None, challenger: ModelVersionRecord | None) -> None:
        self.champion = champion
        self.challenger = challenger
        self.promote_calls: list[tuple[str, list[str]]] = []

    async def get_active(self, sport: str, market_type: str, role: str = "champion") -> ModelVersionRecord | None:
        return self.champion if role == "champion" else self.challenger

    async def get(self, model_id: uuid.UUID) -> ModelVersionRecord | None:
        for record in (self.champion, self.challenger):
            if record is not None and record.id == model_id:
                return record
        return None

    async def promote(self, sport: str, market_types: list[str]) -> list[ModelVersionRecord]:
        self.promote_calls.append((sport, market_types))
        assert self.challenger is not None
        return [replace(self.challenger, model_type=market, role="champion") for market in market_types]


class FakePredictionRepo:
    def __init__(self, rows_by_model: dict[uuid.UUID, list[PredictionRecord]]) -> None:
        self._rows = rows_by_model

    async def latest_rows_for_model(self, model_version_id: uuid.UUID) -> list[PredictionRecord]:
        return self._rows.get(model_version_id, [])


class FakeStatistics:
    def __init__(self, games: dict[str, Game]) -> None:
        self._games = games

    async def get_game(self, game_id: str) -> Game:
        try:
            return self._games[game_id]
        except KeyError:
            raise RuntimeError(f"statistics-service 500 for {game_id}") from None


class FakeRegistry:
    def __init__(self) -> None:
        self.reloaded: list[str] = []

    async def reload_sport(self, sport: str) -> None:
        self.reloaded.append(sport)


def make_service(
    champion: ModelVersionRecord | None,
    challenger: ModelVersionRecord | None,
    rows_by_model: dict[uuid.UUID, list[PredictionRecord]] | None = None,
    games: dict[str, Game] | None = None,
    min_samples: int = 300,
) -> tuple[ExperimentService, FakeModelRepo, FakeRegistry]:
    model_repo = FakeModelRepo(champion, challenger)
    registry = FakeRegistry()
    service = ExperimentService(
        model_repo=model_repo,  # type: ignore[arg-type]
        prediction_repo=FakePredictionRepo(rows_by_model or {}),  # type: ignore[arg-type]
        statistics=FakeStatistics(games or {}),  # type: ignore[arg-type]
        registry=registry,  # type: ignore[arg-type]
        promotion_min_samples=min_samples,
    )
    return service, model_repo, registry


class TestReport:
    async def test_missing_champion_is_a_404(self) -> None:
        service, _, _ = make_service(champion=None, challenger=make_model("challenger"))
        with pytest.raises(NotFoundError, match="No active champion"):
            await service.report("BASKETBALL", "MONEYLINE")

    async def test_missing_challenger_is_a_404(self) -> None:
        service, _, _ = make_service(champion=make_model("champion"), challenger=None)
        with pytest.raises(NotFoundError, match="No active challenger"):
            await service.report("BASKETBALL", "MONEYLINE")

    async def test_grades_shared_pairs_against_final_scores(self) -> None:
        champion, challenger = make_model("champion"), make_model("challenger")
        # game-1: home wins (outcome 1.0); game-2: home loses (outcome 0.0)
        rows = {
            champion.id: [make_row(champion.id, "game-1", 0.6), make_row(champion.id, "game-2", 0.6)],
            challenger.id: [make_row(challenger.id, "game-1", 0.9), make_row(challenger.id, "game-2", 0.2)],
        }
        games = {"game-1": make_game("game-1", 110, 100), "game-2": make_game("game-2", 95, 100)}
        service, _, _ = make_service(champion, challenger, rows, games, min_samples=3)

        report = await service.report("BASKETBALL", "MONEYLINE")

        assert report.graded_pairs == 2
        # brier: champion ((0.6-1)^2 + (0.6-0)^2)/2, challenger ((0.9-1)^2 + (0.2-0)^2)/2
        assert report.champion.brier_score == pytest.approx(0.26)
        assert report.challenger.brier_score == pytest.approx(0.025)
        assert report.promotion_ready is False
        assert any("only 2 graded pairs" in blocker for blocker in report.blockers)

    async def test_rows_only_one_model_scored_are_not_pairs(self) -> None:
        champion, challenger = make_model("champion"), make_model("challenger")
        rows = {
            champion.id: [make_row(champion.id, "game-1", 0.6)],
            challenger.id: [make_row(challenger.id, "game-2", 0.9)],
        }
        games = {"game-1": make_game("game-1", 110, 100), "game-2": make_game("game-2", 95, 100)}
        service, _, _ = make_service(champion, challenger, rows, games)

        report = await service.report("BASKETBALL", "MONEYLINE")
        assert report.graded_pairs == 0
        assert report.champion.brier_score is None
        assert report.challenger.brier_score is None

    async def test_unfetchable_games_stay_ungraded(self) -> None:
        champion, challenger = make_model("champion"), make_model("challenger")
        rows = {
            champion.id: [make_row(champion.id, "game-gone", 0.6)],
            challenger.id: [make_row(challenger.id, "game-gone", 0.9)],
        }
        service, _, _ = make_service(champion, challenger, rows, games={})  # get_game raises

        report = await service.report("BASKETBALL", "MONEYLINE")
        assert report.graded_pairs == 0

    async def test_non_final_games_stay_ungraded(self) -> None:
        champion, challenger = make_model("champion"), make_model("challenger")
        rows = {
            champion.id: [make_row(champion.id, "game-live", 0.6)],
            challenger.id: [make_row(challenger.id, "game-live", 0.9)],
        }
        games = {"game-live": make_game("game-live", status="SCHEDULED")}
        service, _, _ = make_service(champion, challenger, rows, games)

        report = await service.report("BASKETBALL", "MONEYLINE")
        assert report.graded_pairs == 0

    async def test_pushes_drop_the_pair(self) -> None:
        champion, challenger = make_model("champion"), make_model("challenger")
        spread = {"market_type": "SPREAD", "selection": "Los Angeles Lakers -5.0"}
        rows = {
            champion.id: [make_row(champion.id, "game-push", 0.55, **spread)],
            challenger.id: [make_row(challenger.id, "game-push", 0.60, **spread)],
        }
        # home wins by exactly 5: the -5.0 spread pushes, no binary signal
        games = {"game-push": make_game("game-push", 105, 100)}
        service, _, _ = make_service(champion, challenger, rows, games)

        report = await service.report("BASKETBALL", "MONEYLINE")
        assert report.graded_pairs == 0

    async def test_unsettleable_rows_are_skipped_not_fatal(self) -> None:
        champion, challenger = make_model("champion"), make_model("challenger")
        # a DRAW side on a two-way NBA moneyline cannot settle (ValueError)
        rows = {
            champion.id: [
                make_row(champion.id, "game-1", 0.1, side="DRAW", selection="Draw"),
                make_row(champion.id, "game-1", 0.6),
            ],
            challenger.id: [
                make_row(challenger.id, "game-1", 0.2, side="DRAW", selection="Draw"),
                make_row(challenger.id, "game-1", 0.9),
            ],
        }
        games = {"game-1": make_game("game-1", 110, 100)}
        service, _, _ = make_service(champion, challenger, rows, games)

        report = await service.report("BASKETBALL", "MONEYLINE")
        assert report.graded_pairs == 1  # only the HOME pair graded


class TestPromote:
    async def test_blocked_promotion_is_422_with_the_blockers(self) -> None:
        champion, challenger = make_model("champion"), make_model("challenger")
        service, model_repo, registry = make_service(champion, challenger)  # zero graded pairs

        with pytest.raises(UnprocessableError) as exc_info:
            await service.promote("BASKETBALL", "MONEYLINE", force=False)
        assert exc_info.value.details["blockers"]
        assert model_repo.promote_calls == []
        assert registry.reloaded == []

    async def test_force_bypasses_the_criteria_and_reloads_the_registry(self) -> None:
        champion, challenger = make_model("champion"), make_model("challenger")
        service, model_repo, registry = make_service(champion, challenger)

        records = await service.promote("BASKETBALL", "MONEYLINE", force=True)

        assert model_repo.promote_calls == [("BASKETBALL", MARKET_TYPES)]
        assert registry.reloaded == ["BASKETBALL"]
        assert all(record.role == "champion" for record in records)

    async def test_a_promotion_ready_challenger_promotes_without_force(self) -> None:
        champion, challenger = make_model("champion"), make_model("challenger")
        # one graded pair where the challenger beats the champion on every criterion
        rows = {
            champion.id: [make_row(champion.id, "game-1", 0.6)],
            challenger.id: [make_row(challenger.id, "game-1", 0.9)],
        }
        games = {"game-1": make_game("game-1", 110, 100)}
        service, model_repo, registry = make_service(champion, challenger, rows, games, min_samples=1)

        records = await service.promote("BASKETBALL", "MONEYLINE", force=False)
        assert model_repo.promote_calls == [("BASKETBALL", MARKET_TYPES)]
        assert registry.reloaded == ["BASKETBALL"]
        assert len(records) == len(MARKET_TYPES)
