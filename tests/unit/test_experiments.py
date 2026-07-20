"""Experiment report math, promotion criteria boundaries, and forced promotion (Wave 4)."""

import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest

from prediction_engine.api.errors import NotFoundError, UnprocessableError
from prediction_engine.clients.statistics import Game, GameResult, TeamRef
from prediction_engine.core.experiments import ECE_TOLERANCE, build_report
from prediction_engine.core.training.evaluate import brier_score, expected_calibration_error, log_loss
from prediction_engine.db.repository import ModelVersionRecord, PredictionRecord
from prediction_engine.services.experiments import ExperimentService

CHAMPION_ID = str(uuid.uuid4())
CHALLENGER_ID = str(uuid.uuid4())


def constructed_pairs(n: int = 400, seed: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Challenger closer to the truth than the champion on every pair."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.3, 0.7, size=n)
    outcomes = (rng.uniform(size=n) < truth).astype(np.float64)
    challenger = np.clip(truth + rng.normal(0, 0.02, size=n), 0.01, 0.99)
    champion = np.clip(truth + rng.normal(0, 0.15, size=n), 0.01, 0.99)
    return champion, challenger, outcomes


class TestBuildReport:
    def test_metrics_match_the_evaluate_module(self) -> None:
        champion, challenger, outcomes = constructed_pairs()
        report = build_report("BASKETBALL", "SPREAD", CHAMPION_ID, CHALLENGER_ID, champion, challenger, outcomes, 300)
        assert report.graded_pairs == 400
        assert report.champion.brier_score == pytest.approx(round(brier_score(champion, outcomes), 5))
        assert report.champion.log_loss == pytest.approx(round(log_loss(champion, outcomes), 5))
        assert report.challenger.calibration_error == pytest.approx(
            round(expected_calibration_error(challenger, outcomes), 5)
        )
        assert report.promotion_ready
        assert report.blockers == []

    def test_sample_count_boundary(self) -> None:
        champion, challenger, outcomes = constructed_pairs(n=300)
        ready = build_report("BASKETBALL", "SPREAD", CHAMPION_ID, CHALLENGER_ID, champion, challenger, outcomes, 300)
        assert ready.promotion_ready

        short = build_report(
            "BASKETBALL", "SPREAD", CHAMPION_ID, CHALLENGER_ID, champion[:299], challenger[:299], outcomes[:299], 300
        )
        assert not short.promotion_ready
        assert any("299 graded pairs" in blocker for blocker in short.blockers)

    def test_equal_brier_blocks_promotion(self) -> None:
        _, challenger, outcomes = constructed_pairs()
        report = build_report(
            "BASKETBALL", "SPREAD", CHAMPION_ID, CHALLENGER_ID, challenger, challenger.copy(), outcomes, 300
        )
        assert not report.promotion_ready
        assert any("Brier" in blocker for blocker in report.blockers)
        assert any("log loss" in blocker for blocker in report.blockers)

    def test_ece_worse_than_tolerance_blocks_despite_better_accuracy(self) -> None:
        # champion: constant 0.5 on a balanced set -> Brier 0.25, ECE 0.0
        # challenger: sharp but miscalibrated (0.9 / 0.1) -> Brier 0.01, ECE 0.1
        outcomes = np.array([1.0, 0.0] * 200)
        champion = np.full(400, 0.5)
        challenger = np.where(outcomes == 1.0, 0.9, 0.1)
        report = build_report("BASKETBALL", "SPREAD", CHAMPION_ID, CHALLENGER_ID, champion, challenger, outcomes, 300)
        assert report.champion.calibration_error == pytest.approx(0.0, abs=1e-9)
        assert report.challenger.calibration_error == pytest.approx(0.1, abs=1e-6)
        assert not report.promotion_ready
        assert len(report.blockers) == 1
        assert "ECE" in report.blockers[0]

    def test_ece_exactly_at_tolerance_passes(self) -> None:
        # challenger ECE lands exactly at champion + ECE_TOLERANCE: not a blocker
        outcomes = np.array([1.0, 0.0] * 200)
        champion = np.full(400, 0.5)
        challenger = np.where(outcomes == 1.0, 1.0 - ECE_TOLERANCE, ECE_TOLERANCE)
        report = build_report("BASKETBALL", "SPREAD", CHAMPION_ID, CHALLENGER_ID, champion, challenger, outcomes, 300)
        assert report.challenger.calibration_error == pytest.approx(ECE_TOLERANCE, abs=1e-9)
        assert report.promotion_ready

    def test_zero_pairs_yields_null_metrics_and_blocks(self) -> None:
        empty = np.array([], dtype=np.float64)
        report = build_report("BASKETBALL", "SPREAD", CHAMPION_ID, CHALLENGER_ID, empty, empty, empty, 300)
        assert report.graded_pairs == 0
        assert report.champion.brier_score is None
        assert report.challenger.log_loss is None
        assert not report.promotion_ready

    def test_mismatched_arrays_rejected(self) -> None:
        with pytest.raises(ValueError, match="paired"):
            build_report(
                "BASKETBALL",
                "SPREAD",
                CHAMPION_ID,
                CHALLENGER_ID,
                np.array([0.5]),
                np.array([0.5, 0.6]),
                np.array([1.0]),
                300,
            )


# --- service-level: pairing, grading, and promotion mechanics ----------------


def make_model_record(model_id: uuid.UUID, role: str) -> ModelVersionRecord:
    return ModelVersionRecord(
        id=model_id,
        sport="BASKETBALL",
        model_type="MONEYLINE",
        version=f"NBA_unified_{role}",
        algorithm="xgboost",
        trained_at=datetime.now(tz=UTC),
        training_samples=100,
        evaluation_metrics={},
        feature_names=[],
        is_active=True,
        artifact_path="/tmp/none",
        notes=None,
        role=role,
    )


def make_prediction(game_id: str, model_id: uuid.UUID, probability: float, is_shadow: bool) -> PredictionRecord:
    return PredictionRecord(
        id=uuid.uuid4(),
        game_external_id=game_id,
        model_version_id=model_id,
        league="NBA",
        market_type="MONEYLINE",
        side="HOME",
        selection="Los Angeles Lakers ML",
        predicted_probability=probability,
        simulation_probability=probability,
        implied_probability=None,
        edge=None,
        confidence_lower=None,
        confidence_upper=None,
        feature_importance={},
        created_at=datetime.now(tz=UTC),
        is_shadow=is_shadow,
    )


class FakeModelRepo:
    def __init__(self, champion: ModelVersionRecord | None, challenger: ModelVersionRecord | None) -> None:
        self._by_role = {"champion": champion, "challenger": challenger}
        self.promoted_with: tuple[str, list[str]] | None = None

    async def get_active(self, sport: str, market_type: str, role: str = "champion") -> ModelVersionRecord | None:
        return self._by_role.get(role)

    async def promote(self, sport: str, market_types: list[str]) -> list[ModelVersionRecord]:
        self.promoted_with = (sport, market_types)
        challenger = self._by_role["challenger"]
        assert challenger is not None
        return [make_model_record(challenger.id, "champion")]


class FakePredictionRepo:
    def __init__(self, rows_by_model: dict[uuid.UUID, list[PredictionRecord]]) -> None:
        self._rows = rows_by_model

    async def latest_rows_for_model(self, model_version_id: uuid.UUID) -> list[PredictionRecord]:
        return self._rows.get(model_version_id, [])


class FakeStatistics:
    def __init__(self, games: dict[str, Game]) -> None:
        self._games = games

    async def get_game(self, game_id: str) -> Game:
        return self._games[game_id]


class FakeRegistry:
    def __init__(self) -> None:
        self.reloaded: list[str] = []

    async def reload_sport(self, sport: str) -> None:
        self.reloaded.append(sport)


def final_game(game_id: str, home: int, away: int) -> Game:
    return Game(
        id=game_id,
        league="NBA",
        status="FINAL",
        home_team=TeamRef(id="t-home", name="Los Angeles Lakers"),
        away_team=TeamRef(id="t-away", name="Boston Celtics"),
        scheduled_start="2026-03-01T00:00:00Z",
        season=2026,
        result=GameResult(home_score=home, away_score=away),
    )


def make_service(
    rows_by_model: dict[uuid.UUID, list[PredictionRecord]],
    games: dict[str, Game],
    champion: ModelVersionRecord | None,
    challenger: ModelVersionRecord | None,
    min_samples: int = 2,
) -> tuple[ExperimentService, FakeModelRepo, FakeRegistry]:
    model_repo = FakeModelRepo(champion, challenger)
    registry = FakeRegistry()
    service = ExperimentService(
        model_repo,  # type: ignore[arg-type]
        FakePredictionRepo(rows_by_model),  # type: ignore[arg-type]
        FakeStatistics(games),  # type: ignore[arg-type]
        registry,  # type: ignore[arg-type]
        promotion_min_samples=min_samples,
    )
    return service, model_repo, registry


class TestExperimentService:
    def _fixtures(self, challenger_better: bool = True) -> Any:
        champion_model = make_model_record(uuid.uuid4(), "champion")
        challenger_model = make_model_record(uuid.uuid4(), "challenger")
        games = {f"g{i}": final_game(f"g{i}", 100 + (i % 3), 100) for i in range(6)}
        # home wins games where i % 3 != 0; probabilities near truth for the
        # challenger, anti-correlated for the champion when challenger_better
        rows_champion, rows_challenger = [], []
        for i, game_id in enumerate(games):
            outcome_win = (100 + (i % 3)) > 100
            good = 0.9 if outcome_win else 0.1
            bad = 0.4 if outcome_win else 0.6
            rows_champion.append(make_prediction(game_id, champion_model.id, bad if challenger_better else good, False))
            rows_challenger.append(make_prediction(game_id, challenger_model.id, good, True))
        rows = {champion_model.id: rows_champion, challenger_model.id: rows_challenger}
        return champion_model, challenger_model, games, rows

    async def test_report_counts_only_final_shared_pairs(self) -> None:
        champion_model, challenger_model, games, rows = self._fixtures()
        # one game not final, one extra challenger row without a champion twin
        games["g-live"] = final_game("g-live", 1, 0).model_copy(update={"status": "LIVE", "result": None})
        rows[challenger_model.id].append(make_prediction("g-live", challenger_model.id, 0.5, True))
        service, _, _ = make_service(rows, games, champion_model, challenger_model)

        report = await service.report("BASKETBALL", "MONEYLINE")
        # 6 shared games; i % 3 == 0 games are ties -> two-way pushes drop (i = 0, 3)
        assert report.graded_pairs == 4
        assert report.challenger.brier_score is not None
        assert report.champion.brier_score is not None
        assert report.challenger.brier_score < report.champion.brier_score
        assert report.promotion_ready

    async def test_report_404s_without_a_challenger(self) -> None:
        champion_model = make_model_record(uuid.uuid4(), "champion")
        service, _, _ = make_service({}, {}, champion_model, None)
        with pytest.raises(NotFoundError, match="No active challenger"):
            await service.report("BASKETBALL", "MONEYLINE")

    async def test_promote_flips_roles_and_reloads_registry(self) -> None:
        champion_model, challenger_model, games, rows = self._fixtures()
        service, model_repo, registry = make_service(rows, games, champion_model, challenger_model)

        records = await service.promote("BASKETBALL", "MONEYLINE", force=False)
        assert model_repo.promoted_with == ("BASKETBALL", ["SPREAD", "TOTAL", "MONEYLINE"])
        assert registry.reloaded == ["BASKETBALL"]
        assert records[0].role == "champion"
        assert records[0].id == challenger_model.id

    async def test_promote_blocks_when_not_ready_unless_forced(self) -> None:
        champion_model, challenger_model, games, rows = self._fixtures(challenger_better=False)
        service, model_repo, registry = make_service(rows, games, champion_model, challenger_model)

        with pytest.raises(UnprocessableError, match="not promotion-ready"):
            await service.promote("BASKETBALL", "MONEYLINE", force=False)
        assert model_repo.promoted_with is None
        assert registry.reloaded == []

        records = await service.promote("BASKETBALL", "MONEYLINE", force=True)
        assert model_repo.promoted_with is not None
        assert registry.reloaded == ["BASKETBALL"]
        assert records[0].role == "champion"
