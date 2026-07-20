"""Shadow scoring, AB split, and shadow read-path exclusion (Phase 7 Wave 4)."""

import hashlib
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from prediction_engine.api.schemas import PredictionRequest
from prediction_engine.clients.simulation import SimulationResult, SimulationRun
from prediction_engine.clients.statistics import Game, TeamRef
from prediction_engine.core.features.builder import FeatureBundle
from prediction_engine.core.predictor import Predictor
from prediction_engine.db.repository import PredictionRecord, _latest_for_game_stmt

GAME_ID = "game-1"


def make_game() -> Game:
    return Game(
        id=GAME_ID,
        league="NBA",
        status="SCHEDULED",
        home_team=TeamRef(id="t-home", name="Los Angeles Lakers"),
        away_team=TeamRef(id="t-away", name="Boston Celtics"),
        scheduled_start="2026-07-10T00:00:00Z",
    )


def make_run() -> SimulationRun:
    return SimulationRun(
        simulation_run_id="run-1",
        game_id=GAME_ID,
        status="completed",
        converged=True,
        result=SimulationResult(
            home_win_probability=0.7,
            away_win_probability=0.3,
            mean_margin=6.0,
            mean_total=220.0,
            spread_cover_probabilities={"-5.5": 0.55, "-6.5": 0.5},
            total_over_probabilities={"219.5": 0.52, "220.5": 0.5},
        ),
    )


class FakeStatistics:
    async def get_game(self, game_id: str) -> Game:
        return make_game()


class FakeSimulation:
    async def get_run(self, run_id: str) -> SimulationRun:
        return make_run()


class FakeLines:
    async def best_lines(self, game_external_id: str, market_type: str | None = None) -> list[Any]:
        return []


class FakeFeatureBuilder:
    async def build(self, game: Game) -> FeatureBundle:
        return FeatureBundle(features={"net_rating_diff": 1.0}, sources={"stats": None}, lines_game_external_id=None)


class FakeModel:
    def __init__(self, adjustment: float, fail: bool = False) -> None:
        self._adjustment = adjustment
        self._fail = fail

    def predict_adjustment(self, features: dict[str, Any]) -> float:
        if self._fail:
            raise RuntimeError("corrupt challenger artifact")
        return self._adjustment

    def feature_importance(self, features: dict[str, Any]) -> dict[str, float]:
        return {"sim_probability": 1.0}


class IdentityCalibrator:
    def apply(self, probability: float) -> float:
        return probability


class FakeConformal:
    def interval(self, probability: float) -> tuple[float, float]:
        return max(probability - 0.05, 0.0), min(probability + 0.05, 1.0)


def make_loaded(adjustment: float, fail: bool = False) -> Any:
    bundle = SimpleNamespace(
        model=FakeModel(adjustment, fail), calibrator=IdentityCalibrator(), conformal=FakeConformal()
    )
    return SimpleNamespace(record=SimpleNamespace(id=uuid.uuid4()), bundle=bundle)


class FakeRegistry:
    def __init__(self, champion: Any, challenger: Any = None) -> None:
        self.champion = champion
        self.challenger = challenger

    async def get_active(self, sport: str, market_type: str) -> Any:
        return self.champion

    async def get_challenger(self, sport: str, market_type: str) -> Any:
        return self.challenger


class CapturingRepo:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.row_features: list[dict[str, Any]] | None = None

    async def insert_predictions(
        self,
        rows: list[dict[str, Any]],
        features: dict[str, Any],
        feature_sources: dict[str, Any],
        row_features: list[dict[str, Any]] | None = None,
    ) -> list[PredictionRecord]:
        self.rows = rows
        self.row_features = row_features
        return [
            PredictionRecord(
                id=uuid.uuid4(),
                game_external_id=row["game_external_id"],
                model_version_id=row["model_version_id"],
                league=row["league"],
                market_type=row["market_type"],
                side=row["side"],
                selection=row["selection"],
                predicted_probability=row["predicted_probability"],
                simulation_probability=row["simulation_probability"],
                implied_probability=row["implied_probability"],
                edge=row["edge"],
                confidence_lower=row["confidence_lower"],
                confidence_upper=row["confidence_upper"],
                feature_importance=row["feature_importance"],
                created_at=datetime.now(tz=UTC),
                is_shadow=row["is_shadow"],
            )
            for row in rows
        ]


class FakeRedis:
    async def get(self, key: str) -> None:
        return None

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        return None

    async def publish(self, channel: str, payload: str) -> None:
        return None


def make_predictor(
    registry: FakeRegistry,
    repo: CapturingRepo,
    shadow_scoring_enabled: bool = True,
    ab_split_pct: int = 0,
) -> Predictor:
    return Predictor(
        statistics=FakeStatistics(),  # type: ignore[arg-type]
        lines=FakeLines(),  # type: ignore[arg-type]
        simulation=FakeSimulation(),  # type: ignore[arg-type]
        reconciler=SimpleNamespace(),  # type: ignore[arg-type]
        features=FakeFeatureBuilder(),  # type: ignore[arg-type]
        registry=registry,  # type: ignore[arg-type]
        repo=repo,  # type: ignore[arg-type]
        redis_client=FakeRedis(),  # type: ignore[arg-type]
        shadow_scoring_enabled=shadow_scoring_enabled,
        ab_split_pct=ab_split_pct,
    )


REQUEST = PredictionRequest(game_id=GAME_ID, simulation_run_id="run-1", market_types=["MONEYLINE"])


class TestShadowScoring:
    async def test_challenger_emits_flagged_shadow_row_primary_unchanged(self) -> None:
        champion, challenger = make_loaded(0.0), make_loaded(0.05)
        repo = CapturingRepo()
        response = await make_predictor(FakeRegistry(champion, challenger), repo).create_predictions(REQUEST)

        assert len(repo.rows) == 2
        primary, shadow = repo.rows
        assert primary["is_shadow"] is False
        assert primary["model_version_id"] == champion.record.id
        assert primary["predicted_probability"] == pytest.approx(0.7)
        assert shadow["is_shadow"] is True
        assert shadow["model_version_id"] == challenger.record.id
        assert shadow["predicted_probability"] == pytest.approx(0.75)
        # shadow and primary share side/selection (the comparison key)
        assert shadow["side"] == primary["side"]
        assert shadow["selection"] == primary["selection"]
        # response only surfaces the primary row
        assert len(response.predictions) == 1
        assert response.predictions[0].model_version_id == str(champion.record.id)

    async def test_shadow_rows_reuse_the_primary_feature_vector_content(self) -> None:
        repo = CapturingRepo()
        await make_predictor(FakeRegistry(make_loaded(0.0), make_loaded(0.05)), repo).create_predictions(REQUEST)
        assert repo.row_features is not None
        assert len(repo.row_features) == 2
        assert repo.row_features[0] == repo.row_features[1]

    async def test_no_challenger_means_no_shadow_rows(self) -> None:
        repo = CapturingRepo()
        await make_predictor(FakeRegistry(make_loaded(0.0), None), repo).create_predictions(REQUEST)
        assert [row["is_shadow"] for row in repo.rows] == [False]

    async def test_challenger_failure_never_breaks_the_primary_path(self) -> None:
        champion = make_loaded(0.0)
        repo = CapturingRepo()
        registry = FakeRegistry(champion, make_loaded(0.05, fail=True))
        response = await make_predictor(registry, repo).create_predictions(REQUEST)

        assert [row["is_shadow"] for row in repo.rows] == [False]
        assert repo.rows[0]["model_version_id"] == champion.record.id
        assert len(response.predictions) == 1

    async def test_challenger_lookup_failure_is_isolated_too(self) -> None:
        class ExplodingRegistry(FakeRegistry):
            async def get_challenger(self, sport: str, market_type: str) -> Any:
                raise RuntimeError("db down")

        repo = CapturingRepo()
        response = await make_predictor(ExplodingRegistry(make_loaded(0.0)), repo).create_predictions(REQUEST)
        assert [row["is_shadow"] for row in repo.rows] == [False]
        assert len(response.predictions) == 1

    async def test_shadow_scoring_disabled_writes_no_shadow_rows(self) -> None:
        repo = CapturingRepo()
        predictor = make_predictor(
            FakeRegistry(make_loaded(0.0), make_loaded(0.05)), repo, shadow_scoring_enabled=False
        )
        await predictor.create_predictions(REQUEST)
        assert [row["is_shadow"] for row in repo.rows] == [False]


class TestAbSplit:
    async def test_full_split_serves_challenger_as_primary(self) -> None:
        champion, challenger = make_loaded(0.0), make_loaded(0.05)
        repo = CapturingRepo()
        predictor = make_predictor(FakeRegistry(champion, challenger), repo, ab_split_pct=100)
        response = await predictor.create_predictions(REQUEST)

        primary, shadow = repo.rows
        assert primary["is_shadow"] is False
        assert primary["model_version_id"] == challenger.record.id
        assert primary["predicted_probability"] == pytest.approx(0.75)
        # the champion becomes the shadow row
        assert shadow["is_shadow"] is True
        assert shadow["model_version_id"] == champion.record.id
        assert response.predictions[0].model_version_id == str(challenger.record.id)

    async def test_zero_split_always_serves_champion(self) -> None:
        champion, challenger = make_loaded(0.0), make_loaded(0.05)
        repo = CapturingRepo()
        predictor = make_predictor(FakeRegistry(champion, challenger), repo, ab_split_pct=0)
        await predictor.create_predictions(REQUEST)
        assert repo.rows[0]["model_version_id"] == champion.record.id

    def test_bucketing_is_deterministic_sha256(self) -> None:
        repo = CapturingRepo()
        bucket = int(hashlib.sha256(GAME_ID.encode()).hexdigest(), 16) % 100
        for pct in (0, bucket, bucket + 1, 100):
            predictor = make_predictor(FakeRegistry(make_loaded(0.0)), repo, ab_split_pct=pct)
            expected = pct > bucket  # served iff bucket < pct
            assert predictor._serves_challenger(GAME_ID) is expected
            # repeated calls never flip
            assert predictor._serves_challenger(GAME_ID) is expected


class TestShadowReadPathExclusion:
    def _sql(self, include_shadow: bool) -> str:
        from sqlalchemy.dialects import postgresql

        stmt = _latest_for_game_stmt(GAME_ID, None, None, include_shadow)
        return str(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})).lower()

    def test_latest_for_game_filters_shadows_by_default(self) -> None:
        assert "is_shadow = false" in self._sql(include_shadow=False)

    def test_include_shadow_lifts_the_filter(self) -> None:
        # the column is still selected; only the WHERE filter disappears
        assert "is_shadow = false" not in self._sql(include_shadow=True)
