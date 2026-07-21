"""Model route contract paths not covered by the Wave 4 flow tests:
validation 422s, model 404/detail, list filters, and the experiment report
envelope over a seeded graded shadow pair.
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest
from httpx import Response

from prediction_engine.db.engine import create_engine
from prediction_engine.db.repository import ModelVersionRepository, PredictionRepository
from tests.integration.conftest import STATS_URL, enveloped, game_payload


@pytest.fixture
async def repos(migrated_database_url: str):
    engine = create_engine(migrated_database_url)
    yield ModelVersionRepository(engine), PredictionRepository(engine)
    await engine.dispose()


async def _deactivate_version(database_url: str, version: str) -> None:
    conn = await asyncpg.connect(database_url.split("?")[0].replace("postgres://", "postgresql://"))
    try:
        await conn.execute("UPDATE predictions.model_versions SET is_active = FALSE WHERE version = $1", version)
    finally:
        await conn.close()


class TestModelValidation:
    def test_unknown_sport_is_422(self, client) -> None:
        response = client.get("/api/v1/predict/models/experiments/CRICKET/MONEYLINE")
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "UNPROCESSABLE_ENTITY"

    def test_unknown_market_type_is_422(self, client) -> None:
        response = client.get("/api/v1/predict/models/experiments/BASKETBALL/PARLAY")
        assert response.status_code == 422
        assert "market_type must be one of" in response.json()["error"]["message"]

    def test_unknown_model_id_is_404(self, client) -> None:
        response = client.get(f"/api/v1/predict/models/{uuid.uuid4()}")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "RESOURCE_NOT_FOUND"


class TestModelListing:
    def test_role_and_active_filters_apply(self, client) -> None:
        response = client.get(
            "/api/v1/predict/models", params={"sport": "basketball", "role": "CHAMPION", "is_active": "true"}
        )
        assert response.status_code == 200
        models = response.json()["data"]
        assert models, "startup bootstrap must have registered active champions"
        assert all(model["sport"] == "BASKETBALL" for model in models)
        assert all(model["role"] == "champion" for model in models)
        assert all(model["is_active"] for model in models)

    def test_model_detail_includes_the_ordered_feature_names(self, client) -> None:
        model_id = client.get("/api/v1/predict/models").json()["data"][0]["id"]
        response = client.get(f"/api/v1/predict/models/{model_id}")
        assert response.status_code == 200
        detail = response.json()["data"]
        assert detail["id"] == model_id
        assert isinstance(detail["feature_names"], list) and detail["feature_names"]


class TestExperimentReportEndpoint:
    async def _register_challenger(self, model_repo: ModelVersionRepository, artifact_path: str) -> str:
        """An active challenger sharing the bootstrap champion's artifact."""
        version = f"exp_routes_{uuid.uuid4().hex[:8]}"
        now = datetime.now(tz=UTC)
        await model_repo.register(
            sport="BASKETBALL",
            market_types=["SPREAD", "TOTAL", "MONEYLINE"],
            version=version,
            trained_at=now,
            training_range=(now - timedelta(days=365), now),
            training_samples=100,
            evaluation_metrics={"brier_score": 0.2},
            feature_names=["sim_probability"],
            artifact_path=artifact_path,
            role="challenger",
        )
        return version

    async def test_report_grades_seeded_shadow_pairs(self, client, repos, upstream, migrated_database_url: str) -> None:
        model_repo, prediction_repo = repos
        champion = await model_repo.get_active("BASKETBALL", "MONEYLINE", role="champion")
        assert champion is not None, "startup bootstrap must have registered a champion"
        # registered (not fixture-injected) in the test body: the engine's
        # pooled asyncpg connections must all live on this test's event loop
        challenger_version = await self._register_challenger(model_repo, champion.artifact_path)
        try:
            await self._assert_report(client, repos, upstream, champion, challenger_version)
        finally:
            # deactivate so later modules keep the bootstrap champion-only state
            await _deactivate_version(migrated_database_url, challenger_version)

    async def _assert_report(self, client, repos, upstream, champion, challenger_version: str) -> None:
        model_repo, prediction_repo = repos
        challenger = await model_repo.get_active("BASKETBALL", "MONEYLINE", role="challenger")
        assert challenger is not None and challenger.version == challenger_version

        # one shadow pair: champion 0.6 vs challenger 0.9 on a home win
        game_id = f"exp-routes-{uuid.uuid4().hex[:8]}"
        row: dict[str, Any] = {
            "game_external_id": game_id,
            "league": "NBA",
            "market_type": "MONEYLINE",
            "side": "HOME",
            "selection": "Los Angeles Lakers ML",
            "simulation_probability": 0.55,
            "implied_probability": None,
            "edge": None,
            "confidence_lower": 0.5,
            "confidence_upper": 0.7,
            "feature_importance": {},
        }
        await prediction_repo.insert_predictions(
            [
                {**row, "model_version_id": champion.id, "predicted_probability": 0.6, "is_shadow": False},
                {**row, "model_version_id": challenger.id, "predicted_probability": 0.9, "is_shadow": True},
            ],
            features={"net_rating_diff": 2.0},
            feature_sources={},
        )
        final_payload = game_payload(game_id)
        final_payload["status"] = "FINAL"
        final_payload["result"] = {"home_score": 110, "away_score": 105, "overtime": False}
        upstream.get(f"{STATS_URL}/api/v1/stats/games/{game_id}").mock(
            return_value=Response(200, json=enveloped(final_payload))
        )

        response = client.get("/api/v1/predict/models/experiments/BASKETBALL/MONEYLINE")
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["sport"] == "BASKETBALL"
        assert data["market_type"] == "MONEYLINE"
        assert data["graded_pairs"] == 1
        assert data["champion"]["role"] == "champion"
        assert data["champion"]["brier_score"] == pytest.approx(0.16)
        assert data["challenger"]["role"] == "challenger"
        assert data["challenger"]["brier_score"] == pytest.approx(0.01)
        assert data["challenger"]["version_tag"] == challenger_version
        assert data["champion"]["version_tag"] == champion.version
        # one pair can never satisfy the promotion sample floor
        assert data["promotion_ready"] is False
        assert any("graded pairs" in blocker for blocker in data["blockers"])
