"""Wave 4 integration: migration 0004, champion/challenger rows, shadow writes,
the retrain job, and promotion -- against real Postgres + Redis.

NOTE: written during the Wave 4 implementation session; per the wave plan
these run in the verification session (pre-push / CI), not inline.
"""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest
from httpx import Response

from prediction_engine.db.engine import create_engine
from prediction_engine.db.repository import ModelVersionRepository, PredictionRepository
from tests.integration.conftest import (
    STATS_URL,
    enveloped,
    game_payload,
    mock_happy_path,
)


async def _fetch(database_url: str, query: str, *args: Any) -> list[Any]:
    conn = await asyncpg.connect(database_url.split("?")[0].replace("postgres://", "postgresql://"))
    try:
        return list(await conn.fetch(query, *args))
    finally:
        await conn.close()


@pytest.fixture(scope="module")
def real_artifact_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """One real (tiny) trained artifact shared by every registered row.

    Registered rows are LOADED by the registry (startup bootstrap, shadow
    scoring, promotion reloads), so fake paths poison every later test
    session against the same database.
    """
    from prediction_engine.core.training.synthetic import generate_synthetic_dataset
    from prediction_engine.core.training.train import save_artifact, train_model

    result = train_model(generate_synthetic_dataset(n_rows=1500), n_rounds=30, data_label="itest")
    return str(save_artifact(result, tmp_path_factory.mktemp("wave4-artifacts")))


def _final_game_payload(game_id: str, home: int, away: int, season: int = 2026) -> dict[str, Any]:
    payload = game_payload(game_id)
    payload["status"] = "FINAL"
    payload["season"] = season
    payload["result"] = {"home_score": home, "away_score": away, "overtime": False}
    return payload


class TestMigration0004:
    """Runs first in this module: downgrade/upgrade must land back on head."""

    def test_downgrade_and_upgrade_round_trip(self, migrated_database_url: str) -> None:
        from alembic.config import Config

        from alembic import command

        config = Config("alembic.ini")
        command.downgrade(config, "0003")
        columns = asyncio.run(
            _fetch(
                migrated_database_url,
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'predictions' AND table_name = 'model_versions'",
            )
        )
        names = {row["column_name"] for row in columns}
        assert "role" not in names

        command.upgrade(config, "head")
        columns = asyncio.run(
            _fetch(
                migrated_database_url,
                "SELECT column_name, column_default FROM information_schema.columns "
                "WHERE table_schema = 'predictions' AND table_name IN ('model_versions', 'predictions')",
            )
        )
        names = {row["column_name"] for row in columns}
        assert "role" in names
        assert "is_shadow" in names

    def test_active_index_covers_role(self, migrated_database_url: str) -> None:
        rows = asyncio.run(
            _fetch(
                migrated_database_url,
                "SELECT indexdef FROM pg_indexes WHERE schemaname = 'predictions' "
                "AND indexname = 'uq_model_versions_active'",
            )
        )
        assert len(rows) == 1
        indexdef = rows[0]["indexdef"]
        assert "role" in indexdef
        assert "UNIQUE" in indexdef


@pytest.fixture
async def repos(migrated_database_url: str):
    engine = create_engine(migrated_database_url)
    yield ModelVersionRepository(engine), PredictionRepository(engine)
    await engine.dispose()


def _register_kwargs(sport: str = "BASKETBALL") -> dict[str, Any]:
    now = datetime.now(tz=UTC)
    return {
        "sport": sport,
        "market_types": ["SPREAD", "TOTAL", "MONEYLINE"],
        "trained_at": now,
        "training_range": (now - timedelta(days=365), now),
        "training_samples": 100,
        "evaluation_metrics": {"brier_score": 0.2},
        "feature_names": ["sim_probability"],
    }


class TestChampionChallengerCoexistence:
    async def test_champion_and_challenger_are_both_active_under_the_new_index(
        self, repos, real_artifact_path: str
    ) -> None:
        model_repo, _ = repos
        version = f"itest_{uuid.uuid4().hex[:8]}"
        champions = await model_repo.register(
            version=f"{version}_champ", artifact_path=real_artifact_path, **_register_kwargs()
        )
        challengers = await model_repo.register(
            version=f"{version}_chal", artifact_path=real_artifact_path, role="challenger", **_register_kwargs()
        )
        assert all(record.is_active for record in champions + challengers)

        champion = await model_repo.get_active("BASKETBALL", "MONEYLINE", role="champion")
        challenger = await model_repo.get_active("BASKETBALL", "MONEYLINE", role="challenger")
        assert champion is not None and champion.version == f"{version}_champ"
        assert challenger is not None and challenger.version == f"{version}_chal"

    async def test_registering_a_second_challenger_deactivates_only_the_first_challenger(
        self, repos, real_artifact_path: str
    ) -> None:
        model_repo, _ = repos
        version = f"itest_{uuid.uuid4().hex[:8]}"
        await model_repo.register(
            version=f"{version}_c1", artifact_path=real_artifact_path, role="challenger", **_register_kwargs()
        )
        champion_before = await model_repo.get_active("BASKETBALL", "MONEYLINE", role="champion")
        await model_repo.register(
            version=f"{version}_c2", artifact_path=real_artifact_path, role="challenger", **_register_kwargs()
        )

        challenger = await model_repo.get_active("BASKETBALL", "MONEYLINE", role="challenger")
        champion_after = await model_repo.get_active("BASKETBALL", "MONEYLINE", role="champion")
        assert challenger is not None and challenger.version == f"{version}_c2"
        assert champion_before is not None and champion_after is not None
        assert champion_after.id == champion_before.id  # champion untouched


class TestShadowWrites:
    async def test_post_predictions_with_active_challenger_writes_shadow_rows(
        self, client, repos, upstream, migrated_database_url: str
    ) -> None:
        model_repo, _ = repos
        game_id = f"game-shadow-{uuid.uuid4().hex[:8]}"
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        mock_happy_path(upstream, game_id, run_id)

        # the challenger reuses the bootstrap champion's on-disk artifact
        champion = await model_repo.get_active("BASKETBALL", "MONEYLINE", role="champion")
        assert champion is not None, "startup bootstrap must have registered a champion"
        challengers = await model_repo.register(
            version=f"shadow_test_{uuid.uuid4().hex[:8]}",
            artifact_path=champion.artifact_path,
            role="challenger",
            **_register_kwargs(),
        )
        challenger_ids = {record.id for record in challengers}

        response = client.post(
            "/api/v1/predict/predictions",
            json={"game_id": game_id, "simulation_run_id": run_id, "market_types": ["SPREAD", "TOTAL", "MONEYLINE"]},
        )
        assert response.status_code == 201
        data = response.json()["data"]
        # the response only surfaces primary rows
        assert len(data["predictions"]) == 3
        assert all(not p["is_shadow"] for p in data["predictions"])

        rows = await _fetch(
            migrated_database_url,
            "SELECT model_version_id, market_type, is_shadow FROM predictions.predictions WHERE game_external_id = $1",
            game_id,
        )
        primary = [row for row in rows if not row["is_shadow"]]
        shadows = [row for row in rows if row["is_shadow"]]
        assert len(primary) == 3
        assert len(shadows) == 3
        assert {row["model_version_id"] for row in shadows} <= challenger_ids
        # every shadow row has its own feature vector row (FK layout)
        vectors = await _fetch(
            migrated_database_url,
            "SELECT count(*) AS n FROM predictions.feature_vectors fv "
            "JOIN predictions.predictions p ON p.id = fv.prediction_id WHERE p.game_external_id = $1",
            game_id,
        )
        assert vectors[0]["n"] == 6

        # read paths exclude shadows
        latest = client.get(f"/api/v1/predict/games/{game_id}/latest")
        assert latest.status_code == 200
        latest_predictions = latest.json()["data"]["predictions"]
        assert len(latest_predictions) == 3
        assert all(not p["is_shadow"] for p in latest_predictions)


class TestRetrainJob:
    async def test_retrain_registers_a_challenger_and_status_progresses(
        self, client, repos, upstream, migrated_database_url: str
    ) -> None:
        model_repo, prediction_repo = repos
        champion = await model_repo.get_active("BASKETBALL", "MONEYLINE", role="champion")
        assert champion is not None

        # Seed graded history: 2 seasons x 20 games, mixed outcomes. Each
        # game gets one MONEYLINE row whose stored vector is minimal (missing
        # features vectorize to NaN, which XGBoost handles natively).
        prefix = f"retrain-{uuid.uuid4().hex[:8]}"
        for i in range(40):
            game_id = f"{prefix}-g{i}"
            season = 2025 if i < 20 else 2026
            home_wins = i % 2 == 0
            # both sides per game: every assembled slice then contains both
            # outcome classes regardless of row ordering, so Platt can fit
            await prediction_repo.insert_predictions(
                [
                    {
                        "game_external_id": game_id,
                        "model_version_id": champion.id,
                        "league": "NBA",
                        "market_type": "MONEYLINE",
                        "side": side,
                        "selection": f"{'Los Angeles Lakers' if side == 'HOME' else 'Boston Celtics'} ML",
                        "predicted_probability": 0.6 if side == "HOME" else 0.4,
                        "simulation_probability": 0.55 if home_wins else 0.45,
                        "implied_probability": None,
                        "edge": None,
                        "confidence_lower": 0.5,
                        "confidence_upper": 0.7,
                        "feature_importance": {},
                        "is_shadow": False,
                    }
                    for side in ("HOME", "AWAY")
                ],
                features={"net_rating_diff": 2.0 if home_wins else -2.0},
                feature_sources={},
            )
            upstream.get(f"{STATS_URL}/api/v1/stats/games/{game_id}").mock(
                return_value=Response(
                    200,
                    json=enveloped(_final_game_payload(game_id, 110 if home_wins else 100, 105, season=season)),
                )
            )

        accepted = client.post(
            "/api/v1/predict/models/retrain",
            json={"sport": "BASKETBALL", "market": "game", "ensemble": True, "min_rows": 10},
        )
        assert accepted.status_code == 202
        job_id = accepted.json()["data"]["job_id"]

        # TestClient runs BackgroundTasks before returning, so the terminal
        # status is already in Redis.
        status = client.get(f"/api/v1/predict/models/retrain/{job_id}")
        assert status.status_code == 200
        body = status.json()["data"]
        assert body["status"] == "registered"
        assert body["rows"] >= 10
        assert body["model_version_id"]

        challenger = await model_repo.get_active("BASKETBALL", "MONEYLINE", role="challenger")
        assert challenger is not None
        assert str(challenger.id) == body["model_version_id"] or challenger.version in body["detail"]

    def test_retrain_rejects_player_props_with_deferral(self, client) -> None:
        response = client.post("/api/v1/predict/models/retrain", json={"sport": "BASKETBALL", "market": "player_prop"})
        assert response.status_code == 422
        assert "deferred" in response.json()["error"]["message"]

    def test_insufficient_data_status(self, client) -> None:
        # HOCKEY has a feature registry but no seeded prediction history here
        accepted = client.post(
            "/api/v1/predict/models/retrain", json={"sport": "HOCKEY", "market": "game", "min_rows": 999999}
        )
        assert accepted.status_code == 202
        job_id = accepted.json()["data"]["job_id"]
        status = client.get(f"/api/v1/predict/models/retrain/{job_id}")
        assert status.json()["data"]["status"] == "insufficient_data"

    def test_unknown_job_is_404(self, client) -> None:
        assert client.get(f"/api/v1/predict/models/retrain/{uuid.uuid4()}").status_code == 404


class TestPromotion:
    async def test_promote_flips_roles_atomically(self, client, repos) -> None:
        model_repo, _ = repos
        champion_before = await model_repo.get_active("BASKETBALL", "MONEYLINE", role="champion")
        challenger_before = await model_repo.get_active("BASKETBALL", "MONEYLINE", role="challenger")
        assert champion_before is not None
        if challenger_before is None:
            challengers = await model_repo.register(
                version=f"promo_{uuid.uuid4().hex[:8]}",
                artifact_path=champion_before.artifact_path,
                role="challenger",
                **_register_kwargs(),
            )
            challenger_before = challengers[-1]

        # without enough graded pairs the criteria block promotion
        blocked = client.post("/api/v1/predict/models/experiments/BASKETBALL/MONEYLINE/promote", json={"force": False})
        assert blocked.status_code == 422

        promoted = client.post("/api/v1/predict/models/experiments/BASKETBALL/MONEYLINE/promote", json={"force": True})
        assert promoted.status_code == 200
        data = promoted.json()["data"]
        assert data["forced"] is True
        assert len(data["model_version_ids"]) == 3

        new_champion = await model_repo.get_active("BASKETBALL", "MONEYLINE", role="champion")
        assert new_champion is not None
        assert new_champion.version == challenger_before.version
        assert await model_repo.get_active("BASKETBALL", "MONEYLINE", role="challenger") is None
        # the outgoing champion is retired: inactive with role=shadow
        retired = await model_repo.get(champion_before.id)
        assert retired is not None
        assert retired.role == "shadow"
        assert not retired.is_active

    def test_experiments_endpoint_404s_without_challenger(self, client) -> None:
        # promotion above consumed the challenger
        response = client.get("/api/v1/predict/models/experiments/BASKETBALL/MONEYLINE")
        assert response.status_code == 404
