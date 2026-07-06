"""Migration 0002 checks: the side column accepts DRAW and rejects junk.

Runs against the alembic-migrated testcontainers Postgres, exercising the
Phase 6 enum values (SOCCER sport, FIFA_WC league) end to end.
"""

import uuid

import asyncpg
import pytest


async def _connect(database_url: str) -> asyncpg.Connection:
    return await asyncpg.connect(database_url.split("?")[0].replace("postgres://", "postgresql://"))


async def _insert_prediction(conn: asyncpg.Connection, model_version_id: uuid.UUID, side: str | None) -> None:
    await conn.execute(
        """
        INSERT INTO predictions.predictions
            (game_external_id, model_version_id, league, market_type, side, selection, predicted_probability)
        VALUES ($1, $2, 'FIFA_WC', 'MONEYLINE', $3, 'Draw', 0.27)
        """,
        f"game-{uuid.uuid4()}",
        model_version_id,
        side,
    )


@pytest.fixture
async def soccer_model_version(migrated_database_url: str) -> uuid.UUID:
    conn = await _connect(migrated_database_url)
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO predictions.model_versions
                (sport, model_type, version, trained_at, training_data_range, training_samples, artifact_path)
            VALUES ('SOCCER', 'MONEYLINE', 'test', NOW(), tstzrange(NOW() - interval '1 year', NOW()), 100, '/tmp/x')
            RETURNING id
            """
        )
        return row["id"]
    finally:
        await conn.close()


class TestSideColumn:
    async def test_draw_row_accepted(self, migrated_database_url: str, soccer_model_version: uuid.UUID) -> None:
        conn = await _connect(migrated_database_url)
        try:
            await _insert_prediction(conn, soccer_model_version, "DRAW")
            side = await conn.fetchval(
                "SELECT side FROM predictions.predictions WHERE model_version_id = $1 AND side = 'DRAW'",
                soccer_model_version,
            )
            assert side == "DRAW"
        finally:
            await conn.close()

    async def test_null_side_accepted_for_legacy_rows(
        self, migrated_database_url: str, soccer_model_version: uuid.UUID
    ) -> None:
        conn = await _connect(migrated_database_url)
        try:
            await _insert_prediction(conn, soccer_model_version, None)
        finally:
            await conn.close()

    async def test_invalid_side_rejected(self, migrated_database_url: str, soccer_model_version: uuid.UUID) -> None:
        conn = await _connect(migrated_database_url)
        try:
            with pytest.raises(asyncpg.CheckViolationError, match="chk_predictions_side"):
                await _insert_prediction(conn, soccer_model_version, "MIDDLE")
        finally:
            await conn.close()


class TestSideInApi:
    def test_prediction_rows_carry_sides(self, client, upstream) -> None:
        from tests.integration.conftest import mock_happy_path

        game_id = f"game-{uuid.uuid4()}"
        run_id = f"run-{uuid.uuid4()}"
        mock_happy_path(upstream, game_id, run_id)

        response = client.post("/api/v1/predict/predictions", json={"game_id": game_id, "simulation_run_id": run_id})
        assert response.status_code == 201, response.text
        sides = {p["market_type"]: p["side"] for p in response.json()["data"]["predictions"]}
        assert sides == {"SPREAD": "HOME", "MONEYLINE": "HOME", "TOTAL": "OVER"}

        detail_id = response.json()["data"]["predictions"][0]["id"]
        detail = client.get(f"/api/v1/predict/predictions/{detail_id}")
        assert detail.status_code == 200
        assert detail.json()["data"]["side"] in {"HOME", "OVER"}
