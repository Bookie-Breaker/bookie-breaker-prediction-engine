"""Migration 0003 + player-prop API checks (Phase 7 Wave 3).

Runs against the alembic-migrated testcontainers Postgres with respx-mocked
upstreams: the new player columns round-trip through POST /predictions,
YES/NO sides pass the widened check constraint, and the 0003 migration
downgrades/upgrades cleanly.
"""

import asyncio
import uuid
from typing import Any

import asyncpg
import pytest
from httpx import Response

from tests.integration.conftest import SIM_URL, enveloped, mock_happy_path

PLAYER_ID = "11111111-2222-3333-4444-555555555555"


async def _connect(database_url: str) -> asyncpg.Connection:
    return await asyncpg.connect(database_url.split("?")[0].replace("postgres://", "postgresql://"))


def player_distributions_payload(run_id: str, game_id: str) -> dict[str, Any]:
    return {
        "simulation_run_id": run_id,
        "game_id": game_id,
        "players": {
            PLAYER_ID: {
                "name": "LeBron James",
                "team": "HOME",
                "stats": {
                    "player_points": {
                        "distribution": {"mean": 26.1, "std": 6.8},
                        "over_probabilities": {"24.5": 0.58, "25.5": 0.52, "26.5": 0.46},
                    }
                },
            }
        },
    }


def mock_prop_happy_path(router: Any, game_id: str, run_id: str) -> None:
    mock_happy_path(router, game_id, run_id)
    router.get(f"{SIM_URL}/api/v1/sim/simulations/{run_id}/player-distributions").mock(
        return_value=Response(200, json=enveloped(player_distributions_payload(run_id, game_id)))
    )


@pytest.fixture
async def basketball_model_version(migrated_database_url: str) -> uuid.UUID:
    conn = await _connect(migrated_database_url)
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO predictions.model_versions
                (sport, model_type, version, trained_at, training_data_range, training_samples, artifact_path)
            VALUES ('BASKETBALL', 'PLAYER_PROP', 'test-props', NOW(),
                    tstzrange(NOW() - interval '1 year', NOW()), 100, '/tmp/x')
            RETURNING id
            """
        )
        return row["id"]
    finally:
        await conn.close()


class TestPropColumns:
    async def test_yes_and_no_sides_accepted(
        self, migrated_database_url: str, basketball_model_version: uuid.UUID
    ) -> None:
        conn = await _connect(migrated_database_url)
        try:
            for side in ("YES", "NO"):
                await conn.execute(
                    """
                    INSERT INTO predictions.predictions
                        (game_external_id, model_version_id, league, market_type, side,
                         player_external_id, stat_type, selection, predicted_probability)
                    VALUES ($1, $2, 'NBA', 'PLAYER_PROP', $3, $4, 'player_anytime_td', 'x', 0.5)
                    """,
                    f"game-{uuid.uuid4()}",
                    basketball_model_version,
                    side,
                    PLAYER_ID,
                )
        finally:
            await conn.close()

    async def test_prop_columns_round_trip(
        self, migrated_database_url: str, basketball_model_version: uuid.UUID
    ) -> None:
        conn = await _connect(migrated_database_url)
        game_id = f"game-{uuid.uuid4()}"
        try:
            await conn.execute(
                """
                INSERT INTO predictions.predictions
                    (game_external_id, model_version_id, league, market_type, side,
                     player_external_id, stat_type, prop_line, selection, predicted_probability)
                VALUES ($1, $2, 'NBA', 'PLAYER_PROP', 'OVER', $3, 'player_points', 25.5,
                        'LeBron James player_points Over 25.5', 0.55)
                """,
                game_id,
                basketball_model_version,
                PLAYER_ID,
            )
            row = await conn.fetchrow(
                "SELECT player_external_id, stat_type, prop_line FROM predictions.predictions "
                "WHERE game_external_id = $1",
                game_id,
            )
            assert row["player_external_id"] == PLAYER_ID
            assert row["stat_type"] == "player_points"
            assert float(row["prop_line"]) == 25.5
        finally:
            await conn.close()

    async def test_invalid_side_still_rejected(
        self, migrated_database_url: str, basketball_model_version: uuid.UUID
    ) -> None:
        conn = await _connect(migrated_database_url)
        try:
            with pytest.raises(asyncpg.CheckViolationError, match="chk_predictions_side"):
                await conn.execute(
                    """
                    INSERT INTO predictions.predictions
                        (game_external_id, model_version_id, league, market_type, side, selection,
                         predicted_probability)
                    VALUES ($1, $2, 'NBA', 'PLAYER_PROP', 'MAYBE', 'x', 0.5)
                    """,
                    f"game-{uuid.uuid4()}",
                    basketball_model_version,
                )
        finally:
            await conn.close()


class TestMigration0003RoundTrip:
    async def test_downgrade_and_upgrade(self, migrated_database_url: str) -> None:
        """0003 downgrades cleanly (columns gone, old side vocabulary back)
        and re-upgrades to head, leaving the schema as the other tests
        expect."""
        from alembic.config import Config

        from alembic import command

        config = Config("alembic.ini")
        # alembic's async env calls asyncio.run internally; run it off-loop
        await asyncio.to_thread(command.downgrade, config, "0002")
        conn = await _connect(migrated_database_url)
        try:
            columns = {
                row["column_name"]
                for row in await conn.fetch(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = 'predictions' AND table_name = 'predictions'"
                )
            }
            assert {"player_external_id", "stat_type", "prop_line"}.isdisjoint(columns)
        finally:
            await conn.close()

        await asyncio.to_thread(command.upgrade, config, "head")
        conn = await _connect(migrated_database_url)
        try:
            columns = {
                row["column_name"]
                for row in await conn.fetch(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = 'predictions' AND table_name = 'predictions'"
                )
            }
            assert {"player_external_id", "stat_type", "prop_line"} <= columns
        finally:
            await conn.close()


class TestPropPredictionsApi:
    async def test_post_predictions_with_props_persists_player_columns(
        self, client, upstream, migrated_database_url: str
    ) -> None:
        game_id = f"game-{uuid.uuid4()}"
        run_id = f"run-{uuid.uuid4()}"
        mock_prop_happy_path(upstream, game_id, run_id)

        response = client.post(
            "/api/v1/predict/predictions",
            json={
                "game_id": game_id,
                "simulation_run_id": run_id,
                "market_types": ["MONEYLINE"],
                "props": [
                    {
                        "player_external_id": PLAYER_ID,
                        "player_name": "LeBron James",
                        "stat_type": "player_points",
                        "line": 25.5,
                    }
                ],
            },
        )
        assert response.status_code == 201, response.text
        predictions = response.json()["data"]["predictions"]
        by_key = {(p["market_type"], p["side"]): p for p in predictions}
        assert set(by_key) == {("MONEYLINE", "HOME"), ("PLAYER_PROP", "OVER"), ("PLAYER_PROP", "UNDER")}
        over = by_key[("PLAYER_PROP", "OVER")]
        under = by_key[("PLAYER_PROP", "UNDER")]
        assert over["player_external_id"] == PLAYER_ID
        assert over["stat_type"] == "player_points"
        assert over["prop_line"] == 25.5
        assert over["predicted_probability"] + under["predicted_probability"] == pytest.approx(1.0, abs=1e-4)

        conn = await _connect(migrated_database_url)
        try:
            rows = await conn.fetch(
                "SELECT side, player_external_id, stat_type, prop_line FROM predictions.predictions "
                "WHERE game_external_id = $1 AND market_type = 'PLAYER_PROP' ORDER BY side",
                game_id,
            )
            assert [row["side"] for row in rows] == ["OVER", "UNDER"]
            assert all(row["player_external_id"] == PLAYER_ID for row in rows)
            assert all(row["stat_type"] == "player_points" for row in rows)
            assert all(float(row["prop_line"]) == 25.5 for row in rows)
        finally:
            await conn.close()

    async def test_line_off_grid_is_422(self, client, upstream) -> None:
        game_id = f"game-{uuid.uuid4()}"
        run_id = f"run-{uuid.uuid4()}"
        mock_prop_happy_path(upstream, game_id, run_id)

        response = client.post(
            "/api/v1/predict/predictions",
            json={
                "game_id": game_id,
                "simulation_run_id": run_id,
                "market_types": [],
                "props": [{"player_external_id": PLAYER_ID, "stat_type": "player_points", "line": 30.0}],
            },
        )
        assert response.status_code == 422
        assert "outside the simulated" in response.json()["error"]["message"]

    async def test_run_without_player_capture_is_422(self, client, upstream) -> None:
        game_id = f"game-{uuid.uuid4()}"
        run_id = f"run-{uuid.uuid4()}"
        mock_happy_path(upstream, game_id, run_id)
        upstream.get(f"{SIM_URL}/api/v1/sim/simulations/{run_id}/player-distributions").mock(
            return_value=Response(404, json={"error": {"code": "RESOURCE_NOT_FOUND", "message": "no players"}})
        )

        response = client.post(
            "/api/v1/predict/predictions",
            json={
                "game_id": game_id,
                "simulation_run_id": run_id,
                "market_types": [],
                "props": [{"player_external_id": PLAYER_ID, "stat_type": "player_points", "line": 25.5}],
            },
        )
        assert response.status_code == 422
        assert "player distributions" in response.json()["error"]["message"]
