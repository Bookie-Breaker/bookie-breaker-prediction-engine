"""The roadmap task-18 pipeline: simulation -> prediction -> edge detection.

Runs against real Postgres (alembic-migrated) and Redis with respx-mocked
statistics/lines/simulation services.
"""

import json
import time
import uuid

import redis as sync_redis


def post_predictions(client, game_id: str, run_id: str, **kwargs):
    body = {"game_id": game_id, "simulation_run_id": run_id, **kwargs}
    return client.post("/api/v1/predict/predictions", json=body)


class TestPredictionPipeline:
    def test_full_pipeline_predict_then_edges(self, client, upstream, redis_url) -> None:
        from tests.integration.conftest import mock_happy_path

        game_id = f"game-{uuid.uuid4()}"
        run_id = f"run-{uuid.uuid4()}"
        mock_happy_path(upstream, game_id, run_id)

        redis_client = sync_redis.Redis.from_url(redis_url, decode_responses=True)
        pubsub = redis_client.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe("events:prediction.completed")

        response = post_predictions(client, game_id, run_id)
        assert response.status_code == 201, response.text
        data = response.json()["data"]

        assert data["game_id"] == game_id
        assert len(data["predictions"]) == 3
        by_market = {p["market_type"]: p for p in data["predictions"]}
        assert by_market["SPREAD"]["selection"] == "Los Angeles Lakers -3.5"
        assert by_market["TOTAL"]["selection"] == "Over 220.5"
        assert by_market["MONEYLINE"]["selection"] == "Los Angeles Lakers ML"
        for prediction in data["predictions"]:
            assert 0.0 < prediction["predicted_probability"] < 1.0
            assert prediction["confidence_lower"] <= prediction["predicted_probability"]
            assert prediction["predicted_probability"] <= prediction["confidence_upper"]
            assert prediction["model_version_id"]
        assert data["features_used"]["home_rest_days"] == 1.0
        assert data["feature_source_versions"]["simulation_run_id"] == run_id

        # prediction.completed event published
        message = None
        deadline = time.monotonic() + 5.0
        while message is None and time.monotonic() < deadline:
            message = pubsub.get_message(timeout=0.5)
        assert message is not None
        event = json.loads(message["data"])
        assert event["event"] == "prediction.completed"
        assert event["game_ids"] == [game_id]
        assert event["predictions_count"] == 3
        pubsub.close()

        # GET latest returns one row per market
        latest = client.get(f"/api/v1/predict/games/{game_id}/latest")
        assert latest.status_code == 200
        assert len(latest.json()["data"]["predictions"]) == 3

        # GET detail includes the stored feature vector
        detail = client.get(f"/api/v1/predict/predictions/{by_market['SPREAD']['id']}")
        assert detail.status_code == 200
        detail_data = detail.json()["data"]
        assert detail_data["feature_vector"]["features"]["home_offensive_rating"] == 115.0

        # Edge detection: the sim says 75% ML vs 60% implied -> a positive
        # moneyline edge must survive calibration
        edges = client.get(f"/api/v1/predict/games/{game_id}/edges")
        assert edges.status_code == 200
        edge_data = edges.json()["data"]
        ml_edges = [e for e in edge_data["edges"] if e["market_type"] == "MONEYLINE"]
        assert ml_edges, f"expected a moneyline edge, got {edge_data}"
        assert ml_edges[0]["edge_percentage"] > 0
        assert ml_edges[0]["implied_probability"] == 0.6
        assert ml_edges[0]["sportsbook_key"] == "draftkings"

        # min_edge filter excludes everything at an absurd threshold
        filtered = client.get(f"/api/v1/predict/games/{game_id}/edges?min_edge=99")
        assert filtered.json()["data"]["edges"] == []

    def test_idempotency_replay_and_conflict(self, client, upstream) -> None:
        from tests.integration.conftest import mock_happy_path

        game_id = f"game-{uuid.uuid4()}"
        run_id = f"run-{uuid.uuid4()}"
        mock_happy_path(upstream, game_id, run_id)
        key = str(uuid.uuid4())

        first = post_predictions(client, game_id, run_id)
        assert first.status_code == 201
        replayed = client.post(
            "/api/v1/predict/predictions",
            json={"game_id": game_id, "simulation_run_id": run_id},
            headers={"X-Idempotency-Key": key},
        )
        assert replayed.status_code == 201

        again = client.post(
            "/api/v1/predict/predictions",
            json={"game_id": game_id, "simulation_run_id": run_id},
            headers={"X-Idempotency-Key": key},
        )
        assert again.status_code == 201
        assert again.json()["data"]["predictions"][0]["id"] == replayed.json()["data"]["predictions"][0]["id"]

        conflict = client.post(
            "/api/v1/predict/predictions",
            json={"game_id": game_id, "simulation_run_id": run_id, "market_types": ["SPREAD"]},
            headers={"X-Idempotency-Key": key},
        )
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "DUPLICATE_RESOURCE"

    def test_mismatched_simulation_run_rejected(self, client, upstream) -> None:
        from httpx import Response

        from tests.integration.conftest import SIM_URL, STATS_URL, enveloped, game_payload, simulation_run_payload

        game_id = f"game-{uuid.uuid4()}"
        run_id = f"run-{uuid.uuid4()}"
        upstream.get(f"{STATS_URL}/api/v1/stats/games/{game_id}").mock(
            return_value=Response(200, json=enveloped(game_payload(game_id)))
        )
        upstream.get(f"{SIM_URL}/api/v1/sim/simulations/{run_id}").mock(
            return_value=Response(200, json=enveloped(simulation_run_payload(run_id, "some-other-game")))
        )
        response = post_predictions(client, game_id, run_id)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "UNPROCESSABLE_ENTITY"

    def test_unknown_simulation_run_404(self, client, upstream) -> None:
        from httpx import Response

        from tests.integration.conftest import SIM_URL, STATS_URL, enveloped, game_payload

        game_id = f"game-{uuid.uuid4()}"
        run_id = f"run-{uuid.uuid4()}"
        upstream.get(f"{STATS_URL}/api/v1/stats/games/{game_id}").mock(
            return_value=Response(200, json=enveloped(game_payload(game_id)))
        )
        upstream.get(f"{SIM_URL}/api/v1/sim/simulations/{run_id}").mock(
            return_value=Response(404, json={"error": {"code": "RESOURCE_NOT_FOUND", "message": "no"}, "meta": {}})
        )
        assert post_predictions(client, game_id, run_id).status_code == 404


class TestModelsAndHealth:
    def test_bootstrap_registered_three_active_models(self, client) -> None:
        response = client.get("/api/v1/predict/models?is_active=true")
        assert response.status_code == 200
        models = response.json()["data"]
        assert {m["market_type"] for m in models} == {"SPREAD", "TOTAL", "MONEYLINE"}
        assert all(m["sport"] == "BASKETBALL" and m["is_active"] for m in models)
        assert all(m["version_tag"].startswith("NBA_unified_") for m in models)

        detail = client.get(f"/api/v1/predict/models/{models[0]['id']}")
        assert detail.status_code == 200
        assert detail.json()["data"]["feature_names"][0] == "sim_probability"

    def test_retrain_accepted(self, client) -> None:
        response = client.post("/api/v1/predict/models/retrain", json={"sport": "BASKETBALL", "market_type": "SPREAD"})
        assert response.status_code == 202
        assert response.json()["data"]["status"] == "started"

    def test_health_lists_active_models(self, client, upstream) -> None:
        from httpx import Response

        from tests.integration.conftest import LINES_URL, STATS_URL, enveloped

        upstream.get(f"{STATS_URL}/api/v1/stats/health").mock(
            return_value=Response(200, json=enveloped({"status": "healthy"}))
        )
        upstream.get(f"{LINES_URL}/api/v1/lines/health").mock(
            return_value=Response(200, json=enveloped({"status": "healthy"}))
        )
        response = client.get("/api/v1/predict/health")
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["service"] == "prediction-engine"
        assert data["dependencies"]["postgres"] == "healthy"
        assert set(data["active_models"]) == {"BASKETBALL_SPREAD", "BASKETBALL_TOTAL", "BASKETBALL_MONEYLINE"}
