"""prediction.completed event publishing (events/publisher.py)."""

import json
import logging

import pytest

from prediction_engine.events.publisher import CHANNEL, publish_prediction_completed


class CapturingRedis:
    def __init__(self) -> None:
        self.published: list[tuple[str, str]] = []

    async def publish(self, channel: str, payload: str) -> None:
        self.published.append((channel, payload))


class ExplodingRedis:
    async def publish(self, channel: str, payload: str) -> None:
        raise ConnectionError("redis is down")


async def test_publishes_the_documented_payload_on_the_channel() -> None:
    redis = CapturingRedis()
    await publish_prediction_completed(
        redis,  # type: ignore[arg-type]
        batch_id="batch-1",
        game_ids=["game-1", "game-2"],
        league="NBA",
        market_types=["SPREAD", "TOTAL"],
        predictions_count=4,
        edges_found=1,
    )

    assert len(redis.published) == 1
    channel, raw = redis.published[0]
    assert channel == CHANNEL
    payload = json.loads(raw)
    assert payload["event"] == "prediction.completed"
    assert payload["batch_id"] == "batch-1"
    assert payload["game_ids"] == ["game-1", "game-2"]
    assert payload["league"] == "NBA"
    assert payload["market_types"] == ["SPREAD", "TOTAL"]
    assert payload["predictions_count"] == 4
    assert payload["edges_found"] == 1
    assert payload["timestamp"].endswith("Z")


async def test_publish_failures_are_swallowed_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    # fire-and-forget: a dead redis must never fail the request path
    with caplog.at_level(logging.WARNING, logger="prediction_engine.events.publisher"):
        await publish_prediction_completed(
            ExplodingRedis(),  # type: ignore[arg-type]
            batch_id="batch-2",
            game_ids=[],
            league="NBA",
            market_types=[],
            predictions_count=0,
            edges_found=0,
        )
    assert any("failed to publish" in record.message for record in caplog.records)
