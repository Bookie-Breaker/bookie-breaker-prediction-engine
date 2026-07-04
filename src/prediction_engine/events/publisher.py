"""Publish prediction.completed events per redis-schemas.md.

Fire-and-forget: publish failures are logged and never fail the request.
"""

import json
import logging
from datetime import UTC, datetime

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

CHANNEL = "events:prediction.completed"


async def publish_prediction_completed(
    redis_client: "aioredis.Redis",
    batch_id: str,
    game_ids: list[str],
    league: str,
    market_types: list[str],
    predictions_count: int,
    edges_found: int,
) -> None:
    payload = {
        "event": "prediction.completed",
        "timestamp": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "batch_id": batch_id,
        "game_ids": game_ids,
        "league": league,
        "market_types": market_types,
        "predictions_count": predictions_count,
        "edges_found": edges_found,
    }
    try:
        await redis_client.publish(CHANNEL, json.dumps(payload))
    except Exception:  # noqa: BLE001 - pub/sub is best-effort by design
        logger.warning("failed to publish %s for batch %s", CHANNEL, batch_id, exc_info=True)
