"""Health aggregation across dependencies and active models."""

import asyncio
import time

import redis.asyncio as aioredis

from prediction_engine import __version__
from prediction_engine.api.schemas import HealthData
from prediction_engine.clients.lines import LinesClient
from prediction_engine.clients.statistics import StatisticsClient
from prediction_engine.core.model.registry import ModelRegistry
from prediction_engine.db.repository import PredictionRepository


class HealthService:
    def __init__(
        self,
        statistics: StatisticsClient,
        lines: LinesClient,
        repo: PredictionRepository,
        registry: ModelRegistry,
        redis_client: "aioredis.Redis",
    ) -> None:
        self._statistics = statistics
        self._lines = lines
        self._repo = repo
        self._registry = registry
        self._redis = redis_client
        self._started = time.monotonic()

    async def _redis_ok(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except Exception:  # noqa: BLE001 - any redis failure means unhealthy
            return False

    async def health(self) -> HealthData:
        stats_ok, lines_ok, redis_ok, db_ok = await asyncio.gather(
            self._statistics.health(), self._lines.health(), self._redis_ok(), self._repo.is_healthy()
        )
        healthy = stats_ok and lines_ok and redis_ok and db_ok
        return HealthData(
            status="healthy" if healthy else "degraded",
            version=__version__,
            uptime_seconds=int(time.monotonic() - self._started),
            dependencies={
                "statistics_service": "healthy" if stats_ok else "unhealthy",
                "lines_service": "healthy" if lines_ok else "unhealthy",
                "redis": "healthy" if redis_ok else "unhealthy",
                "postgres": "healthy" if db_ok else "unhealthy",
            },
            active_models=self._registry.active_map(),
        )
