"""Dependency health aggregation (services/health.py)."""

from prediction_engine.services.health import HealthService


class FakeDependency:
    def __init__(self, healthy: bool) -> None:
        self._healthy = healthy

    async def health(self) -> bool:
        return self._healthy


class FakeRepo:
    def __init__(self, healthy: bool) -> None:
        self._healthy = healthy

    async def is_healthy(self) -> bool:
        return self._healthy


class FakeRegistry:
    def active_map(self) -> dict[str, str]:
        return {"BASKETBALL/MONEYLINE": "v1"}


class PingingRedis:
    async def ping(self) -> bool:
        return True


class DeadRedis:
    async def ping(self) -> bool:
        raise ConnectionError("redis unreachable")


def make_service(stats: bool = True, lines: bool = True, db: bool = True, redis=None) -> HealthService:
    return HealthService(
        statistics=FakeDependency(stats),  # type: ignore[arg-type]
        lines=FakeDependency(lines),  # type: ignore[arg-type]
        repo=FakeRepo(db),  # type: ignore[arg-type]
        registry=FakeRegistry(),  # type: ignore[arg-type]
        redis_client=redis if redis is not None else PingingRedis(),  # type: ignore[arg-type]
    )


async def test_all_dependencies_up_reports_healthy() -> None:
    data = await make_service().health()
    assert data.status == "healthy"
    assert data.dependencies == {
        "statistics_service": "healthy",
        "lines_service": "healthy",
        "redis": "healthy",
        "postgres": "healthy",
    }
    assert data.uptime_seconds >= 0
    assert data.active_models == {"BASKETBALL/MONEYLINE": "v1"}


async def test_redis_failure_degrades_only_redis() -> None:
    data = await make_service(redis=DeadRedis()).health()
    assert data.status == "degraded"
    assert data.dependencies["redis"] == "unhealthy"
    assert data.dependencies["postgres"] == "healthy"


async def test_any_single_dependency_failure_degrades_overall_status() -> None:
    data = await make_service(lines=False).health()
    assert data.status == "degraded"
    assert data.dependencies["lines_service"] == "unhealthy"
    assert data.dependencies["statistics_service"] == "healthy"
