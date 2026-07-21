"""Retrain job status lifecycle and failure handling (services/retrain.py)."""

import json
from pathlib import Path

import pytest

from prediction_engine.services import retrain as retrain_module
from prediction_engine.services.retrain import RETRAIN_STATUS_PREFIX, RetrainService


class DictRedis:
    """Minimal get/set redis stand-in recording TTLs."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int | None] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value
        self.ttls[key] = ex


def make_service(redis: DictRedis, status_ttl_seconds: int = 86_400) -> RetrainService:
    return RetrainService(
        model_repo=object(),  # type: ignore[arg-type]
        prediction_repo=object(),  # type: ignore[arg-type]
        statistics=object(),  # type: ignore[arg-type]
        redis_client=redis,  # type: ignore[arg-type]
        model_dir=Path("/nonexistent"),
        status_ttl_seconds=status_ttl_seconds,
    )


async def test_status_is_none_for_unknown_job() -> None:
    assert await make_service(DictRedis()).status("no-such-job") is None


async def test_start_writes_a_queued_status_with_the_configured_ttl() -> None:
    redis = DictRedis()
    service = make_service(redis, status_ttl_seconds=1234)
    job_id = await service.start("BASKETBALL", ensemble=True, min_rows=500)

    status = await service.status(job_id)
    assert status is not None
    assert status["status"] == "queued"
    assert status["sport"] == "BASKETBALL"
    assert status["ensemble"] is True
    assert status["min_rows"] == 500
    assert redis.ttls[f"{RETRAIN_STATUS_PREFIX}{job_id}"] == 1234


async def test_run_failure_lands_in_the_status_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # job failures must be observable via GET /models/retrain/{job_id}, not
    # just the server log
    async def exploding_assemble(*args, **kwargs):
        raise RuntimeError("statistics-service is down")

    monkeypatch.setattr(retrain_module, "assemble_training_rows", exploding_assemble)
    redis = DictRedis()
    service = make_service(redis)
    job_id = await service.start("BASKETBALL", ensemble=False, min_rows=100)

    await service.run(job_id, "BASKETBALL", ensemble=False, min_rows=100)

    status = json.loads(redis.store[f"{RETRAIN_STATUS_PREFIX}{job_id}"])
    assert status["status"] == "failed"
    assert "statistics-service is down" in status["detail"]
    assert status["sport"] == "BASKETBALL"
