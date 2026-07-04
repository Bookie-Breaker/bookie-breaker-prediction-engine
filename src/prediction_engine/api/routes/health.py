"""Health endpoint with dependency status and the active model map."""

from typing import Annotated

from fastapi import APIRouter, Depends

from prediction_engine.api.dependencies import get_health_service
from prediction_engine.api.envelope import Envelope, envelope
from prediction_engine.api.schemas import HealthData
from prediction_engine.services.health import HealthService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=Envelope[HealthData])
async def get_health(service: Annotated[HealthService, Depends(get_health_service)]) -> Envelope[HealthData]:
    """Liveness plus dependency status. Returns 200 even when degraded so
    container healthchecks measure this service, not its dependencies."""
    return envelope(await service.health())
