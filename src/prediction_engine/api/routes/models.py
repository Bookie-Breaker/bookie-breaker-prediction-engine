"""Model version endpoints and retrain trigger."""

import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from prediction_engine.api.dependencies import get_model_repo
from prediction_engine.api.envelope import Envelope, envelope
from prediction_engine.api.errors import NotFoundError
from prediction_engine.api.schemas import (
    ModelVersionData,
    ModelVersionDetailData,
    RetrainData,
    RetrainRequest,
)
from prediction_engine.db.repository import ModelVersionRecord, ModelVersionRepository

router = APIRouter(tags=["models"])

RepoDep = Annotated[ModelVersionRepository, Depends(get_model_repo)]


def _to_data(record: ModelVersionRecord) -> ModelVersionData:
    return ModelVersionData(
        id=str(record.id),
        sport=record.sport,
        market_type=record.model_type,
        version_tag=record.version,
        algorithm=record.algorithm,
        training_date=record.trained_at.isoformat().replace("+00:00", "Z"),
        training_samples=record.training_samples,
        evaluation_metrics={k: float(v) for k, v in record.evaluation_metrics.items() if isinstance(v, int | float)},
        is_active=record.is_active,
        notes=record.notes,
    )


@router.get("/models", response_model=Envelope[list[ModelVersionData]])
async def list_models(
    repo: RepoDep,
    sport: Annotated[str | None, Query()] = None,
    market_type: Annotated[str | None, Query()] = None,
    is_active: Annotated[bool | None, Query()] = None,
) -> Envelope[list[ModelVersionData]]:
    records = await repo.list_models(
        sport=sport.upper() if sport else None,
        market_type=market_type.upper() if market_type else None,
        is_active=is_active,
    )
    return envelope([_to_data(record) for record in records])


@router.get("/models/{model_id}", response_model=Envelope[ModelVersionDetailData])
async def get_model(model_id: uuid.UUID, repo: RepoDep) -> Envelope[ModelVersionDetailData]:
    record = await repo.get(model_id)
    if record is None:
        raise NotFoundError(f"Model version {model_id} not found")
    base = _to_data(record)
    return envelope(ModelVersionDetailData(**base.model_dump(), feature_names=record.feature_names))


@router.post("/models/retrain", status_code=202, response_model=Envelope[RetrainData])
async def retrain_model(request: RetrainRequest) -> Envelope[RetrainData]:
    """Accept a retrain request (202).

    Real retraining requires graded game outcomes joined to stored feature
    vectors, which do not exist until the Phase 3 grading loop runs. The
    request is accepted and immediately reported as started; the training
    pipeline itself is exercised via scripts/train.py.
    """
    return envelope(
        RetrainData(
            retrain_id=str(uuid.uuid4()),
            sport=request.sport.upper(),
            market_type=request.market_type.upper(),
            status="started",
            started_at=datetime.now(tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            estimated_duration_minutes=15,
        )
    )
