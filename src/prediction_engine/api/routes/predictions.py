"""Prediction endpoints per api-contracts/prediction-engine-api.md."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query

from prediction_engine.api.dependencies import get_prediction_repo, get_predictor
from prediction_engine.api.envelope import Envelope, envelope
from prediction_engine.api.errors import NotFoundError
from prediction_engine.api.schemas import (
    FeatureVectorData,
    LatestPredictionsData,
    PredictionDetailData,
    PredictionGroupData,
    PredictionRequest,
)
from prediction_engine.core.predictor import Predictor, _to_item
from prediction_engine.db.repository import PredictionRepository

router = APIRouter(tags=["predictions"])

PredictorDep = Annotated[Predictor, Depends(get_predictor)]
RepoDep = Annotated[PredictionRepository, Depends(get_prediction_repo)]


@router.post("/predictions", status_code=201, response_model=Envelope[PredictionGroupData])
async def create_predictions(
    request: PredictionRequest,
    predictor: PredictorDep,
    x_idempotency_key: Annotated[str | None, Header()] = None,
) -> Envelope[PredictionGroupData]:
    return envelope(await predictor.create_predictions(request, idempotency_key=x_idempotency_key))


@router.get("/predictions/{prediction_id}", response_model=Envelope[PredictionDetailData])
async def get_prediction(prediction_id: uuid.UUID, repo: RepoDep) -> Envelope[PredictionDetailData]:
    found = await repo.get(prediction_id)
    if found is None:
        raise NotFoundError(f"Prediction {prediction_id} not found")
    record, vector, sources = found
    item = _to_item(record)
    return envelope(
        PredictionDetailData(
            id=item.id,
            game_id=record.game_external_id,
            model_version_id=item.model_version_id,
            market_type=item.market_type,
            selection=item.selection,
            predicted_probability=item.predicted_probability,
            simulation_probability=item.simulation_probability,
            adjustment_magnitude=item.adjustment_magnitude,
            confidence_lower=item.confidence_lower,
            confidence_upper=item.confidence_upper,
            feature_importance=item.feature_importance,
            feature_vector=FeatureVectorData(
                id=vector["id"],
                features=vector["features"],
                feature_source_versions=sources,
            ),
            created_at=item.created_at,
        )
    )


@router.get("/games/{game_id}/latest", response_model=Envelope[LatestPredictionsData])
async def latest_predictions(
    game_id: str,
    repo: RepoDep,
    market_type: Annotated[str | None, Query()] = None,
    model_version: Annotated[uuid.UUID | None, Query()] = None,
) -> Envelope[LatestPredictionsData]:
    markets = [m.strip().upper() for m in market_type.split(",")] if market_type else None
    records = await repo.latest_for_game(game_id, market_types=markets, model_version_id=model_version)
    if not records:
        raise NotFoundError(f"No predictions exist for game {game_id}")
    return envelope(LatestPredictionsData(game_id=game_id, predictions=[_to_item(r) for r in records]))
