"""Edge convenience endpoint combining predictions with current lines."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from prediction_engine.api.dependencies import get_predictor
from prediction_engine.api.envelope import Envelope, envelope
from prediction_engine.api.schemas import EdgesData
from prediction_engine.core.predictor import Predictor

router = APIRouter(tags=["edges"])


@router.get("/games/{game_id}/edges", response_model=Envelope[EdgesData])
async def edges_for_game(
    game_id: str,
    predictor: Annotated[Predictor, Depends(get_predictor)],
    min_edge: Annotated[float, Query()] = 0.0,
    market_type: Annotated[str | None, Query()] = None,
) -> Envelope[EdgesData]:
    market = market_type.strip().upper() if market_type else None
    return envelope(await predictor.edges_for_game(game_id, min_edge=min_edge, market_type=market))
