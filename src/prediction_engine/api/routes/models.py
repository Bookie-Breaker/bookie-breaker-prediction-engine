"""Model version endpoints, retraining jobs, and champion/challenger experiments."""

import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Path, Query

from prediction_engine.api.dependencies import get_experiment_service, get_model_repo, get_retrain_service
from prediction_engine.api.envelope import Envelope, envelope
from prediction_engine.api.errors import NotFoundError, UnprocessableError
from prediction_engine.api.schemas import (
    ExperimentData,
    ExperimentModelData,
    ModelVersionData,
    ModelVersionDetailData,
    PromoteRequest,
    PromotionData,
    RetrainAcceptedData,
    RetrainRequest,
    RetrainStatusData,
)
from prediction_engine.core.experiments import ExperimentReport
from prediction_engine.core.features.registry import get_features
from prediction_engine.core.model.registry import MARKET_TYPES
from prediction_engine.db.repository import ModelVersionRecord, ModelVersionRepository
from prediction_engine.services.experiments import ExperimentService
from prediction_engine.services.retrain import RetrainService

router = APIRouter(tags=["models"])

RepoDep = Annotated[ModelVersionRepository, Depends(get_model_repo)]
RetrainDep = Annotated[RetrainService, Depends(get_retrain_service)]
ExperimentDep = Annotated[ExperimentService, Depends(get_experiment_service)]


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
        role=record.role,
    )


def _validated_sport(sport: str) -> str:
    """Uppercase and validate a model key (sport or NCAA_BB) against the feature registry."""
    key = sport.upper()
    try:
        get_features(key)
    except ValueError as exc:
        raise UnprocessableError(str(exc)) from exc
    return key


def _validated_market(market_type: str) -> str:
    market = market_type.upper()
    if market not in MARKET_TYPES:
        raise UnprocessableError(f"market_type must be one of {MARKET_TYPES}, got {market_type!r}")
    return market


@router.get("/models", response_model=Envelope[list[ModelVersionData]])
async def list_models(
    repo: RepoDep,
    sport: Annotated[str | None, Query(description="Filter by sport (e.g. BASKETBALL).")] = None,
    market_type: Annotated[str | None, Query(description="Filter by market type.")] = None,
    is_active: Annotated[bool | None, Query(description="Filter by active status.")] = None,
    role: Annotated[str | None, Query(description="Filter by role (champion, challenger, shadow).")] = None,
) -> Envelope[list[ModelVersionData]]:
    """List model versions, newest first."""
    records = await repo.list_models(
        sport=sport.upper() if sport else None,
        market_type=market_type.upper() if market_type else None,
        is_active=is_active,
        role=role.lower() if role else None,
    )
    return envelope([_to_data(record) for record in records])


@router.post("/models/retrain", status_code=202, response_model=Envelope[RetrainAcceptedData])
async def retrain_model(
    request: RetrainRequest, retrainer: RetrainDep, background_tasks: BackgroundTasks
) -> Envelope[RetrainAcceptedData]:
    """Enqueue a real retraining job (202); poll GET /models/retrain/{job_id}.

    The job assembles graded outcomes from stored predictions + feature
    vectors, trains walk-forward, and registers the result as an ACTIVE
    CHALLENGER that shadow-scores until promoted.
    """
    if request.market == "player_prop":
        raise UnprocessableError(
            "player_prop retraining is deferred: prop rows settle against per-player box scores, "
            "which the assembly pipeline does not join yet (v1 retrains game markets only)"
        )
    sport = _validated_sport(request.sport)
    job_id = await retrainer.start(sport, request.ensemble, request.min_rows)
    background_tasks.add_task(retrainer.run, job_id, sport, request.ensemble, request.min_rows)
    return envelope(RetrainAcceptedData(job_id=job_id, sport=sport))


@router.get("/models/retrain/{job_id}", response_model=Envelope[RetrainStatusData])
async def retrain_status(
    job_id: Annotated[str, Path(description="The retrain job identifier.")], retrainer: RetrainDep
) -> Envelope[RetrainStatusData]:
    """Get a retrain job's status (kept for 24 hours)."""
    status = await retrainer.status(job_id)
    if status is None:
        raise NotFoundError(f"Retrain job {job_id} not found (statuses expire after 24 hours)")
    return envelope(RetrainStatusData.model_validate(status))


def _experiment_data(report: ExperimentReport) -> tuple[ExperimentModelData, ExperimentModelData]:
    champion = ExperimentModelData(
        model_version_id=report.champion.model_version_id,
        version_tag="",
        role="champion",
        brier_score=report.champion.brier_score,
        log_loss=report.champion.log_loss,
        calibration_error=report.champion.calibration_error,
    )
    challenger = ExperimentModelData(
        model_version_id=report.challenger.model_version_id,
        version_tag="",
        role="challenger",
        brier_score=report.challenger.brier_score,
        log_loss=report.challenger.log_loss,
        calibration_error=report.challenger.calibration_error,
    )
    return champion, challenger


@router.get("/models/experiments/{sport}/{market_type}", response_model=Envelope[ExperimentData])
async def get_experiment(
    sport: Annotated[str, Path(description="Model key (sport, or NCAA_BB).")],
    market_type: Annotated[str, Path(description="Game market type (SPREAD, TOTAL, MONEYLINE).")],
    experiments: ExperimentDep,
    repo: RepoDep,
) -> Envelope[ExperimentData]:
    """Champion vs challenger over graded shadow pairs (404 without an active challenger)."""
    sport_key = _validated_sport(sport)
    market = _validated_market(market_type)
    report = await experiments.report(sport_key, market)
    champion, challenger = _experiment_data(report)
    champion_record = await repo.get(uuid.UUID(report.champion.model_version_id))
    challenger_record = await repo.get(uuid.UUID(report.challenger.model_version_id))
    if champion_record is not None:
        champion = champion.model_copy(update={"version_tag": champion_record.version})
    if challenger_record is not None:
        challenger = challenger.model_copy(update={"version_tag": challenger_record.version})
    return envelope(
        ExperimentData(
            sport=sport_key,
            market_type=market,
            graded_pairs=report.graded_pairs,
            champion=champion,
            challenger=challenger,
            promotion_ready=report.promotion_ready,
            blockers=report.blockers,
        )
    )


@router.post("/models/experiments/{sport}/{market_type}/promote", response_model=Envelope[PromotionData])
async def promote_challenger(
    sport: Annotated[str, Path(description="Model key (sport, or NCAA_BB).")],
    market_type: Annotated[str, Path(description="Game market type the criteria are evaluated on.")],
    request: PromoteRequest,
    experiments: ExperimentDep,
) -> Envelope[PromotionData]:
    """Promote the active challenger to champion (atomic role flip across the unified model's markets).

    Fails 422 with the blocking criteria unless force=true. The outgoing
    champion is retired (deactivated, role=shadow).
    """
    sport_key = _validated_sport(sport)
    market = _validated_market(market_type)
    records = await experiments.promote(sport_key, market, force=request.force)
    return envelope(
        PromotionData(
            sport=sport_key,
            market_type=market,
            promoted_version=records[0].version,
            model_version_ids=[str(record.id) for record in records],
            forced=request.force,
        )
    )


@router.get("/models/{model_id}", response_model=Envelope[ModelVersionDetailData])
async def get_model(
    model_id: Annotated[uuid.UUID, Path(description="The model version identifier.")], repo: RepoDep
) -> Envelope[ModelVersionDetailData]:
    """Get a model version including its ordered feature names."""
    record = await repo.get(model_id)
    if record is None:
        raise NotFoundError(f"Model version {model_id} not found")
    base = _to_data(record)
    return envelope(ModelVersionDetailData(**base.model_dump(), feature_names=record.feature_names))
