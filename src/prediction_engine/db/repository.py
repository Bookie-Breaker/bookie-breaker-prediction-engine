"""Repositories over the predictions schema (SQLAlchemy Core, async)."""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Row, insert, select, update
from sqlalchemy.dialects.postgresql import Range
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from prediction_engine.db.tables import feature_vectors, model_versions, predictions


@dataclass(frozen=True)
class ModelVersionRecord:
    id: uuid.UUID
    sport: str
    model_type: str
    version: str
    algorithm: str
    trained_at: datetime
    training_samples: int
    evaluation_metrics: dict[str, Any]
    feature_names: list[str]
    is_active: bool
    artifact_path: str
    notes: str | None


@dataclass(frozen=True)
class PredictionRecord:
    id: uuid.UUID
    game_external_id: str
    model_version_id: uuid.UUID
    league: str
    market_type: str
    side: str | None
    selection: str
    predicted_probability: float
    simulation_probability: float | None
    implied_probability: float | None
    edge: float | None
    confidence_lower: float | None
    confidence_upper: float | None
    feature_importance: dict[str, float]
    created_at: datetime


def _model_version_from_row(row: Row[Any]) -> ModelVersionRecord:
    return ModelVersionRecord(
        id=row.id,
        sport=row.sport,
        model_type=row.model_type,
        version=row.version,
        algorithm=row.algorithm,
        trained_at=row.trained_at,
        training_samples=row.training_samples,
        evaluation_metrics=dict(row.evaluation_metrics),
        feature_names=list(row.feature_names),
        is_active=row.is_active,
        artifact_path=row.artifact_path,
        notes=row.notes,
    )


def _prediction_from_row(row: Row[Any]) -> PredictionRecord:
    return PredictionRecord(
        id=row.id,
        game_external_id=row.game_external_id,
        model_version_id=row.model_version_id,
        league=row.league,
        market_type=row.market_type,
        side=row.side,
        selection=row.selection,
        predicted_probability=float(row.predicted_probability),
        simulation_probability=float(row.simulation_probability) if row.simulation_probability is not None else None,
        implied_probability=float(row.implied_probability) if row.implied_probability is not None else None,
        edge=float(row.edge) if row.edge is not None else None,
        confidence_lower=float(row.confidence_lower) if row.confidence_lower is not None else None,
        confidence_upper=float(row.confidence_upper) if row.confidence_upper is not None else None,
        feature_importance=dict(row.feature_importance),
        created_at=row.created_at,
    )


class ModelVersionRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def list_models(
        self,
        sport: str | None = None,
        market_type: str | None = None,
        is_active: bool | None = None,
    ) -> list[ModelVersionRecord]:
        stmt = select(model_versions).order_by(model_versions.c.created_at.desc())
        if sport is not None:
            stmt = stmt.where(model_versions.c.sport == sport)
        if market_type is not None:
            stmt = stmt.where(model_versions.c.model_type == market_type)
        if is_active is not None:
            stmt = stmt.where(model_versions.c.is_active == is_active)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).fetchall()
        return [_model_version_from_row(row) for row in rows]

    async def get(self, model_id: uuid.UUID) -> ModelVersionRecord | None:
        stmt = select(model_versions).where(model_versions.c.id == model_id)
        async with self._engine.connect() as conn:
            row = (await conn.execute(stmt)).one_or_none()
        return _model_version_from_row(row) if row is not None else None

    async def active_models(self) -> list[ModelVersionRecord]:
        return await self.list_models(is_active=True)

    async def register(
        self,
        sport: str,
        market_types: list[str],
        version: str,
        trained_at: datetime,
        training_range: tuple[datetime, datetime],
        training_samples: int,
        evaluation_metrics: dict[str, Any],
        feature_names: list[str],
        artifact_path: str,
        notes: str | None = None,
        activate: bool = True,
    ) -> list[ModelVersionRecord]:
        """Register one row per market type sharing a single artifact.

        The unified model covers all market types (market type is a feature),
        so SPREAD/TOTAL/MONEYLINE rows share artifact_path. When activating,
        previous active rows for the same (sport, market_type) are
        deactivated in the same transaction to satisfy the partial unique
        index.
        """
        records: list[ModelVersionRecord] = []
        async with self._engine.begin() as conn:
            for market_type in market_types:
                if activate:
                    await conn.execute(
                        update(model_versions)
                        .where(
                            model_versions.c.sport == sport,
                            model_versions.c.model_type == market_type,
                            model_versions.c.is_active == True,  # noqa: E712
                        )
                        .values(is_active=False)
                    )
                result = await conn.execute(
                    insert(model_versions)
                    .values(
                        sport=sport,
                        model_type=market_type,
                        version=version,
                        algorithm="xgboost",
                        trained_at=trained_at,
                        training_data_range=Range(training_range[0], training_range[1]),
                        training_samples=training_samples,
                        evaluation_metrics=evaluation_metrics,
                        feature_names=feature_names,
                        is_active=activate,
                        artifact_path=artifact_path,
                        notes=notes,
                    )
                    .returning(model_versions)
                )
                records.append(_model_version_from_row(result.one()))
        return records


class PredictionRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def insert_predictions(
        self,
        rows: list[dict[str, Any]],
        features: dict[str, Any],
        feature_sources: dict[str, Any],
    ) -> list[PredictionRecord]:
        """Insert predictions and their shared feature vector atomically."""
        records: list[PredictionRecord] = []
        async with self._engine.begin() as conn:
            for row in rows:
                result = await conn.execute(insert(predictions).values(**row).returning(predictions))
                record = _prediction_from_row(result.one())
                records.append(record)
                await conn.execute(
                    insert(feature_vectors).values(
                        prediction_id=record.id,
                        features=features,
                        feature_sources=feature_sources,
                    )
                )
        return records

    async def get(self, prediction_id: uuid.UUID) -> tuple[PredictionRecord, dict[str, Any], dict[str, Any]] | None:
        stmt = (
            select(
                predictions,
                feature_vectors.c.id.label("fv_id"),
                feature_vectors.c.features,
                feature_vectors.c.feature_sources,
            )
            .join(feature_vectors, feature_vectors.c.prediction_id == predictions.c.id, isouter=True)
            .where(predictions.c.id == prediction_id)
        )
        async with self._engine.connect() as conn:
            row = (await conn.execute(stmt)).one_or_none()
        if row is None:
            return None
        vector = {"id": str(row.fv_id), "features": dict(row.features or {})}
        sources = dict(row.feature_sources or {})
        return _prediction_from_row(row), vector, sources

    async def latest_for_game(
        self,
        game_external_id: str,
        market_types: list[str] | None = None,
        model_version_id: uuid.UUID | None = None,
    ) -> list[PredictionRecord]:
        """Most recent prediction per (market type, side) for a game.

        Side is part of the distinct key so three-way moneyline batches
        (HOME/DRAW/AWAY rows sharing one market type, ADR-027) are returned
        in full; two-way markets emit exactly one side per batch, so their
        behavior is unchanged.
        """
        stmt = (
            select(predictions)
            .where(predictions.c.game_external_id == game_external_id)
            .order_by(predictions.c.market_type, predictions.c.side, predictions.c.created_at.desc())
            .distinct(predictions.c.market_type, predictions.c.side)
        )
        if market_types:
            stmt = stmt.where(predictions.c.market_type.in_(market_types))
        if model_version_id is not None:
            stmt = stmt.where(predictions.c.model_version_id == model_version_id)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).fetchall()
        return [_prediction_from_row(row) for row in rows]

    async def is_healthy(self) -> bool:
        try:
            async with self._engine.connect() as conn:
                await conn.execute(select(1))
            return True
        except Exception:  # noqa: BLE001 - any DB failure means unhealthy
            return False


async def ping(conn: AsyncConnection) -> None:
    await conn.execute(select(1))


def utc_now() -> datetime:
    return datetime.now(tz=UTC)
