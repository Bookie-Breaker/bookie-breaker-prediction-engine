"""SQLAlchemy Core table definitions matching schemas/database-schemas/prediction-engine.md.

The enum types (sport_enum, market_type_enum, league_enum) live in the
``public`` schema and are owned by infra-ops init-db scripts, so they are
referenced with create_type=False. DDL itself is applied by Alembic.
"""

import uuid
from typing import Any

from sqlalchemy import (
    TIMESTAMP,
    Boolean,
    CheckConstraint,
    Column,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    Table,
    Text,
    text,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import JSONB, TSTZRANGE, UUID

metadata = MetaData(schema="predictions")


# Values mirror infra-ops init-db/02-create-enums.sql; declared here so
# SQLAlchemy can bind and validate parameters (the types are NOT created by
# this service).
_ENUM_VALUES: dict[str, tuple[str, ...]] = {
    "sport_enum": ("FOOTBALL", "BASKETBALL", "BASEBALL"),
    "market_type_enum": ("SPREAD", "TOTAL", "MONEYLINE", "PLAYER_PROP", "TEAM_PROP", "GAME_PROP", "FUTURE", "LIVE"),
    "league_enum": ("NFL", "NBA", "MLB", "NCAA_FB", "NCAA_BB", "NCAA_BSB"),
}


def _enum(name: str) -> "postgresql.ENUM":
    return postgresql.ENUM(*_ENUM_VALUES[name], name=name, schema="public", create_type=False)


def _uuid_pk() -> Any:
    return Column(
        "id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()")
    )


model_versions = Table(
    "model_versions",
    metadata,
    _uuid_pk(),
    Column("sport", _enum("sport_enum"), nullable=False),
    Column("model_type", _enum("market_type_enum"), nullable=False),
    Column("version", Text, nullable=False),
    Column("algorithm", Text, nullable=False, server_default=text("'xgboost'")),
    Column("trained_at", TIMESTAMP(timezone=True), nullable=False),
    Column("training_data_range", TSTZRANGE, nullable=False),
    Column("training_samples", Integer, nullable=False),
    Column("evaluation_metrics", JSONB, nullable=False, server_default=text("'{}'")),
    Column("feature_names", JSONB, nullable=False, server_default=text("'[]'")),
    Column("is_active", Boolean, nullable=False, server_default=text("FALSE")),
    Column("artifact_path", Text, nullable=False),
    Column("notes", Text),
    Column("created_at", TIMESTAMP(timezone=True), nullable=False, server_default=text("NOW()")),
    Index(
        "uq_model_versions_active",
        "sport",
        "model_type",
        unique=True,
        postgresql_where=text("is_active = TRUE"),
    ),
    Index("idx_model_versions_sport_type", "sport", "model_type", text("created_at DESC")),
)

predictions = Table(
    "predictions",
    metadata,
    _uuid_pk(),
    Column("game_external_id", Text, nullable=False),
    Column("model_version_id", UUID(as_uuid=True), ForeignKey("model_versions.id"), nullable=False),
    Column("league", _enum("league_enum"), nullable=False),
    Column("market_type", _enum("market_type_enum"), nullable=False),
    Column("selection", Text, nullable=False),
    Column("predicted_probability", Numeric(6, 5), nullable=False),
    Column("simulation_probability", Numeric(6, 5)),
    Column("implied_probability", Numeric(6, 5)),
    Column("edge", Numeric(6, 5)),
    Column("confidence_lower", Numeric(6, 5)),
    Column("confidence_upper", Numeric(6, 5)),
    Column("feature_importance", JSONB, nullable=False, server_default=text("'{}'")),
    Column("created_at", TIMESTAMP(timezone=True), nullable=False, server_default=text("NOW()")),
    CheckConstraint(
        "predicted_probability >= 0 AND predicted_probability <= 1",
        name="chk_predictions_probability_range",
    ),
    CheckConstraint(
        "confidence_lower >= 0 AND confidence_upper <= 1 AND confidence_lower <= confidence_upper",
        name="chk_predictions_confidence_range",
    ),
    Index("idx_predictions_game_market", "game_external_id", "market_type", text("created_at DESC")),
    Index("idx_predictions_model_version", "model_version_id", text("created_at DESC")),
    Index("idx_predictions_created", text("created_at DESC")),
    Index("idx_predictions_league", "league", text("created_at DESC")),
)

feature_vectors = Table(
    "feature_vectors",
    metadata,
    _uuid_pk(),
    Column(
        "prediction_id",
        UUID(as_uuid=True),
        ForeignKey("predictions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    ),
    Column("features", JSONB, nullable=False),
    Column("feature_sources", JSONB),
    Column("created_at", TIMESTAMP(timezone=True), nullable=False, server_default=text("NOW()")),
    Index("idx_feature_vectors_created", text("created_at DESC")),
)
