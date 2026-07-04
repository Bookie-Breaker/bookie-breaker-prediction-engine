"""Initial predictions schema: model_versions, predictions, feature_vectors.

DDL follows schemas/database-schemas/prediction-engine.md verbatim. The
sport_enum/market_type_enum/league_enum types are owned by infra-ops
init-db scripts in the public schema and are referenced, not created.

Revision ID: 0001
Revises:
Create Date: 2026-07-04

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _enum(name: str) -> postgresql.ENUM:
    return postgresql.ENUM(name=name, schema="public", create_type=False)


def upgrade() -> None:
    op.create_table(
        "model_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("sport", _enum("sport_enum"), nullable=False),
        sa.Column("model_type", _enum("market_type_enum"), nullable=False),
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column("algorithm", sa.Text(), nullable=False, server_default=sa.text("'xgboost'")),
        sa.Column("trained_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("training_data_range", postgresql.TSTZRANGE(), nullable=False),
        sa.Column("training_samples", sa.Integer(), nullable=False),
        sa.Column("evaluation_metrics", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("feature_names", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("FALSE")),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("notes", sa.Text()),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        schema="predictions",
    )
    op.create_index(
        "uq_model_versions_active",
        "model_versions",
        ["sport", "model_type"],
        unique=True,
        schema="predictions",
        postgresql_where=sa.text("is_active = TRUE"),
    )
    op.create_index(
        "idx_model_versions_sport_type",
        "model_versions",
        ["sport", "model_type", sa.text("created_at DESC")],
        schema="predictions",
    )

    op.create_table(
        "predictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("game_external_id", sa.Text(), nullable=False),
        sa.Column(
            "model_version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("predictions.model_versions.id"),
            nullable=False,
        ),
        sa.Column("league", _enum("league_enum"), nullable=False),
        sa.Column("market_type", _enum("market_type_enum"), nullable=False),
        sa.Column("selection", sa.Text(), nullable=False),
        sa.Column("predicted_probability", sa.Numeric(6, 5), nullable=False),
        sa.Column("simulation_probability", sa.Numeric(6, 5)),
        sa.Column("implied_probability", sa.Numeric(6, 5)),
        sa.Column("edge", sa.Numeric(6, 5)),
        sa.Column("confidence_lower", sa.Numeric(6, 5)),
        sa.Column("confidence_upper", sa.Numeric(6, 5)),
        sa.Column("feature_importance", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.CheckConstraint(
            "predicted_probability >= 0 AND predicted_probability <= 1",
            name="chk_predictions_probability_range",
        ),
        sa.CheckConstraint(
            "confidence_lower >= 0 AND confidence_upper <= 1 AND confidence_lower <= confidence_upper",
            name="chk_predictions_confidence_range",
        ),
        schema="predictions",
    )
    op.create_index(
        "idx_predictions_game_market",
        "predictions",
        ["game_external_id", "market_type", sa.text("created_at DESC")],
        schema="predictions",
    )
    op.create_index(
        "idx_predictions_model_version",
        "predictions",
        ["model_version_id", sa.text("created_at DESC")],
        schema="predictions",
    )
    op.create_index("idx_predictions_created", "predictions", [sa.text("created_at DESC")], schema="predictions")
    op.create_index(
        "idx_predictions_league", "predictions", ["league", sa.text("created_at DESC")], schema="predictions"
    )

    op.create_table(
        "feature_vectors",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "prediction_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("predictions.predictions.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("features", postgresql.JSONB(), nullable=False),
        sa.Column("feature_sources", postgresql.JSONB()),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        schema="predictions",
    )
    op.create_index("idx_feature_vectors_prediction", "feature_vectors", ["prediction_id"], schema="predictions")
    op.create_index(
        "idx_feature_vectors_created", "feature_vectors", [sa.text("created_at DESC")], schema="predictions"
    )


def downgrade() -> None:
    op.drop_table("feature_vectors", schema="predictions")
    op.drop_table("predictions", schema="predictions")
    op.drop_table("model_versions", schema="predictions")
