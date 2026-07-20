"""Add model_versions.role and predictions.is_shadow (Phase 7 Wave 4).

Champion/challenger serving: every model_versions row gains a role
(champion | challenger | shadow) and the one-active-per-(sport, model_type)
partial unique index widens to (sport, model_type, role), so a champion and
a challenger can be active side by side. Existing rows become champions
(the served role), preserving current behavior exactly.

Shadow scoring writes a second predictions row per emitted prediction with
the challenger's model_version_id; is_shadow marks those rows so every read
path (latest, edges) can exclude them.

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-19

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "model_versions",
        sa.Column("role", sa.Text(), nullable=False, server_default=sa.text("'champion'")),
        schema="predictions",
    )
    op.create_check_constraint(
        "chk_model_versions_role",
        "model_versions",
        "role IN ('champion', 'challenger', 'shadow')",
        schema="predictions",
    )
    op.drop_index("uq_model_versions_active", table_name="model_versions", schema="predictions")
    op.create_index(
        "uq_model_versions_active",
        "model_versions",
        ["sport", "model_type", "role"],
        unique=True,
        schema="predictions",
        postgresql_where=sa.text("is_active = TRUE"),
    )
    op.add_column(
        "predictions",
        sa.Column("is_shadow", sa.Boolean(), nullable=False, server_default=sa.text("FALSE")),
        schema="predictions",
    )


def downgrade() -> None:
    op.drop_column("predictions", "is_shadow", schema="predictions")
    op.drop_index("uq_model_versions_active", table_name="model_versions", schema="predictions")
    # The narrower pre-Wave-4 index cannot build while a champion AND a
    # challenger are active for the same (sport, model_type); active
    # non-champions are deactivated (the pre-Wave-4 serving convention)
    # rather than deleting model history.
    op.execute("UPDATE predictions.model_versions SET is_active = FALSE WHERE is_active AND role <> 'champion'")
    op.create_index(
        "uq_model_versions_active",
        "model_versions",
        ["sport", "model_type"],
        unique=True,
        schema="predictions",
        postgresql_where=sa.text("is_active = TRUE"),
    )
    op.drop_constraint("chk_model_versions_role", "model_versions", schema="predictions", type_="check")
    op.drop_column("model_versions", "role", schema="predictions")
