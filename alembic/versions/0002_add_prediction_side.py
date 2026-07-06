"""Add the nullable side column to predictions (ADR-027).

Three-way moneylines emit HOME/DRAW/AWAY rows; every new prediction row
carries its side (HOME for spread/moneyline home rows, OVER for totals).
Rows created before Phase 6 keep NULL.

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-05

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("predictions", sa.Column("side", sa.Text(), nullable=True), schema="predictions")
    op.create_check_constraint(
        "chk_predictions_side",
        "predictions",
        "side IN ('HOME', 'AWAY', 'DRAW', 'OVER', 'UNDER')",
        schema="predictions",
    )


def downgrade() -> None:
    op.drop_constraint("chk_predictions_side", "predictions", schema="predictions", type_="check")
    op.drop_column("predictions", "side", schema="predictions")
