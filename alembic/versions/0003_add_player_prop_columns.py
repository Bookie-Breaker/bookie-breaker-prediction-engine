"""Add player-prop columns to predictions and YES/NO to the side vocabulary.

Player-prop rows (Phase 7 Wave 3, market_type = PLAYER_PROP) carry the
statistics-service player UUID, the canonical prop stat key, and the prop
line; all three stay NULL on game-market rows (Wave 0 additive style).
Yes/no props (anytime scorer / anytime TD) emit YES/NO sides, so the side
check constraint gains both values.

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-19

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("predictions", sa.Column("player_external_id", sa.Text(), nullable=True), schema="predictions")
    op.add_column("predictions", sa.Column("stat_type", sa.Text(), nullable=True), schema="predictions")
    op.add_column("predictions", sa.Column("prop_line", sa.Numeric(8, 2), nullable=True), schema="predictions")
    op.drop_constraint("chk_predictions_side", "predictions", schema="predictions", type_="check")
    op.create_check_constraint(
        "chk_predictions_side",
        "predictions",
        "side IN ('HOME', 'AWAY', 'DRAW', 'OVER', 'UNDER', 'YES', 'NO')",
        schema="predictions",
    )


def downgrade() -> None:
    op.drop_constraint("chk_predictions_side", "predictions", schema="predictions", type_="check")
    # YES/NO rows would violate the restored pre-prop constraint; their side
    # becomes NULL (the pre-Phase-6 legacy-row convention) rather than
    # deleting prediction history.
    op.execute("UPDATE predictions.predictions SET side = NULL WHERE side IN ('YES', 'NO')")
    op.create_check_constraint(
        "chk_predictions_side",
        "predictions",
        "side IN ('HOME', 'AWAY', 'DRAW', 'OVER', 'UNDER')",
        schema="predictions",
    )
    op.drop_column("predictions", "prop_line", schema="predictions")
    op.drop_column("predictions", "stat_type", schema="predictions")
    op.drop_column("predictions", "player_external_id", schema="predictions")
