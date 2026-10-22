"""F3 FinOps alert thresholds: FM-only revisioned configuration, additive.

Revision ID: f3_0006
Revises: f3_0005
"""
from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "f3_0006"
down_revision = "f3_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "core_ai_finops_alert_policies",
        sa.Column("tenant_id", sa.String(128), primary_key=True),
        sa.Column("unit_id", sa.String(128), primary_key=True),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False),
        sa.Column("maximum_known_cost", sa.String(80), nullable=False),
        sa.Column("modified_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("revision >= 1", name="ck_finops_policy_revision"),
    )


def downgrade() -> None:
    # Destructive: disposable internal CI databases only.
    op.drop_table("core_ai_finops_alert_policies")
