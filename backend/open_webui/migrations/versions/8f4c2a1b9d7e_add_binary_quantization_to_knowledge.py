"""Add binary quantization to knowledge

Revision ID: 8f4c2a1b9d7e
Revises: d31026856c01
Create Date: 2025-08-08 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

revision = "8f4c2a1b9d7e"
down_revision = "d31026856c01"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("knowledge", sa.Column("is_binary_quantized", sa.Boolean(), nullable=True))


def downgrade():
    op.drop_column("knowledge", "is_binary_quantized")