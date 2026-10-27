"""Ítems de impacto funcional (PHQ-9 ítem 10, GAD-7 ítem 8; opcionales y no puntuados).

Revision ID: 0005
Revises: 0004
"""
import importlib

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    importlib.import_module("migrations.versions.0002_seed_catalog").upgrade()


def downgrade() -> None:
    from alembic import op
    op.get_bind().exec_driver_sql(
        "DELETE FROM core.instrument_items WHERE (instrument_code, item_no) IN (('PHQ9', 10), ('GAD7', 8));"
        "UPDATE core.instruments SET n_items = 9 WHERE code = 'PHQ9';"
        "UPDATE core.instruments SET n_items = 7 WHERE code = 'GAD7';"
    )
