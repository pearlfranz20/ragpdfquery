"""Baseline schema (mirrors db/init.sql)

Revision ID: 0001
Revises:
Create Date: 2026-11-15

This revision applies db/init.sql verbatim so that a fresh database can be
built with `alembic upgrade head`, and a database that was already created
from init.sql can adopt Alembic with `alembic stamp 0001`. All future schema
changes should be real migrations generated against app/models.py.
"""
from pathlib import Path

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

INIT_SQL = Path(__file__).resolve().parents[2] / "db" / "init.sql"


def upgrade() -> None:
    op.execute(INIT_SQL.read_text(encoding="utf-8"))


def downgrade() -> None:
    for table in ("eval_runs", "query_log", "ingestion_jobs", "chunks", "documents"):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
