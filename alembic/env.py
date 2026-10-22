"""Controlled Alembic environment for FM Core AI Control Plane."""

from __future__ import annotations

from logging.config import fileConfig

from sqlalchemy import create_engine, pool

import core.ai_admin_audit
import core.ai_circuit_state_store
import core.ai_finops_policy_store
import core.ai_provider_evidence_store
import core.ai_quota_store
import core.ai_usage_ledger  # noqa: F401 - registra tabelas do ledger
from alembic import context
from core.ai_control_plane import ControlPlaneBase

config = context.config
if config.config_file_name and config.get_section("loggers"):
    fileConfig(config.config_file_name)

target_metadata = ControlPlaneBase.metadata


def database_url() -> str:
    url = config.get_main_option("sqlalchemy.url", "").strip()
    if not url:
        raise RuntimeError("FM Core migration requires explicit sqlalchemy.url")
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
        return
    engine = create_engine(database_url(), poolclass=pool.NullPool)
    try:
        with engine.connect() as connection:
            context.configure(connection=connection, target_metadata=target_metadata)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
