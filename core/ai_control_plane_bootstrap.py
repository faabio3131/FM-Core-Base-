"""Explicit F3.5 database-backed Core startup; never runs production migrations.

The database must already be migrated by a separately authorized process.
No tenant, secret or provider credentials are loaded into application globals.
"""

from __future__ import annotations

from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

import core.ai_admin_audit as admin_audit
import core.ai_circuit_state_store
import core.ai_quota_store  # noqa: F401 - registers quota reservation tables
from core.ai_control_plane import AIControlPlaneStore, ControlPlaneBase
from core.ai_control_plane_runtime import (
    CoreControlPlaneAIRuntime,
    CoreCredentialResolver,
    FMManagedCredentialResolver,
)
from core.ai_plan_usage_guard import PlanPolicyAuthority, PlanUsageGuard
from core.ai_quota_store import QuotaReservations
from core.ai_tenant_credential_resolver import (
    GovernedCredentialResolver,
    TenantCredentialGateway,
)
from core.ai_usage_ledger import PersistentAIFinOps
from core.seguranca.segredos import ReferenceSecretStore

REQUIRED_REVISION = "f3_0006"
REQUIRED_TABLES = frozenset(ControlPlaneBase.metadata.tables) | {
    admin_audit.CommandAdminAuditRecord.__tablename__,
    "core_ai_price_snapshots",
    "core_ai_price_tariffs",
    "core_ai_usage_events",
}


def control_plane_is_ready(engine: Engine) -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            inspector = inspect(conn)
            if not REQUIRED_TABLES.issubset(inspector.get_table_names()):
                return False
            if "alembic_version" not in inspector.get_table_names():
                return False
            revision = conn.execute(text("SELECT version_num FROM alembic_version")).scalars().all()
            return revision == [REQUIRED_REVISION]
    except SQLAlchemyError:
        return False


def build_persistent_core(
    database_url: str,
    *,
    credential_resolver: CoreCredentialResolver | None = None,
    tenant_gateway: TenantCredentialGateway | None = None,
    plan_authority: PlanPolicyAuthority | None = None,
    max_audio_bytes: int = 10 * 1024 * 1024,
) -> tuple[CoreControlPlaneAIRuntime, Engine]:
    # O gateway de tenant deve ser injetado explicitamente por integrador confiável.
    # O runtime comercial não constrói gateway nem aceita BYOK por padrão.
    if credential_resolver is not None and tenant_gateway is not None:
        raise ValueError("core_credential_resolver_ambiguous")
    # The caller validates the URL and commercial environment before invoking.
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        if not control_plane_is_ready(engine):
            raise RuntimeError("core_control_plane_database_not_ready")
        sessions = sessionmaker(engine, expire_on_commit=False)
        fm_credentials = FMManagedCredentialResolver(ReferenceSecretStore())
        credentials: CoreCredentialResolver = (
            credential_resolver
            or (
                GovernedCredentialResolver(
                    fm_credentials=fm_credentials, tenant_gateway=tenant_gateway
                )
                if tenant_gateway is not None
                else fm_credentials
            )
        )
        runtime = CoreControlPlaneAIRuntime(
            control_plane=AIControlPlaneStore(sessions),
            credentials=credentials,
            metering=PersistentAIFinOps(sessions),
            plan_usage=(
                PlanUsageGuard(
                    quota=QuotaReservations(sessions), authority=plan_authority,
                    max_audio_bytes=max_audio_bytes,
                ) if plan_authority is not None else None
            ),
        )
        return runtime, engine
    except BaseException:
        engine.dispose()
        raise
