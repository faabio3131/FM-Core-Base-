"""PostgreSQL 16 concurrency gate for F3.8 quotas (disposable CI only)."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path
from threading import Barrier

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from alembic import command
from core.ai_circuit_breaker import CircuitDenied, CircuitPhase, CircuitPolicy
from core.ai_circuit_state_store import PersistentCircuit, ProviderCircuitRecord
from core.ai_control_plane import AIProvider
from core.ai_quota_store import (
    AIQuotaPolicyRecord,
    AIQuotaReservationRecord,
    QuotaDenied,
    QuotaRequirement,
    QuotaReservations,
)
from core.ai_usage_ledger import AIUsageRecord, PersistentAIFinOps

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)


@pytest.fixture
def postgres_quota():
    url = os.environ.get("FM_CORE_F3_POSTGRES_TEST_URL", "")
    if not url:
        pytest.skip("requires disposable PostgreSQL CI database")
    if not url.startswith("postgresql+psycopg://") or "/core_f3" not in url:
        pytest.fail("unsafe test database URL")
    root = Path(__file__).resolve().parents[2]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")
    engine = create_engine(url, pool_size=6, max_overflow=4, pool_timeout=10)
    sessions = sessionmaker(engine, expire_on_commit=False)
    with sessions.begin() as db:
        db.query(AIQuotaReservationRecord).delete()
        db.query(AIQuotaPolicyRecord).delete()
        db.add(AIQuotaPolicyRecord(
            tenant_id="tenant-a", unit_id="unit-a", capability="text",
            limit_units=1, window_seconds=60, revision=1,
        ))
    try:
        yield QuotaReservations(sessions), sessions
    finally:
        engine.dispose()


def test_pg_concurrent_reservation_serializes_budget(postgres_quota):
    store, sessions = postgres_quota
    barrier = Barrier(2)

    def claim(request_id: str) -> str:
        barrier.wait(timeout=8)
        try:
            store.reserve(
                tenant_id="tenant-a", unit_id="unit-a", capability="text",
                request_id=request_id, units=1, at=NOW,
            )
            return "admitted"
        except QuotaDenied:
            return "denied"

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(claim, "req-a")
        b = pool.submit(claim, "req-b")
        outcomes = [a.result(timeout=20), b.result(timeout=20)]
    assert sorted(outcomes) == ["admitted", "denied"]
    with sessions() as db:
        rows = db.scalars(select(AIQuotaReservationRecord)).all()
        assert len(rows) == 1
        assert rows[0].reserved_units == 1


def test_pg_same_request_is_idempotent_across_two_workers(postgres_quota):
    store, sessions = postgres_quota
    barrier = Barrier(2)

    def claim() -> str:
        barrier.wait(timeout=8)
        return store.reserve(
            tenant_id="tenant-a", unit_id="unit-a", capability="text",
            request_id="stable-id", units=1, at=NOW,
        ).request_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(claim), pool.submit(claim)
        assert a.result(timeout=20) == b.result(timeout=20) == "stable-id"
    with sessions() as db:
        assert len(db.scalars(select(AIQuotaReservationRecord)).all()) == 1


def test_pg_circuit_only_one_half_open_claim(postgres_quota):
    """Two independent workers contend for a single provider probe lease."""
    _, sessions = postgres_quota
    with sessions.begin() as db:
        provider = db.get(AIProvider, "provider-ci")
        if provider is None:
            db.add(AIProvider(
                provider_id="provider-ci", adapter="gemini",
                base_url="https://invalid.example", enabled=True,
            ))
        db.flush()
        row = db.get(ProviderCircuitRecord, "provider-ci")
        if row is None:
            db.add(ProviderCircuitRecord(
                provider_id="provider-ci", phase="open", failures=2,
                until=NOW, probe_id=None, version=1,
            ))
        else:
            row.phase = "open"
            row.failures = 2
            row.until = NOW
            row.probe_id = None
            row.version += 1
    store = PersistentCircuit(
        sessions, CircuitPolicy(threshold=2, cooldown_seconds=30, probe_seconds=5),
    )
    barrier = Barrier(2)

    def attempt():
        barrier.wait(timeout=8)
        try:
            return "claimed" if store.claim(
                provider_id="provider-ci", now=NOW,
            )[0] else "unexpected"
        except CircuitDenied:
            return "denied"

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(attempt), pool.submit(attempt)
        assert sorted((a.result(timeout=20), b.result(timeout=20))) == [
            "claimed", "denied",
        ]
    with sessions() as db:
        row = db.get(ProviderCircuitRecord, "provider-ci")
        assert row.phase == CircuitPhase.HALF_OPEN
        assert row.probe_id


def test_pg_shared_tenant_budget_across_units_is_atomic(postgres_quota):
    """Parallel units cannot evade a shared plan-wide token ceiling."""
    store, sessions = postgres_quota
    with sessions.begin() as db:
        for scope, metric, limit in (
            ("__tenant__", "ai.total_tokens", 1000),
            ("unit-a", "ai.requests", 5),
            ("unit-b", "ai.requests", 5),
        ):
            key = ("tenant-budget-ci", scope, metric)
            old = db.get(AIQuotaPolicyRecord, key)
            if old is None:
                db.add(AIQuotaPolicyRecord(
                    tenant_id=key[0], unit_id=key[1], capability=key[2],
                    limit_units=limit, window_seconds=60, revision=1,
                ))
        db.flush()
    barrier = Barrier(2)

    def reserve(unit: str) -> str:
        barrier.wait(timeout=8)
        try:
            store.reserve_bundle(
                request_id=f"request-{unit}", at=NOW,
                needs=(
                    QuotaRequirement(
                        tenant_id="tenant-budget-ci", unit_id=unit,
                        capability="ai.requests", units=1,
                    ),
                    QuotaRequirement(
                        tenant_id="tenant-budget-ci", unit_id="__tenant__",
                        capability="ai.total_tokens", units=700,
                    ),
                ),
            )
            return "accepted"
        except QuotaDenied:
            return "denied"

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(reserve, "unit-a")
        second = pool.submit(reserve, "unit-b")
        assert sorted((first.result(timeout=20), second.result(timeout=20))) == [
            "accepted", "denied",
        ]
    with sessions() as db:
        rows = db.scalars(select(AIQuotaReservationRecord).where(
            AIQuotaReservationRecord.tenant_id == "tenant-budget-ci",
        )).all()
        assert len(rows) == 2
        assert {item.capability for item in rows} == {
            "ai.requests", "ai.total_tokens",
        }
        assert len({item.request_id for item in rows}) == 1


def test_pg_signed_projection_rejects_window_change_with_history(postgres_quota):
    """Revision update and prior usage must not land in different periods."""
    store, sessions = postgres_quota
    grants = (
        ("ai.requests", "tenant", 4, 60),
        ("ai.total_tokens", "tenant", 500, 60),
    )
    store.project_signed_grants(
        tenant_id="tenant-a", unit_id="unit-a", revision=1,
        grants=grants,
    )
    store.reserve_bundle(
        needs=(QuotaRequirement(
            tenant_id="tenant-a", unit_id="__tenant__",
            capability="ai.total_tokens", units=300, policy_revision=1,
        ),), request_id="prior-history", at=NOW,
    )
    with pytest.raises(QuotaDenied, match="projection_window_change_blocked"):
        store.project_signed_grants(
            tenant_id="tenant-a", unit_id="unit-a", revision=2,
            grants=(("ai.requests", "tenant", 5, 60),
                    ("ai.total_tokens", "tenant", 700, 3600)),
        )
    with sessions() as db:
        req = db.get(AIQuotaPolicyRecord, ("tenant-a", "__tenant__", "ai.requests"))
        tokens = db.get(AIQuotaPolicyRecord, ("tenant-a", "__tenant__", "ai.total_tokens"))
        assert (req.revision, req.limit_units) == (1, 4)
        assert (tokens.revision, tokens.window_seconds) == (1, 60)
    usage = store.usage_snapshot(
        tenant_id="tenant-a", unit_id="__tenant__",
        capability="ai.total_tokens", at=NOW,
    )
    assert (usage.reserved_units, usage.remaining_units) == (300, 200)


def test_pg_ambiguous_recovery_is_atomic_across_concurrent_workers(postgres_quota):
    """One human-reviewed provider proof may settle only once, never double-refund."""
    from core.ai_admin_audit import CommandAdminAuditRecord
    from core.ai_command_authority import VerifiedCommandAdmin

    store, sessions = postgres_quota
    store.reserve(
        tenant_id="tenant-a", unit_id="unit-a", capability="text",
        request_id="recovery-contended", units=1, at=NOW,
    )
    barrier = Barrier(2)
    evidence = (QuotaRequirement(
        tenant_id="tenant-a", unit_id="unit-a", capability="text",
        units=0, policy_revision=1,
    ),)

    def resolve(jti: str) -> str:
        principal = VerifiedCommandAdmin(
            user_id="platform-human", session_id="session",
            token_id=jti, issued_at=int(NOW.timestamp()) - 1,
            expires_at=int(NOW.timestamp()) + 90,
            step_up_at=int(NOW.timestamp()) - 1,
            permissions=frozenset({"core.ai.admin.mutate"}),
        )
        barrier.wait(timeout=8)
        try:
            store.reconcile_ambiguous_bundle(
                usage=evidence, request_id="recovery-contended",
                principal=principal, evidence_digest="b" * 64,
                evidence_kind="provider_no_charge_confirmed", at=NOW,
            )
            return "resolved"
        except QuotaDenied:
            return "denied"

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(resolve, "recovery-token-1")
        b = pool.submit(resolve, "recovery-token-2")
        assert sorted([a.result(timeout=20), b.result(timeout=20)]) == [
            "denied", "resolved",
        ]
    with sessions() as db:
        entry = db.get(
            AIQuotaReservationRecord,
            ("tenant-a", "unit-a", "text", "recovery-contended"),
        )
        assert entry.state == "settled"
        assert entry.consumed_units == 0
        audits = db.scalars(select(CommandAdminAuditRecord).where(
            CommandAdminAuditRecord.resource_id == "tenant-a:recovery-contended",
        )).all()
        assert len(audits) == 1


def test_pg_signed_command_recovery_exactly_once_with_service_fence(postgres_quota):
    """Two workers, one provider proof, protected by transport + human authority."""
    from decimal import Decimal

    import jwt
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from fastapi.testclient import TestClient

    from core.ai_admin_audit import CommandAdminAuditRecord
    from core.ai_provider_evidence_store import (
        TrustedProviderEvidenceStore,
        sign_verified_gateway_attestation,
    )
    from core.ai_service_api import create_app
    from core.ai_service_settings import CoreEnvironment, CoreSettings
    from core.ai_verified_usage_receipt import VerifiedProviderUsageReceipt

    store, sessions = postgres_quota
    url = os.environ["FM_CORE_F3_POSTGRES_TEST_URL"]
    tenant = "tenant-recovery-pg"
    request_id = "pg-recovery-command-v1"
    at = datetime.now(UTC)
    with sessions.begin() as db:
        for cap in ("ai.requests", "ai.total_tokens"):
            db.add(AIQuotaPolicyRecord(
                tenant_id=tenant, unit_id="__tenant__", capability=cap,
                limit_units=200, window_seconds=3600, revision=1,
            ))
    store.reserve_bundle(
        request_id=request_id, at=at,
        needs=(
            QuotaRequirement(
                tenant_id=tenant, unit_id="__tenant__",
                capability="ai.requests", units=1, policy_revision=1,
            ),
            QuotaRequirement(
                tenant_id=tenant, unit_id="__tenant__",
                capability="ai.total_tokens", units=100, policy_revision=1,
            ),
        ),
    )
    provider_secret = b"synthetic-fm-pg-evidence-gateway-secret"
    recovery_service_key = "synthetic-fm-command-to-core-transport-12345"
    source_id = "fm-ci-gateway"
    evidence_id = "provider-evidence-pg-1"
    receipt = VerifiedProviderUsageReceipt(
        provider="gemini", model="fixture", request_id=request_id,
        tenant_id=tenant, unit_id="unit-1", input_tokens=9, output_tokens=6,
        provider_cost_usd=Decimal("0.0015"), evidence_digest="c" * 64,
    )
    evidence_store = TrustedProviderEvidenceStore(
        sessions, gateway_secret=provider_secret, source_id=source_id,
    )
    # The FM gateway proof is SYNTHETIC, not a claim about live Gemini source.
    assert evidence_store.record_attested(
        evidence_id=evidence_id, receipt=receipt,
        gateway_signature=sign_verified_gateway_attestation(
            evidence_id=evidence_id, source_id=source_id,
            receipt=receipt, secret=provider_secret,
        ),
    )
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = private.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    cfg = CoreSettings(
        environment=CoreEnvironment.TEST, service_key="service-ci-key",
        routes=(), control_plane_database_url=url,
        governed_recovery_enabled=True, governed_recovery_issuer="fm-command",
        governed_recovery_audience="fm-core-recovery",
        governed_recovery_public_key_pem=public,
        governed_recovery_gateway_secret=provider_secret.decode(),
        governed_recovery_source_id=source_id,
        governed_recovery_transport_key=recovery_service_key,
    )
    path = "/internal/v1/command/ai/usage/reconcile"
    body = {
        "request_id": request_id, "tenant_id": tenant, "unit_id": "unit-1",
        "evidence_id": evidence_id,
        "requests_revision": 1, "tokens_revision": 1,
    }
    now = int(at.timestamp())

    def signed_request(seq: int) -> tuple[dict[str, str], dict[str, str]]:
        claims = {
            "iss": "fm-command", "aud": "fm-core-recovery",
            "sub": "platform-human", "session_id": "fm-session",
            "jti": f"pg-unique-recovery-jti-{seq}",
            "iat": now, "nbf": now, "exp": now + 120, "step_up_at": now,
            "actor_type": "human", "role": "platform_admin",
            "scope": "fm_core_platform", "permissions": ["core.ai.admin.mutate"],
        }
        return body, {
            "Authorization": "Bearer " + jwt.encode(claims, private, algorithm="RS256"),
            "X-FM-Recovery-Service-Key": recovery_service_key,
        }

    with TestClient(create_app(settings=cfg)) as client:
        payload, headers = signed_request(0)
        assert client.post(path, json=payload, headers={
            "Authorization": headers["Authorization"],
        }).status_code == 401
        assert client.post(path, json={**payload, "tenant_id": "wrong-tenant"},
                           headers=headers).status_code == 409

        barrier = Barrier(2)

        def worker(seq: int) -> int:
            body2, headers2 = signed_request(seq + 1)
            barrier.wait(timeout=8)
            return client.post(path, json=body2, headers=headers2).status_code

        with ThreadPoolExecutor(max_workers=2) as pool:
            a, b = pool.submit(worker, 1), pool.submit(worker, 2)
            assert sorted((a.result(timeout=20), b.result(timeout=20))) == [204, 409]

    with sessions() as db:
        settled = db.scalars(select(AIQuotaReservationRecord).where(
            AIQuotaReservationRecord.tenant_id == tenant,
            AIQuotaReservationRecord.request_id == request_id,
        )).all()
        assert len(settled) == 2
        assert {row.capability: row.consumed_units for row in settled} == {
            "ai.requests": 1, "ai.total_tokens": 15,
        }
        assert all(row.state == "settled" for row in settled)
        audits = db.scalars(select(CommandAdminAuditRecord).where(
            CommandAdminAuditRecord.resource_id == f"{tenant}:{request_id}",
        )).all()
        assert len(audits) == 1


def test_pg_finops_pending_reconciliation_is_bounded_and_tenant_scoped(postgres_quota):
    """Run the same SQL GROUP BY/HAVING query against disposable PostgreSQL."""
    _, sessions = postgres_quota
    tenant = "pg-p01-backlog"
    unit = "unit-a"
    with sessions.begin() as db:
        db.query(AIUsageRecord).filter(
            AIUsageRecord.tenant_id == tenant,
        ).delete(synchronize_session=False)
        for event_id, request_id, tenant_scope, unit_scope, cost, inp in (
            ("pg-backlog-1", "req-known", tenant, unit, "0.004", 12),
            ("pg-backlog-2", "req-pending", tenant, unit, "0.004", 12),
            ("pg-backlog-3", "req-pending", tenant, unit, None, None),
            ("pg-backlog-4", "req-foreign", tenant, "other", None, None),
            ("pg-backlog-5", "req-stranger", "another-tenant", unit, None, None),
        ):
            db.add(AIUsageRecord(
                event_id=event_id, tenant_id=tenant_scope, unit_id=unit_scope,
                request_id=request_id, correlation_id="pg-backlog",
                capability="atendimento_interpretacao", provider="gemini",
                model="model-synthetic", route_reason="test", fallback_used=cost is None,
                fallback_reason=None, input_tokens=inp, output_tokens=3 if inp else None,
                cached_tokens=None, latency_ms=1, outcome="success",
                cost_decimal=cost, currency="USD" if cost else "XXX",
                cost_origin="estimated" if cost else "unavailable",
                price_snapshot_id="snapshot-test", event_time=NOW,
            ))
    backlog, truncated = PersistentAIFinOps(sessions).listar_pendencias(
        tenant_id=tenant, unit_id=unit, inicio=date(2026, 10, 18),
        fim=date(2026, 10, 18), limit=1,
    )
    assert truncated is False
    assert len(backlog) == 1
    assert backlog[0].request_id == "req-pending"
    assert backlog[0].observed_attempts == 2
    assert backlog[0].unknown_cost_attempts == 1
    assert backlog[0].incomplete_token_attempts == 1
    assert backlog[0].fallback_attempts == 1
    other, _ = PersistentAIFinOps(sessions).listar_pendencias(
        tenant_id=tenant, unit_id="other", inicio=date(2026, 10, 18),
        fim=date(2026, 10, 18),
    )
    assert [item.request_id for item in other] == ["req-foreign"]
    with sessions.begin() as db:
        db.query(AIUsageRecord).filter(
            AIUsageRecord.tenant_id.in_((tenant, "another-tenant")),
        ).delete(synchronize_session=False)
