from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from alembic import command
from core.ai_command_authority import VerifiedCommandAdmin
from core.ai_control_plane import AIControlPlaneStore, RouteDefinition
from core.ai_finops_policy_store import PersistentFinOpsAlertPolicies
from core.ai_router import AIUsageEvent, CapabilityIA, OutcomeIA
from core.ai_service_api import create_app
from core.ai_service_settings import CoreEnvironment, CoreSettings
from core.ai_usage_ledger import PersistentAIFinOps


@pytest.fixture
def fixture(tmp_path):
    root = Path(__file__).resolve().parents[2]
    url = f"sqlite:///{tmp_path / 'operator.db'}"
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")
    engine = create_engine(url)
    sessions = sessionmaker(engine, expire_on_commit=False)
    store = AIControlPlaneStore(sessions)
    store.register_provider(
        provider_id="openai", adapter="openai_compatible",
        base_url="https://private-provider.invalid/v1",
    )
    store.register_model(model_id="model-id", provider_id="openai", name="model-test")
    cap = CapabilityIA.ATENDIMENTO_INTERPRETACAO
    store.grant_capability(model_id="model-id", capability=cap)
    store.set_provider_health(provider_id="openai", status="healthy", circuit="closed")
    store.add_route(RouteDefinition(
        policy_id="route", scope="tenant", tenant_id="tenant-a", unit_id=None,
        model_id="model-id", capability=cap, priority=99,
        price_snapshot_id="p1", credential_source="tenant_managed",
        secret_reference="integration:tenant-a:very-private-key-ref",
    ))
    PersistentAIFinOps(sessions).registrar(AIUsageEvent(
        tenant_id="tenant-a", unidade_id="unit-1", request_id="req",
        correlation_id="corr", capability=cap, provider="openai",
        model="model-test", route_reason="reason", fallback_used=False,
        fallback_reason=None, input_tokens=None, output_tokens=None,
        cached_tokens=None, latency_ms=1, outcome=OutcomeIA.SUCESSO,
        custo_real_calculado=None, moeda=None, price_snapshot_id="missing",
        timestamp=datetime(2026, 10, 18, 10, tzinfo=UTC),
    ))
    yield url
    engine.dispose()


def _cfg(db: str, *, with_admin: bool = True) -> CoreSettings:
    return CoreSettings(
        environment=CoreEnvironment.TEST,
        service_key="consumer-only",
        admin_key="operator-secret" if with_admin else None,
        routes=(), control_plane_database_url=db,
    ).validate()


def test_operator_endpoints_are_readonly_and_require_separate_key(fixture):
    with TestClient(create_app(_cfg(fixture))) as client:
        base = "/v1/admin/ai/routes?tenant_id=tenant-a&unit_id=unit-1"
        assert client.get(base).status_code == 401
        assert client.get(base, headers={"X-Core-Service-Key": "consumer-only"}).status_code == 401
        assert client.get(base, headers={"X-Core-Admin-Key": "invalid"}).status_code == 401
        auth = {"X-Core-Admin-Key": "operator-secret"}
        routes = client.get(base, headers=auth)
        assert routes.status_code == 200
        assert routes.json()[0]["policy_id"] == "route"
        assert "private" not in routes.text
        assert client.get(
            "/v1/admin/ai/providers/health", headers=auth
        ).json()[0]["health"] == "healthy"
        usage = client.get(
            "/v1/admin/ai/usage",
            params={
                "tenant_id": "tenant-a", "unit_id": "unit-1",
                "start_date": "2026-10-18", "end_date": "2026-10-18",
            },
            headers=auth,
        )
        assert usage.status_code == 200
        assert usage.json()[0]["cost_unknown_events"] == 1
        assert usage.json()[0]["cost_total"] == "0"
        assert client.get(
            "/v1/admin/ai/usage",
            params={
                "tenant_id": "tenant-b", "unit_id": "unit-1",
                "start_date": "2026-10-18", "end_date": "2026-10-18",
            },
            headers=auth,
        ).json() == []
        assert client.post("/v1/admin/ai/routes", headers=auth).status_code == 405


def test_operator_read_api_disabled_without_admin_key(fixture):
    with TestClient(create_app(_cfg(fixture, with_admin=False))) as client:
        response = client.get(
            "/v1/admin/ai/routes?tenant_id=tenant-a&unit_id=unit-1",
            headers={"X-Core-Admin-Key": "operator-secret"},
        )
        assert response.status_code == 404


def test_operator_finops_alert_requires_admin_key_and_preserves_unknown_cost(fixture):
    with TestClient(create_app(_cfg(fixture))) as client:
        url = "/v1/admin/ai/usage/request"
        parameters = {
            "tenant_id": "tenant-a", "unit_id": "unit-1", "request_id": "req",
        }
        assert client.get(url, params=parameters).status_code == 401
        assert client.get(
            url, params=parameters,
            headers={"X-Core-Service-Key": "consumer-only"},
        ).status_code == 401
        authorized = {"X-Core-Admin-Key": "operator-secret"}
        result = client.get(url, params=parameters, headers=authorized)
        assert result.status_code == 200
        payload = result.json()
        assert payload["status"] == "needs_attention"
        assert payload["reasons"] == ["provider_cost_unverified"]
        assert payload["unknown_attempts"] == 1
        assert payload["known_cost_subtotal"] == "0"
        assert payload["all_costs_known"] is False
        cross_tenant = client.get(
            url, params={**parameters, "tenant_id": "tenant-b"},
            headers=authorized,
        )
        assert cross_tenant.status_code == 200
        assert cross_tenant.json()["status"] == "needs_attention"
        assert cross_tenant.json()["reasons"] == ["usage_evidence_unavailable"]
        assert cross_tenant.json()["observed_attempts"] == 0
        mismatch = client.get(
            url, params={**parameters, "maximum_known_cost": "0.05", "currency": "BRL"},
            headers=authorized,
        )
        assert mismatch.status_code == 200
        assert mismatch.json()["unknown_attempts"] == 1


def test_operator_pending_usage_is_scoped_read_only_and_uncertified(fixture):
    with TestClient(create_app(_cfg(fixture))) as client:
        path = "/v1/admin/ai/usage/pending"
        params = {
            "tenant_id": "tenant-a", "unit_id": "unit-1",
            "start_date": "2026-10-18", "end_date": "2026-10-18",
            "limit": 1,
        }
        assert client.get(path, params=params).status_code == 401
        assert client.get(
            path, params=params, headers={"X-Core-Service-Key": "consumer-only"},
        ).status_code == 401
        auth = {"X-Core-Admin-Key": "operator-secret"}
        response = client.get(path, params=params, headers=auth)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        item = response.json()
        assert item["status"] == "needs_attention"
        assert item["provider_invoice_verified"] is False
        assert item["truncated"] is False
        assert item["items"] == [{
            "request_id": "req",
            "first_seen": "2026-10-18T10:00:00+00:00",
            "last_seen": "2026-10-18T10:00:00+00:00",
            "observed_attempts": 1,
            "unknown_cost_attempts": 1,
            "incomplete_token_attempts": 1,
            "fallback_attempts": 0,
        }]
        assert "private-provider" not in response.text
        assert "very-private" not in response.text
        other = client.get(
            path, params={**params, "tenant_id": "tenant-b"}, headers=auth,
        )
        assert other.status_code == 200
        assert other.json()["items"] == []
        assert other.json()["status"] == "no_pending_observations"
        assert other.json()["provider_invoice_verified"] is False
        assert client.get(
            path, params={**params, "limit": 101}, headers=auth,
        ).status_code == 422
        assert client.get(
            path, params={**params, "end_date": "2026-12-01"}, headers=auth,
        ).status_code == 422
        assert client.post(path, params=params, headers=auth).status_code == 405


def test_pending_usage_operator_api_disabled_by_default(fixture):
    with TestClient(create_app(_cfg(fixture, with_admin=False))) as client:
        response = client.get(
            "/v1/admin/ai/usage/pending",
            params={
                "tenant_id": "tenant-a", "unit_id": "unit-1",
                "start_date": "2026-10-18", "end_date": "2026-10-18",
            },
            headers={"X-Core-Admin-Key": "operator-secret"},
        )
        assert response.status_code == 404


def test_operator_reads_persisted_fm_threshold_and_cannot_override_it(fixture):
    engine = create_engine(fixture)
    sessions = sessionmaker(engine, expire_on_commit=False)
    principal = VerifiedCommandAdmin(
        user_id="fm-admin", session_id="fm-session",
        token_id="operator-policy-once-jti-123456",
        issued_at=1, expires_at=2, step_up_at=1,
        permissions=frozenset({"core.ai.admin.mutate"}),
    )
    store = PersistentFinOpsAlertPolicies(sessions)
    assert store.update(
        tenant_id="tenant-a", unit_id="unit-1", expected_revision=0,
        enabled=True, currency="USD", maximum_known_cost=Decimal("0.03"),
        principal=principal,
    ).revision == 1
    with TestClient(create_app(_cfg(fixture))) as client:
        auth = {"X-Core-Admin-Key": "operator-secret"}
        params = {"tenant_id": "tenant-a", "unit_id": "unit-1"}
        endpoint = "/v1/admin/ai/finops/alert-policy"
        assert client.get(endpoint, params=params).status_code == 401
        assert client.get(endpoint, params=params, headers={
            "X-Core-Service-Key": "consumer-only",
        }).status_code == 401
        config = client.get(endpoint, params=params, headers=auth)
        assert config.status_code == 200
        assert config.headers["cache-control"] == "no-store"
        assert config.json()["revision"] == 1
        assert config.json()["maximum_known_cost"] == "0.03"
        assert client.post(endpoint, json={}, headers=auth).status_code == 405
        assert client.get(endpoint, params={**params, "tenant_id": "foreign"},
                          headers=auth).json()["configured"] is False

        usage_params = {**params, "request_id": "req"}
        alert = client.get(
            "/v1/admin/ai/usage/request", params=usage_params, headers=auth,
        )
        assert alert.status_code == 200
        assert alert.json()["policy_revision"] == 1
        assert alert.json()["policy_enabled"] is True
        assert alert.json()["reasons"] == ["provider_cost_unverified"]
        assert client.get(
            "/v1/admin/ai/usage/request",
            params={**usage_params, "maximum_known_cost": "999", "currency": "USD"},
            headers=auth,
        ).status_code == 422
    engine.dispose()
