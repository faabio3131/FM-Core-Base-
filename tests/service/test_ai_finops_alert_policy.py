"""F3 corporate alert configuration: transactional, signed, adversarial."""
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from core.ai_admin_audit import CommandAdminAuditRecord
from core.ai_command_authority import CommandTrustPolicy, VerifiedCommandAdmin
from core.ai_control_plane import ControlPlaneBase, ControlPlaneError
from core.ai_finops_policy_api import finops_alert_policy_router
from core.ai_finops_policy_store import (
    AIFinOpsAlertPolicyRecord,
    PersistentFinOpsAlertPolicies,
)


@pytest.fixture
def store(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'alerts.db'}")
    ControlPlaneBase.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    yield PersistentFinOpsAlertPolicies(sessions), sessions
    engine.dispose()


def principal(jti="synthetic-jti-unique-0123456789"):
    return VerifiedCommandAdmin(
        user_id="human-fm", session_id="fm-session", token_id=jti,
        issued_at=100, expires_at=200, step_up_at=100,
        permissions=frozenset({"core.ai.admin.mutate"}),
    )


def set_policy(store, *, jti="synthetic-jti-unique-0123456789", expected=0, **change):
    data = {"tenant_id": "tenant-a", "unit_id": "unit-a",
            "expected_revision": expected, "enabled": True,
            "currency": "USD", "maximum_known_cost": Decimal("0.05"),
            "principal": principal(jti)}
    data.update(change)
    return store.update(**data)


def test_policy_cas_audit_scope_replay_and_disable_atomic(store):
    policies, sessions = store
    assert policies.get(tenant_id="tenant-a", unit_id="unit-a") is None
    result = set_policy(policies)
    assert result.revision == 1 and result.enabled
    assert policies.get(tenant_id="tenant-a", unit_id="other-unit") is None
    assert policies.get(tenant_id="foreign", unit_id="unit-a") is None
    with pytest.raises(ControlPlaneError, match="alert_revision_conflict"):
        set_policy(policies, jti="another-unique-jti-00000")
    with pytest.raises(ControlPlaneError, match="replayed"):
        set_policy(policies, expected=1)
    assert policies.get(tenant_id="tenant-a", unit_id="unit-a").revision == 1
    updated = set_policy(
        policies, jti="third-unique-jti-0000000", expected=1,
        enabled=False, maximum_known_cost=Decimal(10),
    )
    assert updated.revision == 2 and not updated.enabled
    with sessions() as db:
        rows = db.scalars(select(CommandAdminAuditRecord)).all()
        assert len(rows) == 2
        assert all(row.action == "finops.alert_policy.update" for row in rows)
        assert len(db.scalars(select(AIFinOpsAlertPolicyRecord)).all()) == 1


@pytest.mark.parametrize("changes", [
    {"enabled": 1},
    {"currency": "usd"},
    {"currency": "DOLLAR"},
    {"maximum_known_cost": Decimal("NaN")},
    {"maximum_known_cost": Decimal("Infinity")},
    {"maximum_known_cost": Decimal(0)},
    {"maximum_known_cost": Decimal("-0.1")},
    {"maximum_known_cost": Decimal(1000000001)},
    {"tenant_id": "foreign tenant"},
    {"expected_revision": True},
    {"expected_revision": -1},
    {"principal": replace(principal(), permissions=frozenset())},
])
def test_policy_rejects_malformed_or_unauthorized_mutations(store, changes):
    policies, _ = store
    with pytest.raises(ControlPlaneError):
        set_policy(policies, **changes)


def test_http_requires_both_service_identity_and_recent_human_step_up(store):
    from fastapi import FastAPI

    policies, sessions = store
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = private.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("ascii")
    trust = CommandTrustPolicy(
        issuer="fm-command", audience="fm-core-recovery", public_key_pem=public,
    )
    transport = "synthetic-independent-command-service-key-123456789"
    app = FastAPI()
    app.state.control_plane_engine = sessions.kw["bind"]
    app.include_router(finops_alert_policy_router(trust=trust, transport_key=transport))
    body = {
        "tenant_id": "tenant-a", "unit_id": "unit-a", "expected_revision": 0,
        "enabled": True, "currency": "USD", "maximum_known_cost": "0.05",
    }
    path = "/internal/v1/command/ai/finops/alert-policy"
    now = int(datetime.now(UTC).timestamp())

    def jwt_for(jti, **overrides):
        claims = {
            "iss": "fm-command", "aud": "fm-core-recovery", "sub": "fm-admin",
            "jti": jti, "session_id": "fm-session", "iat": now, "nbf": now, "exp": now + 120,
            "step_up_at": now, "actor_type": "human", "role": "platform_admin",
            "scope": "fm_core_platform", "permissions": ["core.ai.admin.mutate"],
        }
        claims.update(overrides)
        return jwt.encode(claims, private, algorithm="RS256")

    headers = {
        "X-FM-Recovery-Service-Key": transport,
        "Authorization": "Bearer " + jwt_for("synthetic-write-jti-12345"),
    }
    http = TestClient(app)
    assert http.post(path, json=body).status_code == 401
    assert http.post(path, json=body, headers={
        "Authorization": headers["Authorization"],
    }).status_code == 401
    assert http.post(path, json=body, headers={
        **headers, "X-FM-Recovery-Service-Key": "wrong",
    }).status_code == 401
    assert http.post(path, json=body, headers={
        **headers, "Authorization": "Bearer " +
        jwt_for("wrong-role-jti-1234567", role="tenant_owner"),
    }).status_code == 403
    for value in (True, 0.0, "1", None):
        assert http.post(path, json={**body, "expected_revision": value},
                         headers=headers).status_code == 422
    for value in ("NaN", "-1", "Infinity", "0", "1e2", "0.0000000000001"):
        assert http.post(path, json={**body, "maximum_known_cost": value},
                         headers=headers).status_code == 422
    assert http.post(path, json=body, headers=headers).status_code == 204
    assert policies.get(tenant_id="tenant-a", unit_id="unit-a").revision == 1
    assert http.post(path, json={**body, "expected_revision": 1},
                     headers=headers).status_code == 409
    assert policies.get(tenant_id="tenant-a", unit_id="unit-a").revision == 1
