"""F9 Command FinOps read API: synthetic only, no actual provider/corporate key."""
from __future__ import annotations

import json
import secrets
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.ai_command_finops_api import (
    PATH,
    SCHEMA,
    command_finops_router,
    sign_finops_request,
)
from core.ai_control_plane import ControlPlaneBase
from core.ai_finops import AIFinOpsBucket
from core.ai_usage_ledger import AIUsageRecord, PersistentAIFinOps

SECRET = b"synthetic-command-finops-key-isolated-123456789"


class FakeFinOps:
    def __init__(self, buckets=()):
        self.buckets = tuple(buckets)
        self.calls: list[dict] = []

    def listar(self, *, tenant_id, unidade_id, inicio, fim):
        self.calls.append({
            "tenant_id": tenant_id, "unit_id": unidade_id,
            "start": inicio, "end": fim,
        })
        return self.buckets


def _bucket(
    *, known: int = 1, unknown: int = 1, currency: str = "USD",
    cost: str = "0.00250",
) -> AIFinOpsBucket:
    return AIFinOpsBucket(
        tenant_id="tenant-a", unidade_id="unit-a",
        bucket_date=datetime.now(UTC).date(),
        provider="gemini", model="synthetic-model",
        capability="atendimento_interpretacao", outcome="success",
        moeda=currency, attempts=known + unknown,
        fallback_attempts=1, input_tokens=135, output_tokens=20,
        cached_tokens=15, latency_ms_total=200, latency_ms_max=150,
        cost_known_events=known, cost_unknown_events=unknown,
        cost_total=Decimal(cost),
    )


def _app(model):
    app = FastAPI()
    app.include_router(command_finops_router(read_model=model, shared_secret=SECRET))
    return TestClient(app)


def _body(*, tenant="tenant-a", unit="unit-a",
          start: date | None = None, end: date | None = None,
          issued_at: int | None = None):
    today = datetime.now(UTC).date()
    return {
        "schema": SCHEMA, "tenant_id": tenant, "unit_id": unit,
        "start_date": (start or today).isoformat(),
        "end_date": (end or today).isoformat(),
        "issued_at": (
            int(datetime.now(UTC).timestamp()) if issued_at is None else issued_at
        ),
        "nonce": secrets.token_urlsafe(24),
    }


def _send(client, payload, *, key=SECRET):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return client.post(
        PATH, content=raw,
        headers={
            "Content-Type": "application/json",
            "X-FM-Command-FinOps-Signature": sign_finops_request(
                secret=key, raw_body=raw,
            ),
        },
    )


def test_mixed_known_unknown_preserves_subtotals_and_provenance():
    model = FakeFinOps((_bucket(), _bucket(
        known=1, unknown=0, currency="BRL", cost="2.0001",
    )))
    client = _app(model)
    response = _send(client, _body())
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    result = response.json()
    assert result["schema"] == SCHEMA
    assert result["coverage"] == "partial_unknown_costs"
    assert result["known_attempts"] == 2
    assert result["unknown_attempts"] == 1
    assert result["internal_fm_cost_only"] is True
    assert result["commercial_revenue_included"] is False
    assert result["actual_provider_spend_verified"] is False
    assert len(result["buckets"]) == 2  # USD and BRL never silently summed
    assert result["buckets"][0]["known_cost_subtotal"] == "0.00250"
    assert result["buckets"][1]["known_cost_subtotal"] == "2.0001"
    assert all(x["token_counters_complete"] is False for x in result["buckets"])
    assert model.calls == [{
        "tenant_id": "tenant-a", "unit_id": "unit-a",
        "start": datetime.now(UTC).date(), "end": datetime.now(UTC).date(),
    }]


def test_empty_ledger_does_not_claim_zero_actual_spend():
    result = _send(_app(FakeFinOps()), _body())
    assert result.status_code == 200
    data = result.json()
    assert data["coverage"] == "no_observations"
    assert data["actual_provider_spend_verified"] is False
    assert data["buckets"] == []


def test_wrong_missing_or_replayed_body_signature_is_denied_before_read():
    model = FakeFinOps((_bucket(),))
    client = _app(model)
    payload = _body()
    raw = json.dumps(payload).encode()
    assert client.post(PATH, content=raw, headers={
        "Content-Type": "application/json",
    }).status_code == 401
    signed = sign_finops_request(secret=SECRET, raw_body=raw)
    altered = json.dumps({**payload, "tenant_id": "victim-tenant"}).encode()
    assert client.post(PATH, content=altered, headers={
        "Content-Type": "application/json",
        "X-FM-Command-FinOps-Signature": signed,
    }).status_code == 401
    assert _send(client, _body(issued_at=int(datetime.now(UTC).timestamp()) - 100)).status_code == 400
    assert model.calls == []


def test_rejects_bad_input_before_read():
    model = FakeFinOps()
    client = _app(model)
    today = datetime.now(UTC).date()
    invalid = [
        {**_body(), "tenant_id": "another/tenant"},
        {**_body(), "unit_id": ""},
        {**_body(), "schema": "unknown"},
        {**_body(), "nonce": "short"},
        {**_body(), "issued_at": True},
        {**_body(), "start_date": "2026-02-30"},
        {**_body(), "start_date": (today + timedelta(days=1)).isoformat()},
        {**_body(), "start_date": (today - timedelta(days=35)).isoformat()},
        {**_body(), "end_date": (today + timedelta(days=1)).isoformat()},
        {**_body(), "unknown_field": True},
    ]
    for body in invalid:
        response = _send(client, body)
        assert response.status_code == 400, (body, response.text)
    raw = json.dumps(_body()).encode()
    duplicated = raw[:-1] + b',"tenant_id":"tenant-b"}'
    signed = sign_finops_request(secret=SECRET, raw_body=duplicated)
    response = client.post(PATH, content=duplicated, headers={
        "Content-Type": "application/json",
        "X-FM-Command-FinOps-Signature": signed,
    })
    assert response.status_code == 400
    assert model.calls == []


def test_rejects_oversized_and_wrong_media_type():
    client = _app(FakeFinOps())
    oversized = b"x" * 2049
    response = client.post(PATH, content=oversized, headers={
        "Content-Type": "application/json",
    })
    assert response.status_code == 413
    response = client.post(PATH, content=b"{}", headers={
        "Content-Type": "text/plain",
    })
    assert response.status_code == 415


def test_ledger_cross_tenant_or_inconsistent_counts_fail_closed():
    # Construct valid immutable buckets, then simulate a corrupted upstream
    # read adapter mutating the object *after* the constructor guard.
    bad_counts = _bucket()
    object.__setattr__(bad_counts, "cost_known_events", 300)
    bad_fallback = _bucket()
    object.__setattr__(bad_fallback, "fallback_attempts", 300)
    mismatches = [
        replace(_bucket(), tenant_id="tenant-b"),
        replace(_bucket(), unidade_id="unit-b"),
        bad_counts,
        bad_fallback,
        replace(_bucket(), bucket_date=date(2020, 1, 1)),
    ]
    for invalid_bucket in mismatches:
        result = _send(_app(FakeFinOps((invalid_bucket,))), _body())
        assert result.status_code == 409
        assert result.json()["detail"] == "core_finops.ledger_unavailable"


def test_excessive_bucket_cardinality_fails_closed():
    client = _app(FakeFinOps((_bucket(known=1, unknown=0),) * 501))
    response = _send(client, _body())
    assert response.status_code == 409


def test_non_negative_known_only_is_a_priced_internal_subtotal_not_invoice():
    response = _send(_app(FakeFinOps((_bucket(
        known=1, unknown=0, cost="0.004",
    ),))), _body())
    assert response.status_code == 200
    data = response.json()
    assert data["coverage"] == "priced_internal_observations"
    assert data["buckets"][0]["actual_provider_spend_verified"] is False
    assert data["unknown_attempts"] == 0


def test_real_persistent_finops_ledger_keeps_tenant_and_unit_isolated(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'finops.sqlite3'}")
    ControlPlaneBase.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    at = datetime.now(UTC)
    try:
        with sessions.begin() as db:
            for number, tenant, unit, price in (
                (1, "tenant-a", "unit-a", "0.0012"),
                (2, "tenant-a", "unit-a", None),
                (3, "tenant-b", "unit-a", "999.99"),
                (4, "tenant-a", "unit-secret", "500.00"),
            ):
                db.add(AIUsageRecord(
                    event_id=f"event-{number}",tenant_id=tenant,unit_id=unit,
                    request_id=f"request-{number}",correlation_id=f"corr-{number}",
                    capability="atendimento_interpretacao",provider="gemini",
                    model="synthetic-model",route_reason="synthetic",
                    fallback_used=number == 2,fallback_reason=None,
                    input_tokens=10 if price is not None else None,
                    output_tokens=5 if price is not None else None,
                    cached_tokens=None,latency_ms=42,
                    outcome="success" if price is not None else "transient_failure",
                    cost_decimal=price,currency="USD" if price is not None else "XXX",
                    cost_origin="estimated" if price is not None else "unavailable",
                    price_snapshot_id="synthetic-price",event_time=at,
                ))
        client = _app(PersistentAIFinOps(sessions))
        response = _send(client, _body())
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["coverage"] == "partial_unknown_costs"
        assert result["known_attempts"] == 1
        assert result["unknown_attempts"] == 1
        assert sum(bucket["attempts"] for bucket in result["buckets"]) == 2
        assert "999.99" not in response.text
        assert "500.00" not in response.text
        assert "tenant-b" not in response.text
        assert "unit-secret" not in response.text
        assert any(bucket["known_cost_subtotal"] == "0.0012"
                   for bucket in result["buckets"])
        assert all(bucket["actual_provider_spend_verified"] is False
                   for bucket in result["buckets"])
    finally:
        engine.dispose()


def test_finops_database_outage_is_not_reported_as_zero():
    class Unavailable:
        def listar(self, **kwargs):
            raise RuntimeError("INTERNAL_DB_PASSWORD_DO_NOT_LEAK")

    response = _send(_app(Unavailable()), _body())
    assert response.status_code == 503
    assert response.json()["detail"] == "core_finops.source_unavailable"
    assert "INTERNAL_DB_PASSWORD_DO_NOT_LEAK" not in response.text


def test_invalid_signer_key_cannot_activate_internal_finops_port():
    from pytest import raises
    with raises(ValueError):
        command_finops_router(read_model=FakeFinOps(), shared_secret=b"short")
    with raises(ValueError):
        sign_finops_request(secret=b"short", raw_body=b"{}")


def test_shared_node_python_hmac_interoperability_golden_vector():
    # Same static vector also asserted by FM Command TypeScript; a changed
    # route, method, body serialization or signature must fail both projects.
    key=b"synthetic-finops-contract-key-32-bytes-123"
    body=(b'{"schema":"fm.core.finops.read.v1","tenant_id":"tenant-a",'
          b'"unit_id":"unit-a","start_date":"2026-10-21",'
          b'"end_date":"2026-10-22","issued_at":1792627200,'
          b'"nonce":"abcdefghijklmnopqrstuvwxABCDEFGH"}')
    assert sign_finops_request(secret=key,raw_body=body)==(
        "b3e81412631dcb6de6ea42e8d866e58db84598d70b651db7c68b36bca2270e30"
    )
