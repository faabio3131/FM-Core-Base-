"""Signed provider-source reconciliation with FM's immutable per-attempt ledger."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from core.ai_control_plane import ControlPlaneBase, ControlPlaneError
from core.ai_provider_source_verifier import ProviderSigningPolicy
from core.ai_provider_usage_match import match_signed_provider_attempt
from core.ai_quota_store import QuotaDenied
from core.ai_usage_ledger import AIUsageRecord


@pytest.fixture
def ledger(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'provider-usage.db'}")
    ControlPlaneBase.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    with sessions.begin() as db:
        db.add(AIUsageRecord(
            event_id="attempt-1", tenant_id="tenant-a", unit_id="unit-a",
            request_id="req-1", correlation_id="corr", capability="text",
            provider="verified-provider", model="model-a", route_reason="initial",
            fallback_used=False, fallback_reason=None,
            input_tokens=100, output_tokens=50, cached_tokens=None,
            latency_ms=20, outcome="success", cost_decimal="0.005",
            currency="USD", cost_origin="estimated", price_snapshot_id="p1",
            event_time=datetime(2026, 10, 22, 12, tzinfo=UTC),
        ))
    yield sessions
    engine.dispose()


def signed_statement(*, overrides=None):
    now = datetime.now(UTC)
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw,
    )
    policy = ProviderSigningPolicy(issuer="provider-statement-fixture", public_key=public)
    statement = {
        "issuer": policy.issuer, "request_id": "req-1", "tenant_id": "tenant-a",
        "unit_id": "unit-a", "provider": "verified-provider",
        "model": "model-a", "input_tokens": 100, "output_tokens": 50,
        "cost_usd": "0.006", "no_charge": False, "observed_at": int(now.timestamp()),
    }
    statement.update(overrides or {})
    raw = json.dumps(statement, sort_keys=True, separators=(",", ":")).encode()
    signature = private.sign(b"FM_PROVIDER_USAGE_SOURCE_V1\n" + raw)
    return policy, raw, signature, now


def check(sessions, policy, raw, signature, now, **scope):
    args = {
        "sessions": sessions, "source_payload": raw, "source_signature": signature,
        "source_policy": policy, "tenant_id": "tenant-a", "unit_id": "unit-a",
        "request_id": "req-1", "at": now,
    }
    args.update(scope)
    return match_signed_provider_attempt(**args)


def test_exact_signed_source_can_match_single_attempt_without_asserting_invoice(ledger):
    policy, raw, signature, now = signed_statement()
    matched = check(ledger, policy, raw, signature, now)
    assert matched.attempt_id == "attempt-1"
    assert matched.status == "signed_statement_matches_single_attempt"
    assert matched.invoice_verified is False
    assert matched.provider_statement_cost_usd == Decimal("0.006")
    assert matched.internally_estimated_cost_usd == Decimal("0.005")


@pytest.mark.parametrize("tamper", [
    {"input_tokens": 99}, {"output_tokens": 51},
    {"model": "other-model"}, {"provider": "other-provider"},
])
def test_signed_but_inconsistent_statement_cannot_be_reconciled(ledger, tamper):
    policy, raw, signature, now = signed_statement(overrides=tamper)
    with pytest.raises(ControlPlaneError, match="provider_match_mismatch"):
        check(ledger, policy, raw, signature, now)


def test_fallback_multi_attempt_and_no_charge_are_separate_paths(ledger):
    policy, raw, signature, now = signed_statement()
    with ledger.begin() as db:
        db.add(AIUsageRecord(
            event_id="attempt-2", tenant_id="tenant-a", unit_id="unit-a",
            request_id="req-1", correlation_id="corr", capability="text",
            provider="verified-provider", model="model-a", route_reason="fallback",
            fallback_used=True, fallback_reason="temporary",
            input_tokens=None, output_tokens=None, cached_tokens=None,
            latency_ms=10, outcome="transient_failure", cost_decimal=None,
            currency="XXX", cost_origin="unavailable", price_snapshot_id="p1",
            event_time=now,
        ))
    with pytest.raises(ControlPlaneError, match="provider_match_ambiguous_attempts"):
        check(ledger, policy, raw, signature, now)
    no_charge_policy, no_charge_body, no_charge_sig, at = signed_statement(
        overrides={"no_charge": True, "input_tokens": 0,
                   "output_tokens": 0, "cost_usd": "0"},
    )
    with pytest.raises(ControlPlaneError, match="no_charge_separate"):
        check(ledger, no_charge_policy, no_charge_body, no_charge_sig, at)


def test_bad_signature_wrong_tenant_and_stale_source_are_not_evidence(ledger):
    policy, raw, signature, now = signed_statement()
    with pytest.raises(QuotaDenied):
        check(ledger, policy, raw, b"", now)
    with pytest.raises(QuotaDenied):
        check(ledger, policy, raw, signature, now, tenant_id="tenant-b")
    with pytest.raises(QuotaDenied):
        check(ledger, policy, raw, signature, now.replace(year=now.year + 1))


def test_cost_unknown_stays_unknown_even_with_verified_usage_statement(ledger):
    policy, raw, signature, now = signed_statement()
    with ledger.begin() as db:
        row = db.scalar(select(AIUsageRecord).where(
            AIUsageRecord.event_id == "attempt-1",
        ))
        row.cost_decimal = None
        row.currency = "XXX"
    match = check(ledger, policy, raw, signature, now)
    assert match.internally_estimated_cost_usd is None
    assert match.provider_statement_cost_usd == Decimal("0.006")
    assert match.invoice_verified is False
