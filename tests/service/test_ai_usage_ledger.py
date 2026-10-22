from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from core.ai_control_plane import ControlPlaneBase, ControlPlaneError
from core.ai_pricing import PriceSnapshotIA, SnapshotPrecoDuplicado, TarifaIA
from core.ai_router import AIUsageEvent, CapabilityIA, OutcomeIA
from core.ai_usage_ledger import AIUsageRecord, PersistentAIFinOps


@pytest.fixture
def ledger(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'ledger.db'}")
    ControlPlaneBase.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    yield PersistentAIFinOps(sessions), sessions
    engine.dispose()


def price():
    return PriceSnapshotIA(
        price_snapshot_id="price-1",
        provider="openai",
        model="model-1",
        modalidade="text",
        moeda="USD",
        versao=1,
        vigencia_inicio=datetime(2025, 1, 1, tzinfo=UTC),
        vigencia_fim=None,
        tarifas=(
            TarifaIA(componente="input_tokens", preco=Decimal("0.02"), unidades_por_preco=1000),
            TarifaIA(componente="output_tokens", preco=Decimal("0.03"), unidades_por_preco=1000),
        ),
    )


def event(*, tenant="a", unit="u", request="req-1", price_id="price-1", inp=1000, out=1000):
    return AIUsageEvent(
        tenant_id=tenant,
        unidade_id=unit,
        request_id=request,
        correlation_id="correlation-1",
        capability=CapabilityIA.ATENDIMENTO_INTERPRETACAO,
        provider="openai",
        model="model-1",
        route_reason="capability=text;priority=100",
        fallback_used=False,
        fallback_reason=None,
        input_tokens=inp,
        output_tokens=out,
        cached_tokens=None,
        latency_ms=25,
        outcome=OutcomeIA.SUCESSO,
        custo_real_calculado=None,
        moeda=None,
        price_snapshot_id=price_id,
        timestamp=datetime(2026, 10, 18, 12, tzinfo=UTC),
    )


def _list(ledger, tenant="a", unit="u"):
    return ledger.listar(
        tenant_id=tenant,
        unidade_id=unit,
        inicio=date(2026, 10, 18),
        fim=date(2026, 10, 18),
    )


def test_price_snapshot_and_exact_cost_persist(ledger):
    store, sessions = ledger
    store.add_price_snapshot(price())
    store.registrar(event())
    clone = PersistentAIFinOps(sessions)
    buckets = _list(clone)
    assert len(buckets) == 1
    assert buckets[0].attempts == 1
    assert buckets[0].moeda == "USD"
    assert buckets[0].cost_total == Decimal("0.05")
    assert buckets[0].cost_known_events == 1
    assert buckets[0].cost_unknown_events == 0


def test_events_are_idempotent_per_request_and_attempt(ledger):
    store, _ = ledger
    store.add_price_snapshot(price())
    store.registrar(event())
    store.registrar(event())
    assert _list(store)[0].attempts == 1


def test_missing_prices_and_tokens_are_explicitly_unknown(ledger):
    store, _ = ledger
    store.registrar(event(price_id="missing"))
    unknown = _list(store)[0]
    assert unknown.cost_known_events == 0
    assert unknown.cost_unknown_events == 1
    assert unknown.moeda == "XXX"
    store.add_price_snapshot(price())
    store.registrar(event(request="req-2", inp=None))
    with_tokens_missing = _list(store)
    assert len(with_tokens_missing) == 2
    known_currency_unknown_cost = next(x for x in with_tokens_missing if x.moeda == "USD")
    assert known_currency_unknown_cost.cost_unknown_events == 1
    assert known_currency_unknown_cost.cost_total == 0


def test_tenant_and_unit_isolation_is_mandatory(ledger):
    store, _ = ledger
    store.add_price_snapshot(price())
    store.registrar(event(tenant="a", unit="u", request="x"))
    store.registrar(event(tenant="b", unit="u", request="y"))
    store.registrar(event(tenant="a", unit="other", request="z"))
    assert _list(store, "a", "u")[0].attempts == 1
    assert _list(store, "b", "u")[0].attempts == 1
    assert _list(store, "a", "other")[0].attempts == 1
    assert _list(store, "b", "other") == ()


def test_price_snapshot_is_append_only_through_repository(ledger):
    store, _ = ledger
    store.add_price_snapshot(price())
    with pytest.raises(SnapshotPrecoDuplicado):
        store.add_price_snapshot(price())


def test_invalid_date_range_fails_closed(ledger):
    store, _ = ledger
    with pytest.raises(ValueError, match="invalid_scope_or_interval"):
        store.listar(
            tenant_id="a",
            unidade_id="u",
            inicio=date(2026, 10, 19),
            fim=date(2026, 10, 18),
        )


def test_request_finops_reconciliation_reports_unknown_without_zeroing(ledger):
    store, _ = ledger
    store.add_price_snapshot(price())
    store.registrar(event(request="request-a"))
    store.registrar(replace(
        event(request="request-a", price_id="unknown-snapshot"),
        fallback_used=True, fallback_reason="timeout", input_tokens=None, output_tokens=None,
    ))
    summary = store.reconcile_request(
        tenant_id="a", unit_id="u", request_id="request-a",
    )
    assert summary.total_attempts == 2
    assert summary.known_attempts == 1
    assert summary.unknown_attempts == 1
    assert summary.fallback_attempts == 1
    assert summary.known_total == Decimal("0.05")
    assert summary.currency == "USD"
    assert summary.all_recorded_costs_known is False
    assert store.reconcile_request(
        tenant_id="b", unit_id="u", request_id="request-a",
    ).total_attempts == 0
    assert store.reconcile_request(
        tenant_id="a", unit_id="different", request_id="request-a",
    ).all_recorded_costs_known is False


def test_request_finops_reconciliation_marks_fully_known_attempts(ledger):
    store, _ = ledger
    store.add_price_snapshot(price())
    store.registrar(event(request="complete"))
    complete = store.reconcile_request(
        tenant_id="a", unit_id="u", request_id="complete",
    )
    assert complete.known_attempts == 1
    assert complete.unknown_attempts == 0
    assert complete.all_recorded_costs_known is True
    assert complete.known_total == Decimal("0.05")
    with pytest.raises(ValueError, match="invalid_request_scope"):
        store.reconcile_request(tenant_id="a", unit_id="u", request_id=" ")


def test_distinct_fallback_routes_using_same_provider_model_are_not_deduped(ledger):
    store, _ = ledger
    store.add_price_snapshot(price())
    first = event(request="multi-route")
    second = replace(
        first,
        route_reason="capability=text;priority=90",
        fallback_used=True,
        fallback_reason="primary_timeout",
    )
    third = replace(
        first,
        route_reason="capability=text;priority=80",
        fallback_used=True,
        fallback_reason="secondary_timeout",
    )
    store.registrar(first)
    store.registrar(second)
    store.registrar(third)
    store.registrar(second)  # replay of the same attempt stays idempotent
    summary = store.reconcile_request(
        tenant_id="a", unit_id="u", request_id="multi-route",
    )
    assert summary.total_attempts == 3
    assert summary.fallback_attempts == 2
    assert summary.known_total == Decimal("0.15")
    assert summary.all_recorded_costs_known is True


def test_finops_alert_marks_unknown_provider_attempt_as_manual_reconciliation(ledger):
    from core.ai_finops_alerts import evaluate_request_cost_alert

    store, _ = ledger
    store.add_price_snapshot(price())
    store.registrar(event(request="needs-reconcile"))
    store.registrar(replace(
        event(request="needs-reconcile", price_id="no-evidence"),
        fallback_used=True, fallback_reason="timeout",
        input_tokens=None, output_tokens=None,
    ))
    summary = store.reconcile_request(
        tenant_id="a", unit_id="u", request_id="needs-reconcile",
    )
    alert = evaluate_request_cost_alert(
        summary, maximum_known_cost=Decimal("0.04"), currency="USD",
    )
    assert alert.status == "needs_attention"
    assert set(alert.reasons) == {
        "provider_cost_unverified", "known_subtotal_above_limit",
    }
    assert alert.known_cost == Decimal("0.05")
    assert alert.unknown_attempts == 1
    assert alert.all_costs_known is False


def test_finops_alert_requires_evidence_and_exact_currency(ledger):
    from core.ai_finops_alerts import evaluate_request_cost_alert

    store, _ = ledger
    unknown = store.reconcile_request(
        tenant_id="a", unit_id="u", request_id="not-observed",
    )
    absent = evaluate_request_cost_alert(unknown)
    assert absent.status == "needs_attention"
    assert absent.reasons == ("usage_evidence_unavailable",)
    assert not absent.all_costs_known
    store.add_price_snapshot(price())
    store.registrar(event(request="known"))
    summary = store.reconcile_request(
        tenant_id="a", unit_id="u", request_id="known",
    )
    clean = evaluate_request_cost_alert(
        summary, maximum_known_cost=Decimal("0.10"), currency="USD",
    )
    assert clean.status == "reconciled"
    assert clean.known_cost == Decimal("0.05")
    with pytest.raises(ValueError, match="alert_policy_invalid"):
        evaluate_request_cost_alert(
            summary, maximum_known_cost=Decimal("0.10"), currency="BRL",
        )
    with pytest.raises(ValueError, match="alert_policy_invalid"):
        evaluate_request_cost_alert(
            summary, maximum_known_cost=Decimal("NaN"), currency="USD",
        )


@pytest.mark.parametrize("tampered_cost", ["-0.01", "NaN", "Infinity", "corrupt"])
def test_corrupt_attempt_cost_cannot_be_offset_by_valid_positive_cost(ledger, tampered_cost):
    store, sessions = ledger
    store.add_price_snapshot(price())
    first = event(request="bad-cost")
    second = replace(
        first,
        route_reason="capability=text;priority=90",
        fallback_used=True,
        fallback_reason="retry",
    )
    store.registrar(first)
    store.registrar(second)
    with sessions.begin() as db:
        row = db.scalars(select(AIUsageRecord).where(
            AIUsageRecord.request_id == "bad-cost",
            AIUsageRecord.fallback_used.is_(True),
        )).one()
        row.cost_decimal = tampered_cost

    # An invalid attempt must not be hidden by a valid +0.05 subtotal.
    with pytest.raises(ControlPlaneError, match="invalid_known_cost"):
        store.reconcile_request(tenant_id="a", unit_id="u", request_id="bad-cost")
    with pytest.raises(ControlPlaneError, match="invalid_known_cost"):
        _list(store)
