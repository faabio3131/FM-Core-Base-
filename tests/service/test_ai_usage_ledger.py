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


def test_two_physical_provider_calls_same_request_and_route_are_both_accounted(ledger):
    store, _ = ledger
    store.add_price_snapshot(price())
    first = replace(event(request="two-physical-calls"), attempt_id="physical-001")
    second = replace(first, attempt_id="physical-002")
    store.registrar(first)
    store.registrar(first)  # duplicate delivery of SAME physical attempt
    store.registrar(second)  # separate provider call, identical route/request/counters
    summary = store.reconcile_request(
        tenant_id="a", unit_id="u", request_id="two-physical-calls",
    )
    assert summary.total_attempts == 2
    assert summary.known_attempts == 2
    assert summary.known_total == Decimal("0.10")
    assert summary.all_recorded_costs_known is True
    entries = _list(store)
    assert sum(item.attempts for item in entries) == 2


def test_reconciliation_backlog_is_scoped_bounded_and_groups_fallback_attempts(ledger):
    store, _ = ledger
    store.add_price_snapshot(price())
    store.registrar(event(request="request-a"))  # known internal estimate
    store.registrar(replace(
        event(request="request-a", price_id="unknown-a"),
        fallback_used=True, route_reason="capability=text;priority=90",
        input_tokens=None, output_tokens=None,
    ))
    store.registrar(event(request="request-b", inp=None))
    store.registrar(event(request="verified"))  # no unresolved attempts
    store.registrar(event(request="foreign", tenant="other", inp=None))
    store.registrar(event(request="other-unit", unit="other", inp=None))

    first, truncated = store.listar_pendencias(
        tenant_id="a", unit_id="u", inicio=date(2026, 10, 18),
        fim=date(2026, 10, 18), limit=1,
    )
    assert truncated is True
    assert len(first) == 1
    assert first[0].request_id == "request-a"
    assert first[0].observed_attempts == 2
    assert first[0].fallback_attempts == 1
    assert first[0].unknown_cost_attempts == 1
    assert first[0].incomplete_token_attempts == 1

    all_items, truncated = store.listar_pendencias(
        tenant_id="a", unit_id="u", inicio=date(2026, 10, 18),
        fim=date(2026, 10, 18), limit=100,
    )
    assert truncated is False
    assert [item.request_id for item in all_items] == ["request-a", "request-b"]
    assert all_items[1].unknown_cost_attempts == 1
    assert store.listar_pendencias(
        tenant_id="a", unit_id="u", inicio=date(2026, 10, 19),
        fim=date(2026, 10, 19),
    ) == ((), False)
    assert store.listar_pendencias(
        tenant_id="other", unit_id="u", inicio=date(2026, 10, 18),
        fim=date(2026, 10, 18),
    )[0][0].request_id == "foreign"


@pytest.mark.parametrize("changes", [
    {"tenant_id": " "}, {"unit_id": ""}, {"limit": True},
    {"limit": 101}, {"limit": 0}, {"limit": 1.5},
    {"inicio": date(2026, 10, 20)}, {"fim": date(2026, 11, 25)},
    {"fim": date.max}, {"inicio": datetime(2026, 10, 18, 12)},
])
def test_reconciliation_backlog_rejects_unbounded_or_ambiguous_inputs(ledger, changes):
    store, _ = ledger
    valid = {
        "tenant_id": "a", "unit_id": "u",
        "inicio": date(2026, 10, 18), "fim": date(2026, 10, 18),
        "limit": 50,
    }
    with pytest.raises(ControlPlaneError, match="pending_scope_invalid"):
        store.listar_pendencias(**{**valid, **changes})


def test_finops_alert_inbox_paginates_stably_and_never_crosses_tenant(ledger):
    store, _ = ledger
    store.add_price_snapshot(price())
    for request_id in ("a", "b", "c"):
        store.registrar(event(request=request_id))
    store.registrar(event(request="other-unit", unit="other"))
    store.registrar(event(request="other-tenant", tenant="b"))
    params = {
        "tenant_id": "a", "unit_id": "u",
        "inicio": date(2026, 10, 18), "fim": date(2026, 10, 18),
        "limit": 2,
    }
    first, cursor = store.listar_ids_solicitacoes(**params)
    assert first == ("a", "b")
    assert cursor == "b"
    second, next_cursor = store.listar_ids_solicitacoes(
        **params, after_request_id=cursor,
    )
    assert second == ("c",)
    assert next_cursor is None
    assert store.listar_ids_solicitacoes(
        **{**params, "inicio": date(2026, 10, 19),
           "fim": date(2026, 10, 19)},
    ) == ((), None)


@pytest.mark.parametrize("changes", [
    {"tenant_id": " "}, {"unit_id": ""}, {"limit": True},
    {"limit": 0}, {"limit": 101}, {"limit": 1.5},
    {"fim": date.max}, {"fim": date(2026, 11, 30)},
    {"inicio": date(2026, 10, 19)},
    {"after_request_id": " "},
    {"after_request_id": "x "},
    {"after_request_id": "x" * 129},
])
def test_finops_alert_inbox_rejects_unsafe_scope_and_cursor(ledger, changes):
    store, _ = ledger
    valid = {
        "tenant_id": "a", "unit_id": "u",
        "inicio": date(2026, 10, 18),
        "fim": date(2026, 10, 18), "limit": 50,
    }
    with pytest.raises(ControlPlaneError, match="alert_inbox_scope_invalid"):
        store.listar_ids_solicitacoes(**{**valid, **changes})
