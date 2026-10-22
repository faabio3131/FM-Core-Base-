"""F3/F9: fail-closed cost/usage bucket invariants, synthetic only."""
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from core.ai_finops import AIFinOpsBucket


def bucket() -> AIFinOpsBucket:
    return AIFinOpsBucket(
        tenant_id="tenant-a", unidade_id="unit-a", bucket_date=date(2026, 10, 22),
        provider="gemini", model="test-model", capability="atendimento_interpretacao",
        outcome="success", moeda="USD", attempts=3, fallback_attempts=1,
        input_tokens=300, output_tokens=100, cached_tokens=25,
        latency_ms_total=350, latency_ms_max=200,
        cost_known_events=2, cost_unknown_events=1, cost_total=Decimal("0.02"),
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"attempts": 0, "cost_known_events": 0, "cost_unknown_events": 0},
        {"attempts": 4},
        {"cost_known_events": 3},
        {"cost_unknown_events": 0},
        {"fallback_attempts": 4},
        {"fallback_attempts": -1},
        {"cost_unknown_events": True},
        {"attempts": True},
        {"input_tokens": 1.5},
        {"output_tokens": "100"},
        {"latency_ms_max": 351},
        {"latency_ms_total": -1},
        {"bucket_date": datetime(2026, 10, 22, tzinfo=UTC)},
        {"cost_total": float("nan")},
        {"cost_total": Decimal("NaN")},
        {"cost_total": Decimal("-0.01")},
    ],
)
def test_corrupt_finops_bucket_is_rejected(changes):
    with pytest.raises(ValueError):
        replace(bucket(), **changes)


def test_unknown_cost_is_separately_counted_without_zero_valuing_it():
    current = bucket()
    assert current.attempts == 3
    assert current.cost_known_events == 2
    assert current.cost_unknown_events == 1
    assert current.cost_total == Decimal("0.02")
    assert current.fallback_attempts == 1
