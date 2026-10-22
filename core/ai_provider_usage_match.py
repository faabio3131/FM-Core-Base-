"""Cross-check independently signed usage against exactly one physical Core attempt.

Signature confirms a pinned *statement* issuer, not payment, invoice, settlement
or provider adoption. Unsigned vendor responses never enter this adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from core.ai_control_plane import ControlPlaneError
from core.ai_provider_source_verifier import (
    ProviderSigningPolicy,
    verified_signed_usage_source,
)
from core.ai_usage_ledger import AIUsageRecord, _observed_cost


@dataclass(frozen=True, kw_only=True)
class MatchedProviderStatement:
    tenant_id: str
    unit_id: str
    request_id: str
    provider: str
    model: str
    attempt_id: str
    source_issuer: str
    provider_statement_cost_usd: Decimal | None
    internally_estimated_cost_usd: Decimal | None
    matched_fallback_attempt: bool = False
    status: str = "signed_statement_matches_single_attempt"
    invoice_verified: bool = False


def match_signed_provider_attempt(
    *, sessions: sessionmaker[Session], source_payload: bytes,
    source_signature: bytes, source_policy: ProviderSigningPolicy,
    tenant_id: str, unit_id: str, request_id: str,
    at: datetime | None = None,
) -> MatchedProviderStatement:
    # Cryptography, freshness, pinned issuer and exact scope checked FIRST.
    receipt = verified_signed_usage_source(
        payload=source_payload, signature=source_signature,
        policy=source_policy, expected_request_id=request_id,
        expected_tenant_id=tenant_id, expected_unit_id=unit_id, at=at,
    )
    if receipt.no_charge:
        # No-charge attestations need a distinct transactional recovery path,
        # never classified as a billed attempt from read-only matching.
        raise ControlPlaneError("core_finops.provider_match_no_charge_separate")
    with sessions() as db:
        events = db.scalars(select(AIUsageRecord).where(
            AIUsageRecord.tenant_id == tenant_id,
            AIUsageRecord.unit_id == unit_id,
            AIUsageRecord.request_id == request_id,
        ).limit(101)).all()
    # Never infer a provider receipt from a truncated routing history.
    if len(events) > 100:
        raise ControlPlaneError("core_finops.provider_match_unbounded_attempts")
    if not events:
        raise ControlPlaneError("core_finops.provider_match_ambiguous_attempts")

    same_route = [
        event for event in events
        if event.provider == receipt.provider and event.model == receipt.model
    ]
    # An unknown/unmetered event on the SAME physical provider/model may
    # represent the signed statement. Do not attribute it to another event.
    if any(
        event.input_tokens is None or event.output_tokens is None
        for event in same_route
    ):
        raise ControlPlaneError("core_finops.provider_match_ambiguous_attempts")
    matches = [
        event for event in same_route
        if event.input_tokens == receipt.input_tokens
        and event.output_tokens == receipt.output_tokens
    ]
    if not matches and len(events) == 1:
        raise ControlPlaneError("core_finops.provider_match_mismatch")
    if len(matches) != 1:
        raise ControlPlaneError("core_finops.provider_match_ambiguous_attempts")
    event = matches[0]
    # Even with a unique usage tuple, a failed or unidentifiable event
    # cannot prove its billed attempt. A successfully billed fallback can.
    if event.outcome != "success" or not event.event_id:
        raise ControlPlaneError("core_finops.provider_match_mismatch")
    known = _observed_cost(event.cost_decimal)
    if event.currency != "USD" and known is not None:
        # Not even a USD estimate can be inferred from other currencies.
        raise ControlPlaneError("core_finops.provider_match_currency_mismatch")
    return MatchedProviderStatement(
        tenant_id=tenant_id, unit_id=unit_id, request_id=request_id,
        provider=receipt.provider, model=receipt.model,
        attempt_id=event.event_id, source_issuer=source_policy.issuer,
        provider_statement_cost_usd=receipt.provider_cost_usd,
        internally_estimated_cost_usd=known,
        matched_fallback_attempt=bool(event.fallback_used),
    )
