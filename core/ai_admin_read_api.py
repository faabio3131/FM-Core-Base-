"""Read-only technical operations API, separate from consumer credentials.

This is NOT a human platform_admin authority. Until a signed Command identity
and governed admin mutation/audit policy exist, no write API is exposed.
"""

from __future__ import annotations

import hmac
from datetime import date
from decimal import Decimal

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from core.ai_control_plane import (
    AIControlPlaneStore,
    AIProvider,
    AIProviderHealth,
    ControlPlaneError,
)
from core.ai_finops_alerts import evaluate_request_cost_alert
from core.ai_router import CapabilityIA
from core.ai_usage_ledger import PersistentAIFinOps


def require_operator_key(
    request: Request,
    x_core_admin_key: str | None = Header(default=None, alias="X-Core-Admin-Key"),
) -> None:
    cfg = request.app.state.settings
    if not cfg.persistent_control_plane or not cfg.admin_key:
        raise HTTPException(status_code=404, detail="core_operator_api_disabled")
    if not x_core_admin_key or not hmac.compare_digest(cfg.admin_key, x_core_admin_key):
        raise HTTPException(status_code=401, detail="core_operator_unauthorized")


def _sessions(request: Request):
    engine = request.app.state.control_plane_engine
    if engine is None:
        raise HTTPException(status_code=503, detail="core_control_plane_not_ready")
    return sessionmaker(engine, expire_on_commit=False)


def register_admin_read_routes(app: FastAPI) -> None:
    router = APIRouter(
        prefix="/v1/admin/ai",
        tags=["internal-core-operator-readonly"],
        dependencies=[Depends(require_operator_key)],
    )

    @router.get("/providers/health")
    def provider_health(request: Request) -> list[dict[str, str | bool]]:
        try:
            with _sessions(request)() as db:
                records = db.execute(
                    select(AIProvider, AIProviderHealth)
                    .outerjoin(
                        AIProviderHealth, AIProvider.provider_id == AIProviderHealth.provider_id
                    )
                    .order_by(AIProvider.provider_id)
                    .limit(1000)
                ).all()
                return [
                    {
                        "provider_id": provider.provider_id,
                        "enabled": provider.enabled,
                        "health": state.status if state is not None else "unknown",
                        "circuit": state.circuit if state is not None else "unknown",
                    }
                    for provider, state in records
                ]
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail="core_operator_query_unavailable") from exc

    @router.get("/routes")
    def resolved_routes(
        request: Request,
        tenant_id: str = Query(min_length=1, max_length=128),
        unit_id: str = Query(min_length=1, max_length=128),
        capability: CapabilityIA = CapabilityIA.ATENDIMENTO_INTERPRETACAO,
    ) -> list[dict[str, str | int]]:
        try:
            routes = AIControlPlaneStore(_sessions(request)).resolve(
                tenant_id=tenant_id, unit_id=unit_id, capability=capability
            )
        except (ControlPlaneError, SQLAlchemyError) as exc:
            raise HTTPException(status_code=503, detail="core_operator_route_unavailable") from exc
        return [
            {
                "policy_id": item.route.configuracao_id,
                "scope": item.scope,
                "provider": item.route.provider,
                "model": item.route.model,
                "priority": item.route.prioridade,
                "price_snapshot_id": item.route.price_snapshot_id,
            }
            for item in routes
        ]

    @router.get("/usage")
    def usage(
        request: Request,
        tenant_id: str = Query(min_length=1, max_length=128),
        unit_id: str = Query(min_length=1, max_length=128),
        start_date: date = Query(),
        end_date: date = Query(),
    ) -> list[dict[str, str | int]]:
        try:
            buckets = PersistentAIFinOps(_sessions(request)).listar(
                tenant_id=tenant_id,
                unidade_id=unit_id,
                inicio=start_date,
                fim=end_date,
            )
        except (ControlPlaneError, SQLAlchemyError) as exc:
            raise HTTPException(status_code=422, detail="core_operator_usage_invalid") from exc
        return [
            {
                "day": item.bucket_date.isoformat(),
                "provider": item.provider,
                "model": item.model,
                "capability": item.capability,
                "outcome": item.outcome,
                "currency": item.moeda,
                "attempts": item.attempts,
                "fallback_attempts": item.fallback_attempts,
                "input_tokens": item.input_tokens,
                "output_tokens": item.output_tokens,
                "cost_known_events": item.cost_known_events,
                "cost_unknown_events": item.cost_unknown_events,
                "cost_total": str(item.cost_total),
            }
            for item in buckets
        ]

    @router.get("/usage/pending")
    def pending_usage_reconciliation(
        request: Request,
        tenant_id: str = Query(min_length=1, max_length=128),
        unit_id: str = Query(min_length=1, max_length=128),
        start_date: date = Query(),
        end_date: date = Query(),
        limit: int = Query(default=50, ge=1, le=100),
    ) -> JSONResponse:
        """FM operator-only queue; observations do not attest provider invoices."""
        try:
            items, truncated = PersistentAIFinOps(_sessions(request)).listar_pendencias(
                tenant_id=tenant_id, unit_id=unit_id,
                inicio=start_date, fim=end_date, limit=limit,
            )
        except ControlPlaneError as exc:
            raise HTTPException(
                status_code=422, detail="core_operator_pending_scope_invalid",
            ) from exc
        except (SQLAlchemyError, ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=503, detail="core_operator_pending_unavailable",
            ) from exc
        return JSONResponse(
            {
                "tenant_id": tenant_id, "unit_id": unit_id,
                "start_date": start_date.isoformat(), "end_date": end_date.isoformat(),
                "status": "needs_attention" if items else "no_pending_observations",
                "truncated": truncated,
                "provider_invoice_verified": False,
                "items": [
                    {
                        "request_id": item.request_id,
                        "first_seen": item.first_seen.isoformat(),
                        "last_seen": item.last_seen.isoformat(),
                        "observed_attempts": item.observed_attempts,
                        "unknown_cost_attempts": item.unknown_cost_attempts,
                        "incomplete_token_attempts": item.incomplete_token_attempts,
                        "fallback_attempts": item.fallback_attempts,
                    }
                    for item in items
                ],
            },
            headers={"Cache-Control": "no-store"},
        )

    @router.get("/usage/request")
    def request_finops_alert(
        request: Request,
        tenant_id: str = Query(min_length=1, max_length=128),
        unit_id: str = Query(min_length=1, max_length=128),
        request_id: str = Query(min_length=1, max_length=128),
        maximum_known_cost: Decimal | None = Query(default=None, gt=0),
        currency: str | None = Query(default=None, min_length=3, max_length=3),
    ) -> dict[str, object]:
        """Pure operator read: unknown cost remains UNKNOWN, not accounting zero."""
        try:
            summary = PersistentAIFinOps(_sessions(request)).reconcile_request(
                tenant_id=tenant_id, unit_id=unit_id, request_id=request_id,
            )
            alert = evaluate_request_cost_alert(
                summary, maximum_known_cost=maximum_known_cost,
                currency=currency,
            )
        except (ControlPlaneError, SQLAlchemyError) as exc:
            raise HTTPException(
                status_code=503, detail="core_operator_usage_unavailable",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail="core_operator_alert_policy_invalid",
            ) from exc
        return {
            "tenant_id": tenant_id, "unit_id": unit_id,
            "request_id": request_id,
            "status": alert.status,
            "reasons": list(alert.reasons),
            "known_cost_subtotal": str(alert.known_cost),
            "currency": alert.cost_currency,
            "unknown_attempts": alert.unknown_attempts,
            "observed_attempts": alert.observed_attempts,
            "all_costs_known": alert.all_costs_known,
        }

    app.include_router(router)
