"""Opt-in FM Command alert-policy writes; never tenant/self-service writes."""
from __future__ import annotations

import hmac
import re
from decimal import Decimal, InvalidOperation

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from core.ai_command_authority import (
    CommandAuthorityDenied,
    CommandTrustPolicy,
    verify_command_platform_admin,
)
from core.ai_control_plane import ControlPlaneError
from core.ai_finops_policy_store import PersistentFinOpsAlertPolicies

_NONNEGATIVE_DECIMAL = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]{1,12})?$")


class AlertPolicyUpdatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    tenant_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,128}$")
    unit_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,128}$")
    expected_revision: int = Field(ge=0)
    enabled: bool
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    maximum_known_cost: str = Field(min_length=1, max_length=80)


def finops_alert_policy_router(
    *, trust: CommandTrustPolicy, transport_key: str,
) -> APIRouter:
    if len(transport_key.encode()) < 32:
        raise ValueError("core_finops.alert_transport_invalid")
    router = APIRouter(prefix="/internal/v1/command", tags=["command-finops"])

    @router.post("/ai/finops/alert-policy", status_code=204)
    def update(
        payload: AlertPolicyUpdatePayload,
        request: Request,
        authorization: str | None = Header(default=None),
        x_fm_recovery_service_key: str | None = Header(
            default=None, alias="X-FM-Recovery-Service-Key",
        ),
    ) -> None:
        # Both a separate S2S credential and a recently step-upped human token
        # are required; Core admin read key and Assistant key are not sufficient.
        if (
            not x_fm_recovery_service_key
            or not hmac.compare_digest(transport_key, x_fm_recovery_service_key)
            or not authorization or not authorization.startswith("Bearer ")
        ):
            raise HTTPException(status_code=401, detail="core_finops.alert_denied")
        try:
            principal = verify_command_platform_admin(
                authorization.removeprefix("Bearer "), trust,
            )
        except CommandAuthorityDenied as exc:
            raise HTTPException(status_code=403, detail="core_finops.alert_denied") from exc
        if not _NONNEGATIVE_DECIMAL.fullmatch(payload.maximum_known_cost):
            raise HTTPException(status_code=422, detail="core_finops.alert_cost_invalid")
        try:
            amount = Decimal(payload.maximum_known_cost)
        except InvalidOperation as exc:
            raise HTTPException(status_code=422, detail="core_finops.alert_cost_invalid") from exc
        if not amount.is_finite() or not Decimal(0) < amount <= Decimal(1000000000):
            raise HTTPException(status_code=422, detail="core_finops.alert_cost_invalid")
        engine = getattr(request.app.state, "control_plane_engine", None)
        if engine is None:
            raise HTTPException(status_code=503, detail="core_finops.alert_unavailable")
        try:
            PersistentFinOpsAlertPolicies(
                sessionmaker(engine, expire_on_commit=False),
            ).update(
                tenant_id=payload.tenant_id, unit_id=payload.unit_id,
                expected_revision=payload.expected_revision,
                enabled=payload.enabled, currency=payload.currency,
                maximum_known_cost=amount, principal=principal,
            )
        except (ControlPlaneError, InvalidOperation) as exc:
            raise HTTPException(status_code=409, detail="core_finops.alert_conflict") from exc
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail="core_finops.alert_unavailable") from exc

    return router
