"""FM-owned, versioned FinOps alert thresholds (never commercial tenant pricing).

Only a Command human platform administrator with RS256 step-up can mutate.
Every change and its one-use JWT jti is audited in the SAME transaction.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation

from sqlalchemy import Boolean, DateTime, Integer, String, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, Session, mapped_column, sessionmaker

from core.ai_admin_audit import stage_admin_mutation_audit
from core.ai_command_authority import VerifiedCommandAdmin
from core.ai_control_plane import ControlPlaneBase, ControlPlaneError

_SCOPE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_CURRENCY = re.compile(r"^[A-Z]{3}$")


class AIFinOpsAlertPolicyRecord(ControlPlaneBase):
    __tablename__ = "core_ai_finops_alert_policies"

    tenant_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    unit_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    maximum_known_cost: Mapped[str] = mapped_column(String(80), nullable=False)
    modified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


@dataclass(frozen=True, kw_only=True)
class FinOpsAlertPolicy:
    tenant_id: str
    unit_id: str
    revision: int
    enabled: bool
    currency: str
    maximum_known_cost: Decimal

    def __post_init__(self) -> None:
        if (
            any(type(x) is not str or not _SCOPE.fullmatch(x)
                for x in (self.tenant_id, self.unit_id))
            or type(self.revision) is not int or self.revision < 1
            or type(self.enabled) is not bool
            or type(self.currency) is not str or not _CURRENCY.fullmatch(self.currency)
            or not isinstance(self.maximum_known_cost, Decimal)
            or not self.maximum_known_cost.is_finite()
            or not Decimal(0) < self.maximum_known_cost <= Decimal(1000000000)
        ):
            raise ControlPlaneError("core_finops.alert_policy_invalid")


class PersistentFinOpsAlertPolicies:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    @staticmethod
    def _row(row: AIFinOpsAlertPolicyRecord) -> FinOpsAlertPolicy:
        try:
            value = Decimal(row.maximum_known_cost)
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ControlPlaneError("core_finops.alert_policy_corrupt") from exc
        try:
            return FinOpsAlertPolicy(
                tenant_id=row.tenant_id, unit_id=row.unit_id,
                revision=row.revision, enabled=row.enabled, currency=row.currency,
                maximum_known_cost=value,
            )
        except ControlPlaneError as exc:
            raise ControlPlaneError("core_finops.alert_policy_corrupt") from exc

    def get(self, *, tenant_id: str, unit_id: str) -> FinOpsAlertPolicy | None:
        if any(type(s) is not str or not _SCOPE.fullmatch(s)
               for s in (tenant_id, unit_id)):
            raise ControlPlaneError("core_finops.alert_policy_scope_invalid")
        with self._sessions() as db:
            row = db.get(AIFinOpsAlertPolicyRecord, (tenant_id, unit_id))
            return self._row(row) if row is not None else None

    def update(
        self, *, tenant_id: str, unit_id: str,
        expected_revision: int, enabled: bool, currency: str,
        maximum_known_cost: Decimal, principal: VerifiedCommandAdmin,
    ) -> FinOpsAlertPolicy:
        if type(expected_revision) is not int or expected_revision < 0:
            raise ControlPlaneError("core_finops.alert_revision_invalid")
        if not isinstance(principal, VerifiedCommandAdmin) or (
            "core.ai.admin.mutate" not in principal.permissions
        ):
            raise ControlPlaneError("core_finops.alert_authority_denied")
        policy = FinOpsAlertPolicy(
            tenant_id=tenant_id, unit_id=unit_id,
            revision=expected_revision + 1, enabled=enabled, currency=currency,
            maximum_known_cost=maximum_known_cost,
        )
        canonical = json.dumps({
            "tenant_id": tenant_id, "unit_id": unit_id,
            "expected_revision": expected_revision, "enabled": enabled,
            "currency": currency, "maximum_known_cost": str(maximum_known_cost),
        }, sort_keys=True, separators=(",", ":")).encode()
        try:
            with self._sessions.begin() as db:
                # For existing rows, FOR UPDATE serializes conflicting FM editors.
                row = db.scalar(select(AIFinOpsAlertPolicyRecord).where(
                    AIFinOpsAlertPolicyRecord.tenant_id == tenant_id,
                    AIFinOpsAlertPolicyRecord.unit_id == unit_id,
                ).with_for_update())
                current_revision = row.revision if row is not None else 0
                if current_revision != expected_revision:
                    raise ControlPlaneError("core_finops.alert_revision_conflict")
                if row is None:
                    db.add(AIFinOpsAlertPolicyRecord(
                        tenant_id=tenant_id, unit_id=unit_id, revision=policy.revision,
                        enabled=enabled, currency=currency,
                        maximum_known_cost=str(maximum_known_cost),
                        modified_at=datetime.now(UTC),
                    ))
                else:
                    row.revision = policy.revision
                    row.enabled = enabled
                    row.currency = currency
                    row.maximum_known_cost = str(maximum_known_cost)
                    row.modified_at = datetime.now(UTC)
                stage_admin_mutation_audit(
                    db, principal=principal, action="finops.alert_policy.update",
                    resource_id=f"{tenant_id}:{unit_id}",
                    canonical_request=canonical,
                )
        except IntegrityError as exc:
            raise ControlPlaneError("core_finops.alert_concurrent_or_replayed") from exc
        return policy
