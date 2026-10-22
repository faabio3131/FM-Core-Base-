"""API interna do FM Core Base para consumidores como FM Assistant Base."""

from __future__ import annotations

import base64
import binascii
import hmac
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from core.ai_admin_read_api import register_admin_read_routes
from core.ai_assistant_credential_gateway import AssistantSignedCredentialGateway
from core.ai_command_authority import CommandTrustPolicy, VerifiedCommandAdmin
from core.ai_command_quota_readback import command_core_quota_readback_router
from core.ai_control_plane import ControlPlaneError
from core.ai_control_plane_bootstrap import build_persistent_core, control_plane_is_ready
from core.ai_finops_policy_api import finops_alert_policy_router
from core.ai_governed_recovery_api import governed_recovery_router
from core.ai_provider_evidence_store import TrustedProviderEvidenceStore
from core.ai_quota_store import QuotaDenied, QuotaReservations
from core.ai_router import FalhaRotaDefinitiva, FalhaRotaTransitoria, SemRotaCompativel
from core.ai_service_contracts import CoreAIResult, TextMessage
from core.ai_service_runtime import CoreAIRuntime
from core.ai_service_settings import CoreSettings, load_settings
from core.ai_signed_plan_authority import AssistantPlanHTTPSFetch, SignedAssistantPlanAuthority
from core.ai_verified_usage_receipt import VerifiedProviderUsageReceipt
from core.seguranca.erros import ReferenciaSegredoInvalida, SegredoAusente


class TextMessageRequest(BaseModel):
    role: str = Field(pattern=r"^(user|assistant)$")
    content: str = Field(min_length=1, max_length=50_000)


class TextAIRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=128)
    unit_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)
    correlation_id: str = Field(min_length=1, max_length=128)
    system_prompt: str = Field(min_length=1, max_length=50_000)
    messages: list[TextMessageRequest] = Field(min_length=1, max_length=100)


class AudioAIRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=128)
    unit_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)
    correlation_id: str = Field(min_length=1, max_length=128)
    mime_type: str = Field(pattern=r"^audio/")
    audio_base64: str = Field(min_length=1)
    instruction: str = Field(min_length=1, max_length=4_000)


class AIResponse(BaseModel):
    content: str
    provider: str
    model: str
    fallback_used: bool
    fallback_reason: str | None
    price_snapshot_id: str


def _response(result: CoreAIResult) -> AIResponse:
    return AIResponse(
        content=result.content,
        provider=result.provider,
        model=result.model,
        fallback_used=result.fallback_used,
        fallback_reason=result.fallback_reason,
        price_snapshot_id=result.price_snapshot_id,
    )


def create_app(
    settings: CoreSettings | None = None,
    runtime: CoreAIRuntime | None = None,
) -> FastAPI:
    cfg = (settings or load_settings()).validate()
    recovery_trust = (
        CommandTrustPolicy(
            issuer=cfg.governed_recovery_issuer or "",
            audience=cfg.governed_recovery_audience or "",
            public_key_pem=cfg.governed_recovery_public_key_pem or "",
        ) if cfg.governed_recovery_enabled else None
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = cfg
        db_engine = None
        policy_client = None
        if cfg.persistent_control_plane:
            tenant_gateway = (
                AssistantSignedCredentialGateway(
                    base_url=cfg.assistant_bridge_url or "",
                    key_id=cfg.assistant_bridge_key_id or "",
                    signing_key=cfg.assistant_bridge_signing_key or "",
                )
                if cfg.assistant_bridge_url
                else None
            )
            plan_authority = None
            if cfg.assistant_plan_url and cfg.assistant_plan_signing_key:
                policy_client = httpx.Client(trust_env=False)
                secret = cfg.assistant_plan_signing_key.encode()
                plan_authority = SignedAssistantPlanAuthority(
                    secret=secret,
                    fetch=AssistantPlanHTTPSFetch(
                        origin=cfg.assistant_plan_url, secret=secret,
                        bounds=lambda capability: {
                            name: (inp, out)
                            for name, inp, out in cfg.assistant_plan_bounds
                        }[capability],
                        client=policy_client,
                    ),
                )
            ai_runtime, db_engine = build_persistent_core(
                cfg.control_plane_database_url or "",
                tenant_gateway=tenant_gateway,
                plan_authority=plan_authority,
                max_audio_bytes=cfg.assistant_max_audio_bytes,
            )
            if cfg.command_quota_readback_enabled:
                if plan_authority is None or db_engine is None:
                    raise RuntimeError("core_command_quota_readback_unavailable")
                app.state.command_quota_authority = plan_authority
                app.state.command_quota_ledger = QuotaReservations(
                    sessionmaker(db_engine, expire_on_commit=False)
                )
        else:
            ai_runtime = runtime or CoreAIRuntime.from_settings(cfg)
        if recovery_trust is not None:
            if db_engine is None:
                raise RuntimeError("core_recovery_requires_persistent_database")
            recovery_sessions = sessionmaker(db_engine, expire_on_commit=False)
            app.state.governed_recovery_evidence = TrustedProviderEvidenceStore(
                recovery_sessions,
                gateway_secret=(
                    cfg.governed_recovery_gateway_secret or ""
                ).encode(),
                source_id=cfg.governed_recovery_source_id or "",
            )
            app.state.governed_recovery_ledger = QuotaReservations(recovery_sessions)
        app.state.ai_runtime = ai_runtime
        app.state.control_plane_engine = db_engine
        try:
            yield
        finally:
            if db_engine is not None:
                db_engine.dispose()
            if policy_client is not None:
                policy_client.close()

    app = FastAPI(
        title="FM Core Base AI Service",
        version="0.1.0",
        description="Roteamento multimodelo e capacidades de IA da FM Tecnologia.",
        lifespan=lifespan,
    )

    # Do not even register the corporate quota read port unless FM explicitly
    # enables it with a dedicated key. Delegates are resolved at request time
    # from lifespan state, not a stale or client-supplied quota source.
    if cfg.command_quota_readback_enabled:
        class _LivePlan:
            def resolve(self, *, tenant_id, unit_id, capability, now):
                authority = getattr(app.state, "command_quota_authority", None)
                if authority is None:
                    raise QuotaDenied("core_quota_readback.authority_unavailable")
                return authority.resolve(
                    tenant_id=tenant_id, unit_id=unit_id,
                    capability=capability, now=now,
                )

        class _LiveQuota:
            def usage_snapshot(self, *, tenant_id, unit_id, capability, at):
                ledger = getattr(app.state, "command_quota_ledger", None)
                if ledger is None:
                    raise QuotaDenied("core_quota_readback.ledger_unavailable")
                return ledger.usage_snapshot(
                    tenant_id=tenant_id, unit_id=unit_id,
                    capability=capability, at=at,
                )

        app.include_router(command_core_quota_readback_router(
            authority=_LivePlan(), quota=_LiveQuota(),
            shared_secret=(cfg.command_quota_readback_secret or "").encode(),
        ))

    register_admin_read_routes(app)
    # The mutation route does not exist unless an FM-owned explicit opt-in,
    # distinct trusted evidence gateway key and Command RS256 issuer are
    # configured. No client can pass a provider receipt to this endpoint.
    if recovery_trust is not None:
        class _BoundEvidence:
            def verify(
                self, *, evidence_id: str, request_id: str,
                tenant_id: str, unit_id: str,
            ) -> VerifiedProviderUsageReceipt:
                evidence = getattr(app.state, "governed_recovery_evidence", None)
                if evidence is None:
                    raise QuotaDenied("core_recovery_unavailable")
                return evidence.verify(
                    evidence_id=evidence_id, request_id=request_id,
                    tenant_id=tenant_id, unit_id=unit_id,
                )

        class _BoundQuota:
            def reconcile_verified_provider_receipt(
                self, *, receipt: VerifiedProviderUsageReceipt,
                request_policy_revision: int, tokens_policy_revision: int,
                principal: VerifiedCommandAdmin, at: datetime,
                request_scope: str = "__tenant__", tokens_scope: str = "__tenant__",
            ) -> None:
                ledger = getattr(app.state, "governed_recovery_ledger", None)
                if ledger is None:
                    raise QuotaDenied("core_recovery_unavailable")
                ledger.reconcile_verified_provider_receipt(
                    receipt=receipt, request_policy_revision=request_policy_revision,
                    tokens_policy_revision=tokens_policy_revision,
                    principal=principal, at=at, request_scope=request_scope,
                    tokens_scope=tokens_scope,
                )

        recovery_router = governed_recovery_router(
            trust=recovery_trust, evidence=_BoundEvidence(), ledger=_BoundQuota(),
            transport_key=cfg.governed_recovery_transport_key or "",
        )
        # Register the already-constructed APIRoute directly. This does not
        # alter its explicit HMAC/RS256 authorization or evidence dependencies.
        # Verify the exact installed path before exposing any mutation.
        expected_path = "/internal/v1/command/ai/usage/reconcile"
        if len(recovery_router.routes) != 1 or (
            getattr(recovery_router.routes[0], "path", None) != expected_path
        ):
            raise RuntimeError("core_recovery_route_invalid")
        if any(getattr(r, "path", None) == expected_path for r in app.router.routes):
            raise RuntimeError("core_recovery_route_duplicate")
        app.router.routes.extend(recovery_router.routes)

        # Same RS256 human FM authority + distinct S2S identity; never
        # enabled automatically for tenants or by the operator read key.
        policy_router = finops_alert_policy_router(
            trust=recovery_trust,
            transport_key=cfg.governed_recovery_transport_key or "",
        )
        policy_path = "/internal/v1/command/ai/finops/alert-policy"
        if len(policy_router.routes) != 1 or (
            getattr(policy_router.routes[0], "path", None) != policy_path
        ):
            raise RuntimeError("core_finops_policy_route_invalid")
        if any(getattr(r, "path", None) == policy_path for r in app.router.routes):
            raise RuntimeError("core_finops_policy_route_duplicate")
        app.router.routes.extend(policy_router.routes)

    def require_service_key(
        request: Request,
        x_core_service_key: str | None = Header(default=None, alias="X-Core-Service-Key"),
    ) -> None:
        expected = request.app.state.settings.service_key
        if not x_core_service_key or not hmac.compare_digest(expected, x_core_service_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="core_service_unauthorized",
            )

    @app.get("/health")
    def health() -> dict[str, object]:
        if cfg.persistent_control_plane:
            return {
                "status": "ok",
                "service": "fm-core-base",
                "mode": "control_plane",
            }
        return {
            "status": "ok",
            "service": "fm-core-base",
            "routes": len(cfg.routes),
            "providers": sorted({route.provider for route in cfg.routes}),
        }

    @app.get("/ready")
    def readiness(request: Request) -> dict[str, str]:
        engine = request.app.state.control_plane_engine
        if cfg.persistent_control_plane and (
            engine is None or not control_plane_is_ready(engine)
        ):
            raise HTTPException(status_code=503, detail="core_control_plane_not_ready")
        return {"status": "ready"}

    @app.post(
        "/v1/ai/text",
        response_model=AIResponse,
        dependencies=[Depends(require_service_key)],
    )
    def text_ai(payload: TextAIRequest, request: Request) -> AIResponse:
        try:
            result = request.app.state.ai_runtime.generate_text(
                tenant_id=payload.tenant_id,
                unit_id=payload.unit_id,
                request_id=payload.request_id,
                correlation_id=payload.correlation_id,
                system_prompt=payload.system_prompt,
                messages=tuple(
                    TextMessage(role=item.role, content=item.content)
                    for item in payload.messages
                ),
            )
        except (
            ControlPlaneError, QuotaDenied, SegredoAusente, ReferenciaSegredoInvalida,
            SQLAlchemyError,
        ) as exc:
            raise HTTPException(
                status_code=503, detail="core_control_plane_dependency_unavailable"
            ) from exc
        except (SemRotaCompativel, FalhaRotaTransitoria) as exc:
            detail = (
                "core_control_plane_dependency_unavailable"
                if exc.codigo == "core_ai_credential_unavailable"
                else exc.codigo
            )
            raise HTTPException(status_code=503, detail=detail) from exc
        except FalhaRotaDefinitiva as exc:
            raise HTTPException(status_code=502, detail=exc.codigo) from exc
        return _response(result)

    @app.post(
        "/v1/ai/transcribe",
        response_model=AIResponse,
        dependencies=[Depends(require_service_key)],
    )
    def transcribe_ai(payload: AudioAIRequest, request: Request) -> AIResponse:
        try:
            audio = base64.b64decode(payload.audio_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=422, detail="audio_base64_invalid") from exc
        if not audio or len(audio) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="audio_size_invalid")
        try:
            result = request.app.state.ai_runtime.transcribe_audio(
                tenant_id=payload.tenant_id,
                unit_id=payload.unit_id,
                request_id=payload.request_id,
                correlation_id=payload.correlation_id,
                audio=audio,
                mime_type=payload.mime_type,
                instruction=payload.instruction,
            )
        except (
            ControlPlaneError, QuotaDenied, SegredoAusente, ReferenciaSegredoInvalida,
            SQLAlchemyError,
        ) as exc:
            raise HTTPException(
                status_code=503, detail="core_control_plane_dependency_unavailable"
            ) from exc
        except (SemRotaCompativel, FalhaRotaTransitoria) as exc:
            detail = (
                "core_control_plane_dependency_unavailable"
                if exc.codigo == "core_ai_credential_unavailable"
                else exc.codigo
            )
            raise HTTPException(status_code=503, detail=detail) from exc
        except FalhaRotaDefinitiva as exc:
            raise HTTPException(status_code=502, detail=exc.codigo) from exc
        return _response(result)

    @app.get(
        "/v1/plan/usage", dependencies=[Depends(require_service_key)],
    )
    def read_plan_usage(
        request: Request,
        tenant_id: str = Query(min_length=1, max_length=128),
        unit_id: str = Query(min_length=1, max_length=128),
        resource: str = Query(min_length=1, max_length=80),
        scope: str = Query(pattern="^(tenant|unit)$"),
    ) -> dict[str, object]:
        """Exact-scoped read of committed quota ledger; no default zero."""
        engine = request.app.state.control_plane_engine
        if engine is None:
            raise HTTPException(status_code=503, detail="plan_usage_unavailable")
        try:
            ledger = QuotaReservations(sessionmaker(engine, expire_on_commit=False))
            scoped_unit = "__tenant__" if scope == "tenant" else unit_id
            snapshot = ledger.usage_snapshot(
                tenant_id=tenant_id, unit_id=scoped_unit,
                capability=resource, at=datetime.now(UTC),
            )
        except (QuotaDenied, SQLAlchemyError) as exc:
            raise HTTPException(
                status_code=503, detail="plan_usage_unavailable",
            ) from exc
        return {
            "tenant_id": tenant_id,
            "unit_id": unit_id,
            "resource": resource,
            "scope": scope,
            "policy_revision": snapshot.policy_revision,
            "limit_units": snapshot.limit_units,
            "window_seconds": snapshot.window_seconds,
            "consumed_units": snapshot.consumed_units,
            "reserved_units": snapshot.reserved_units,
            "remaining_units": snapshot.remaining_units,
            "period_ends_at": snapshot.period_ends_at.isoformat(),
            "observed_at": datetime.now(UTC).isoformat(),
        }

    return app
