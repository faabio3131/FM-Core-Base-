"""F3: ledger de consumo e custos estimados com origem explicita.

O custo desconhecido nunca vira 0 contabil: cost_total em FinOps representa
somente os eventos conhecidos, e cost_unknown_events e obrigatorio para leitura.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal, InvalidOperation
from hashlib import sha256

from sqlalchemy import DateTime, ForeignKey, Integer, String, select
from sqlalchemy.orm import Mapped, Session, mapped_column, sessionmaker

from core.ai_control_plane import ControlPlaneBase, ControlPlaneError
from core.ai_cost import CalculadoraCustoIA, OrigemCustoIA
from core.ai_finops import AIFinOpsBucket
from core.ai_pricing import (
    CatalogoPrecosIAEmMemoria,
    PriceSnapshotIA,
    SnapshotPrecoAusente,
    TarifaIA,
)
from core.ai_router import AIUsageEvent


class AIPriceSnapshotRecord(ControlPlaneBase):
    __tablename__ = "core_ai_price_snapshots"

    price_snapshot_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    modalidade: Mapped[str] = mapped_column(String(40), nullable=False)
    moeda: Mapped[str] = mapped_column(String(3), nullable=False)
    versao: Mapped[int] = mapped_column(Integer, nullable=False)
    vigencia_inicio: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    vigencia_fim: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AIPriceTariffRecord(ControlPlaneBase):
    __tablename__ = "core_ai_price_tariffs"

    tariff_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    price_snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("core_ai_price_snapshots.price_snapshot_id"), nullable=False
    )
    componente: Mapped[str] = mapped_column(String(80), nullable=False)
    preco_decimal: Mapped[str] = mapped_column(String(80), nullable=False)
    unidades_por_preco: Mapped[int] = mapped_column(Integer, nullable=False)


class AIUsageRecord(ControlPlaneBase):
    __tablename__ = "core_ai_usage_events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    correlation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    capability: Mapped[str] = mapped_column(String(80), nullable=False)
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    route_reason: Mapped[str] = mapped_column(String(500), nullable=False)
    fallback_used: Mapped[bool] = mapped_column(nullable=False)
    fallback_reason: Mapped[str | None] = mapped_column(String(200))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    cached_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    outcome: Mapped[str] = mapped_column(String(40), nullable=False)
    cost_decimal: Mapped[str | None] = mapped_column(String(80))
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    cost_origin: Mapped[str] = mapped_column(String(40), nullable=False)
    price_snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


def _utc(value: datetime) -> datetime:
    # SQLite nao devolve timezone, embora PostgreSQL preserve TIMESTAMPTZ.
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _observed_cost(value: str | None) -> Decimal | None:
    """Check each provider attempt separately, before cost aggregation.

    Offsetting a negative or corrupt record with a positive one would forge
    a plausible subtotal and hide a damaged ledger in FM Command.
    """
    if value is None:
        return None
    try:
        amount = Decimal(value)
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ControlPlaneError("core_finops.invalid_known_cost") from exc
    if not amount.is_finite() or amount < 0:
        raise ControlPlaneError("core_finops.invalid_known_cost")
    return amount


@dataclass(frozen=True, kw_only=True)
class AIRequestCostSummary:
    """No false cost=zero when provider attempts lack trustworthy usage."""

    tenant_id: str
    unit_id: str
    request_id: str
    currency: str | None
    known_total: Decimal
    known_attempts: int
    unknown_attempts: int
    total_attempts: int
    fallback_attempts: int
    all_recorded_costs_known: bool


class PersistentAIFinOps:
    """Catalogo imutavel por API e ledger idempotente, filtrado por tenant/unidade."""

    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions
        self._calculator = CalculadoraCustoIA()

    @staticmethod
    def _snapshot(db: Session, snapshot_id: str) -> PriceSnapshotIA:
        record = db.get(AIPriceSnapshotRecord, snapshot_id)
        if record is None:
            raise SnapshotPrecoAusente("price_snapshot_id_nao_encontrado")
        tariffs = db.scalars(
            select(AIPriceTariffRecord).where(
                AIPriceTariffRecord.price_snapshot_id == snapshot_id
            )
        ).all()
        return PriceSnapshotIA(
            price_snapshot_id=record.price_snapshot_id,
            provider=record.provider,
            model=record.model,
            modalidade=record.modalidade,
            moeda=record.moeda,
            versao=record.versao,
            vigencia_inicio=_utc(record.vigencia_inicio),
            vigencia_fim=_utc(record.vigencia_fim) if record.vigencia_fim else None,
            tarifas=tuple(
                TarifaIA(
                    componente=item.componente,
                    preco=Decimal(item.preco_decimal),
                    unidades_por_preco=item.unidades_por_preco,
                )
                for item in tariffs
            ),
        )

    def add_price_snapshot(self, snapshot: PriceSnapshotIA) -> None:
        with self._sessions.begin() as db:
            ids = db.scalars(select(AIPriceSnapshotRecord.price_snapshot_id)).all()
            snapshots = tuple(self._snapshot(db, identity) for identity in ids)
            # Reutiliza o catalogo canônico, inclusive verificacao de vigencias.
            CatalogoPrecosIAEmMemoria((*snapshots, snapshot))
            db.add(
                AIPriceSnapshotRecord(
                    price_snapshot_id=snapshot.price_snapshot_id,
                    provider=snapshot.provider,
                    model=snapshot.model,
                    modalidade=snapshot.modalidade,
                    moeda=snapshot.moeda,
                    versao=snapshot.versao,
                    vigencia_inicio=snapshot.vigencia_inicio,
                    vigencia_fim=snapshot.vigencia_fim,
                )
            )
            for tariff in snapshot.tarifas:
                db.add(
                    AIPriceTariffRecord(
                        price_snapshot_id=snapshot.price_snapshot_id,
                        componente=tariff.componente,
                        preco_decimal=str(tariff.preco),
                        unidades_por_preco=tariff.unidades_por_preco,
                    )
                )

    @staticmethod
    def _event_id(event: AIUsageEvent) -> str:
        fields = (
            event.tenant_id,
            event.unidade_id,
            event.request_id,
            event.capability.value,
            event.provider,
            event.model,
            event.route_reason,
            event.price_snapshot_id,
            event.outcome.value,
            str(event.fallback_used),
        )
        return sha256("\x1f".join(fields).encode("utf-8")).hexdigest()

    def registrar(self, evento: AIUsageEvent) -> None:
        event_id = self._event_id(evento)
        with self._sessions.begin() as db:
            if db.get(AIUsageRecord, event_id) is not None:
                return
            snapshot: PriceSnapshotIA | None = None
            try:
                snapshot = self._snapshot(db, evento.price_snapshot_id)
            except SnapshotPrecoAusente:
                pass
            valid_snapshot = (
                snapshot is not None
                and snapshot.provider == evento.provider
                and snapshot.model == evento.model
                and snapshot.vigente_em(evento.timestamp)
            )
            if valid_snapshot and snapshot is not None:
                calculated = self._calculator.estimar_tokens(
                    snapshot=snapshot,
                    input_tokens=evento.input_tokens,
                    output_tokens=evento.output_tokens,
                    cached_tokens=evento.cached_tokens,
                )
                value = calculated.valor
                currency = snapshot.moeda
                origin = calculated.origem.value
            else:
                value = None
                currency = "XXX"  # ISO 4217: sem moeda conhecida
                origin = OrigemCustoIA.INDISPONIVEL.value
            db.add(
                AIUsageRecord(
                    event_id=event_id,
                    tenant_id=evento.tenant_id,
                    unit_id=evento.unidade_id,
                    request_id=evento.request_id,
                    correlation_id=evento.correlation_id,
                    capability=evento.capability.value,
                    provider=evento.provider,
                    model=evento.model,
                    route_reason=evento.route_reason,
                    fallback_used=evento.fallback_used,
                    fallback_reason=evento.fallback_reason,
                    input_tokens=evento.input_tokens,
                    output_tokens=evento.output_tokens,
                    cached_tokens=evento.cached_tokens,
                    latency_ms=evento.latency_ms,
                    outcome=evento.outcome.value,
                    cost_decimal=str(value) if value is not None else None,
                    currency=currency,
                    cost_origin=origin,
                    price_snapshot_id=evento.price_snapshot_id,
                    event_time=evento.timestamp,
                )
            )

    def reconcile_request(
        self, *, tenant_id: str, unit_id: str, request_id: str,
    ) -> AIRequestCostSummary:
        """Summarize ALL persisted routing attempts in one scoped request.

        Known_total is a subtotal, never a proof of final spend when unknown
        attempts exist. Mixing currencies is not silently converted.
        """
        if any(
            not isinstance(value, str) or not value.strip() or len(value) > 128
            for value in (tenant_id, unit_id, request_id)
        ):
            raise ControlPlaneError("core_finops.invalid_request_scope")
        with self._sessions() as db:
            rows = db.scalars(select(AIUsageRecord).where(
                AIUsageRecord.tenant_id == tenant_id,
                AIUsageRecord.unit_id == unit_id,
                AIUsageRecord.request_id == request_id,
            )).all()
            currencies = {
                item.currency for item in rows
                if item.cost_decimal is not None
            }
            if len(currencies) > 1:
                raise ControlPlaneError("core_finops.mixed_currency")
            known_amounts = [_observed_cost(item.cost_decimal) for item in rows]
            known = [amount for amount in known_amounts if amount is not None]
            total = sum(known, Decimal(0))
            unknown = len(rows) - len(known)
            return AIRequestCostSummary(
                tenant_id=tenant_id,
                unit_id=unit_id,
                request_id=request_id,
                currency=next(iter(currencies), None),
                known_total=total,
                known_attempts=len(known),
                unknown_attempts=unknown,
                total_attempts=len(rows),
                fallback_attempts=sum(bool(item.fallback_used) for item in rows),
                all_recorded_costs_known=bool(rows) and unknown == 0,
            )

    def listar(
        self, *, tenant_id: str, unidade_id: str, inicio: date, fim: date
    ) -> tuple[AIFinOpsBucket, ...]:
        if not tenant_id.strip() or not unidade_id.strip() or fim < inicio:
            raise ControlPlaneError("core_finops.invalid_scope_or_interval")
        from_date = datetime.combine(inicio, time.min, tzinfo=UTC)
        until_date = datetime.combine(date.fromordinal(fim.toordinal() + 1), time.min, tzinfo=UTC)
        with self._sessions() as db:
            rows = db.scalars(
                select(AIUsageRecord).where(
                    AIUsageRecord.tenant_id == tenant_id,
                    AIUsageRecord.unit_id == unidade_id,
                    AIUsageRecord.event_time >= from_date,
                    AIUsageRecord.event_time < until_date,
                )
            ).all()
            buckets: dict[tuple, list[AIUsageRecord]] = defaultdict(list)
            for item in rows:
                key = (
                    _utc(item.event_time).date(),
                    item.provider, item.model, item.capability, item.outcome, item.currency,
                )
                buckets[key].append(item)
            result = []
            for key, events in sorted(buckets.items()):
                day, provider, model, capability, outcome, currency = key
                result.append(
                    AIFinOpsBucket(
                        tenant_id=tenant_id,
                        unidade_id=unidade_id,
                        bucket_date=day,
                        provider=provider,
                        model=model,
                        capability=capability,
                        outcome=outcome,
                        moeda=currency,
                        attempts=len(events),
                        fallback_attempts=sum(int(x.fallback_used) for x in events),
                        input_tokens=sum(x.input_tokens or 0 for x in events),
                        output_tokens=sum(x.output_tokens or 0 for x in events),
                        cached_tokens=sum(x.cached_tokens or 0 for x in events),
                        latency_ms_total=sum(x.latency_ms for x in events),
                        latency_ms_max=max(x.latency_ms for x in events),
                        cost_known_events=sum(x.cost_decimal is not None for x in events),
                        cost_unknown_events=sum(x.cost_decimal is None for x in events),
                        cost_total=sum(
                            (
                                amount for item in events
                                if (amount := _observed_cost(item.cost_decimal)) is not None
                            ),
                            Decimal(0),
                        ),
                    )
                )
            return tuple(result)
