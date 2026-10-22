"""Contratos provider-neutral do AI FinOps Read Model do Kordena V1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Protocol


@dataclass(frozen=True, kw_only=True)
class AIFinOpsBucket:
    tenant_id: str
    unidade_id: str
    bucket_date: date
    provider: str
    model: str
    capability: str
    outcome: str
    moeda: str
    attempts: int
    fallback_attempts: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    latency_ms_total: int
    latency_ms_max: int
    cost_known_events: int
    cost_unknown_events: int
    cost_total: Decimal

    def __post_init__(self) -> None:
        for nome in (
            "tenant_id",
            "unidade_id",
            "provider",
            "model",
            "capability",
            "outcome",
            "moeda",
        ):
            valor = getattr(self, nome)
            if not isinstance(valor, str) or not valor.strip():
                raise ValueError(f"{nome}_obrigatorio")

        if len(self.moeda) != 3 or not self.moeda.isascii() or not self.moeda.isalpha():
            raise ValueError("moeda_finops_invalida")

        if type(self.bucket_date) is not date:
            raise ValueError("dia_finops_invalido")

        inteiros = (
            self.attempts,
            self.fallback_attempts,
            self.input_tokens,
            self.output_tokens,
            self.cached_tokens,
            self.latency_ms_total,
            self.latency_ms_max,
            self.cost_known_events,
            self.cost_unknown_events,
        )
        # bool is an int subclass in Python; never accept it as an
        # observed count in a corporate cost/usage report.
        if any(type(valor) is not int or valor < 0 for valor in inteiros):
            raise ValueError("metrica_finops_invalida")
        if (
            self.attempts == 0
            or self.fallback_attempts > self.attempts
            or self.cost_known_events + self.cost_unknown_events != self.attempts
            or self.latency_ms_max > self.latency_ms_total
        ):
            raise ValueError("cobertura_finops_inconsistente")

        if (
            not isinstance(self.cost_total, Decimal)
            or not self.cost_total.is_finite()
            or self.cost_total < 0
        ):
            raise ValueError("custo_finops_invalido")


class PortaAIFinOpsReadModel(Protocol):
    def listar(
        self,
        *,
        tenant_id: str,
        unidade_id: str,
        inicio: date,
        fim: date,
    ) -> tuple[AIFinOpsBucket, ...]: ...
