from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.ai_router import (
    AIModelRouter,
    CapabilityIA,
    ConteudoAudioIA,
    FalhaRotaDefinitiva,
    FalhaRotaTransitoria,
    MedidorUsoIAEmMemoria,
    OutcomeIA,
    PoliticaRoteamentoAmbigua,
    RespostaModeloIA,
    RotaIA,
    SolicitacaoIA,
)

AGORA = datetime(2026, 8, 28, 18, 0, tzinfo=timezone.utc)


class ExecutorFake:
    def __init__(self, resultados: dict[str, object]) -> None:
        self.resultados = resultados
        self.chamadas: list[str] = []

    def executar(
        self,
        *,
        rota: RotaIA,
        solicitacao: SolicitacaoIA,
    ) -> RespostaModeloIA:
        self.chamadas.append(rota.provider)
        resultado = self.resultados[rota.provider]

        if isinstance(resultado, Exception):
            raise resultado

        assert isinstance(resultado, RespostaModeloIA)
        return resultado


def solicitacao() -> SolicitacaoIA:
    return SolicitacaoIA(
        tenant_id="tenant-a",
        unidade_id="loja-1",
        request_id="req-1",
        correlation_id="corr-1",
        capability=CapabilityIA.TOOL_PLANNING,
        conteudo={"user": "conteudo-sensivel"},
    )


def rota(provider: str, prioridade: int) -> RotaIA:
    return RotaIA(
        configuracao_id=f"config-{provider}",
        provider=provider,
        model=f"model-{provider}",
        capability=CapabilityIA.TOOL_PLANNING,
        prioridade=prioridade,
        price_snapshot_id=f"price-{provider}-v1",
    )


def relogio():
    valores = iter((10.0, 10.012, 20.0, 20.025))
    return lambda: next(valores)


def test_af16_seleciona_maior_prioridade() -> None:
    meter = MedidorUsoIAEmMemoria()

    executor = ExecutorFake(
        {
            "a": RespostaModeloIA(
                conteudo="ok",
                input_tokens=10,
                output_tokens=4,
            ),
            "b": RespostaModeloIA(conteudo="nao"),
        }
    )

    router = AIModelRouter(
        rotas=(rota("b", 50), rota("a", 100)),
        executor=executor,
        metering=meter,
        monotonic=relogio(),
        now=lambda: AGORA,
    )

    resultado = router.executar(solicitacao())

    assert resultado.provider == "a"
    assert executor.chamadas == ["a"]
    assert meter.eventos[0].outcome is OutcomeIA.SUCESSO
    assert meter.eventos[0].input_tokens == 10


def test_af17_fallback_apenas_para_falha_transitoria() -> None:
    meter = MedidorUsoIAEmMemoria()

    executor = ExecutorFake(
        {
            "a": FalhaRotaTransitoria(
                "ai_router.provider_transient_failure"
            ),
            "b": RespostaModeloIA(conteudo="fallback-ok"),
        }
    )

    router = AIModelRouter(
        rotas=(rota("a", 100), rota("b", 50)),
        executor=executor,
        metering=meter,
        monotonic=relogio(),
        now=lambda: AGORA,
    )

    resultado = router.executar(solicitacao())

    assert executor.chamadas == ["a", "b"]
    assert resultado.provider == "b"
    assert resultado.fallback_used is True
    assert [
        evento.outcome
        for evento in meter.eventos
    ] == [
        OutcomeIA.FALHA_TRANSITORIA,
        OutcomeIA.SUCESSO,
    ]


def test_af17_falha_definitiva_nao_faz_fallback() -> None:
    meter = MedidorUsoIAEmMemoria()

    executor = ExecutorFake(
        {
            "a": FalhaRotaDefinitiva(
                "ai_router.provider_definitive_failure"
            ),
            "b": RespostaModeloIA(conteudo="nao"),
        }
    )

    router = AIModelRouter(
        rotas=(rota("a", 100), rota("b", 50)),
        executor=executor,
        metering=meter,
        monotonic=relogio(),
        now=lambda: AGORA,
    )

    with pytest.raises(FalhaRotaDefinitiva):
        router.executar(solicitacao())

    assert executor.chamadas == ["a"]
    assert meter.eventos[0].outcome is OutcomeIA.FALHA_DEFINITIVA


def test_af16_empate_de_prioridade_falha_fechado() -> None:
    meter = MedidorUsoIAEmMemoria()

    executor = ExecutorFake(
        {
            "a": RespostaModeloIA(conteudo="a"),
            "b": RespostaModeloIA(conteudo="b"),
        }
    )

    router = AIModelRouter(
        rotas=(rota("a", 100), rota("b", 100)),
        executor=executor,
        metering=meter,
    )

    with pytest.raises(PoliticaRoteamentoAmbigua):
        router.executar(solicitacao())

    assert executor.chamadas == []


def test_af19_af20_telemetria_nao_contem_payload() -> None:
    meter = MedidorUsoIAEmMemoria()

    executor = ExecutorFake(
        {
            "a": RespostaModeloIA(
                conteudo="resposta-secreta",
                input_tokens=7,
                output_tokens=3,
                cached_tokens=1,
            )
        }
    )

    router = AIModelRouter(
        rotas=(rota("a", 100),),
        executor=executor,
        metering=meter,
        monotonic=relogio(),
        now=lambda: AGORA,
    )

    router.executar(solicitacao())

    evento = meter.eventos[0]
    campos = set(evento.__dataclass_fields__)

    assert "conteudo" not in campos
    assert "prompt" not in campos
    assert "response" not in campos
    assert "api_key" not in campos
    assert "resposta-secreta" not in repr(evento)


def test_audio_contract_hides_bytes_and_validates_mime() -> None:
    conteudo = ConteudoAudioIA(
        audio=b"segredo-audio-binario",
        mime_type="audio/ogg",
        instrucao="Transcreva fielmente.",
    )

    assert "segredo-audio-binario" not in repr(conteudo)
    assert conteudo.mime_type == "audio/ogg"

    with pytest.raises(ValueError, match="mime_type_audio_invalido"):
        ConteudoAudioIA(
            audio=b"x",
            mime_type="application/octet-stream",
            instrucao="Transcreva.",
        )


def test_audio_transcription_is_routed_as_own_capability() -> None:
    meter = MedidorUsoIAEmMemoria()
    executor = ExecutorFake(
        {
            "a": RespostaModeloIA(conteudo="quero dois x-bacon"),
        }
    )
    rota_audio = RotaIA(
        configuracao_id="config-audio",
        provider="a",
        model="model-audio",
        capability=CapabilityIA.ATENDIMENTO_TRANSCRICAO,
        prioridade=100,
        price_snapshot_id="price-audio-v1",
    )
    router = AIModelRouter(
        rotas=(rota_audio,),
        executor=executor,
        metering=meter,
        monotonic=relogio(),
        now=lambda: AGORA,
    )
    solicitacao_audio = SolicitacaoIA(
        tenant_id="tenant-a",
        unidade_id="loja-1",
        request_id="req-audio",
        correlation_id="corr-audio",
        capability=CapabilityIA.ATENDIMENTO_TRANSCRICAO,
        conteudo=ConteudoAudioIA(
            audio=b"audio-binario",
            mime_type="audio/ogg",
            instrucao="Transcreva.",
        ),
    )

    resultado = router.executar(solicitacao_audio)

    assert resultado.conteudo == "quero dois x-bacon"
    assert resultado.capability is CapabilityIA.ATENDIMENTO_TRANSCRICAO
    assert executor.chamadas == ["a"]
    assert meter.eventos[0].capability is CapabilityIA.ATENDIMENTO_TRANSCRICAO
    assert "audio-binario" not in repr(meter.eventos[0])


def test_router_generates_distinct_ids_for_repeated_physical_calls():
    meter = MedidorUsoIAEmMemoria()
    executor = ExecutorFake({
        "a": RespostaModeloIA(conteudo="ok", input_tokens=10, output_tokens=5),
    })
    router = AIModelRouter(
        rotas=(rota("a", 100),), executor=executor, metering=meter,
        monotonic=lambda: 1.0, now=lambda: AGORA,
    )
    router.executar(solicitacao())
    router.executar(solicitacao())
    assert len(meter.eventos) == 2
    assert meter.eventos[0].request_id == meter.eventos[1].request_id
    assert meter.eventos[0].route_reason == meter.eventos[1].route_reason
    assert meter.eventos[0].timestamp == meter.eventos[1].timestamp
    assert meter.eventos[0].attempt_id
    assert meter.eventos[1].attempt_id
    assert meter.eventos[0].attempt_id != meter.eventos[1].attempt_id


@pytest.mark.parametrize("bad_id", ["", " ", "/foreign", "a" * 129, True])
def test_router_usage_rejects_invalid_explicit_attempt_id(bad_id):
    from dataclasses import replace

    meter = MedidorUsoIAEmMemoria()
    router = AIModelRouter(
        rotas=(rota("a", 100),),
        executor=ExecutorFake({"a": RespostaModeloIA(conteudo="ok")}),
        metering=meter, monotonic=lambda: 1.0, now=lambda: AGORA,
    )
    router.executar(solicitacao())
    with pytest.raises(ValueError, match="attempt_id_invalid"):
        replace(meter.eventos[0], attempt_id=bad_id)
