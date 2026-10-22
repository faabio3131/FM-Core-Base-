# F3 / Prompt Mestre 01/04 — fila operacional de uso incerto (22/10/2026)

## Problema e fechamento parcial
O Core já persistia tentativas por provider/modelo/fallback e oferecia leitura por **request_id**. Faltava descobrir de forma governada quais solicitações precisavam de reconciliação sem pesquisar identificadores conhecidos previamente. Este WP implementa consulta read-only e paginada por tenant/unidade e intervalo, **sem inventar comprovantes externos**.

- `PersistentAIFinOps.listar_pendencias`: agregação SQL por request, contabilizando separadamente custos desconhecidos, entrada/saída de tokens incompletas e tentativas de fallback. Filtra 1–31 dias, limite de 1–100 registros, ordenação estável por primeira observação e ID; sinaliza truncamento. Nunca busca eventos globais em memória nem vaza outros tenants.
- `GET /v1/admin/ai/usage/pending`: API opcional protegida pela credencial segregada `X-Core-Admin-Key`, somente quando o Control Plane persistente está ativo. Resposta `Cache-Control: no-store`, status observacional e `provider_invoice_verified=false` invariável; jamais credencial de provider, preço comercial ou texto de mensagem.
- Negativos: autorização, escopo cruzado, intervalo/limite inválidos, indisponibilidade de credencial, ausência de observações, janela sem evidência, fallback e leituras incompletas.
- Sem nova tabela/migration: usa o ledger imutável `core_ai_usage_events` existente.

## Limites de evidência
Uma janela vazia **não comprova que o provider não cobrou**, e a agregação não atesta custo final. O leitor somente identifica observações internas com dados faltantes. Uma mesma request distribuída ao longo de janelas distintas é contabilizada apenas pelos eventos observados naquela janela; a ação determinística de recuperação permanece separada e bloqueada por comprovação independente + autorização FM. Não criar uma rotina que liquide automaticamente saldo ou custo.

## Gate e promoção
PR isolada sobre `feat/ai-service-v1` do Core. Exigir Ruff, suíte AI Service e F3 PostgreSQL Security Gate no commit exato, auditoria técnica, merge do WP somente no candidato e CI pós-merge. Depois repinar a Foundation Command ao novo Core SHA e executar o E2E cross-runtime antes de considerar o Prompt 01/04 pronto.

**Pendente de F3 global:** adaptadores de origem verificável por fornecedor, correspondência com extratos/faturas, políticas e alertas persistidos configurados pela FM, liquidação governada, auditoria adversarial final. Credenciais/provedores vivos, infra/PSP e homologação externa continuam posteriores ao marco de construção integral.
