# F3/P01 — Identity for each physical IA attempt (22/10/2026)

## Defect and invariant
The legacy deterministic usage-event hash used tenant/unit/request/route/outcome/fallback but no unique physical invocation identity. If a model is actually invoked twice with the same request identifier and route, the second provider attempt can be silently collapsed by the usage ledger. That is unsafe for FM-owned cost accounting, even when Core admission policies should ordinarily reject retries.

## Internal construction
- The Core router now creates a fresh cryptographically random opaque `attempt_id` (UUIDv4) for every actual model invocation, including transient failures, definitive failures and fallback.
- `AIUsageEvent.attempt_id` is optional to preserve existing synthetic/legacy producers, validated as a bounded ASCII token.
- Persistent FinOps event key includes the attempt token only when present. Historical IDs with absent attempt token are byte-for-byte unchanged, and re-recording the **same** attempt remains idempotent.
- Real physical attempts with otherwise identical request/route/outcome/timestamp/usage count as two observed attempts, preserving known or unknown costs separately; no DB migration, real provider, new secret or customer mutation.

## Tests/gates
Unit router tests prove distinct IDs even with identical request ID, static clock and route. Ledger tests prove two identical physical attempts record two charges, while replay of one evidence event records only one. AI Service CI + PostgreSQL Quota gate required green on exact SHA, review, merge into F3 candidate branch and post-merge validation.

## Limits
This does **not** imply actual-provider signed financial receipts or exactly-once remote execution under a network crash; those remain separate F3 reconciliation work. The main Core PR #1 stays Draft. External credential/deploy/production DB/homologation strictly deferred.
