# F3 / P01 — FinOps accounting integrity and CI contract (2026-10-22)

## Scope: internal construction only
- Canonical `AIFinOpsBucket` fails closed for malformed counters, boolean/float/string counts, zero-attempt buckets, mismatched known/unknown attempt coverage, fallback overflow, invalid day, inconsistent latency and non-Decimal/non-finite/negative costs.
- The persistent Core ledger validates **every** known provider-attempt amount individually before accumulating request or period subtotals. A negative, NaN, infinite or corrupt cost cannot be compensated by a positive event and mislabeled as a valid subtotal.
- The Core AI Service CI executes the previously cross-repo-only Command FinOps API contract and lint-checks the relevant sources, so regressions cannot silently enter the Core candidate.
- Regression coverage: typed budget/count controls, unknown cost not represented as verified zero, positive-offset corruption and both request/day aggregations.
- This change preserves the corporate FM/tenant-unit ownership boundaries and API schema; it does not settle provider invoices, mutate grants/licences, create prices or call external providers.

## Certification
Candidate PR targets `feat/ai-service-v1` (still under Core main PR #1 Draft). Require AI Service CI and F3 PostgreSQL Quota gate **SUCCESS** at exact candidate SHA, internal code/security review, merge only to the candidate F3 branch, then verify post-merge checks. The Command cross-runtime pin and final F3 gate require separate evidence at the integrated Core SHA.

## Explicitly outstanding
Provider-specific signed/billable evidence reconciliation, configured/persisted alert policies, commercial end-to-end final adversarial suite, Core/Assistant monolithic PR audits, and production/external homologation (deferred until software build completion).
