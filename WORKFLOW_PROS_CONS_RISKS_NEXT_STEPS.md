# RCA Workflow: Pros, Cons, Risks, and Next Steps

This document summarizes the current multi-agent RCA workflow (API + orchestrator + dashboard) from an engineering and operations perspective.

## Biggest Pros

- Strong end-to-end structure: intake -> retrieval/research -> hypothesis -> test plan -> stats -> critique -> report.
- Good auditability: workflow logs, agent logs, citations, report artifacts, and markdown trace outputs are persisted.
- Practical statistical toolkit: control charts, t-tests, ANOVA, capability study, and correlation are already integrated.
- Flexible deployment shape: local API, worker CLI path, and Docker-based infrastructure options.
- Fast iteration model: mock mode and in-memory flow make development/testing easy before production hardening.
- Product-guide + catalog hybrid retrieval helps ground analysis in domain documentation and known fields.

## Biggest Cons

- In-memory storage for cases/reports/feedback means no durability across API restarts.
- Default mock behavior in core data tools can hide production integration gaps until late.
- Single-process background task execution limits scalability and queue reliability under load.
- Limited automated test coverage relative to system breadth (mostly regression-focused tests).
- API and dashboard usability are good for development but still light on enterprise controls and workflows.

## Biggest Risks

## 1) Reliability and Data Loss Risk

- API process restart loses active/previous case state because storage is process memory.
- Background tasks have no durable retry/queue semantics; long runs can fail silently if process dies.

## 2) Decision Quality Risk

- LLM quality and consistency can vary by provider/model, impacting hypothesis ranking and narrative quality.
- Mock-mode defaults can produce confidence in behavior that does not match real data integrations.

## 3) Statistical Misuse Risk

- Statistical methods are available, but misuse risk remains if pull plans, assumptions, or interpretation are weak.
- Small sample or poor subgrouping conditions can still produce unstable signals despite warnings.

## 4) Security and Compliance Risk

- CORS is permissive in current API setup, which is acceptable for local dev but risky for shared environments.
- Report/trace artifacts may contain sensitive operational context and need retention/access controls.

## 5) Operational Scaling Risk

- Current architecture can become bottlenecked with concurrent submissions (CPU-bound stats + LLM latency).
- Observability is present but still basic for production SLO management (alerts, distributed tracing, queue health).

## Recommended Next Steps (Priority Order)

## Near Term (1-2 weeks)

- Move case/report/feedback state to persistent storage (Postgres) while keeping API contract stable.
- Add a durable job queue/worker model for RCA runs (with retries, timeout policy, dead-letter handling).
- Introduce environment-based safe defaults (disable mock mode in non-dev, tighten CORS outside localhost).
- Add smoke tests for critical endpoints and one full workflow integration test path.

## Mid Term (2-6 weeks)

- Expand automated tests around agent transitions, failure branches, and artifact generation.
- Add structured observability: request IDs, queue metrics, per-stage latency dashboards, and alert thresholds.
- Add role-based access and report/trace retention policy controls for compliance-sensitive deployments.
- Formalize dataset readiness checks before statistical execution (minimum sample, schema/quality gates).

## Longer Term

- Add evaluation harness metrics to release criteria (hypothesis ranking quality, citation quality, regression scorecards).
- Improve human-in-the-loop controls (approval gates before final recommendation publishing).
- Define production playbooks: incident response, model fallback policy, and rollback procedures.

## Suggested Success Metrics

- Workflow completion rate and median/p95 case turnaround time.
- Percentage of runs with usable citations and complete artifacts.
- LLM call reliability/latency and case-level LLM usage consistency.
- Engineer usefulness feedback rate and top-3 root-cause hit rate.
- Failure recovery rate after transient infra/model errors.
