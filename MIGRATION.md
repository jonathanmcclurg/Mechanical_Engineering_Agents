# Production Migration Guide

This file tracks development-hardcoded data and defaults that should be reviewed/changed before production deployment.

## 1) Runtime blockers (change before prod)

| Location | Current dev-hardcoded behavior | Production change |
|---|---|---|
| `config/settings.py:34` | `DATABASE_URL` fallback is `postgresql://user:password@localhost:5432/rca_db`. | Set `DATABASE_URL` in environment/secret manager to your real DB endpoint and credentials. Avoid relying on this fallback in production. |
| `apps/dashboard/src/App.tsx:5` | Frontend API fallback is `http://localhost:8000`. | Set `VITE_API_BASE` at build/deploy time to the production API base URL (for example `https://api.company.com`). |
| `apps/api/main.py:24` | API stores `_cases`, `_reports`, `_feedback` in process memory only. Data is lost on restart and not shared across replicas. | Replace with persistent storage (Postgres/Redis/etc.) and wire endpoints/background workflow to DB-backed repositories. |
| `apps/api/main.py:73` | CORS `allow_origins=["*"]`. | Restrict to known frontend origins in production (for example `["https://rca.company.com"]`). |
| `src/orchestrator/crew.py:58` | `SQLTool(mock_mode=True)` default. | Inject a real `SQLTool` with production DB query implementation and `mock_mode=False`. |
| `src/orchestrator/crew.py:61` | `DataFetchTool(mock_mode=True)` default. | Inject a real `DataFetchTool` integration and `mock_mode=False`. |
| `src/agents/research_agent.py:39` | Agent defaults to `SQLTool(mock_mode=True)`. | Pass a non-mock `SQLTool` when constructing this agent. |
| `src/agents/research_agent.py:40` | Agent defaults to `DataFetchTool(mock_mode=True)`. | Pass a non-mock `DataFetchTool` when constructing this agent. |
| `src/agents/stats_agent.py:45` | Agent defaults to `DataFetchTool(mock_mode=True)`. | Pass a non-mock `DataFetchTool` when constructing this agent. |
| `src/agents/stats_agent.py:314` | If expected data is missing, agent fabricates mock data via `_generate_mock_data`. | In production, fail fast instead of synthesizing data (raise an error when required columns are absent). |
| `src/tools/sql_tool.py:151` | `SQLTool` defaults to mock mode. | Change default to `mock_mode=False` for production path or ensure production wiring always overrides this. |
| `src/tools/sql_tool.py:305` | Real DB query path is `NotImplementedError`. | Implement parameterized DB query execution in non-mock mode. |
| `src/tools/data_fetch_tool.py:146` | `DataFetchTool` defaults to mock mode. | Change default to `mock_mode=False` for production path or ensure production wiring always overrides this. |
| `src/tools/data_fetch_tool.py:395` | Real API fetch path is `NotImplementedError`. | Implement actual internal data API client/auth/retry in `_fetch_from_api`. |
| `src/tools/data_fetch_tool.py:410` | Real outlier fetch path is `NotImplementedError`. | Implement real outlier service call in `_fetch_test_outliers_from_api`. |

## 2) Deployment defaults to harden

| Location | Current value | Production change |
|---|---|---|
| `infra/docker-compose.yaml:13` | Postgres password fallback is `${POSTGRES_PASSWORD:-changeme}`. | Provide a strong secret (`POSTGRES_PASSWORD`) via secret management; never use fallback. |
| `infra/docker-compose.yaml:34` | LLM base URL fallback is local Ollama (`http://ollama:11434/v1`). | Point `LLM_BASE_URL` to your production LLM endpoint if not running local Ollama in-cluster. |
| `src/llm/adapter.py:59` | `HTTP-Referer` is placeholder `https://github.com/your-org/mechanical-engineering-agents`. | Replace with your org/app URL used for provider attribution/allowlisting. |
| `src/llm/adapter.py:104` | Local adapter uses sentinel key `"not-needed"` when no key is provided. | For production hosted endpoints that require auth, ensure `LLM_API_KEY` is always set and validated at startup. |

## 3) Sample/dev UX and tooling values (safe to keep, but dev-oriented)

| Location | Dev/sample hardcoded data | Production recommendation |
|---|---|---|
| `apps/worker/run_case.py:45` | Worker initializes `SQLTool(mock_mode=True)`. | Use real DB tool in production worker runs. |
| `apps/worker/run_case.py:120` | CLI default failure type `leak_test_fail`. | Keep for local demos, but require explicit case input in production job runners. |
| `apps/worker/run_case.py:126` | CLI default part number `TEST-PART-001`. | Same as above. |
| `apps/worker/run_case.py:132` | CLI default description `Test failure for development`. | Same as above. |
| `apps/worker/run_case.py:154` | Creates synthetic case when `--case-file` is not supplied. | Disable this path in production automation; enforce external case payloads. |
| `apps/dashboard/src/App.tsx:188` | “Generate random case” uses synthetic failure types/parts/test names. | Keep only in non-prod UI, or gate behind debug/admin feature flag. |
| `scripts/test_llm_connectivity.py:62` | Default API base is localhost. | Pass `--api-base` explicitly in CI/prod checks. |
| `scripts/test_case_llm_usage.py:30` | Default API base is localhost. | Pass `--api-base` explicitly in CI/prod checks. |
| `scripts/test_case_llm_usage.py:35` | Script submits fixed sample payload (`HYD-VALVE-200`, `SN-LLM-0001`, etc.). | Keep as integration probe only; do not reuse for production workload generation. |

## 4) Suggested migration order

1. Implement real non-mock integrations in `SQLTool` and `DataFetchTool`.
2. Update constructor defaults/wiring so production never enters mock paths.
3. Replace API in-memory stores with persistent storage.
4. Lock down CORS and deploy environment variables/secrets.
5. Gate or remove sample-data generation paths in worker/dashboard for production environments.
