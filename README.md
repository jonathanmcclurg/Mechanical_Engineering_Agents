# Manufacturing RCA Multi-Agent System

An on-premises, air-gapped multi-agent system for automated Root Cause Analysis (RCA) in manufacturing environments. Uses CrewAI for orchestration with statistical hypothesis testing and cited evidence.

## Features

- **Multi-agent orchestration**: Specialized agents for intake, research, hypothesis generation, statistical analysis, and reporting
- **Statistical analysis**: X̄–S control charts, 2-sample t-tests, effect sizes, assumption checks
- **Product guide integration**: First-class RAG retrieval with citations
- **Hybrid field selection**: Product-guide-informed semantic catalog search + LLM field selection, with recipe guardrails
- **Self-improving**: Feedback-driven improvements to retrieval, prompts, and analysis recipes (no model training)
- **Fully auditable**: Every claim is cited; every decision is logged

## Architecture

```
FailureEvent → IntakeTriage → ProductGuideAgent → PrivateDataResearch
                                    ↓
                              HypothesisAgent → TestPlanAgent → StatsAnalysisAgent
                                                                      ↓
                                                              CriticEvidenceAgent → ReportAgent → Engineer
```

`PrivateDataResearchAgent` now uses a hybrid approach for analysis data pulls:
1) Build an enriched query from failure context + product guide findings  
2) Retrieve top catalog candidates via semantic search and part-family filtering  
3) Let the LLM select relevant fields from candidates  
4) Merge recipe `must_include_fields` for known failure modes  

## Project Structure

```
├── apps/
│   ├── api/           # REST API endpoints
│   ├── dashboard/     # React + Vite frontend web app
│   └── worker/        # Background job runner / CLI workflow runner
├── src/
│   ├── agents/        # Agent definitions and prompts
│   ├── orchestrator/  # CrewAI setup
│   ├── tools/         # Tool wrappers (SQL, APIs, stats, RAG, data catalog)
│   ├── schemas/       # Pydantic models
│   └── evals/         # Evaluation harness
├── config/
│   └── analysis_recipes/  # Statistical analysis recipes per failure type
├── data/
│   └── catalog/       # Master field catalog (test IDs, ROA params, buyoffs, process params)
├── infra/             # Docker manifests (compose + Dockerfile)
└── tests/
```

## Run Locally (localhost)

### Prerequisites
- Python 3.10+
- Node.js 18+ (for the frontend dashboard)
- Optional: Docker Desktop (if you want local Postgres/pgvector via `infra/docker-compose.yaml`)

### 1) Install backend dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment (optional but recommended)
Create a `.env` file in the repo root to override defaults. Example:

```bash
cat > .env << 'EOF'
LLM_PROVIDER=local
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL_NAME=llama3:8b
LLM_API_KEY=not-needed-for-local
EMBEDDING_LOCAL_ONLY=true
EMBEDDING_MODEL_NAME=./data/models/all-MiniLM-L6-v2
CATALOG_EMBEDDING_TOP_K=50
PRODUCT_GUIDE_AUTO_INGEST=true
PRODUCT_GUIDE_REBUILD_ON_STARTUP=false
EOF
```

If no valid LLM config is available, the API starts with mock-response behavior.

### 2.5) Download local embedding model (one-time)
```bash
python scripts/download_embedding_model.py
```

This caches `sentence-transformers/all-MiniLM-L6-v2` into
`./data/models/all-MiniLM-L6-v2` so retrieval embeddings run locally.

### 3) (Optional) Start local infrastructure
```bash
cd infra
docker compose up -d postgres
cd ..
```

### 4) Start the backend API (Terminal 1)
```bash
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs become available at:
- `http://localhost:8000/docs`
- `http://localhost:8000/redoc`

### 5) Start the frontend dashboard (Terminal 2)
```bash
cd apps/dashboard
npm install
npm run dev
```

By default, the dashboard points to `http://localhost:8000` via `VITE_API_BASE`.
If needed, override it:

```bash
cd apps/dashboard
VITE_API_BASE=http://localhost:8000 npm run dev
```

Open:
- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`

### 6) Submit a test failure case
You can submit from the dashboard UI or by API:

```bash
curl -X POST http://localhost:8000/cases \
  -H "Content-Type: application/json" \
  -d '{
    "failure_type": "leak_test_fail",
    "failure_description": "Unit failed helium leak test",
    "part_number": "HYD-VALVE-200"
  }'
```

Then check status with:
```bash
curl http://localhost:8000/cases
curl http://localhost:8000/reports
```

### 7) Optional health and LLM checks
```bash
curl http://localhost:8000/health
curl http://localhost:8000/health/llm
curl http://localhost:8000/health/llm/metrics
```

Or run the helper scripts:
```bash
python scripts/test_llm_connectivity.py
python scripts/test_case_llm_usage.py
```

## Frontend Web App

The webapp lives in `apps/dashboard` (React + TypeScript + Vite) and provides:
- Case submission form for new failure events
- Case and report lists with status tracking
- Report detail view
- Markdown reasoning trace viewer
- Dashboard metrics (`/metrics`) for quick monitoring

Core backend endpoints used by the frontend:
- `GET /metrics`
- `GET /cases`
- `GET /cases/{case_id}`
- `POST /cases`
- `GET /reports`
- `GET /reports/{report_id}`
- `GET /reports/{report_id}/trace`
- `GET /cases/{case_id}/trace`

## Report Generation

Report artifacts are generated automatically by the API workflow in `apps/api/main.py`:

- **Structured report JSON (in memory)** is available through `GET /reports/{report_id}`.
- **PDF report** is generated with `src/tools/pdf_report.py` and saved under `data/reports/<report_id>.pdf`.
- **Markdown reasoning trace** is generated with `src/tools/markdown_trace.py` and saved under `data/reports/<report_id>.md`.

Useful report endpoints:
- `GET /reports/{report_id}`: report payload
- `GET /reports/{report_id}/pdf`: download PDF
- `GET /reports/{report_id}/trace`: view markdown trace
- `GET /cases/{case_id}/trace`: view trace for failed case runs

Additional service endpoints:
- `GET /`: service info
- `GET /health`: API health
- `GET /health/llm`: API-to-LLM connectivity probe
- `GET /health/llm/metrics`: aggregate LLM adapter usage metrics
- `POST /feedback`: submit engineer feedback
- `GET /feedback`: list feedback entries

## Configuration

### On-Prem LLM
Set `LLM_BASE_URL` to your internal model server (vLLM, Ollama, TGI, etc.).

### Database
Configure `DATABASE_URL` for PostgreSQL with pgvector extension.

### Product Guide
Place your product guide PDF/DOCX/MD/TXT in `data/product_guides/` and run the ingestion script:

```bash
python scripts/ingest_product_guides.py --rebuild
```

This creates a persisted RAG store in `data/rag_store/` (chunks + embeddings).
Auto-ingestion can be controlled with `PRODUCT_GUIDE_AUTO_INGEST` and
`PRODUCT_GUIDE_REBUILD_ON_STARTUP` in `.env`.

### Data Catalog (Hybrid Selection)
The master data catalog lives under `data/catalog/`:
- `test_ids.json`
- `roa_parameters.json`
- `operator_buyoffs.json`
- `process_parameters.json`

Each entry includes metadata (description, category, tags, applicable part families)
used for semantic field retrieval.

Relevant settings:
- `CATALOG_DIR` (default: `./data/catalog`)
- `CATALOG_DB_URL` (optional, for production DB-backed catalog loading)
- `CATALOG_EMBEDDING_TOP_K` (default: `50`)

For known failure modes, analysis recipes can define `must_include_fields` to ensure
critical fields are always pulled even if not selected by the LLM.

### Current Integration Status
`DataFetchTool` and `SQLTool` run in `mock_mode=True` by default in local/dev flows.
Real internal API and DB integrations can be injected in production via tool wiring.

## Worker CLI

You can also run a case directly without the API:

```bash
python -m apps.worker.run_case --case-file ./data/sample_case.json
```

Other useful flags:
- `--failure-type`, `--part-number`, `--description` to generate a synthetic input
- `--output-dir` to change where JSON/PDF reports are written
- `--quiet` to suppress progress logs

## Tests

Run regression tests:

```bash
python -m unittest tests.test_audit_regressions
```

## License

Internal use only.
