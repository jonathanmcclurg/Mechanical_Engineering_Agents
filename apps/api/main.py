"""FastAPI application for the RCA system."""

from datetime import datetime
from time import perf_counter
from pathlib import Path
from typing import Any, Optional
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from src.schemas.case import FailureCase
from src.schemas.feedback import EngineerFeedback
from src.orchestrator.crew import RCACrew
from src.llm.adapter import create_llm
from src.tools.pdf_report import generate_pdf_report
from src.tools.markdown_trace import generate_markdown_trace
from config.settings import get_settings


# In-memory storage (replace with database in production)
_cases: dict[str, dict] = {}
_reports: dict[str, dict] = {}
_feedback: dict[str, dict] = {}
_crew: Optional[RCACrew] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _crew
    # Initialize LLM based on settings
    llm = create_llm()
    if llm is None:
        print("Warning: No LLM configured. Using mock responses.")
    
    # Initialize the RCA crew on startup
    _crew = RCACrew(llm=llm, verbose=True)

    # Auto-ingest product guides on startup if configured
    settings = get_settings()
    if settings.product_guide_auto_ingest and _crew is not None:
        try:
            rag_tool = _crew.rag_tool
            existing_docs = rag_tool.list_documents()
            if settings.product_guide_rebuild_on_startup or not existing_docs:
                ingested = rag_tool.ingest_directory(
                    settings.product_guide_dir,
                    rebuild=settings.product_guide_rebuild_on_startup,
                )
                print(f"Product guide ingestion complete: {len(ingested)} documents")
            else:
                print("Product guide ingestion skipped: store already populated")
        except Exception as e:
            print(f"Warning: Product guide ingestion failed: {e}")
    yield
    # Cleanup on shutdown
    _crew = None


app = FastAPI(
    title="Manufacturing RCA Multi-Agent System",
    description="On-premises root cause analysis using orchestrated AI agents",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models

class CaseSubmission(BaseModel):
    """Request model for submitting a failure case."""
    
    failure_type: str = Field(..., description="Type of failure")
    failure_description: str = Field(..., description="Description of the failure")
    part_number: str = Field(..., description="Part number")
    serial_number: Optional[str] = Field(None, description="Serial number if available")
    test_name: Optional[str] = Field(None, description="Name of the failing test")
    test_value: Optional[float] = Field(None, description="Measured test value")
    spec_lower: Optional[float] = Field(None, description="Lower spec limit")
    spec_upper: Optional[float] = Field(None, description="Upper spec limit")

    class Config:
        extra = "forbid"
        json_schema_extra = {
            "example": {
                "failure_type": "leak_test_fail",
                "failure_description": "Unit failed helium leak test. Leak rate 2.3e-6 mbar*l/s, spec max 1e-6.",
                "part_number": "HYD-VALVE-200",
                "serial_number": "SN-789456",
                "test_name": "Helium_Leak_Test",
                "test_value": 2.3e-6,
                "spec_lower": 0,
                "spec_upper": 1e-6
            }
        }


class CaseResponse(BaseModel):
    """Response model for case submission."""
    
    case_id: str
    status: str
    message: str


class ReportResponse(BaseModel):
    """Response model for report retrieval."""
    
    report_id: str
    case_id: str
    status: str
    report: Optional[dict] = None
    error: Optional[str] = None
    llm_usage: Optional[dict] = None
    artifacts: Optional[dict] = None


class FeedbackSubmission(BaseModel):
    """Request model for submitting engineer feedback."""
    
    report_id: str
    engineer_id: str
    report_useful: bool
    
    # Hypothesis feedback
    correct_hypothesis_id: Optional[str] = None
    actual_root_cause: Optional[str] = None
    root_cause_was_in_top_3: bool = False
    
    # Free-form feedback
    what_worked_well: Optional[str] = None
    what_to_improve: Optional[str] = None
    additional_notes: Optional[str] = None


# Background task to run RCA workflow
def _new_id(prefix: str) -> str:
    """Generate IDs that remain sortable while avoiding same-second collisions."""
    return f"{prefix}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"


def run_rca_workflow(case_id: str, case_data: dict):
    """Run the RCA workflow in background."""
    global _crew, _cases, _reports
    
    if _crew is None:
        _cases[case_id]["status"] = "failed"
        _cases[case_id]["error"] = "RCA crew not initialized"
        return
    
    try:
        start_usage = {}
        llm = getattr(_crew, "llm", None)
        if llm is not None and hasattr(llm, "get_usage_metrics"):
            start_usage = llm.get_usage_metrics()

        _cases[case_id]["status"] = "processing"
        
        # Run the workflow
        result = _crew.run(case_data)
        total_time_seconds = result.get("total_time_seconds")
        
        if result.get("success"):
            _cases[case_id]["status"] = "completed"
            _cases[case_id]["report_id"] = result.get("report_id")

            # Prefer end-to-end workflow duration for report metadata.
            if isinstance(total_time_seconds, (int, float)) and result.get("report"):
                result["report"]["processing_time_seconds"] = float(total_time_seconds)
            
            # Store the report
            report_id = result.get("report_id")
            _reports[report_id] = {
                "report_id": report_id,
                "case_id": case_id,
                "report": result.get("report"),
                "created_at": datetime.utcnow().isoformat(),
                "workflow_outputs": result.get("outputs"),
                "workflow_log": result.get("workflow_log"),
                "agent_logs": result.get("agent_logs"),
            }
            
            # Generate PDF
            try:
                pdf_dir = Path("./data/reports")
                pdf_dir.mkdir(parents=True, exist_ok=True)
                pdf_path = pdf_dir / f"{report_id}.pdf"
                generate_pdf_report(
                    report_data=result.get("report", {}),
                    output_path=str(pdf_path),
                    workflow_outputs=result.get("outputs"),
                    workflow_log=result.get("workflow_log"),
                    processing_time_seconds=total_time_seconds,
                )
                _reports[report_id]["pdf_path"] = str(pdf_path)
            except Exception as e:
                print(f"Warning: PDF generation failed: {e}")

            # Generate markdown trace for workflow reasoning
            try:
                md_dir = Path("./data/reports")
                md_dir.mkdir(parents=True, exist_ok=True)
                md_path = md_dir / f"{report_id}.md"
                md_content = generate_markdown_trace(
                    case_id=case_id,
                    case_data=case_data,
                    result=result,
                )
                md_path.write_text(md_content, encoding="utf-8")
                _reports[report_id]["trace_md_path"] = str(md_path)
            except Exception as e:
                print(f"Warning: Markdown trace generation failed: {e}")
        else:
            _cases[case_id]["status"] = "failed"
            _cases[case_id]["error"] = result.get("error", "Unknown error")
            
            # Still persist a trace for debugging failures
            try:
                md_dir = Path("./data/reports")
                md_dir.mkdir(parents=True, exist_ok=True)
                md_path = md_dir / f"{case_id}.md"
                md_content = generate_markdown_trace(
                    case_id=case_id,
                    case_data=case_data,
                    result=result,
                )
                md_path.write_text(md_content, encoding="utf-8")
                _cases[case_id]["trace_md_path"] = str(md_path)
            except Exception as e:
                print(f"Warning: Markdown trace generation failed: {e}")
            
        # Persist per-case LLM usage delta for debugging/observability
        end_usage = {}
        llm = getattr(_crew, "llm", None)
        if llm is not None and hasattr(llm, "get_usage_metrics"):
            end_usage = llm.get_usage_metrics()

        if start_usage or end_usage:
            delta_calls = int(end_usage.get("call_count", 0)) - int(start_usage.get("call_count", 0))
            delta_total_ms = float(end_usage.get("total_latency_ms", 0.0)) - float(
                start_usage.get("total_latency_ms", 0.0)
            )
            case_llm_usage = {
                "calls_during_case": max(0, delta_calls),
                "total_latency_ms_during_case": round(max(0.0, delta_total_ms), 2),
                "avg_latency_ms_during_case": round(
                    (delta_total_ms / delta_calls) if delta_calls > 0 else 0.0, 2
                ),
                "adapter_call_count_total": int(end_usage.get("call_count", 0)),
                "adapter_last_error": end_usage.get("last_error"),
            }
            _cases[case_id]["llm_usage"] = case_llm_usage
            report_id = _cases[case_id].get("report_id")
            if report_id and report_id in _reports:
                _reports[report_id]["llm_usage"] = case_llm_usage

    except Exception as e:
        _cases[case_id]["status"] = "failed"
        _cases[case_id]["error"] = str(e)


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Manufacturing RCA Multi-Agent System",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "crew_initialized": _crew is not None,
    }


@app.get("/health/llm")
async def health_llm():
    """Check API-to-LLM connectivity using the configured adapter."""
    if _crew is None or getattr(_crew, "llm", None) is None:
        raise HTTPException(
            status_code=503,
            detail="LLM not configured. Check LLM_PROVIDER / LLM_API_KEY / LLM_BASE_URL.",
        )

    settings = get_settings()
    prompt = "Respond with exactly: LLM_CONNECTION_OK"
    system_prompt = "You are a connectivity check. Return only the requested token."

    started = perf_counter()
    try:
        response = _crew.llm.invoke(prompt, system_prompt=system_prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM connectivity check failed: {e}")

    latency_ms = round((perf_counter() - started) * 1000, 2)
    response_text = str(response).strip()

    return {
        "status": "ok",
        "latency_ms": latency_ms,
        "provider": settings.llm_provider,
        "model": settings.llm_model_name,
        "base_url": settings.llm_base_url,
        "api_key_configured": bool(settings.llm_api_key),
        "expected_token_found": "LLM_CONNECTION_OK" in response_text,
        "response_preview": response_text[:200],
    }


@app.get("/health/llm/metrics")
async def health_llm_metrics():
    """Expose aggregate LLM adapter usage metrics for this API process."""
    if _crew is None or getattr(_crew, "llm", None) is None:
        raise HTTPException(
            status_code=503,
            detail="LLM not configured. Check LLM_PROVIDER / LLM_API_KEY / LLM_BASE_URL.",
        )

    llm = _crew.llm
    if not hasattr(llm, "get_usage_metrics"):
        raise HTTPException(status_code=500, detail="LLM adapter does not expose usage metrics")

    settings = get_settings()
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "model": settings.llm_model_name,
        "base_url": settings.llm_base_url,
        "usage": llm.get_usage_metrics(),
    }


@app.post("/cases", response_model=CaseResponse)
async def submit_case(
    submission: CaseSubmission,
    background_tasks: BackgroundTasks,
):
    """Submit a new failure case for RCA.
    
    The workflow runs asynchronously. Use GET /cases/{case_id} to check status
    and GET /reports/{report_id} to retrieve the final report.
    """
    # Generate case ID
    case_id = _new_id("CASE")
    
    # Convert submission to dict
    case_data = submission.model_dump()
    case_data["case_id"] = case_id
    case_data["failure_datetime"] = datetime.utcnow().isoformat()
    
    # Store case
    _cases[case_id] = {
        "case_id": case_id,
        "data": case_data,
        "status": "pending",
        "submitted_at": datetime.utcnow().isoformat(),
    }
    
    # Start background workflow
    background_tasks.add_task(run_rca_workflow, case_id, case_data)
    
    return CaseResponse(
        case_id=case_id,
        status="pending",
        message="Case submitted. RCA workflow started.",
    )


@app.get("/cases/{case_id}")
async def get_case(case_id: str):
    """Get the status of a submitted case."""
    if case_id not in _cases:
        raise HTTPException(status_code=404, detail="Case not found")
    
    case = _cases[case_id]
    return {
        "case_id": case_id,
        "status": case["status"],
        "submitted_at": case["submitted_at"],
        "report_id": case.get("report_id"),
        "error": case.get("error"),
        "llm_usage": case.get("llm_usage"),
    }


@app.get("/cases")
async def list_cases(
    status: Optional[str] = None,
    limit: int = 50,
):
    """List submitted cases."""
    cases = list(_cases.values())
    
    if status:
        cases = [c for c in cases if c["status"] == status]
    
    # Sort by submission time (newest first)
    cases.sort(key=lambda x: x["submitted_at"], reverse=True)
    
    return {
        "total": len(cases),
        "cases": cases[:limit],
    }


@app.get("/cases/{case_id}/trace")
async def get_case_trace(case_id: str):
    """Get the markdown trace for a failed case (if available)."""
    if case_id not in _cases:
        raise HTTPException(status_code=404, detail="Case not found")
    
    trace_path = _cases[case_id].get("trace_md_path")
    if not trace_path or not Path(trace_path).exists():
        raise HTTPException(status_code=404, detail="Trace not found")
    
    return PlainTextResponse(Path(trace_path).read_text(encoding="utf-8"))


@app.get("/reports")
async def list_reports(limit: int = 50):
    """List generated reports."""
    reports = list(_reports.values())
    reports.sort(key=lambda x: x["created_at"], reverse=True)
    summarized_reports = [
        {
            "report_id": r["report_id"],
            "case_id": r["case_id"],
            "created_at": r["created_at"],
            "pdf_path": r.get("pdf_path"),
            "trace_md_path": r.get("trace_md_path"),
            "llm_usage": r.get("llm_usage"),
        }
        for r in reports[:limit]
    ]
    
    return {
        "total": len(reports),
        "reports": summarized_reports,
    }


@app.get("/reports/{report_id}", response_model=ReportResponse)
async def get_report(report_id: str):
    """Get a generated RCA report."""
    if report_id not in _reports:
        raise HTTPException(status_code=404, detail="Report not found")
    
    report_data = _reports[report_id]
    return ReportResponse(
        report_id=report_id,
        case_id=report_data["case_id"],
        status="completed",
        report=report_data["report"],
        llm_usage=report_data.get("llm_usage"),
        artifacts={
            "pdf_path": report_data.get("pdf_path"),
            "trace_md_path": report_data.get("trace_md_path"),
        },
    )


@app.get("/reports/{report_id}/trace")
async def get_report_trace(report_id: str):
    """Get the markdown trace for a completed report."""
    if report_id not in _reports:
        raise HTTPException(status_code=404, detail="Report not found")
    
    trace_path = _reports[report_id].get("trace_md_path")
    if not trace_path or not Path(trace_path).exists():
        raise HTTPException(status_code=404, detail="Trace not found")
    
    return PlainTextResponse(Path(trace_path).read_text(encoding="utf-8"))


@app.get("/reports/{report_id}/pdf")
async def get_report_pdf(report_id: str):
    """Download the RCA report as a PDF.
    
    Returns a downloadable PDF file with all findings,
    reasoning, statistical charts, and recommendations.
    """
    if report_id not in _reports:
        raise HTTPException(status_code=404, detail="Report not found")
    
    report_data = _reports[report_id]
    pdf_path = report_data.get("pdf_path")
    
    if not pdf_path or not Path(pdf_path).exists():
        # Generate on-the-fly if not cached
        try:
            pdf_dir = Path("./data/reports")
            pdf_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = str(pdf_dir / f"{report_id}.pdf")
            generate_pdf_report(
                report_data=report_data.get("report", {}),
                output_path=pdf_path,
                workflow_outputs=report_data.get("workflow_outputs"),
                workflow_log=report_data.get("workflow_log"),
                processing_time_seconds=report_data.get("report", {}).get("processing_time_seconds"),
            )
            _reports[report_id]["pdf_path"] = pdf_path
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"PDF generation failed: {str(e)}"
            )
    
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"{report_id}.pdf",
    )


@app.post("/feedback")
async def submit_feedback(submission: FeedbackSubmission):
    """Submit engineer feedback on an RCA report.
    
    This feedback is used to improve the system's prompts,
    retrieval, and analysis recipes.
    """
    if submission.report_id not in _reports:
        raise HTTPException(status_code=404, detail="Report not found")
    
    feedback_id = _new_id("FB")
    
    _feedback[feedback_id] = {
        "feedback_id": feedback_id,
        "report_id": submission.report_id,
        "data": submission.model_dump(),
        "submitted_at": datetime.utcnow().isoformat(),
    }
    
    return {
        "feedback_id": feedback_id,
        "message": "Feedback recorded. Thank you!",
    }


@app.get("/feedback")
async def list_feedback(limit: int = 50):
    """List submitted feedback (for learning loop analysis)."""
    feedback_list = list(_feedback.values())
    feedback_list.sort(key=lambda x: x["submitted_at"], reverse=True)
    
    return {
        "total": len(feedback_list),
        "feedback": feedback_list[:limit],
    }


@app.get("/metrics")
async def get_metrics():
    """Get system metrics for monitoring."""
    total_cases = len(_cases)
    completed = sum(1 for c in _cases.values() if c["status"] == "completed")
    failed = sum(1 for c in _cases.values() if c["status"] == "failed")
    
    feedback_useful = sum(
        1 for f in _feedback.values() 
        if f["data"].get("report_useful")
    )
    
    return {
        "cases": {
            "total": total_cases,
            "completed": completed,
            "failed": failed,
            "pending": total_cases - completed - failed,
        },
        "feedback": {
            "total": len(_feedback),
            "useful_reports": feedback_useful,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
