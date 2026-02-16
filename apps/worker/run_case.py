"""Background worker for running RCA cases.

This can be used as a standalone script or integrated with
a job queue (e.g., Celery, RQ, or a simple database-backed queue).
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.orchestrator.crew import RCACrew
from src.tools.stats_tool import StatsTool
from src.tools.rag_tool import RAGTool
from src.tools.sql_tool import SQLTool
from src.tools.pdf_report import generate_pdf_report
from config.settings import get_settings


def run_case(
    case_data: dict,
    output_dir: str = "./data/reports",
    verbose: bool = True,
) -> dict:
    """Run the RCA workflow for a single case.
    
    Args:
        case_data: The failure case data
        output_dir: Directory to save the report
        verbose: Whether to print progress
        
    Returns:
        Dict with the result
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize tools
    stats_tool = StatsTool(artifacts_dir=str(output_path / "artifacts"))
    rag_tool = RAGTool()
    sql_tool = SQLTool(mock_mode=True)  # Use mock mode for development
    
    # Auto-ingest product guides if configured
    settings = get_settings()
    if settings.product_guide_auto_ingest:
        existing_docs = rag_tool.list_documents()
        if settings.product_guide_rebuild_on_startup or not existing_docs:
            try:
                ingested = rag_tool.ingest_directory(
                    settings.product_guide_dir,
                    rebuild=settings.product_guide_rebuild_on_startup,
                )
                if verbose:
                    print(f"Product guide ingestion complete: {len(ingested)} documents")
            except Exception as e:
                if verbose:
                    print(f"Warning: Product guide ingestion failed: {e}")

    # Initialize crew
    crew = RCACrew(
        stats_tool=stats_tool,
        rag_tool=rag_tool,
        sql_tool=sql_tool,
        verbose=verbose,
    )
    
    # Run workflow
    result = crew.run(case_data)
    
    # Save report if successful
    if result.get("success"):
        if result.get("report") and isinstance(result.get("total_time_seconds"), (int, float)):
            result["report"]["processing_time_seconds"] = float(result["total_time_seconds"])
        report_id = result.get("report_id", "unknown")
        report_path = output_path / f"{report_id}.json"
        
        with open(report_path, "w") as f:
            json.dump(result.get("report", {}), f, indent=2, default=str)
        
        if verbose:
            print(f"\nJSON report saved to: {report_path}")
        
        # Generate PDF report
        try:
            pdf_path = output_path / f"{report_id}.pdf"
            generate_pdf_report(
                report_data=result.get("report", {}),
                output_path=str(pdf_path),
                workflow_outputs=result.get("outputs"),
                workflow_log=result.get("workflow_log"),
                processing_time_seconds=result.get("total_time_seconds"),
            )
            result["pdf_path"] = str(pdf_path)
            if verbose:
                print(f"PDF report saved to: {pdf_path}")
        except Exception as e:
            if verbose:
                print(f"Warning: PDF generation failed: {e}")
    
    return result


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Run RCA workflow for a failure case"
    )
    parser.add_argument(
        "--case-file",
        type=str,
        help="Path to JSON file containing case data"
    )
    parser.add_argument(
        "--failure-type",
        type=str,
        default="leak_test_fail",
        help="Type of failure"
    )
    parser.add_argument(
        "--part-number",
        type=str,
        default="TEST-PART-001",
        help="Part number"
    )
    parser.add_argument(
        "--description",
        type=str,
        default="Test failure for development",
        help="Failure description"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/reports",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Load case data
    if args.case_file:
        with open(args.case_file) as f:
            case_data = json.load(f)
    else:
        # Create sample case
        case_data = {
            "failure_type": args.failure_type,
            "failure_description": args.description,
            "part_number": args.part_number,
            "lot_number": f"LOT-{datetime.now().strftime('%Y%m%d')}",
            "test_value": 2.3e-6,
            "spec_upper": 1e-6,
        }
    
    # Run the workflow
    result = run_case(
        case_data=case_data,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
    
    # Print summary
    if result.get("success"):
        print("\n" + "="*50)
        print("RCA WORKFLOW COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Report ID: {result.get('report_id')}")
        if result.get("pdf_path"):
            print(f"PDF Report: {result['pdf_path']}")
        
        report = result.get("report", {})
        if report:
            hypotheses = report.get("ranked_hypotheses", [])
            if hypotheses:
                print(f"\nTop Hypothesis: {hypotheses[0].get('title', 'Unknown')}")
                print(f"Confidence: {hypotheses[0].get('posterior_confidence', 0):.0%}")
    else:
        print("\n" + "="*50)
        print("RCA WORKFLOW FAILED")
        print("="*50)
        print(f"Error: {result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
