"""Generate a markdown trace of RCA workflow reasoning and evidence."""

from __future__ import annotations

from datetime import datetime
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None

MAX_SANITIZE_DEPTH = 6
MAX_SANITIZE_ITEMS = 50


def generate_markdown_trace(
    *,
    case_id: str,
    case_data: dict,
    result: dict,
) -> str:
    """Generate a comprehensive markdown trace for a workflow run."""
    timestamp = datetime.utcnow().isoformat()
    success = result.get("success", False)
    report_id = result.get("report_id")
    total_time = result.get("total_time_seconds")

    outputs = result.get("outputs", {})
    workflow_log = result.get("workflow_log", [])
    agent_logs = result.get("agent_logs", {})

    lines: list[str] = []
    lines.append(f"# RCA Workflow Trace - {case_id}")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append(f"- Timestamp (UTC): {timestamp}")
    lines.append(f"- Success: {success}")
    if report_id:
        lines.append(f"- Report ID: {report_id}")
    if total_time is not None:
        lines.append(f"- Total Time (s): {total_time:.1f}")
    lines.append("")

    lines.append("## Case Data")
    lines.append(_format_json_block(case_data))
    lines.append("")

    lines.append("## Workflow Log")
    if workflow_log:
        for entry in workflow_log:
            lines.append(_format_log_entry(entry))
    else:
        lines.append("- No workflow log entries recorded.")
    lines.append("")

    lines.append("## Agent Reasoning and Outputs")
    for agent_key, output in outputs.items():
        lines.extend(_render_agent_section(agent_key, output, agent_logs.get(agent_key)))
        lines.append("")

    lines.append("## Retrieved Document Chunks")
    product_guide_output = outputs.get("product_guide")
    citations = _extract_citations(product_guide_output)
    if citations:
        for idx, citation in enumerate(citations, start=1):
            lines.extend(_render_citation(idx, citation))
            lines.append("")
    else:
        lines.append("- No document chunks recorded.")
        lines.append("")

    lines.append("## Hypotheses")
    hypothesis_output = outputs.get("hypothesis")
    hypotheses = _extract_hypotheses(hypothesis_output)
    if hypotheses:
        for hyp in hypotheses:
            lines.extend(_render_hypothesis(hyp))
            lines.append("")
    else:
        lines.append("- No hypotheses recorded.")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _render_agent_section(
    agent_key: str,
    output: Any,
    agent_log: list[dict] | None,
) -> list[str]:
    """Render a markdown section for a single agent output."""
    data = _coerce_output(output)
    lines: list[str] = []
    display_name = data.get("agent_name") or agent_key
    lines.append(f"### {display_name}")
    lines.append(f"- Success: {data.get('success', False)}")
    if data.get("error_message"):
        lines.append(f"- Error: {data.get('error_message')}")
    if data.get("confidence") is not None:
        lines.append(f"- Confidence: {data.get('confidence'):.2f}")
    if data.get("reasoning"):
        lines.append(f"- Reasoning: {data.get('reasoning')}")
    lines.append("")

    citations = data.get("citations_used") or []
    lines.append(f"**Citations Used:** {len(citations)}")
    if citations:
        for idx, citation in enumerate(citations[:20], start=1):
            lines.append(f"- {idx}. {citation.get('source_name', 'unknown')} | {citation.get('section_path', 'unknown')}")
        if len(citations) > 20:
            lines.append(f"- ... {len(citations) - 20} more")
    lines.append("")

    if agent_log:
        lines.append("**Execution Log:**")
        for entry in agent_log:
            lines.append(_format_log_entry(entry))
        lines.append("")

    if data.get("data") is not None:
        lines.append("**Output Data:**")
        lines.append(_format_json_block(data.get("data")))

    return lines


def _render_citation(idx: int, citation: dict) -> list[str]:
    lines = [
        f"### Chunk {idx}",
        f"- Source: {citation.get('source_name', 'unknown')}",
        f"- Section: {citation.get('section_path', 'unknown')}",
        f"- Page: {citation.get('page_number', 'n/a')}",
        f"- Score: {citation.get('retrieval_score', 'n/a')}",
        f"- Search Type: {citation.get('search_type', 'n/a')}",
        f"- Matched Query: {citation.get('matched_query', 'n/a')}",
        "",
        "```",
        _safe_text(citation.get("excerpt", "")).strip(),
        "```",
    ]
    return lines


def _render_hypothesis(hypothesis: dict) -> list[str]:
    lines = [
        f"### {hypothesis.get('hypothesis_id', 'unknown')}",
        f"- Title: {hypothesis.get('title', 'unknown')}",
        f"- Rank: {hypothesis.get('rank', 'n/a')}",
        f"- Prior Confidence: {hypothesis.get('prior_confidence', 'n/a')}",
        f"- Description: {hypothesis.get('description', '')}",
        f"- Mechanism: {hypothesis.get('mechanism', '')}",
    ]

    expected = hypothesis.get("expected_signatures") or []
    lines.append(f"- Expected Signatures: {len(expected)}")
    for item in expected:
        lines.append(f"  - {item}")

    tests = hypothesis.get("recommended_tests") or []
    lines.append(f"- Recommended Tests: {len(tests)}")
    for test in tests:
        if isinstance(test, dict):
            label = f"{test.get('test_type', 'unknown')} on {test.get('target_variable', 'unknown')}"
        else:
            label = str(test)
        lines.append(f"  - {label}")

    sources = hypothesis.get("required_data_sources") or []
    lines.append(f"- Required Data Sources: {', '.join(str(s) for s in sources) if sources else 'none'}")
    return lines


def _format_log_entry(entry: dict) -> str:
    timestamp = entry.get("timestamp", "unknown")
    level = entry.get("level", "info")
    message = entry.get("message", "")
    agent = entry.get("agent")
    agent_label = f"{agent} | " if agent else ""
    return f"- {timestamp} [{level}] {agent_label}{message}"


def _extract_citations(output: Any) -> list[dict]:
    data = _coerce_output(output)
    return data.get("citations_used") or []


def _extract_hypotheses(output: Any) -> list[dict]:
    data = _coerce_output(output)
    payload = data.get("data") or {}
    return payload.get("hypotheses") or []


def _coerce_output(output: Any) -> dict:
    if output is None:
        return {}
    if hasattr(output, "model_dump"):
        return output.model_dump()
    if isinstance(output, dict):
        return output
    return {"data": output}


def _format_json_block(payload: Any) -> str:
    sanitized = _sanitize(payload)
    return "```json\n" + _to_pretty_json(sanitized) + "\n```"


def _to_pretty_json(payload: Any) -> str:
    import json

    try:
        return json.dumps(payload, indent=2, sort_keys=True, default=str)
    except Exception:
        return json.dumps(str(payload), indent=2, sort_keys=True)


def _sanitize(value: Any, depth: int = 0) -> Any:
    if depth > MAX_SANITIZE_DEPTH:
        return f"<max_depth_reached:{MAX_SANITIZE_DEPTH}>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if pd is not None and isinstance(value, pd.DataFrame):
        return {
            "row_count": int(value.shape[0]),
            "columns": list(value.columns),
            "head": value.head(5).to_dict(orient="records"),
            "__truncated_rows__": max(0, int(value.shape[0]) - 5),
        }
    if isinstance(value, dict):
        items = list(value.items())
        sanitized_dict = {
            str(k): _sanitize(v, depth + 1) for k, v in items[:MAX_SANITIZE_ITEMS]
        }
        if len(items) > MAX_SANITIZE_ITEMS:
            sanitized_dict["__truncated_keys__"] = len(items) - MAX_SANITIZE_ITEMS
        return sanitized_dict
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        sanitized = [_sanitize(v, depth + 1) for v in items[:MAX_SANITIZE_ITEMS]]
        if len(items) > MAX_SANITIZE_ITEMS:
            sanitized.append({"__truncated_items__": len(items) - MAX_SANITIZE_ITEMS})
        return sanitized
    if hasattr(value, "model_dump"):
        return _sanitize(value.model_dump(), depth + 1)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _sanitize(value.to_dict(), depth + 1)
        except Exception:
            return str(value)
    return str(value)


def _safe_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, str):
        return text
    return str(text)
