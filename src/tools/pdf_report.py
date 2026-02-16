"""PDF Report Generator for RCA Reports.

Converts the structured RCA report dict into a professional,
readable PDF document with embedded charts and formatted tables.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import unicodedata

from fpdf import FPDF


# ── Color Palette ──────────────────────────────────────────
_NAVY       = (0, 40, 85)
_DARK_NAVY  = (0, 28, 60)
_BLUE       = (0, 102, 178)
_LIGHT_BLUE = (220, 235, 250)
_ACCENT     = (0, 136, 204)
_WHITE      = (255, 255, 255)
_OFF_WHITE  = (248, 249, 252)
_LIGHT_GRAY = (235, 238, 242)
_MID_GRAY   = (160, 165, 175)
_DARK_GRAY  = (60, 65, 72)
_TEXT        = (35, 38, 48)
_TEXT_LIGHT  = (90, 95, 105)
_GREEN      = (16, 138, 62)
_GREEN_BG   = (232, 248, 238)
_AMBER      = (180, 120, 0)
_AMBER_BG   = (255, 248, 230)
_RED        = (190, 30, 45)
_RED_BG     = (255, 235, 237)
_PURPLE     = (100, 60, 160)


class RCAReportPDF(FPDF):
    """Custom PDF class for RCA reports with header/footer."""

    def __init__(self, title: str = "RCA Report", case_id: str = ""):
        super().__init__()
        self.report_title = title
        self.case_id = case_id
        self.set_auto_page_break(auto=True, margin=22)
        self._is_cover = True  # suppress header on cover page

    def multi_cell(self, w, h, text="", **kwargs):
        """Reset X only for full-width cells to avoid indentation regressions."""
        if w == 0:
            self.set_x(self.l_margin)
        return super().multi_cell(w, h, text, **kwargs)

    def header(self):
        """Page header – skipped on cover page."""
        if self._is_cover:
            return
        # Thin accent bar at top
        self.set_fill_color(*_NAVY)
        self.rect(0, 0, self.w, 3, "F")
        # Header text
        self.set_y(6)
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*_MID_GRAY)
        half_w = (self.w - 20) / 2
        self.cell(half_w, 5, self.report_title, ln=False, align="L")
        self.set_font("Helvetica", "B", 7.5)
        self.cell(half_w, 5, self.case_id, ln=True, align="R")
        self.set_draw_color(*_LIGHT_GRAY)
        self.set_line_width(0.3)
        self.line(10, self.get_y() + 1, self.w - 10, self.get_y() + 1)
        self.ln(5)

    def footer(self):
        """Page footer."""
        self.set_y(-14)
        self.set_draw_color(*_LIGHT_GRAY)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.set_y(-12)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*_MID_GRAY)
        self.cell(0, 8, f"Page {self.page_no()} of {{nb}}", align="C")


# ────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────

def generate_pdf_report(
    report_data: dict,
    output_path: str,
    workflow_outputs: Optional[dict] = None,
    workflow_log: Optional[list] = None,
    processing_time_seconds: Optional[float] = None,
) -> str:
    """Generate a professional PDF report from the RCA report dict.

    Args:
        report_data: The RCA report dict (from RCAReport.model_dump())
        output_path: Path to write the PDF file
        workflow_outputs: Optional dict of all agent outputs for reasoning
        workflow_log: Optional workflow log entries
        processing_time_seconds: Optional end-to-end workflow duration override

    Returns:
        Path to the generated PDF file
    """
    title = report_data.get("title", "Root Cause Analysis Report")
    case_id = report_data.get("case_id", "")
    report_id = report_data.get("report_id", "")

    pdf = RCAReportPDF(title=title, case_id=case_id)
    pdf.alias_nb_pages()
    pdf.add_page()

    # ── Cover Page ──
    generated_at = report_data.get("generated_at", datetime.utcnow().isoformat())
    process_seconds = processing_time_seconds
    if process_seconds is None:
        process_seconds = report_data.get("processing_time_seconds", 0)
    confidence = report_data.get("overall_confidence", 0)
    agents = report_data.get("agents_involved", [])

    _render_cover_page(pdf, title, case_id, report_id, generated_at,
                       process_seconds, confidence, agents)

    # Turn on header for subsequent pages
    pdf._is_cover = False

    # ── Top Hypothesis Highlight ──
    top_hyp = report_data.get("top_hypothesis")
    if top_hyp:
        _render_top_hypothesis_box(pdf, top_hyp)

    # ── Sections ──
    sections = report_data.get("sections", [])
    for section in sections:
        _render_section(pdf, section, report_data)

    # ── Recommended Actions ──
    imm = report_data.get("immediate_actions", [])
    inv = report_data.get("investigation_actions", [])
    prev = report_data.get("preventive_actions", [])
    if imm or inv or prev:
        _render_action_plan(pdf, imm, inv, prev)

    # ── Ranked Hypotheses Detail Table ──
    ranked = report_data.get("ranked_hypotheses", [])
    if ranked:
        _add_section_heading(pdf, "Hypothesis Ranking Summary")
        _render_hypothesis_table(pdf, ranked)

    # ── Charts / Graphs ──
    chart_paths = _collect_chart_paths(report_data, workflow_outputs)
    if chart_paths:
        _add_section_heading(pdf, "Statistical Charts & Graphs")
        for chart_path in chart_paths:
            _embed_chart(pdf, chart_path)

    # ── Agent Reasoning ──
    if workflow_outputs:
        _add_section_heading(pdf, "Agent Reasoning & Details")
        _render_agent_reasoning(pdf, workflow_outputs)

    # ── Caveats & Data Gaps ──
    caveats = report_data.get("caveats", [])
    data_gaps = report_data.get("data_gaps", [])
    if caveats or data_gaps:
        _render_caveats_section(pdf, caveats, data_gaps)

    # ── Workflow Log ──
    if workflow_log:
        _render_workflow_log(pdf, workflow_log)

    # ── Citations Appendix ──
    citations = report_data.get("all_citations", [])
    if citations:
        _add_section_heading(pdf, "Citations & References")
        _render_citations(pdf, citations)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    return str(output_path)


# ────────────────────────────────────────────────
# Cover page
# ────────────────────────────────────────────────

def _render_cover_page(pdf, title, case_id, report_id, generated_at,
                       process_seconds, confidence, agents):
    """Render a professional cover page with banner and confidence gauge."""
    # Navy banner across top
    pdf.set_fill_color(*_NAVY)
    pdf.rect(0, 0, pdf.w, 62, "F")

    # Accent stripe
    pdf.set_fill_color(*_ACCENT)
    pdf.rect(0, 62, pdf.w, 2.5, "F")

    # Title text in banner
    pdf.set_y(14)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*_LIGHT_BLUE)
    pdf.cell(0, 6, "ROOT CAUSE ANALYSIS", align="C", ln=True)

    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(*_WHITE)
    pdf.multi_cell(0, 13, _sanitize_text(title), align="C")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(180, 200, 225)
    pdf.cell(0, 6, f"Case {case_id}", align="C", ln=True)

    # Move below banner
    pdf.set_y(74)

    # ── Confidence gauge ──
    _render_confidence_gauge(pdf, confidence)
    pdf.ln(6)

    # ── Metadata card ──
    _render_meta_card(pdf, [
        ("Report ID", report_id),
        ("Case ID", case_id),
        ("Generated", str(generated_at)[:19]),
        ("Confidence", f"{confidence:.0%}"),
        ("Processing Time", f"{float(process_seconds):.1f}s"),
    ])

    # Agents tag line
    if agents:
        pdf.ln(5)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*_TEXT_LIGHT)
        pdf.multi_cell(0, 4, f"Agents: {', '.join(agents)}", align="C")


def _render_confidence_gauge(pdf, confidence: float):
    """Draw a horizontal confidence bar with percentage label."""
    bar_w = 100
    bar_h = 8
    x_start = (pdf.w - bar_w) / 2
    y = pdf.get_y()

    # Label
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*_TEXT_LIGHT)
    pdf.cell(0, 5, "Overall Confidence", align="C", ln=True)
    pdf.ln(1)
    y = pdf.get_y()

    # Background track
    pdf.set_fill_color(*_LIGHT_GRAY)
    pdf.rect(x_start, y, bar_w, bar_h, "F")

    # Filled portion – color varies by confidence
    fill_w = max(bar_w * confidence, 0.1)
    if confidence >= 0.7:
        pdf.set_fill_color(*_GREEN)
    elif confidence >= 0.4:
        pdf.set_fill_color(*_AMBER)
    else:
        pdf.set_fill_color(*_RED)
    pdf.rect(x_start, y, fill_w, bar_h, "F")

    # Percentage text overlay
    pdf.set_xy(x_start, y)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*_WHITE)
    pdf.cell(bar_w, bar_h, f"{confidence:.0%}", align="C")
    pdf.set_y(y + bar_h + 1)


def _render_meta_card(pdf, rows):
    """Draw a centered metadata card with subtle background."""
    card_w = 130
    row_h = 7
    card_h = row_h * len(rows) + 8
    x_start = (pdf.w - card_w) / 2
    y = pdf.get_y()

    # Card background
    pdf.set_fill_color(*_OFF_WHITE)
    pdf.set_draw_color(*_LIGHT_GRAY)
    pdf.set_line_width(0.3)
    pdf.rect(x_start, y, card_w, card_h, "DF")

    pdf.set_y(y + 4)
    col_k = 42
    col_v = card_w - col_k - 10

    for key, value in rows:
        pdf.set_x(x_start + 8)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*_TEXT_LIGHT)
        pdf.cell(col_k, row_h, _sanitize_text(key))
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*_TEXT)
        pdf.cell(col_v, row_h, _sanitize_text(value))
        pdf.ln()

    pdf.set_y(y + card_h + 2)


# ────────────────────────────────────────────────
# Section heading
# ────────────────────────────────────────────────

def _add_section_heading(pdf: FPDF, text: str):
    """Add a styled section heading with left accent bar."""
    if pdf.get_y() > pdf.h - 50:
        pdf.add_page()
    pdf.ln(8)

    y = pdf.get_y()
    # Left accent bar
    pdf.set_fill_color(*_ACCENT)
    pdf.rect(10, y, 3, 11, "F")

    pdf.set_x(17)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*_NAVY)
    pdf.cell(0, 11, _sanitize_text(text), ln=True)

    # Subtle underline
    pdf.set_draw_color(*_LIGHT_GRAY)
    pdf.set_line_width(0.3)
    pdf.line(10, pdf.get_y(), pdf.w - 10, pdf.get_y())
    pdf.ln(4)


# ────────────────────────────────────────────────
# Top hypothesis highlight
# ────────────────────────────────────────────────

def _render_top_hypothesis_box(pdf, hyp: dict):
    """Render a highlighted callout box for the top hypothesis."""
    pdf.add_page()
    _add_section_heading(pdf, "Top Hypothesis")

    title = _sanitize_text(str(hyp.get("title", "Unknown")))
    desc = _sanitize_text(str(hyp.get("description", "")))
    mechanism = _sanitize_text(str(hyp.get("mechanism", "")))
    conf = hyp.get("posterior_confidence", hyp.get("prior_confidence", 0))
    status = str(hyp.get("status", "proposed")).upper()

    x = 12
    w = pdf.w - 24
    y = pdf.get_y()

    # Box
    pdf.set_fill_color(*_LIGHT_BLUE)
    pdf.set_draw_color(*_ACCENT)
    pdf.set_line_width(0.6)
    # We'll measure content height after writing, so start with a generous box
    box_top = y

    pdf.set_xy(x + 6, y + 4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*_NAVY)
    pdf.cell(w - 40, 7, title)
    # Confidence badge
    _draw_badge(pdf, pdf.get_x() + 2, pdf.get_y(), f"{conf:.0%}", *_confidence_color(conf))
    pdf.ln(9)

    pdf.set_x(x + 6)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*_TEXT)
    pdf.multi_cell(w - 12, 5.5, desc)

    if mechanism:
        pdf.ln(2)
        pdf.set_x(x + 6)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*_TEXT_LIGHT)
        # Keep mechanism text in one wrapped block to avoid cell/multi_cell overlap.
        pdf.multi_cell(w - 12, 5, f"Mechanism: {mechanism}")

    box_bottom = pdf.get_y() + 4
    # Draw the box retroactively
    pdf.set_fill_color(*_LIGHT_BLUE)
    pdf.set_draw_color(*_ACCENT)
    pdf.set_line_width(0.5)
    pdf.rect(x, box_top, w, box_bottom - box_top, "D")
    # Left accent
    pdf.set_fill_color(*_ACCENT)
    pdf.rect(x, box_top, 3, box_bottom - box_top, "F")

    pdf.set_y(box_bottom + 2)


# ────────────────────────────────────────────────
# Recommended Actions (color-coded)
# ────────────────────────────────────────────────

def _render_action_plan(pdf, immediate, investigation, preventive):
    """Render recommended actions with color-coded priority."""
    _add_section_heading(pdf, "Recommended Actions")

    groups = [
        ("Immediate Actions", immediate, _RED, _RED_BG, "!"),
        ("Investigation Actions", investigation, _AMBER, _AMBER_BG, "?"),
        ("Preventive Actions", preventive, _GREEN, _GREEN_BG, "+"),
    ]
    for label, items, color, bg, icon in groups:
        if not items:
            continue
        _render_action_group(pdf, label, items, color, bg, icon)
        pdf.ln(3)


def _render_action_group(pdf, label, items, color, bg, icon):
    """Render a group of action items with a colored callout."""
    if pdf.get_y() > pdf.h - 40:
        pdf.add_page()

    x = 12
    w = pdf.w - 24
    y = pdf.get_y()

    # Estimate height
    line_h = 6
    est_h = 8 + len(items) * (line_h + 2) + 4

    # Background
    pdf.set_fill_color(*bg)
    pdf.rect(x, y, w, est_h, "F")
    # Left accent
    pdf.set_fill_color(*color)
    pdf.rect(x, y, 3, est_h, "F")

    # Title
    pdf.set_xy(x + 8, y + 3)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*color)
    pdf.cell(0, 6, f"{icon}  {label}", ln=True)

    # Items
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(*_TEXT)
    for item in items:
        pdf.set_x(x + 12)
        pdf.cell(4, line_h, "-")
        pdf.multi_cell(w - 20, line_h, _sanitize_text(item))

    pdf.set_y(y + est_h + 2)


# ────────────────────────────────────────────────
# Caveats & Data Gaps (warning callouts)
# ────────────────────────────────────────────────

def _render_caveats_section(pdf, caveats, data_gaps):
    """Render caveats and data gaps as amber warning callouts."""
    _add_section_heading(pdf, "Caveats & Data Gaps")

    if caveats:
        _render_callout_box(pdf, "Caveats", caveats, _AMBER, _AMBER_BG)
        pdf.ln(4)
    if data_gaps:
        _render_callout_box(pdf, "Data Gaps", data_gaps, _RED, _RED_BG)


def _render_callout_box(pdf, title, items, color, bg):
    """Render a callout box with colored left bar."""
    if pdf.get_y() > pdf.h - 40:
        pdf.add_page()

    x = 12
    w = pdf.w - 24
    y = pdf.get_y()
    line_h = 5.5
    est_h = 8 + len(items) * (line_h + 1.5) + 4

    pdf.set_fill_color(*bg)
    pdf.rect(x, y, w, est_h, "F")
    pdf.set_fill_color(*color)
    pdf.rect(x, y, 3, est_h, "F")

    pdf.set_xy(x + 8, y + 3)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*color)
    pdf.cell(0, 6, title, ln=True)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*_TEXT)
    for item in items:
        pdf.set_x(x + 12)
        pdf.cell(4, line_h, "-")
        pdf.multi_cell(w - 20, line_h, _sanitize_text(item))

    pdf.set_y(y + est_h + 2)


# ────────────────────────────────────────────────
# Workflow Log (formatted)
# ────────────────────────────────────────────────

def _render_workflow_log(pdf, workflow_log):
    """Render workflow log with alternating rows and level colors."""
    _add_section_heading(pdf, "Workflow Log")

    level_colors = {
        "INFO": _TEXT_LIGHT,
        "WARNING": _AMBER,
        "ERROR": _RED,
        "DEBUG": _MID_GRAY,
    }

    for i, entry in enumerate(workflow_log):
        if pdf.get_y() > pdf.h - 12:
            pdf.add_page()

        ts = entry.get("timestamp", "")[:19]
        level = entry.get("level", "info").upper()
        msg = _sanitize_text(entry.get("message", ""))

        # Alternating row background
        if i % 2 == 0:
            pdf.set_fill_color(*_OFF_WHITE)
            pdf.rect(10, pdf.get_y(), pdf.w - 20, 5, "F")

        pdf.set_x(12)
        pdf.set_font("Courier", "", 7)
        pdf.set_text_color(*_MID_GRAY)
        pdf.cell(36, 5, ts)

        color = level_colors.get(level, _TEXT_LIGHT)
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(*color)
        pdf.cell(16, 5, level)

        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(*_TEXT)
        pdf.multi_cell(0, 5, msg)


# ────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────

def _render_section(pdf: FPDF, section: dict, report_data: dict):
    """Render a report section from its markdown-ish content."""
    title = section.get("title", "")
    content = section.get("content", "")
    chart_paths = section.get("chart_paths", [])

    _add_section_heading(pdf, title)
    _render_markdown_content(pdf, content)

    for cp in chart_paths:
        _embed_chart(pdf, cp)


def _sanitize_text(text: str) -> str:
    """Remove or replace characters not supported by core PDF fonts."""
    text = unicodedata.normalize("NFKD", str(text))
    replacements = {
        "\u2022": "-",
        "\u2713": "[OK]",
        "\u2717": "[X]",
        "\u2705": "[OK]",
        "\u274c": "[X]",
        "\u2753": "[?]",
        "\u2014": "--",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u0304": "",
        "\u2212": "-",
        "\u207b": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _render_markdown_content(pdf: FPDF, content: str):
    """Render simplified markdown content to PDF."""
    content = _sanitize_text(content)
    lines = content.split("\n")
    in_table = False
    table_rows = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("## "):
            continue

        if stripped.startswith("### "):
            if in_table:
                _flush_table(pdf, table_rows)
                in_table = False
                table_rows = []
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(*_NAVY)
            heading_text = stripped.lstrip("# ").strip()
            pdf.cell(0, 7, heading_text, ln=True)
            pdf.ln(1)
            continue

        if stripped.startswith("|"):
            in_table = True
            if all(c in "|-: " for c in stripped):
                continue
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            table_rows.append(cells)
            continue

        if in_table:
            _flush_table(pdf, table_rows)
            in_table = False
            table_rows = []

        if stripped == "---":
            pdf.ln(2)
            pdf.set_draw_color(*_LIGHT_GRAY)
            pdf.set_line_width(0.2)
            pdf.line(10, pdf.get_y(), pdf.w - 10, pdf.get_y())
            pdf.ln(3)
            continue

        if stripped.startswith("**") and stripped.endswith("**"):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*_TEXT)
            pdf.multi_cell(0, 6, stripped.replace("**", ""))
            pdf.set_font("Helvetica", "", 10)
            continue

        if stripped.startswith("- ") or stripped.startswith("* "):
            _render_bullet(pdf, stripped[2:])
            continue

        if len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in (".", ")"):
            _render_bullet(pdf, stripped, numbered=True)
            continue

        if stripped == "":
            pdf.ln(3)
            continue

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*_TEXT)
        _render_inline_bold(pdf, stripped)

    if in_table:
        _flush_table(pdf, table_rows)


def _render_inline_bold(pdf: FPDF, text: str):
    """Render a line that may contain **bold** segments."""
    parts = text.split("**")
    if len(parts) <= 1:
        pdf.multi_cell(0, 6, text)
        return

    pdf.set_x(pdf.l_margin)
    for i, part in enumerate(parts):
        if i % 2 == 1:
            pdf.set_font("Helvetica", "B", 10)
        else:
            pdf.set_font("Helvetica", "", 10)
        pdf.write(6, part)
    pdf.ln(6)


def _render_bullet(pdf: FPDF, text: str, numbered: bool = False):
    """Render a bullet point or numbered item."""
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*_TEXT)

    if not numbered:
        # Colored bullet dot
        x = pdf.l_margin + 4
        y = pdf.get_y() + 2.5
        pdf.set_fill_color(*_ACCENT)
        pdf.ellipse(x, y, 1.8, 1.8, "F")
        pdf.set_x(pdf.l_margin + 8)
        pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 8, 6, _sanitize_text(text))
    else:
        pdf.set_x(pdf.l_margin + 4)
        pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 4, 6, _sanitize_text(text))


def _flush_table(pdf: FPDF, rows: list[list[str]]):
    """Render accumulated table rows with styled headers and zebra striping."""
    if not rows:
        return

    num_cols = max(len(r) for r in rows)
    if num_cols == 0:
        return

    usable_w = pdf.w - 24
    col_w = usable_w / num_cols
    x_start = 12

    for i, row in enumerate(rows):
        if pdf.get_y() > pdf.h - 15:
            pdf.add_page()

        if i == 0:
            pdf.set_font("Helvetica", "B", 8.5)
            pdf.set_fill_color(*_NAVY)
            pdf.set_text_color(*_WHITE)
        else:
            pdf.set_font("Helvetica", "", 8.5)
            if i % 2 == 0:
                pdf.set_fill_color(*_OFF_WHITE)
            else:
                pdf.set_fill_color(*_WHITE)
            pdf.set_text_color(*_TEXT)

        pdf.set_x(x_start)
        for j, cell in enumerate(row):
            if j < num_cols:
                pdf.cell(col_w, 7, _sanitize_text(cell[:55]), border=0, fill=True)
        for j in range(len(row), num_cols):
            pdf.cell(col_w, 7, "", border=0, fill=True)
        pdf.ln()

    # Bottom border
    pdf.set_draw_color(*_LIGHT_GRAY)
    pdf.set_line_width(0.3)
    pdf.line(x_start, pdf.get_y(), x_start + usable_w, pdf.get_y())
    pdf.ln(4)


def _render_hypothesis_table(pdf: FPDF, ranked_hypotheses: list[dict]):
    """Render a table summarizing ranked hypotheses with color-coded confidence."""
    col_widths = [10, 62, 28, 28, 28, 34]
    total_w = sum(col_widths)
    x_start = (pdf.w - total_w) / 2
    headers = ["#", "Hypothesis", "Prior", "Posterior", "Status", "Mechanism"]

    # Header
    pdf.set_x(x_start)
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_fill_color(*_NAVY)
    pdf.set_text_color(*_WHITE)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, fill=True)
    pdf.ln()

    # Rows
    for i, hyp in enumerate(ranked_hypotheses[:10]):
        if pdf.get_y() > pdf.h - 15:
            pdf.add_page()

        if i % 2 == 0:
            pdf.set_fill_color(*_OFF_WHITE)
        else:
            pdf.set_fill_color(*_WHITE)

        rank = str(hyp.get("rank", i + 1))
        title = _sanitize_text(str(hyp.get("title", "Unknown")))[:35]
        prior = hyp.get("prior_confidence", 0)
        posterior = hyp.get("posterior_confidence", hyp.get("prior_confidence", 0))
        status = str(hyp.get("status", "N/A"))
        mechanism = _sanitize_text(str(hyp.get("mechanism", "")))[:22]

        pdf.set_x(x_start)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(*_TEXT)
        pdf.cell(col_widths[0], 8, rank, fill=True)

        pdf.set_font("Helvetica", "", 8.5)
        pdf.cell(col_widths[1], 8, title, fill=True)

        # Prior confidence colored
        fg, _ = _confidence_color(prior)
        pdf.set_text_color(*fg)
        pdf.cell(col_widths[2], 8, f"{prior:.0%}", fill=True)

        # Posterior confidence colored
        fg, _ = _confidence_color(posterior)
        pdf.set_text_color(*fg)
        pdf.cell(col_widths[3], 8, f"{posterior:.0%}" if posterior else "-", fill=True)

        # Status colored
        status_color = _status_color(status)
        pdf.set_text_color(*status_color)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(col_widths[4], 8, status.upper()[:12], fill=True)

        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*_TEXT_LIGHT)
        pdf.cell(col_widths[5], 8, mechanism, fill=True)
        pdf.ln()

    # Bottom border
    pdf.set_draw_color(*_LIGHT_GRAY)
    pdf.set_line_width(0.3)
    pdf.line(x_start, pdf.get_y(), x_start + total_w, pdf.get_y())
    pdf.ln(4)


def _collect_chart_paths(
    report_data: dict,
    workflow_outputs: Optional[dict] = None,
) -> list[str]:
    """Collect all chart image paths from the report and outputs."""
    paths = set()

    for section in report_data.get("sections", []):
        for cp in section.get("chart_paths", []):
            if cp:
                paths.add(cp)

    if workflow_outputs:
        stats_output = workflow_outputs.get("stats", None)
        if stats_output:
            data = getattr(stats_output, "data", stats_output) if not isinstance(stats_output, dict) else stats_output
            if isinstance(data, dict):
                inner = data.get("data", data)
                for ev in inner.get("evidence", []):
                    cp = ev.get("chart_path")
                    if cp:
                        paths.add(cp)
                for hr in inner.get("hypothesis_results", []):
                    for test in hr.get("tests_run", []):
                        cp = test.get("chart_path")
                        if cp:
                            paths.add(cp)

    for hyp in report_data.get("ranked_hypotheses", []):
        for ev in hyp.get("evidence_for", []):
            cp = ev.get("chart_path")
            if cp:
                paths.add(cp)
        for ev in hyp.get("evidence_against", []):
            cp = ev.get("chart_path")
            if cp:
                paths.add(cp)

    return [p for p in paths if Path(p).exists()]


def _embed_chart(pdf: FPDF, chart_path: str):
    """Embed a chart image in the PDF with a light frame."""
    path = Path(chart_path)
    if not path.exists():
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*_RED)
        pdf.cell(0, 6, f"[Chart not found: {path.name}]", ln=True)
        return

    if pdf.get_y() > pdf.h - 90:
        pdf.add_page()

    pdf.ln(3)

    # Chart label
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*_TEXT_LIGHT)
    pdf.cell(0, 5, f"Chart: {path.stem}", ln=True)
    pdf.ln(1)

    img_w = pdf.w - 34
    x = 17
    y_before = pdf.get_y()

    try:
        pdf.image(str(path), x=x, w=img_w)
    except Exception as e:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*_RED)
        pdf.cell(0, 6, f"[Error embedding chart: {e}]", ln=True)
        return

    y_after = pdf.get_y()
    # Draw subtle frame around chart
    pdf.set_draw_color(*_LIGHT_GRAY)
    pdf.set_line_width(0.3)
    pdf.rect(x - 1, y_before - 1, img_w + 2, y_after - y_before + 2, "D")

    pdf.ln(6)


def _render_agent_reasoning(pdf: FPDF, workflow_outputs: dict):
    """Render reasoning from each agent as styled cards."""
    agent_order = [
        ("intake", "Intake & Triage"),
        ("product_guide", "Product Guide Retrieval"),
        ("research", "Private Data Research"),
        ("hypothesis", "Hypothesis Generation"),
        ("test_plan", "Test Planning"),
        ("stats", "Statistical Analysis"),
        ("critic", "Evidence Critique"),
        ("report", "Report Generation"),
    ]

    for key, display_name in agent_order:
        output = workflow_outputs.get(key)
        if output is None:
            continue

        if hasattr(output, "reasoning"):
            reasoning = output.reasoning
            confidence = output.confidence
            success = output.success
            error_msg = output.error_message
            data = output.data
        elif isinstance(output, dict):
            reasoning = output.get("reasoning")
            confidence = output.get("confidence")
            success = output.get("success", True)
            error_msg = output.get("error_message")
            data = output.get("data", {})
        else:
            continue

        if pdf.get_y() > pdf.h - 35:
            pdf.add_page()

        _render_agent_card(pdf, display_name, reasoning, confidence,
                           success, error_msg, key, data)


def _render_agent_card(pdf, name, reasoning, confidence, success,
                       error_msg, agent_key, data):
    """Render a single agent's reasoning as a card."""
    x = 12
    w = pdf.w - 24
    y = pdf.get_y()

    # Card header bar
    header_h = 8
    if success:
        pdf.set_fill_color(*_NAVY)
    else:
        pdf.set_fill_color(*_RED)
    pdf.rect(x, y, w, header_h, "F")

    pdf.set_xy(x + 4, y + 1)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*_WHITE)
    pdf.cell(w - 50, header_h - 2, _sanitize_text(name))

    # Status + confidence in header
    status_text = "OK" if success else "FAIL"
    conf_str = f" | {confidence:.0%}" if confidence else ""
    pdf.set_font("Helvetica", "", 8)
    pdf.cell(46, header_h - 2, f"{status_text}{conf_str}", align="R")
    pdf.ln()

    # Card body
    body_top = y + header_h
    pdf.set_y(body_top + 2)

    pdf.set_fill_color(*_OFF_WHITE)
    # We'll draw the body background after measuring content

    if reasoning:
        pdf.set_x(x + 6)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(*_TEXT)
        pdf.multi_cell(w - 12, 4.5, _sanitize_text(f"Reasoning: {reasoning}"))

    if error_msg:
        pdf.set_x(x + 6)
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(*_RED)
        pdf.multi_cell(w - 12, 4.5, _sanitize_text(f"Error: {error_msg}"))

    if isinstance(data, dict):
        _render_agent_data_highlights(pdf, agent_key, data, x + 6, w - 12)

    body_bottom = pdf.get_y() + 2
    # Draw body background
    pdf.set_fill_color(*_OFF_WHITE)
    pdf.set_draw_color(*_LIGHT_GRAY)
    pdf.set_line_width(0.2)
    pdf.rect(x, body_top, w, body_bottom - body_top, "D")

    pdf.set_y(body_bottom + 4)


def _render_agent_data_highlights(pdf: FPDF, agent_key: str, data: dict,
                                   x_offset: float, max_w: float):
    """Render key highlights from agent data."""
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*_TEXT_LIGHT)

    if agent_key == "hypothesis":
        hypotheses = data.get("hypotheses", [])
        if hypotheses:
            pdf.set_x(x_offset)
            pdf.cell(0, 5, f"Generated {len(hypotheses)} hypotheses:", ln=True)
            for h in hypotheses[:5]:
                title = h.get("title", "Unknown") if isinstance(h, dict) else str(h)
                prior = h.get("prior_confidence", "") if isinstance(h, dict) else ""
                conf_str = f" (prior: {prior:.0%})" if prior else ""
                pdf.set_x(x_offset + 4)
                pdf.cell(0, 4, _sanitize_text(f"- {title}{conf_str}"), ln=True)

    elif agent_key == "stats":
        total = data.get("total_tests_run", 0)
        summary = data.get("summary", {})
        pdf.set_x(x_offset)
        pdf.cell(0, 5, f"Total tests run: {total}", ln=True)
        pdf.set_x(x_offset)
        pdf.cell(
            0, 5,
            f"Supported: {summary.get('supported', 0)} | "
            f"Refuted: {summary.get('refuted', 0)} | "
            f"Inconclusive: {summary.get('inconclusive', 0)}",
            ln=True,
        )
        findings = summary.get("key_findings", [])
        if findings:
            for f in findings[:3]:
                pdf.set_x(x_offset + 4)
                pdf.cell(
                    0, 4,
                    _sanitize_text(f"- {f.get('hypothesis', '')}: {f.get('finding', '')}"),
                    ln=True,
                )
        else:
            pdf.set_x(x_offset + 4)
            pdf.cell(0, 4, "- No statistically significant findings detected.", ln=True)

    elif agent_key == "critic":
        issues = data.get("issues_found", [])
        if issues:
            pdf.set_x(x_offset)
            pdf.cell(0, 5, f"Issues found: {len(issues)}", ln=True)
            for issue in issues[:3]:
                sev = issue.get("severity", "?")
                desc = _sanitize_text(issue.get("issue", "Unknown"))
                pdf.set_x(x_offset + 4)
                pdf.cell(0, 4, f"[{sev}] {desc}", ln=True)
        needs_more = data.get("needs_more_evidence", False)
        if needs_more:
            pdf.set_x(x_offset)
            pdf.cell(0, 5, "Verdict: More evidence needed", ln=True)

    elif agent_key == "research":
        sources = data.get("data_sources_queried", [])
        total_records = data.get("total_records", 0)
        if sources:
            pdf.set_x(x_offset)
            pdf.cell(0, 5, f"Data sources: {', '.join(sources[:5])}", ln=True)
        pdf.set_x(x_offset)
        pdf.cell(0, 5, f"Total records analyzed: {total_records}", ln=True)


def _render_citations(pdf: FPDF, citations: list[dict]):
    """Render citations as a formatted reference list."""
    for i, cit in enumerate(citations, 1):
        if pdf.get_y() > pdf.h - 20:
            pdf.add_page()

        source_type = _sanitize_text(cit.get("source_type", "unknown"))
        source_name = _sanitize_text(cit.get("source_name", "Unknown"))
        excerpt = _sanitize_text(cit.get("excerpt", ""))
        source_id = _sanitize_text(cit.get("source_id", ""))

        # Reference number badge
        y = pdf.get_y()
        pdf.set_fill_color(*_ACCENT)
        pdf.set_text_color(*_WHITE)
        pdf.set_font("Helvetica", "B", 7)
        badge_w = 8
        pdf.set_x(12)
        pdf.cell(badge_w, 5, str(i), fill=True, align="C")

        pdf.set_x(22)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*_TEXT)
        pdf.cell(0, 5, f"{source_name} ({source_type})", ln=True)

        if excerpt:
            pdf.set_x(22)
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(*_TEXT_LIGHT)
            pdf.multi_cell(pdf.w - 34, 4, f'"{excerpt[:200]}"')

        if source_id:
            pdf.set_x(22)
            pdf.set_font("Helvetica", "", 7.5)
            pdf.set_text_color(*_MID_GRAY)
            pdf.cell(0, 4, f"ID: {source_id}", ln=True)

        pdf.ln(2)


# ────────────────────────────────────────────────
# Color helpers
# ────────────────────────────────────────────────

def _confidence_color(conf) -> tuple:
    """Return (foreground, background) colors for a confidence value."""
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        return (_TEXT_LIGHT, _LIGHT_GRAY)
    if conf >= 0.7:
        return (_GREEN, _GREEN_BG)
    elif conf >= 0.4:
        return (_AMBER, _AMBER_BG)
    else:
        return (_RED, _RED_BG)


def _status_color(status: str) -> tuple:
    """Return a color for a hypothesis status string."""
    s = str(status).lower()
    if s in ("supported", "confirmed"):
        return _GREEN
    elif s in ("refuted", "rejected"):
        return _RED
    elif s in ("inconclusive", "testing"):
        return _AMBER
    else:
        return _TEXT_LIGHT


def _draw_badge(pdf, x, y, text, fg, bg):
    """Draw a small colored badge at (x, y)."""
    pdf.set_fill_color(*bg)
    badge_w = max(pdf.get_string_width(text) + 5, 14)
    pdf.rect(x, y + 1, badge_w, 6, "F")
    pdf.set_xy(x, y + 1)
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_text_color(*fg)
    pdf.cell(badge_w, 6, text, align="C")
