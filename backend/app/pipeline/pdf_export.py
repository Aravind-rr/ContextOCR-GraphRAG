from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate


def _font_names() -> tuple[str, str]:
    regular = Path("C:/Windows/Fonts/arial.ttf")
    bold = Path("C:/Windows/Fonts/arialbd.ttf")
    if regular.is_file() and bold.is_file():
        if "OCRExport" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("OCRExport", str(regular)))
            pdfmetrics.registerFont(TTFont("OCRExport-Bold", str(bold)))
            pdfmetrics.registerFontFamily(
                "OCRExport",
                normal="OCRExport",
                bold="OCRExport-Bold",
                italic="OCRExport",
                boldItalic="OCRExport-Bold",
            )
        return "OCRExport", "OCRExport-Bold"
    return "Helvetica", "Helvetica-Bold"


def export_corrected_pdf(
    target: Path,
    *,
    tokens: list[dict[str, Any]],
    correction_records: list[dict[str, Any]],
    source_name: str,
    run_id: str,
    domain: str,
    threshold: float,
) -> Path:
    """Create the primary corrected output as a searchable, selectable-text PDF."""
    target.parent.mkdir(parents=True, exist_ok=True)
    font, bold_font = _font_names()
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "OCRTitle", parent=styles["Title"], fontName=bold_font, fontSize=18,
        leading=22, textColor=colors.HexColor("#123D38"), alignment=TA_CENTER,
        spaceAfter=5 * mm,
    )
    meta_style = ParagraphStyle(
        "OCRMeta", parent=styles["Normal"], fontName=font, fontSize=8.5,
        leading=12, textColor=colors.HexColor("#52645F"), alignment=TA_CENTER,
        spaceAfter=7 * mm,
    )
    body_style = ParagraphStyle(
        "OCRBody", parent=styles["BodyText"], fontName=font, fontSize=10.5,
        leading=16, textColor=colors.HexColor("#17201E"), spaceAfter=3 * mm,
    )
    page_style = ParagraphStyle(
        "OCRPage", parent=styles["Heading2"], fontName=bold_font, fontSize=11,
        leading=14, textColor=colors.HexColor("#27675D"), spaceAfter=4 * mm,
    )

    accepted = {
        record["token_id"]: record["final"]
        for record in correction_records
        if record.get("changed") and record.get("status") == "ACCEPTED"
    }
    pages = sorted({int(token.get("page", 1)) for token in tokens}) or [1]
    story: list[Any] = []
    for page_position, page_number in enumerate(pages):
        page_tokens = [token for token in tokens if int(token.get("page", 1)) == page_number]
        if page_position:
            story.append(PageBreak())
        story.append(Paragraph("Corrected OCR Document", title_style))
        story.append(Paragraph(
            f"Source: {escape(source_name)} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Domain: {escape(domain.title())} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Confidence gate: {threshold:.2f}", meta_style,
        ))
        story.append(Paragraph(f"Source page {page_number}", page_style))

        line_groups: dict[int, list[dict[str, Any]]] = {}
        for token_index, token in enumerate(page_tokens):
            line_number = int(token.get("line_index", token_index // 18))
            line_groups.setdefault(line_number, []).append(token)
        for line_tokens in line_groups.values():
            parts = []
            for token in line_tokens:
                original = str(token.get("text", ""))
                final = str(accepted.get(token.get("id"), original))
                rendered = escape(final)
                if token.get("id") in accepted:
                    rendered = f'<font backColor="#FFF1B8"><b>{rendered}</b></font>'
                parts.append(rendered)
            if parts:
                story.append(Paragraph(" ".join(parts), body_style))
        if not page_tokens:
            story.append(Paragraph("No text was extracted from this page.", body_style))

    def decorate(canvas: Any, document: Any) -> None:
        canvas.saveState()
        canvas.setTitle(f"Corrected OCR - {source_name}")
        canvas.setAuthor("ContextOCR GraphRAG Workbench")
        canvas.setSubject("Searchable corrected OCR output")
        canvas.setKeywords("OCR, corrected, searchable, GraphRAG")
        canvas.setFont(font, 7.5)
        canvas.setFillColor(colors.HexColor("#60716D"))
        canvas.drawString(18 * mm, 10 * mm, f"Run {run_id[:8]} - selectable text PDF")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {document.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(target), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=17 * mm,
        title=f"Corrected OCR - {source_name}",
        author="ContextOCR GraphRAG Workbench",
    )
    document.build(story, onFirstPage=decorate, onLaterPages=decorate)
    return target
