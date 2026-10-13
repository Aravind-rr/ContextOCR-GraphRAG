from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from reportlab.lib.colors import Color, HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "dataset"
IMAGE_DIR = DATASET_ROOT / "inputs" / "images"
PDF_DIR = DATASET_ROOT / "inputs" / "pdfs"
GT_IMAGE_DIR = DATASET_ROOT / "ground_truth" / "images"
GT_PDF_DIR = DATASET_ROOT / "ground_truth" / "pdfs"
MANIFEST_DIR = DATASET_ROOT / "manifests"
ACTUAL_RUNS_DIR = ROOT / "actual_runs"

CATEGORY_COLORS = {
    "medical": "#16697A",
    "invoice_receipt": "#7A4E2D",
    "certificate": "#674188",
    "application_form": "#2D6A4F",
    "financial_summary": "#1D4E89",
}


@dataclass(frozen=True)
class DocumentSpec:
    document_id: str
    input_filename: str
    input_format: str
    category: str
    title: str
    subtitle: str
    reference: str
    date: str
    fields: tuple[tuple[str, str], ...]
    table_headers: tuple[str, ...]
    table_rows: tuple[tuple[str, ...], ...]
    notes: tuple[str, ...]
    corruptions: tuple[tuple[str, str, str], ...]
    quality_variations: tuple[str, ...]
    dataset_split: str

    @property
    def has_corruption(self) -> bool:
        return bool(self.corruptions)

    @property
    def ground_truth_lines(self) -> list[str]:
        lines = [
            "SYNTHETIC OCR EVALUATION DOCUMENT",
            self.title,
            self.subtitle,
            f"Reference: {self.reference}",
            f"Date: {self.date}",
        ]
        lines.extend(f"{label}: {value}" for label, value in self.fields)
        lines.append(" | ".join(self.table_headers))
        lines.extend(" | ".join(row) for row in self.table_rows)
        lines.append("NOTES")
        lines.extend(self.notes)
        lines.append("Synthetic evaluation data - not a real record")
        return lines

    @property
    def rendered_lines(self) -> list[str]:
        lines = self.ground_truth_lines
        for correct, corrupted, _kind in self.corruptions:
            matched = False
            for index, line in enumerate(lines):
                if correct in line:
                    lines[index] = line.replace(correct, corrupted, 1)
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Corruption source {correct!r} is absent from {self.document_id}")
        return lines


def _specs_for(prefix: str, fmt: str, extension: str) -> list[DocumentSpec]:
    categories = (
        "medical", "medical", "medical",
        "invoice_receipt", "invoice_receipt", "invoice_receipt",
        "certificate", "certificate", "certificate",
        "application_form", "application_form", "application_form",
        "financial_summary", "financial_summary", "financial_summary",
    )
    titles = (
        "Clinical Follow-up Note", "Outpatient Review Summary", "Medication Review Record",
        "Tax Invoice", "Service Receipt", "Purchase Invoice",
        "Course Completion Certificate", "Workshop Participation Certificate", "Merit Certificate",
        "Library Membership Application", "Research Access Application", "Student Activity Application",
        "Quarterly Financial Summary", "Department Budget Snapshot", "Project Expense Summary",
    )
    subtitles = (
        "General Medicine Unit", "Community Health Clinic", "Medication Safety Desk",
        "Northwind Office Supplies", "Blue River Technical Services", "Orchid Stationery Cooperative",
        "Centre for Applied Computing", "Digital Systems Workshop", "Academic Excellence Programme",
        "Central Learning Resource Centre", "Institutional Research Facility", "Student Development Office",
        "Aurora Community Services", "School of Computing", "ContextOCR Prototype Programme",
    )
    references = (
        "MED-2026-101", "MED-2026-114", "MED-2026-127",
        "INV-2026-410", "RCT-2026-225", "INV-2026-438",
        "CERT-2026-071", "CERT-2026-084", "CERT-2026-096",
        "APP-2026-301", "APP-2026-318", "APP-2026-329",
        "FIN-2026-Q1", "BUD-2026-SEM1", "EXP-2026-044",
    )
    dates = (
        "08 September 2026", "11 September 2026", "14 September 2026",
        "03 September 2026", "06 September 2026", "12 September 2026",
        "02 September 2026", "07 September 2026", "15 September 2026",
        "01 September 2026", "09 September 2026", "13 September 2026",
        "31 August 2026", "05 September 2026", "10 September 2026",
    )
    clean_indices = {1, 4, 7, 10, 13}
    error_profiles = {
        2: (("Patient", "Patlent", "l/1 character substitution"), ("recovery", "recovry", "missing character")),
        3: (("Patient", "Patlent", "l/1 character substitution"), ("dosage", "dossage", "duplicated character")),
        5: (("TOTAL", "T0TAL", "O/0 character substitution"), ("receipt", "reciept", "misspelling")),
        6: (("Stationery", "Statlonery", "i/l character substitution"), ("Quantity", "Quantty", "missing character")),
        8: (("Participation", "Particlpation", "i/l character substitution"), ("Certificate", "Certifficate", "duplicated character")),
        9: (("Excellence", "Excelence", "missing character"), ("Programme", "Programe", "missing character")),
        11: (("Research", "Researeh", "c/e character substitution"), ("approval", "approva1", "l/1 character substitution")),
        12: (("Student", "Studdent", "duplicated character"), ("activity", "activlty", "i/l character substitution")),
        14: (("Balance", "Ba1ance", "l/1 character substitution"), ("Allocated", "Alocated", "missing character")),
        15: (("Expenses", "Expensess", "duplicated character"), ("variance", "varlance", "i/l character substitution")),
    }
    image_quality = {
        1: (), 2: ("mild_noise",), 3: ("slight_rotation_0.8_deg",), 4: (),
        5: ("low_contrast",), 6: ("mild_blur",), 7: (), 8: ("mild_noise",),
        9: ("slight_rotation_-0.7_deg",), 10: (), 11: ("low_contrast",),
        12: ("mild_noise", "mild_blur"), 13: (), 14: ("slight_rotation_0.6_deg",),
        15: ("low_contrast", "mild_noise"),
    }
    pdf_quality = {
        1: (), 2: ("low_contrast_vector_text",), 3: ("slight_rotation_0.6_deg",), 4: (),
        5: ("varied_font_sizes",), 6: ("tight_line_spacing",), 7: (), 8: ("low_contrast_vector_text",),
        9: ("slight_rotation_-0.5_deg",), 10: (), 11: ("tight_line_spacing",),
        12: ("varied_font_sizes",), 13: (), 14: ("slight_rotation_0.5_deg",),
        15: ("low_contrast_vector_text", "varied_font_sizes"),
    }

    category_fields = {
        "medical": (
            (("Patient", "Asha Testwell"), ("Record Type", "Synthetic outpatient record"), ("Visit", "Routine review")),
            (("Patient", "Rohan Sample"), ("Record Type", "Synthetic clinical note"), ("Visit", "Follow-up consultation")),
            (("Patient", "Mira Example"), ("Record Type", "Synthetic medication record"), ("Visit", "Dosage reconciliation")),
        ),
        "invoice_receipt": (
            (("Customer", "Demo Learning Lab"), ("Payment", "Bank transfer"), ("Currency", "INR")),
            (("Customer", "Sample Innovation Cell"), ("Payment", "Card - test transaction"), ("Currency", "INR")),
            (("Customer", "Prototype Design Studio"), ("Payment", "Purchase order"), ("Currency", "INR")),
        ),
        "certificate": (
            (("Recipient", "Ananya Demo"), ("Programme", "Foundations of Data Systems"), ("Result", "Completed")),
            (("Recipient", "Vikram Sample"), ("Programme", "Responsible AI Workshop"), ("Result", "Participated")),
            (("Recipient", "Nila Example"), ("Programme", "Computing Merit Programme"), ("Result", "Distinction")),
        ),
        "application_form": (
            (("Applicant", "Kabir Test"), ("Applicant ID", "SYN-26031"), ("Request", "Annual membership")),
            (("Applicant", "Sara Sample"), ("Applicant ID", "SYN-26044"), ("Request", "Dataset laboratory access")),
            (("Applicant", "Dev Example"), ("Applicant ID", "SYN-26058"), ("Request", "Technical club registration")),
        ),
        "financial_summary": (
            (("Entity", "Aurora Community Services"), ("Period", "April-June 2026"), ("Currency", "INR")),
            (("Entity", "School of Computing"), ("Period", "Semester 1, 2026"), ("Currency", "INR")),
            (("Entity", "ContextOCR Prototype Programme"), ("Period", "August 2026"), ("Currency", "INR")),
        ),
    }
    category_tables = {
        "medical": (
            (("Observation", "Reading", "Status"), (("Blood pressure", "118/76 mmHg", "Normal"), ("Pulse", "72 bpm", "Stable"), ("Oxygen saturation", "98 percent", "Normal"))),
            (("Assessment", "Finding", "Action"), (("Temperature", "36.8 C", "Observe"), ("Symptoms", "Improving", "Continue plan"), ("Review", "14 days", "Scheduled"))),
            (("Medicine", "Dosage", "Instruction"), (("Sample tablet A", "5 mg", "After breakfast"), ("Sample capsule B", "10 mg", "At night"), ("Review", "7 days", "Bring prescription"))),
        ),
        "invoice_receipt": (
            (("Item", "Quantity", "Amount"), (("Archive folders", "12", "1,440.00"), ("Printer labels", "6", "720.00"), ("Desk supplies", "4", "560.00"), ("TOTAL", "22", "2,720.00"))),
            (("Service", "Hours", "Amount"), (("Device inspection", "2", "1,200.00"), ("Configuration support", "3", "2,250.00"), ("Documentation", "1", "600.00"), ("TOTAL", "6", "4,050.00"))),
            (("Item", "Quantity", "Amount"), (("Notebook set", "10", "950.00"), ("Marker pack", "5", "625.00"), ("Stationery tray", "2", "840.00"), ("TOTAL", "17", "2,415.00"))),
        ),
        "certificate": (
            (("Module", "Hours", "Grade"), (("Data modelling", "8", "A"), ("Query design", "6", "A"), ("Assessment", "2", "Passed"))),
            (("Session", "Hours", "Status"), (("AI foundations", "3", "Attended"), ("Ethics laboratory", "2", "Attended"), ("Participation", "5", "Verified"))),
            (("Area", "Score", "Outcome"), (("Problem solving", "92", "Excellent"), ("Programming", "88", "Very good"), ("Overall", "90", "Distinction"))),
        ),
        "application_form": (
            (("Selection", "Choice", "Confirmation"), (("Membership term", "12 months", "Selected"), ("Email notices", "Yes", "Selected"), ("Applicant declaration", "Accepted", "Signed digitally"))),
            (("Resource", "Access level", "Duration"), (("Research dataset", "Read only", "90 days"), ("Secure workspace", "Supervised", "30 hours"), ("Approval", "Pending review", "Not granted"))),
            (("Activity", "Preference", "Availability"), (("Coding club", "Primary", "Friday"), ("Robotics group", "Secondary", "Saturday"), ("Student declaration", "Accepted", "Confirmed"))),
        ),
        "financial_summary": (
            (("Account", "Budget", "Actual"), (("Programme income", "450,000", "462,500"), ("Operating cost", "280,000", "271,400"), ("Balance", "170,000", "191,100"))),
            (("Cost centre", "Allocated", "Used"), (("Laboratory", "320,000", "298,750"), ("Software", "180,000", "171,200"), ("Contingency", "50,000", "12,600"))),
            (("Category", "Plan", "Expenses"), (("Model testing", "95,000", "88,400"), ("Documentation", "42,000", "39,750"), ("Infrastructure", "130,000", "125,900"), ("Net variance", "0", "12,950"))),
        ),
    }
    category_notes = {
        "medical": (
            ("Patient is stable and may continue the current care plan.", "A printed prescription was received for the next review."),
            ("Clinical findings are consistent with routine recovery.", "Return earlier if symptoms change."),
            ("The dosage list was reviewed for clarity and timing.", "This record contains no real medical information."),
        ),
        "invoice_receipt": (
            ("Goods were received in complete condition.", "Payment is due within fourteen days."),
            ("Service completion was confirmed by the demonstration contact.", "This receipt records a test transaction only."),
            ("Quantity and unit prices were checked before issue.", "No real customer account is represented."),
        ),
        "certificate": (
            ("This certificate recognises successful course completion.", "Verification code is valid only inside this synthetic dataset."),
            ("Participation requirements were completed in full.", "The named recipient is fictional."),
            ("Academic excellence was demonstrated across all listed areas.", "This document has no institutional validity."),
        ),
        "application_form": (
            ("The applicant confirms that the submitted details are accurate.", "Membership activation requires manual review."),
            ("Research approval is required before any dataset is accessed.", "No access has been granted by this form."),
            ("Student activity preferences may be updated after registration.", "The applicant and identifier are synthetic."),
        ),
        "financial_summary": (
            ("Revenue remained above plan while operating cost stayed controlled.", "Figures are illustrative and unaudited."),
            ("Allocated funds cover the current semester requirements.", "Balance values are synthetic examples."),
            ("Expenses remained within the approved prototype budget.", "Variance will be reviewed in the next planning cycle."),
        ),
    }

    category_seen: dict[str, int] = {key: 0 for key in CATEGORY_COLORS}
    specs: list[DocumentSpec] = []
    for index, category in enumerate(categories, start=1):
        variant = category_seen[category]
        category_seen[category] += 1
        document_id = f"{prefix}{index:03d}"
        slug = titles[index - 1].lower().replace(" ", "_").replace("-", "_")
        qualities = image_quality[index] if fmt == "image" else pdf_quality[index]
        specs.append(DocumentSpec(
            document_id=document_id,
            input_filename=f"{document_id}_{slug}.{extension}",
            input_format=fmt,
            category=category,
            title=titles[index - 1],
            subtitle=subtitles[index - 1],
            reference=references[index - 1],
            date=dates[index - 1],
            fields=category_fields[category][variant],
            table_headers=category_tables[category][variant][0],
            table_rows=category_tables[category][variant][1],
            notes=category_notes[category][variant],
            corruptions=() if index in clean_indices else error_profiles[index],
            quality_variations=qualities,
            dataset_split="manual_execution" if index <= 5 else "reserved_unprocessed",
        ))
    return specs


def build_specs() -> list[DocumentSpec]:
    return _specs_for("IMG", "image", "png") + _specs_for("PDF", "pdf", "pdf")


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def render_image(spec: DocumentSpec, output: Path) -> None:
    width, height = 1000, 1414
    image = Image.new("RGB", (width, height), "#FAFBFC")
    draw = ImageDraw.Draw(image)
    color = CATEGORY_COLORS[spec.category]
    dark = "#16232A"
    muted = "#52646D"
    body_font = _font(25)
    small_font = _font(20)
    label_font = _font(19, bold=True)
    title_font = _font(39, bold=True)
    subtitle_font = _font(24)
    table_font = _font(18)
    lines = spec.rendered_lines
    rendered_title = lines[1]
    rendered_subtitle = lines[2]
    rendered_reference = lines[3].split(": ", 1)[1]
    rendered_date = lines[4].split(": ", 1)[1]
    rendered_fields = [tuple(line.split(": ", 1)) for line in lines[5:5 + len(spec.fields)]]

    angle = 0.0
    if "slight_rotation_0.8_deg" in spec.quality_variations:
        angle = 0.8
    elif "slight_rotation_-0.7_deg" in spec.quality_variations:
        angle = -0.7
    elif "slight_rotation_0.6_deg" in spec.quality_variations:
        angle = 0.6

    draw.rectangle((0, 0, width, 118), fill=color)
    draw.text((62, 34), "SYNTHETIC OCR EVALUATION", font=_font(24, True), fill="white")
    draw.text((width - 190, 39), spec.document_id, font=_font(22, True), fill="white")
    y = 160
    draw.text((62, y), rendered_title, font=title_font, fill=dark)
    y += 58
    draw.text((62, y), rendered_subtitle, font=subtitle_font, fill=color)
    y += 62
    draw.rounded_rectangle((55, y, width - 55, y + 104), radius=12, fill="#EEF3F4", outline="#D5DFE2", width=2)
    draw.text((76, y + 18), f"Reference: {rendered_reference}", font=body_font, fill=dark)
    draw.text((565, y + 18), f"Date: {rendered_date}", font=small_font, fill=muted)
    y += 136

    field_width = (width - 130) // len(rendered_fields)
    for idx, (label, value) in enumerate(rendered_fields):
        x1 = 55 + idx * field_width
        x2 = 55 + (idx + 1) * field_width - 10
        draw.rounded_rectangle((x1, y, x2, y + 112), radius=10, fill="white", outline="#D9E1E4", width=2)
        draw.text((x1 + 16, y + 14), label.upper(), font=label_font, fill=color)
        value_lines = _wrap(draw, value, small_font, x2 - x1 - 32)
        for line_index, line in enumerate(value_lines[:2]):
            draw.text((x1 + 16, y + 48 + line_index * 25), line, font=small_font, fill=dark)
    y += 150

    header_index = 5 + len(spec.fields)
    rendered_rows = lines[header_index + 1:header_index + 1 + len(spec.table_rows)]
    table_headers = lines[header_index].split(" | ")
    table_rows = [row.split(" | ") for row in rendered_rows]
    x0, x3 = 55, width - 55
    table_width = x3 - x0
    cols = len(table_headers)
    col_width = table_width / cols
    row_height = 58
    draw.rounded_rectangle((x0, y, x3, y + row_height * (len(table_rows) + 1)), radius=8, fill="white", outline="#CBD6DA", width=2)
    draw.rectangle((x0, y, x3, y + row_height), fill=color)
    for col, header in enumerate(table_headers):
        draw.text((int(x0 + col * col_width + 14), y + 17), header, font=label_font, fill="white")
    for row_index, row in enumerate(table_rows, start=1):
        row_y = y + row_index * row_height
        if row_index % 2 == 0:
            draw.rectangle((x0 + 1, row_y, x3 - 1, row_y + row_height), fill="#F3F6F7")
        draw.line((x0, row_y, x3, row_y), fill="#D8E1E4", width=1)
        for col, value in enumerate(row):
            draw.text((int(x0 + col * col_width + 14), row_y + 16), value, font=table_font, fill=dark)
    for col in range(1, cols):
        x = int(x0 + col * col_width)
        draw.line((x, y, x, y + row_height * (len(table_rows) + 1)), fill="#D8E1E4", width=1)
    y += row_height * (len(table_rows) + 1) + 54

    draw.text((62, y), "NOTES", font=_font(22, True), fill=color)
    y += 40
    note_lines = lines[header_index + 2 + len(spec.table_rows):-1]
    for note in note_lines:
        for wrapped in _wrap(draw, note, body_font, width - 145):
            draw.ellipse((65, y + 10, 73, y + 18), fill=color)
            draw.text((90, y), wrapped, font=body_font, fill=dark)
            y += 38
        y += 8

    draw.line((55, height - 108, width - 55, height - 108), fill="#D5DFE2", width=2)
    draw.text((62, height - 80), "Synthetic evaluation data - not a real record", font=small_font, fill=muted)
    draw.text((width - 175, height - 80), "Page 1 of 1", font=small_font, fill=muted)

    if "low_contrast" in spec.quality_variations:
        image = ImageEnhance.Contrast(image).enhance(0.58)
    if "mild_blur" in spec.quality_variations:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.45))
    if "mild_noise" in spec.quality_variations:
        rng = random.Random(f"contextocr-{spec.document_id}")
        pixels = image.load()
        for _ in range(16_000):
            x = rng.randrange(width)
            y_noise = rng.randrange(height)
            r, g, b = pixels[x, y_noise]
            delta = rng.choice((-18, -12, 12, 18))
            pixels[x, y_noise] = tuple(max(0, min(255, channel + delta)) for channel in (r, g, b))
    if angle:
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="white")
    image.save(output, format="PNG", optimize=True)


def _pdf_wrap(text: str, font_name: str, font_size: float, max_width: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if stringWidth(candidate, font_name, font_size) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def render_pdf(spec: DocumentSpec, output: Path) -> None:
    page_width, page_height = A4
    c = canvas.Canvas(str(output), pagesize=A4, pageCompression=1, invariant=1)
    c.setTitle(spec.title)
    c.setAuthor("ContextOCR synthetic evaluation dataset")
    color = HexColor(CATEGORY_COLORS[spec.category])
    dark = HexColor("#16232A")
    muted = HexColor("#52646D")
    low_contrast = "low_contrast_vector_text" in spec.quality_variations
    if low_contrast:
        dark = HexColor("#66747A")
        muted = HexColor("#849197")
        color = Color(color.red, color.green, color.blue, alpha=0.68)
    lines = spec.rendered_lines
    rendered_title = lines[1]
    rendered_subtitle = lines[2]
    rendered_reference = lines[3].split(": ", 1)[1]
    rendered_date = lines[4].split(": ", 1)[1]
    rendered_fields = [tuple(line.split(": ", 1)) for line in lines[5:5 + len(spec.fields)]]

    angle = 0.0
    for variation in spec.quality_variations:
        if variation.startswith("slight_rotation_"):
            angle = float(variation.split("_")[2].replace("deg", ""))
    if angle:
        c.saveState()
        c.translate(page_width / 2, page_height / 2)
        c.rotate(angle)
        c.translate(-page_width / 2, -page_height / 2)

    c.setFillColor(color)
    c.rect(0, page_height - 72, page_width, 72, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(42, page_height - 43, "SYNTHETIC OCR EVALUATION")
    c.drawRightString(page_width - 42, page_height - 43, spec.document_id)

    y = page_height - 108
    title_size = 23 if "varied_font_sizes" not in spec.quality_variations else 26
    c.setFillColor(dark)
    c.setFont("Helvetica-Bold", title_size)
    c.drawString(42, y, rendered_title)
    y -= 25
    c.setFillColor(color)
    c.setFont("Helvetica", 12)
    c.drawString(42, y, rendered_subtitle)
    y -= 44

    c.setFillColor(HexColor("#EEF3F4"))
    c.roundRect(42, y - 48, page_width - 84, 58, 7, fill=1, stroke=0)
    c.setFillColor(dark)
    c.setFont("Helvetica", 11)
    c.drawString(56, y - 10, f"Reference: {rendered_reference}")
    c.drawRightString(page_width - 56, y - 10, f"Date: {rendered_date}")
    y -= 74

    card_gap = 8
    card_width = (page_width - 84 - card_gap * 2) / 3
    for idx, (label, value) in enumerate(rendered_fields):
        x = 42 + idx * (card_width + card_gap)
        c.setFillColor(white)
        c.setStrokeColor(HexColor("#D9E1E4"))
        c.roundRect(x, y - 58, card_width, 64, 6, fill=1, stroke=1)
        c.setFillColor(color)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(x + 10, y - 14, label.upper())
        c.setFillColor(dark)
        c.setFont("Helvetica", 9.5)
        for line_index, line in enumerate(_pdf_wrap(value, "Helvetica", 9.5, card_width - 20)[:2]):
            c.drawString(x + 10, y - 32 - line_index * 12, line)
    y -= 88

    header_index = 5 + len(spec.fields)
    headers = lines[header_index].split(" | ")
    rows = [line.split(" | ") for line in lines[header_index + 1:header_index + 1 + len(spec.table_rows)]]
    x0 = 42
    table_width = page_width - 84
    col_width = table_width / len(headers)
    row_height = 31 if "tight_line_spacing" not in spec.quality_variations else 27
    c.setFillColor(color)
    c.rect(x0, y - row_height, table_width, row_height, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 9)
    for col, header in enumerate(headers):
        c.drawString(x0 + col * col_width + 8, y - 20, header)
    for row_index, row in enumerate(rows, start=1):
        row_top = y - row_height * row_index
        c.setFillColor(HexColor("#F3F6F7") if row_index % 2 == 0 else white)
        c.rect(x0, row_top - row_height, table_width, row_height, fill=1, stroke=0)
        c.setFillColor(dark)
        c.setFont("Helvetica", 8.8)
        for col, value in enumerate(row):
            c.drawString(x0 + col * col_width + 8, row_top - 20, value)
    c.setStrokeColor(HexColor("#CBD6DA"))
    c.rect(x0, y - row_height * (len(rows) + 1), table_width, row_height * (len(rows) + 1), fill=0, stroke=1)
    for col in range(1, len(headers)):
        x = x0 + col * col_width
        c.line(x, y, x, y - row_height * (len(rows) + 1))
    for row_index in range(1, len(rows) + 1):
        row_y = y - row_height * row_index
        c.line(x0, row_y, x0 + table_width, row_y)
    y -= row_height * (len(rows) + 1) + 38

    c.setFillColor(color)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(42, y, "NOTES")
    y -= 20
    note_size = 10.5 if "varied_font_sizes" not in spec.quality_variations else 9.5
    c.setFillColor(dark)
    c.setFont("Helvetica", note_size)
    note_lines = lines[header_index + 2 + len(spec.table_rows):-1]
    for note in note_lines:
        wrapped = _pdf_wrap(note, "Helvetica", note_size, page_width - 108)
        for line in wrapped:
            c.setFillColor(color)
            c.circle(48, y + 3, 2, fill=1, stroke=0)
            c.setFillColor(dark)
            c.drawString(60, y, line)
            y -= 15
        y -= 5

    c.setStrokeColor(HexColor("#D5DFE2"))
    c.line(42, 67, page_width - 42, 67)
    c.setFillColor(muted)
    c.setFont("Helvetica", 8.5)
    c.drawString(42, 48, "Synthetic evaluation data - not a real record")
    c.drawRightString(page_width - 42, 48, "Page 1 of 1")
    if angle:
        c.restoreState()
    c.showPage()
    c.save()


def _manifest_entry(spec: DocumentSpec) -> dict[str, object]:
    input_base = "images" if spec.input_format == "image" else "pdfs"
    gt_base = input_base
    return {
        "document_id": spec.document_id,
        "input_filename": spec.input_filename,
        "input_format": spec.input_format,
        "document_category": spec.category,
        "input_path": f"evaluation/dataset/inputs/{input_base}/{spec.input_filename}",
        "ground_truth_reference_path": f"evaluation/dataset/ground_truth/{gt_base}/{spec.document_id}.txt",
        "intentional_textual_corruption": spec.has_corruption,
        "injected_error_types": [item[2] for item in spec.corruptions],
        "injected_error_details": [
            {"intended": correct, "rendered": corrupted, "type": kind}
            for correct, corrupted, kind in spec.corruptions
        ],
        "image_quality_variations": list(spec.quality_variations) or ["none"],
        "dataset_split": spec.dataset_split,
        "execution_status": "NOT_RUN",
    }


def _prepare_directories() -> None:
    for directory in (IMAGE_DIR, PDF_DIR, GT_IMAGE_DIR, GT_PDF_DIR, MANIFEST_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    for document_id in [f"IMG{i:03d}" for i in range(1, 6)] + [f"PDF{i:03d}" for i in range(1, 6)]:
        run_dir = ACTUAL_RUNS_DIR / document_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / ".gitkeep").touch()


def generate() -> list[dict[str, object]]:
    _prepare_directories()
    specs = build_specs()
    manifest: list[dict[str, object]] = []
    for spec in specs:
        output_dir = IMAGE_DIR if spec.input_format == "image" else PDF_DIR
        gt_dir = GT_IMAGE_DIR if spec.input_format == "image" else GT_PDF_DIR
        output = output_dir / spec.input_filename
        if spec.input_format == "image":
            render_image(spec, output)
        else:
            render_pdf(spec, output)
        (gt_dir / f"{spec.document_id}.txt").write_text(
            "\n".join(spec.ground_truth_lines) + "\n", encoding="utf-8"
        )
        manifest.append(_manifest_entry(spec))

    json_path = MANIFEST_DIR / "dataset_manifest.json"
    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = MANIFEST_DIR / "dataset_manifest.csv"
    fieldnames = [
        "document_id", "input_filename", "input_format", "document_category", "input_path",
        "ground_truth_reference_path", "intentional_textual_corruption", "injected_error_types",
        "injected_error_details", "image_quality_variations", "dataset_split", "execution_status",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in manifest:
            row = dict(entry)
            for key in ("injected_error_types", "injected_error_details", "image_quality_variations"):
                row[key] = json.dumps(row[key], ensure_ascii=False)
            writer.writerow(row)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the deterministic ContextOCR evaluation dataset.")
    parser.parse_args()
    manifest = generate()
    images = sum(entry["input_format"] == "image" for entry in manifest)
    pdfs = sum(entry["input_format"] == "pdf" for entry in manifest)
    clean = sum(not entry["intentional_textual_corruption"] for entry in manifest)
    challenging = len(manifest) - clean
    print(f"Generated {images} images and {pdfs} single-page PDFs.")
    print(f"Composition: {clean} clean, {challenging} intentionally challenging; all NOT_RUN.")


if __name__ == "__main__":
    main()
