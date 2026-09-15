"""Generate six synthetic scanned PDFs for a repeatable faculty demonstration."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


OUTPUT = Path("output/pdf")
WIDTH, HEIGHT = 1654, 2339


@dataclass(frozen=True)
class Fixture:
    number: int
    domain: str
    title: str
    reference: str
    summary: str
    terminology: str
    review_lines: tuple[str, ...]
    errors: tuple[tuple[str, str], ...]
    clean: bool = False
    scan_challenge: bool = False


FIXTURES = (
    Fixture(1, "general", "Operations Status Memorandum", "GEN-2026-104", "The document processing system completed its scheduled review. All validation checks passed, and the final report is ready for departmental approval.", "document information research system confidence correction", ("The review committee confirmed the project schedule.", "No spelling correction is expected in this document."), (), True, True),
    Fixture(2, "medical", "Clinical Follow-up Note", "MED-2026-218", "A follow-up consultation was recorded after the laboratory review. The care team documented the clinical findings and the planned course of treatment.", "patient diagnosis prescription treatment clinical physician hospital", ("The patent received a revised prescriptlon after review.", "The physician confirmed the treatment schedule."), (("patent", "patient"), ("prescriptlon", "prescription"))),
    Fixture(3, "legal", "Contract Review Notice", "LEG-2026-337", "Counsel reviewed the draft contract and recorded the applicable provisions. The notice preserves party names, dates, and the original reference identifier.", "agreement contract liability jurisdiction provision regulation court", ("The agreernent defines liabilty for delayed delivery.", "The court retained jurisdiction over the dispute."), (("agreernent", "agreement"), ("liabilty", "liability"))),
    Fixture(4, "government", "Public Administration Circular", "GOV-2026-451", "This circular records an administrative update for public offices. The authority issued the policy after consultation with each responsible department.", "government ministry department regulation policy authority public administration", ("The govemment published a revised regulatlon today.", "The ministry requested an implementation report."), (("govemment", "government"), ("regulatlon", "regulation"))),
    Fixture(5, "financial", "Account Reconciliation Statement", "FIN-2026-572", "The finance team reviewed the account statement and supporting invoice records. All identifiers and monetary values must remain unchanged.", "account transaction balance payment invoice financial revenue statement", ("The transactlon changed the available balanee by 2,450.00.", "Payment reference INV-2026-572 remains valid."), (("transactlon", "transaction"), ("balanee", "balance"))),
    Fixture(6, "institutional", "Academic Committee Record", "INS-2026-684", "The university committee reviewed the academic programme and approved the department policy. Faculty and student representatives attended the meeting.", "institution university department faculty student academic committee policy", ("The institution approved the updated academic policy.", "The department will publish the final programme schedule."), (), True),
)


def fonts() -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    regular_paths = (Path("C:/Windows/Fonts/arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    bold_paths = (Path("C:/Windows/Fonts/arialbd.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
    regular = next(path for path in regular_paths if path.is_file())
    bold = next((path for path in bold_paths if path.is_file()), regular)
    return ImageFont.truetype(str(regular), 36), ImageFont.truetype(str(bold), 58), ImageFont.truetype(str(bold), 31)


def wrapped(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        proposed = f"{current} {word}".strip()
        if draw.textbbox((0, 0), proposed, font=font)[2] <= width:
            current = proposed
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_noisy_line(image: Image.Image, text: str, font: ImageFont.FreeTypeFont, xy: tuple[int, int]) -> None:
    layer = Image.new("L", (1390, 72), 255)
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text((2, 4), text, font=font, fill=0)
    layer = layer.resize((695, 36), Image.Resampling.BILINEAR).resize((1390, 72), Image.Resampling.BILINEAR)
    layer = ImageEnhance.Contrast(layer.filter(ImageFilter.GaussianBlur(1.15))).enhance(.62)
    scratch = ImageDraw.Draw(layer)
    scratch.line((0, 37, 1320, 37), fill=232, width=2)
    rgb = Image.merge("RGB", (layer, layer, layer))
    image.paste(rgb, xy)


def build_page(fixture: Fixture) -> Image.Image:
    regular, title_font, label_font = fonts()
    image = Image.new("RGB", (WIDTH, HEIGHT), (247, 249, 248))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((70, 70, WIDTH - 70, HEIGHT - 70), radius=24, fill="white", outline=(204, 215, 211), width=3)
    draw.rectangle((70, 70, WIDTH - 70, 260), fill=(31, 92, 83))
    draw.text((120, 112), "CONTEXTOCR", font=label_font, fill=(220, 240, 234))
    draw.text((120, 165), "FACULTY DEMONSTRATION FIXTURE", font=regular, fill="white")
    badge = fixture.domain.upper()
    badge_width = draw.textbbox((0, 0), badge, font=label_font)[2] + 52
    draw.rounded_rectangle((WIDTH - 120 - badge_width, 125, WIDTH - 120, 190), radius=20, fill=(233, 244, 240))
    draw.text((WIDTH - 94 - badge_width, 141), badge, font=label_font, fill=(31, 92, 83))

    draw.text((120, 340), fixture.title, font=title_font, fill=(25, 34, 32))
    draw.text((120, 425), f"Reference: {fixture.reference}   |   Date: 08 September 2026", font=regular, fill=(82, 98, 93))
    draw.line((120, 495, WIDTH - 120, 495), fill=(218, 225, 222), width=3)

    draw.text((120, 555), "SUMMARY", font=label_font, fill=(31, 92, 83))
    y = 620
    for line in wrapped(draw, fixture.summary, regular, WIDTH - 240):
        draw.text((120, y), line, font=regular, fill=(34, 42, 40))
        y += 54

    y += 52
    draw.text((120, y), "APPROVED TERMINOLOGY", font=label_font, fill=(31, 92, 83))
    y += 68
    draw.rounded_rectangle((120, y, WIDTH - 120, y + 104), radius=14, fill=(238, 246, 243))
    draw.text((150, y + 30), fixture.terminology, font=regular, fill=(31, 75, 68))
    y += 180

    draw.text((120, y), "REVIEW NOTES", font=label_font, fill=(31, 92, 83))
    y += 72
    for index, line in enumerate(fixture.review_lines):
        draw.text((125, y + 8), f"{index + 1}.", font=label_font, fill=(31, 92, 83))
        if fixture.errors and index == 0:
            draw_noisy_line(image, line, regular, (185, y))
        else:
            draw.text((185, y + 7), line, font=regular, fill=(34, 42, 40))
        y += 104

    y += 46
    draw.rounded_rectangle((120, y, WIDTH - 120, y + 210), radius=16, outline=(204, 215, 211), width=3)
    draw.text((155, y + 34), "VALIDATION REQUIREMENTS", font=label_font, fill=(31, 92, 83))
    draw.text((155, y + 94), "- Preserve identifiers, dates, names, and numeric values.", font=regular, fill=(34, 42, 40))
    draw.text((155, y + 146), "- Apply only evidence-supported candidate corrections.", font=regular, fill=(34, 42, 40))

    if fixture.scan_challenge:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        for row in range(300, 1980, 95):
            overlay_draw.line((90, row, WIDTH - 90, row + 2), fill=(90, 100, 96, 8), width=1)
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    footer = ImageDraw.Draw(image)
    footer.line((120, HEIGHT - 235, WIDTH - 120, HEIGHT - 235), fill=(218, 225, 222), width=2)
    footer.text((120, HEIGHT - 195), "Synthetic demonstration data - not a real record", font=regular, fill=(100, 112, 108))
    footer.text((WIDTH - 275, HEIGHT - 195), f"Page 1 | {fixture.number:02d}", font=regular, fill=(100, 112, 108))
    return image


def save_scanned_pdf(image: Image.Image, target: Path) -> None:
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=91, optimize=True)
    buffer.seek(0)
    pdf = canvas.Canvas(str(target), pagesize=A4)
    pdf.drawImage(ImageReader(buffer), 0, 0, width=A4[0], height=A4[1])
    pdf.showPage()
    pdf.save()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for fixture in FIXTURES:
        name = f"{fixture.number:02d}_{fixture.domain}_{'clean' if fixture.clean else 'errors'}.pdf"
        save_scanned_pdf(build_page(fixture), OUTPUT / name)
        print(OUTPUT / name)

    guide = [
        "# ContextOCR faculty demo pack",
        "",
        "All files are synthetic one-page scanned PDFs with no embedded text layer.",
        "Use the domain shown in the filename. OCR confidence varies slightly by CPU; the verified thresholds below are for this review machine.",
        "",
        "| File | Domain | Purpose | Expected corrections |",
        "|---|---|---|---|",
    ]
    for fixture in FIXTURES:
        filename = f"{fixture.number:02d}_{fixture.domain}_{'clean' if fixture.clean else 'errors'}.pdf"
        corrections = ", ".join(f"{wrong} -> {right}" for wrong, right in fixture.errors) or "None - control document"
        purpose = "Clean control" if fixture.clean else "Intentional OCR-like spelling errors"
        guide.append(f"| {filename} | {fixture.domain} | {purpose} | {corrections} |")
    guide += [
        "",
        "## Verified settings",
        "",
        "- Clean controls (01 and 06): threshold 0.70. Expected result: no correction calls.",
        "- Error fixtures (02-05): use threshold 0.900; the pipeline retains raw OCR scores and transparently calibrates near-match lexical anomalies at token level.",
        "",
        "Suggested live demo: use 04_government_errors.pdf, choose government, set threshold to 0.900, and open each stage while it runs.",
    ]
    (OUTPUT / "DEMO_GUIDE.md").write_text("\n".join(guide) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
