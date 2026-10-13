from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from calculate_metrics import evaluate_document


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
MANIFEST_PATH = ROOT / "dataset" / "manifests" / "dataset_manifest.json"
RUN_ROOT = ROOT / "illustrative_runs"
PDF_ROOT = PROJECT_ROOT / "output" / "pdf" / "illustrative_evaluation"
RESULT_ROOT = ROOT / "results"
TARGET_IDS = [*(f"IMG{i:03d}" for i in range(1, 6)), *(f"PDF{i:03d}" for i in range(1, 6))]
UNRESOLVED = {("PDF005", "receipt")}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_pdf(path: Path, document_id: str, source_name: str, corrected_text: str, unresolved: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "IllustrativeTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=18, leading=22, alignment=TA_CENTER, textColor=colors.HexColor("#0F5F57"),
        spaceAfter=6,
    )
    badge = ParagraphStyle(
        "Badge", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=9,
        leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#8A3D00"),
        backColor=colors.HexColor("#FFF1D6"), borderColor=colors.HexColor("#E6B85C"),
        borderWidth=0.6, borderPadding=5, spaceAfter=10,
    )
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=10,
        leading=14, textColor=colors.HexColor("#172B28"), spaceAfter=3,
    )
    small = ParagraphStyle(
        "Small", parent=styles["BodyText"], fontName="Helvetica", fontSize=8,
        leading=11, textColor=colors.HexColor("#526662"),
    )
    document = SimpleDocTemplate(
        str(path), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title=f"{document_id} illustrative corrected output",
        author="ContextOCR evaluation fixture",
        subject="Illustrative output - not an actual pipeline result",
    )
    story = [
        Paragraph("ContextOCR - Illustrative Corrected Output", title),
        Paragraph("ILLUSTRATIVE / SYNTHETIC - NOT AN ACTUAL PIPELINE RESULT", badge),
    ]
    metadata = Table(
        [["Document ID", document_id], ["Source fixture", source_name], ["Text layer", "Selectable and searchable"]],
        colWidths=[34 * mm, 130 * mm],
    )
    metadata.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E7F2EF")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#0F5F57")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8CBC7")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([metadata, Spacer(1, 8 * mm), Paragraph("Illustrative output text", styles["Heading2"])])
    for line in corrected_text.splitlines():
        story.append(Paragraph(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") or "&nbsp;", body))
    story.append(Spacer(1, 5 * mm))
    if unresolved:
        detail = ", ".join(f"{item['rendered']} (intended: {item['intended']})" for item in unresolved)
        story.append(Paragraph(f"Intentionally unresolved illustrative error: {detail}", small))
    else:
        story.append(Paragraph("All documented injected errors in this fixture are shown as corrected.", small))
    story.append(Paragraph("This file is for UI/report demonstration only and is excluded from genuine measured-run metrics.", small))
    document.build(story)


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = {entry["document_id"]: entry for entry in manifest}
    generated_at = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, Any]] = []
    total_errors = corrected_errors = 0

    for document_id in TARGET_IDS:
        entry = entries[document_id]
        reference_path = PROJECT_ROOT / entry["ground_truth_reference_path"]
        reference = reference_path.read_text(encoding="utf-8").strip()
        baseline = reference
        traces: list[dict[str, Any]] = []
        for detail in entry["injected_error_details"]:
            baseline = baseline.replace(detail["intended"], detail["rendered"], 1)

        corrected = baseline
        unresolved: list[dict[str, Any]] = []
        for detail in entry["injected_error_details"]:
            total_errors += 1
            leave_unresolved = (document_id, detail["intended"].casefold()) in UNRESOLVED
            if leave_unresolved:
                unresolved.append(detail)
                final = detail["rendered"]
                status = "ILLUSTRATIVE_UNRESOLVED"
            else:
                corrected = corrected.replace(detail["rendered"], detail["intended"], 1)
                corrected_errors += 1
                final = detail["intended"]
                status = "ILLUSTRATIVE_CORRECTED"
            traces.append({
                "original": detail["rendered"], "final": final, "intended": detail["intended"],
                "status": status, "error_type": detail["type"],
                "classification": "ILLUSTRATIVE_NOT_ACTUAL_PIPELINE_EVIDENCE",
            })

        run_dir = RUN_ROOT / document_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "illustrative_ocr_raw.txt").write_text(baseline + "\n", encoding="utf-8")
        (run_dir / "illustrative_corrected.txt").write_text(corrected + "\n", encoding="utf-8")
        write_json(run_dir / "illustrative_trace.json", traces)
        pdf_path = PDF_ROOT / f"{document_id}_illustrative_corrected.pdf"
        render_pdf(pdf_path, document_id, entry["input_filename"], corrected, unresolved)

        metrics = evaluate_document(reference, baseline, corrected)
        record = {
            "document_id": document_id,
            "input_filename": entry["input_filename"],
            "input_format": entry["input_format"],
            "classification": "ILLUSTRATIVE_NOT_ACTUAL_PIPELINE_RESULT",
            "injected_errors": len(entry["injected_error_details"]),
            "illustratively_corrected_errors": sum(t["status"] == "ILLUSTRATIVE_CORRECTED" for t in traces),
            "illustratively_unresolved_errors": len(unresolved),
            "metrics": metrics,
            "pdf_path": str(pdf_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "text_path": str((run_dir / "illustrative_corrected.txt").relative_to(PROJECT_ROOT)).replace("\\", "/"),
        }
        results.append(record)
        write_json(run_dir / "illustrative_metadata.json", {
            "status": "ILLUSTRATIVE_NOT_ACTUAL",
            "generated_at": generated_at,
            **{key: record[key] for key in ("document_id", "input_filename", "injected_errors", "illustratively_corrected_errors", "illustratively_unresolved_errors", "pdf_path", "text_path")},
        })

    correction_accuracy = corrected_errors / total_errors if total_errors else 1.0
    measured = [record["metrics"] for record in results]
    outcome_counts = {name: sum(metric["outcome"] == name for metric in measured) for name in ("IMPROVED", "UNCHANGED", "WORSENED")}
    summary = {
        "classification": "ILLUSTRATIVE_NOT_ACTUAL_PIPELINE_METRICS",
        "documents": len(results),
        "controlled_error_tokens": total_errors,
        "illustratively_corrected_error_tokens": corrected_errors,
        "illustratively_unresolved_error_tokens": total_errors - corrected_errors,
        "illustrative_token_correction_accuracy": correction_accuracy,
        "documents_improved": outcome_counts["IMPROVED"],
        "documents_unchanged": outcome_counts["UNCHANGED"],
        "documents_worsened": outcome_counts["WORSENED"],
        "macro_baseline_cer": sum(metric["baseline_ocr"]["cer"]["rate"] for metric in measured) / len(measured),
        "macro_post_correction_cer": sum(metric["post_correction"]["cer"]["rate"] for metric in measured) / len(measured),
        "macro_baseline_wer": sum(metric["baseline_ocr"]["wer"]["rate"] for metric in measured) / len(measured),
        "macro_post_correction_wer": sum(metric["post_correction"]["wer"]["rate"] for metric in measured) / len(measured),
    }
    write_json(RESULT_ROOT / "illustrative_metrics.json", {"summary": summary, "documents": results})

    csv_path = RESULT_ROOT / "illustrative_metrics.csv"
    fields = [
        "document_id", "input_format", "injected_errors", "corrected_errors", "unresolved_errors",
        "outcome", "baseline_cer", "post_correction_cer", "baseline_wer", "post_correction_wer",
        "classification",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in results:
            metric = record["metrics"]
            writer.writerow({
                "document_id": record["document_id"], "input_format": record["input_format"],
                "injected_errors": record["injected_errors"],
                "corrected_errors": record["illustratively_corrected_errors"],
                "unresolved_errors": record["illustratively_unresolved_errors"],
                "outcome": metric["outcome"],
                "baseline_cer": metric["baseline_ocr"]["cer"]["rate"],
                "post_correction_cer": metric["post_correction"]["cer"]["rate"],
                "baseline_wer": metric["baseline_ocr"]["wer"]["rate"],
                "post_correction_wer": metric["post_correction"]["wer"]["rate"],
                "classification": record["classification"],
            })

    report_lines = [
        "# Illustrative OCR Correction Results",
        "",
        "> **ILLUSTRATIVE / SYNTHETIC - NOT ACTUAL PIPELINE MEASUREMENTS**",
        "",
        f"- Documents: **{len(results)}**",
        f"- Controlled error tokens: **{total_errors}**",
        f"- Illustratively corrected: **{corrected_errors}**",
        f"- Intentionally unresolved: **{total_errors - corrected_errors}**",
        f"- Illustrative token correction accuracy: **{correction_accuracy * 100:.2f}%**",
        f"- Outcomes: **{outcome_counts['IMPROVED']} improved, {outcome_counts['UNCHANGED']} unchanged, {outcome_counts['WORSENED']} worsened**",
        f"- Macro CER: **{summary['macro_baseline_cer'] * 100:.2f}% -> {summary['macro_post_correction_cer'] * 100:.2f}%**",
        f"- Macro WER: **{summary['macro_baseline_wer'] * 100:.2f}% -> {summary['macro_post_correction_wer'] * 100:.2f}%**",
        "",
        "| ID | Errors | Corrected | Unresolved | Outcome | Post CER | Post WER |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for record in results:
        metric = record["metrics"]
        report_lines.append(
            f"| {record['document_id']} | {record['injected_errors']} | {record['illustratively_corrected_errors']} | "
            f"{record['illustratively_unresolved_errors']} | {metric['outcome']} | "
            f"{metric['post_correction']['cer']['rate'] * 100:.2f}% | {metric['post_correction']['wer']['rate'] * 100:.2f}% |"
        )
    report_lines.extend([
        "", "The single intentionally unresolved controlled error is `reciept` in `PDF005`.",
        "These files are stored separately and do not replace genuine `evaluation/actual_runs` evidence or genuine metrics.", "",
    ])
    (RESULT_ROOT / "ILLUSTRATIVE_RESULTS.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
