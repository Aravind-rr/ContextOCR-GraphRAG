from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "results" / "metrics.json"
OUTPUT_PATH = ROOT / "results" / "RESULTS.md"


def percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def pipeline_runtime(metadata: dict[str, Any]) -> float:
    values = metadata.get("stage_timings_ms", {}).values()
    return sum(float(value or 0) for value in values) / 1000


def aggregate(documents: list[dict[str, Any]]) -> dict[str, Any]:
    if not documents:
        return {}
    outcomes = Counter(document["metrics"]["outcome"] for document in documents)
    return {
        "count": len(documents),
        "baseline_cer": sum(document["metrics"]["baseline_ocr"]["cer"]["rate"] for document in documents) / len(documents),
        "corrected_cer": sum(document["metrics"]["post_correction"]["cer"]["rate"] for document in documents) / len(documents),
        "baseline_wer": sum(document["metrics"]["baseline_ocr"]["wer"]["rate"] for document in documents) / len(documents),
        "corrected_wer": sum(document["metrics"]["post_correction"]["wer"]["rate"] for document in documents) / len(documents),
        "improved": outcomes["IMPROVED"],
        "unchanged": outcomes["UNCHANGED"],
        "worsened": outcomes["WORSENED"],
    }


def main() -> None:
    report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    measured = [document for document in report["documents"] if document["metric_status"] == "MEASURED"]
    images = [document for document in measured if document["input_format"] == "image"]
    pdfs = [document for document in measured if document["input_format"] == "pdf"]
    summary = aggregate(measured)

    total_low_confidence = 0
    total_applied = 0
    total_slm_calls = 0
    total_runtime = 0.0
    change_rows: list[str] = []
    warning_rows: list[str] = []
    for document in measured:
        document_id = document["document_id"]
        run_dir = ROOT / "actual_runs" / document_id
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        stats = metadata.get("pipeline_stats", {})
        total_low_confidence += int(stats.get("low_confidence_tokens", 0))
        total_applied += int(stats.get("corrections_applied", 0))
        total_slm_calls += int(stats.get("slm_calls", 0))
        total_runtime += pipeline_runtime(metadata)
        traces = json.loads((run_dir / "corrections.json").read_text(encoding="utf-8"))
        for trace in traces:
            if trace.get("changed"):
                change_rows.append(
                    f"| {document_id} | {trace.get('original', '')} | {trace.get('final', '')} | "
                    f"{trace.get('status', '')} | {float(trace.get('confidence') or 0):.3f} |"
                )
        pipeline = json.loads((run_dir / "pipeline.json").read_text(encoding="utf-8"))
        for stage in pipeline.get("stages", []):
            if stage.get("status") == "WARNING":
                messages = "; ".join(stage.get("warnings", [])) or stage.get("error") or "Unspecified warning"
                warning_rows.append(f"| {document_id} | {stage.get('name')} | {messages} |")

    format_rows = []
    for label, documents in (("Images", images), ("PDFs", pdfs), ("Overall", measured)):
        values = aggregate(documents)
        format_rows.append(
            f"| {label} | {values['count']} | {percent(values['baseline_cer'])} | "
            f"{percent(values['corrected_cer'])} | {percent(values['baseline_wer'])} | "
            f"{percent(values['corrected_wer'])} | {values['improved']} / {values['unchanged']} / {values['worsened']} |"
        )

    document_rows = []
    for document in measured:
        metrics = document["metrics"]
        metadata = document.get("run_metadata", {})
        stats = metadata.get("pipeline_stats", {})
        document_rows.append(
            f"| {document['document_id']} | {document['input_format']} | {metrics['outcome']} | "
            f"{percent(metrics['baseline_ocr']['cer']['rate'])} | {percent(metrics['post_correction']['cer']['rate'])} | "
            f"{percent(metrics['baseline_ocr']['wer']['rate'])} | {percent(metrics['post_correction']['wer']['rate'])} | "
            f"{stats.get('low_confidence_tokens', 0)} | {stats.get('corrections_applied', 0)} | "
            f"{pipeline_runtime(metadata):.1f} |"
        )

    lines = [
        "# Actual OCR Evaluation Results",
        "",
        "These results were calculated from genuine pipeline runs of `IMG001–IMG005` and `PDF001–PDF005` at a confidence threshold of **0.90**. The remaining 20 dataset documents were not processed.",
        "",
        "## Experimental setup",
        "",
        "- Images used PaddleOCR; PDFs used PyMuPDF embedded-text extraction because the generated PDFs contain a text layer.",
        "- Medical documents used the `medical` domain; invoice/receipt documents used `financial`.",
        "- The full image pipeline included DocLayout-YOLO, BGE-M3/FAISS retrieval, Neo4j GraphRAG, Qwen2.5-3B selection, validation, and searchable-PDF export.",
        "- Normalization: Unicode NFKC, lowercase, collapsed whitespace, with punctuation and numbers retained.",
        "",
        "## Aggregate metrics",
        "",
        "| Group | N | Baseline CER | Corrected CER | Baseline WER | Corrected WER | Improved / unchanged / worsened |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *format_rows,
        "",
        f"The ten runs examined **{total_low_confidence}** low-confidence tokens, made **{total_slm_calls}** successful Qwen calls, and applied **{total_applied}** token substitutions. Total summed pipeline-stage time was **{total_runtime:.1f} seconds**.",
        "",
        "## Per-document results",
        "",
        "| ID | Format | Outcome | Baseline CER | Corrected CER | Baseline WER | Corrected WER | Low-confidence | Applied | Runtime (s) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        *document_rows,
        "",
        "## Applied correction trace",
        "",
        "| ID | Original | Final | Validation status | Model confidence |",
        "|---|---|---|---|---:|",
        *(change_rows or ["| — | No token substitutions were applied | — | — | — |"]),
        "",
        "## Pipeline warnings",
        "",
        "| ID | Stage | Warning |",
        "|---|---|---|",
        *(warning_rows or ["| — | — | No stage warnings |"]),
        "",
        "## Evidence-based interpretation",
        "",
        f"- Macro CER changed from **{percent(summary['baseline_cer'])}** to **{percent(summary['corrected_cer'])}**; macro WER changed from **{percent(summary['baseline_wer'])}** to **{percent(summary['corrected_wer'])}**.",
        f"- Macro character-level normalized accuracy (`1 − CER`) changed from **{percent(1 - summary['baseline_cer'])}** to **{percent(1 - summary['corrected_cer'])}**.",
        f"- Outcomes were **{summary['improved']} improved**, **{summary['unchanged']} unchanged**, and **{summary['worsened']} worsened**. These are reported as observed, without replacing or hand-editing model output.",
        "- The five PDFs produced no low-confidence selections because embedded-text extraction assigned no calibrated low-confidence tokens. Their injected spelling corruptions therefore passed through unchanged.",
        "- The image runs demonstrate that the full confidence-aware/Qwen route executed, but three accepted substitutions were not supported by the intended reference (`Clinic → clinical`, `PATLENT → tablet`, and `contact → demonstration`). This exposes a candidate/validation precision limitation rather than a runtime failure.",
        "- Baseline error rates include template/header/footer and OCR punctuation differences because the strict normalization policy intentionally retains punctuation and numbers.",
        "- None of the measured baselines had CER exactly zero, so clean-document preservation is not defined for this ten-document run under the strict metric policy.",
        "",
        "## Artifact locations",
        "",
        "- Per-run raw OCR, corrected text, searchable PDF, trace, evidence, graph, timings, and full pipeline record: `evaluation/actual_runs/<ID>/`",
        "- Machine-readable complete metrics: `evaluation/results/metrics.json`",
        "- Flat per-document metrics: `evaluation/results/metrics.csv`",
        "",
        "## Reproduce",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe evaluation\\run_evaluation_subset.py",
        ".\\.venv\\Scripts\\python.exe evaluation\\calculate_metrics.py",
        ".\\.venv\\Scripts\\python.exe evaluation\\generate_results_report.py",
        ".\\.venv\\Scripts\\python.exe evaluation\\validate_dataset.py",
        "```",
        "",
    ]
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
