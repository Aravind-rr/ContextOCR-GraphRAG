from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the generated dataset and any genuine recorded runs.")
    parser.add_argument(
        "--generation-only",
        action="store_true",
        help="Require the original pre-execution state: all entries NOT_RUN and no output artifacts.",
    )
    args = parser.parse_args()
    image_files = sorted((DATASET / "inputs" / "images").glob("*.png"))
    pdf_files = sorted((DATASET / "inputs" / "pdfs").glob("*.pdf"))
    image_truth = sorted((DATASET / "ground_truth" / "images").glob("*.txt"))
    pdf_truth = sorted((DATASET / "ground_truth" / "pdfs").glob("*.txt"))
    json_manifest_path = DATASET / "manifests" / "dataset_manifest.json"
    csv_manifest_path = DATASET / "manifests" / "dataset_manifest.csv"
    manifest = json.loads(json_manifest_path.read_text(encoding="utf-8"))
    with csv_manifest_path.open(encoding="utf-8", newline="") as handle:
        csv_manifest = list(csv.DictReader(handle))

    require(len(image_files) == 15, f"Expected 15 PNG inputs, found {len(image_files)}")
    require(len(pdf_files) == 15, f"Expected 15 PDF inputs, found {len(pdf_files)}")
    require(len(image_truth) == 15, f"Expected 15 image ground truths, found {len(image_truth)}")
    require(len(pdf_truth) == 15, f"Expected 15 PDF ground truths, found {len(pdf_truth)}")
    require(len(manifest) == 30, f"Expected 30 JSON manifest entries, found {len(manifest)}")
    require(len(csv_manifest) == 30, f"Expected 30 CSV manifest entries, found {len(csv_manifest)}")

    expected_ids = {f"IMG{i:03d}" for i in range(1, 16)} | {f"PDF{i:03d}" for i in range(1, 16)}
    manifest_ids = [entry["document_id"] for entry in manifest]
    require(len(set(manifest_ids)) == 30, "Manifest contains duplicate document IDs")
    require(set(manifest_ids) == expected_ids, "Manifest document IDs do not match IMG001-IMG015 and PDF001-PDF015")
    valid_statuses = {"NOT_RUN", "ACTUALLY_RUN"}
    require(
        all(entry["execution_status"] in valid_statuses for entry in manifest),
        "Manifest execution status must be NOT_RUN or ACTUALLY_RUN",
    )
    require(len({entry["input_filename"] for entry in manifest}) == 30, "Manifest contains duplicate input filenames")
    corruption_events = sum(len(entry["injected_error_details"]) for entry in manifest)
    require(corruption_events == 40, f"Expected 40 injected corruption events, found {corruption_events}")
    require(all(
        len(entry["injected_error_details"]) == (2 if entry["intentional_textual_corruption"] else 0)
        for entry in manifest
    ), "Each challenging document must have two corruption events and each clean document zero")

    for image_path in image_files:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            require(image.width >= 800 and image.height >= 1100, f"Image resolution is too low: {image_path.name}")
    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))
        require(len(reader.pages) == 1, f"PDF must contain exactly one page: {pdf_path.name}")
        require(bool(reader.pages[0].extract_text().strip()), f"PDF has no extractable text: {pdf_path.name}")

    for entry in manifest:
        input_path = ROOT.parent / entry["input_path"]
        truth_path = ROOT.parent / entry["ground_truth_reference_path"]
        require(input_path.is_file(), f"Manifest input is missing: {input_path}")
        require(truth_path.is_file(), f"Manifest ground truth is missing: {truth_path}")
        truth = truth_path.read_text(encoding="utf-8")
        require(bool(truth.strip()), f"Ground truth is empty: {truth_path}")
        for detail in entry["injected_error_details"]:
            require(detail["intended"] in truth, f"Intended text is absent from ground truth: {entry['document_id']}")
            require(detail["rendered"] not in truth, f"Corrupted text leaked into ground truth: {entry['document_id']}")
        if entry["input_format"] == "pdf":
            rendered_text = PdfReader(str(input_path)).pages[0].extract_text()
            for detail in entry["injected_error_details"]:
                require(detail["rendered"].casefold() in rendered_text.casefold(), f"Corrupted text is absent from rendered PDF: {entry['document_id']}")

    expected_run_ids = [f"IMG{i:03d}" for i in range(1, 6)] + [f"PDF{i:03d}" for i in range(1, 6)]
    for document_id in expected_run_ids:
        require((ROOT / "actual_runs" / document_id).is_dir(), f"Manual run directory is missing: {document_id}")
    actually_run = {entry["document_id"] for entry in manifest if entry["execution_status"] == "ACTUALLY_RUN"}
    if args.generation_only:
        require(not actually_run, "Generation-only validation requires every manifest entry to be NOT_RUN")
        prohibited_outputs = []
        for run_dir in (ROOT / "actual_runs").iterdir():
            if run_dir.is_dir():
                prohibited_outputs.extend(path for path in run_dir.iterdir() if path.name != ".gitkeep")
        require(not prohibited_outputs, f"Run outputs exist in generation-only mode: {prohibited_outputs}")
    else:
        require(actually_run.issubset(set(expected_run_ids)), "Only the authorized ten documents may be ACTUALLY_RUN")
        required_artifacts = {
            "run_metadata.json", "pipeline.json", "result.json", "ocr_raw.txt", "corrected.txt",
            "corrected.pdf", "corrections.json", "evidence.json", "graph.json",
        }
        for document_id in actually_run:
            run_dir = ROOT / "actual_runs" / document_id
            missing = sorted(name for name in required_artifacts if not (run_dir / name).is_file())
            require(not missing, f"Missing genuine artifacts for {document_id}: {missing}")
            metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
            require(metadata.get("status") == "ACTUALLY_RUN", f"Invalid run metadata status: {document_id}")
            require(metadata.get("pipeline_status") == "COMPLETED", f"Pipeline was not completed: {document_id}")
            require(bool((run_dir / "ocr_raw.txt").read_text(encoding="utf-8").strip()), f"Empty OCR output: {document_id}")
            require(bool((run_dir / "corrected.txt").read_text(encoding="utf-8").strip()), f"Empty corrected output: {document_id}")
            corrected_reader = PdfReader(str(run_dir / "corrected.pdf"))
            require(len(corrected_reader.pages) >= 1, f"Corrected PDF has no pages: {document_id}")
            require(
                any((page.extract_text() or "").strip() for page in corrected_reader.pages),
                f"Corrected PDF has no selectable text: {document_id}",
            )

    clean = sum(not entry["intentional_textual_corruption"] for entry in manifest)
    challenging = len(manifest) - clean
    image_clean = sum(entry["input_format"] == "image" and not entry["intentional_textual_corruption"] for entry in manifest)
    pdf_clean = sum(entry["input_format"] == "pdf" and not entry["intentional_textual_corruption"] for entry in manifest)
    require((clean, challenging) == (10, 20), f"Expected 10 clean and 20 challenging documents, found {clean}/{challenging}")
    require((image_clean, pdf_clean) == (5, 5), "Clean/challenging distribution is not balanced across formats")

    print("Dataset validation passed.")
    print(
        f"15 readable PNGs; 15 readable one-page PDFs; 30 unique entries; 30 ground truths; "
        f"{len(actually_run)} ACTUALLY_RUN."
    )
    print("Composition: 10 clean and 20 intentionally challenging documents (5 clean per format).")
    if args.generation_only:
        print("No actual OCR or corrected-output files are present.")
    else:
        print("Every ACTUALLY_RUN entry has complete evidence and a searchable corrected PDF.")


if __name__ == "__main__":
    main()
