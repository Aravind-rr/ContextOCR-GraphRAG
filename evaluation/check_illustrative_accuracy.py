from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from calculate_metrics import evaluate_document


EVALUATION_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATION_ROOT.parent
DEFAULT_MANIFEST = EVALUATION_ROOT / "dataset" / "manifests" / "dataset_manifest.json"
DEFAULT_RUNS = EVALUATION_ROOT / "illustrative_runs"


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def resolve_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return {str(record["document_id"]): record for record in records}


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = ("Document", "Errors", "Fixed", "Remaining", "Character acc.", "Word acc.")
    values = [
        (
            row["document_id"],
            str(row["baseline_errors"]),
            str(row["errors_corrected"]),
            str(row["errors_remaining"]),
            percent(row["character_accuracy"]),
            percent(row["word_accuracy"]),
        )
        for row in rows
    ]
    widths = [max(len(headers[index]), *(len(row[index]) for row in values)) for index in range(len(headers))]
    template = "  ".join(f"{{:<{width}}}" for width in widths)
    print(template.format(*headers))
    print(template.format(*(char * width for char, width in zip("------", widths))))
    for row in values:
        print(template.format(*row))


def evaluate(manifest_path: Path, runs_path: Path) -> int:
    if not manifest_path.is_file():
        print(f"ERROR: Manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    if not runs_path.is_dir():
        print(f"ERROR: Illustrative runs directory not found: {runs_path}", file=sys.stderr)
        return 1

    manifest = load_manifest(manifest_path)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for run_dir in sorted(path for path in runs_path.iterdir() if path.is_dir()):
        document_id = run_dir.name
        record = manifest.get(document_id)
        if record is None:
            failures.append(f"{document_id}: not present in the dataset manifest")
            continue

        reference_path = resolve_path(Path(record["ground_truth_reference_path"]), PROJECT_ROOT)
        baseline_path = run_dir / "illustrative_ocr_raw.txt"
        corrected_path = run_dir / "illustrative_corrected.txt"
        missing = [
            str(path)
            for path in (reference_path, baseline_path, corrected_path)
            if not path.is_file()
        ]
        if missing:
            failures.append(f"{document_id}: missing {', '.join(missing)}")
            continue

        reference = reference_path.read_text(encoding="utf-8")
        baseline = baseline_path.read_text(encoding="utf-8")
        corrected = corrected_path.read_text(encoding="utf-8")
        metrics = evaluate_document(reference, baseline, corrected)
        baseline_word = metrics["baseline_ocr"]["wer"]
        corrected_word = metrics["post_correction"]["wer"]
        corrected_char = metrics["post_correction"]["cer"]
        correction = metrics["correction_performance"]

        rows.append(
            {
                "document_id": document_id,
                "baseline_errors": baseline_word["edit_distance"],
                "errors_corrected": correction["errors_corrected"],
                "errors_remaining": corrected_word["edit_distance"],
                "character_edits": corrected_char["edit_distance"],
                "reference_characters": corrected_char["reference_units"],
                "word_edits": corrected_word["edit_distance"],
                "reference_words": corrected_word["reference_units"],
                "character_accuracy": 1.0 - corrected_char["rate"],
                "word_accuracy": 1.0 - corrected_word["rate"],
            }
        )

    if failures:
        print("Validation errors:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
    if not rows:
        print("ERROR: No complete illustrative result sets were found.", file=sys.stderr)
        return 1

    total_baseline_errors = sum(row["baseline_errors"] for row in rows)
    total_corrected_errors = sum(row["errors_corrected"] for row in rows)
    total_remaining_errors = sum(row["errors_remaining"] for row in rows)
    total_reference_characters = sum(row["reference_characters"] for row in rows)
    total_character_edits = sum(row["character_edits"] for row in rows)
    total_reference_words = sum(row["reference_words"] for row in rows)
    total_word_edits = sum(row["word_edits"] for row in rows)

    correction_accuracy = (
        total_corrected_errors / total_baseline_errors if total_baseline_errors else 1.0
    )
    character_accuracy = 1.0 - (total_character_edits / total_reference_characters)
    word_accuracy = 1.0 - (total_word_edits / total_reference_words)

    print("\nContextOCR illustrative result verification")
    print("Compared directly with dataset ground-truth text files.\n")
    print_table(rows)
    print("\nOverall results")
    print(f"  Documents checked:                    {len(rows)}")
    print(
        "  Controlled-error correction accuracy: "
        f"{percent(correction_accuracy)} ({total_corrected_errors}/{total_baseline_errors})"
    )
    print(
        "  Final character accuracy (1 - CER):   "
        f"{percent(character_accuracy)} ({total_character_edits} edits / "
        f"{total_reference_characters} reference characters)"
    )
    print(
        "  Final word accuracy (1 - WER):         "
        f"{percent(word_accuracy)} ({total_word_edits} edits / "
        f"{total_reference_words} reference words)"
    )
    print(f"  Remaining word errors:                 {total_remaining_errors}")
    print("\nNote: these are illustrative/synthetic outputs, not actual pipeline-run measurements.")
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare illustrative OCR outputs with their ground-truth text files."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    args = parser.parse_args()
    raise SystemExit(evaluate(args.manifest.resolve(), args.runs.resolve()))


if __name__ == "__main__":
    main()
