from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


EVALUATION_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATION_ROOT.parent
DEFAULT_MANIFEST = EVALUATION_ROOT / "dataset" / "manifests" / "dataset_manifest.json"
DEFAULT_ACTUAL_RUNS = EVALUATION_ROOT / "actual_runs"
DEFAULT_OUTPUT = EVALUATION_ROOT / "results"


def normalize_text(text: str) -> str:
    """NFKC-normalize, lowercase, and collapse all whitespace to single spaces.

    Punctuation and numeric characters are deliberately retained. Raw files are
    never modified, which keeps the original evidence available for auditing.
    """
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(normalized.lower().split())


def _levenshtein_alignment(reference: Sequence[str], prediction: Sequence[str]) -> dict[str, Any]:
    rows, cols = len(reference) + 1, len(prediction) + 1
    dp = [[0] * cols for _ in range(rows)]
    back: list[list[str | None]] = [[None] * cols for _ in range(rows)]
    for i in range(1, rows):
        dp[i][0], back[i][0] = i, "delete"
    for j in range(1, cols):
        dp[0][j], back[0][j] = j, "insert"
    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == prediction[j - 1]:
                dp[i][j], back[i][j] = dp[i - 1][j - 1], "equal"
                continue
            candidates = (
                (dp[i - 1][j - 1] + 1, 0, "substitute"),
                (dp[i - 1][j] + 1, 1, "delete"),
                (dp[i][j - 1] + 1, 2, "insert"),
            )
            cost, _priority, operation = min(candidates)
            dp[i][j], back[i][j] = cost, operation

    i, j = len(reference), len(prediction)
    alignment: list[dict[str, Any]] = []
    counts = Counter({"substitutions": 0, "deletions": 0, "insertions": 0})
    while i or j:
        operation = back[i][j]
        if operation in ("equal", "substitute"):
            i -= 1
            j -= 1
            alignment.append({
                "operation": operation,
                "reference_index": i,
                "prediction_index": j,
                "reference": reference[i],
                "prediction": prediction[j],
            })
            if operation == "substitute":
                counts["substitutions"] += 1
        elif operation == "delete":
            i -= 1
            alignment.append({
                "operation": "delete", "reference_index": i, "prediction_index": None,
                "reference": reference[i], "prediction": None,
            })
            counts["deletions"] += 1
        elif operation == "insert":
            j -= 1
            alignment.append({
                "operation": "insert", "reference_index": None, "prediction_index": j,
                "reference": None, "prediction": prediction[j], "anchor": i,
            })
            counts["insertions"] += 1
        else:
            raise RuntimeError("Invalid Levenshtein backtrace state")
    alignment.reverse()
    distance = counts["substitutions"] + counts["deletions"] + counts["insertions"]
    return {"distance": distance, **counts, "alignment": alignment}


def _metric(reference: str, prediction: str, level: str) -> dict[str, Any]:
    if level == "character":
        ref_units, pred_units = list(reference), list(prediction)
    elif level == "word":
        ref_units, pred_units = reference.split(), prediction.split()
    else:
        raise ValueError(f"Unknown metric level: {level}")
    if not ref_units:
        return {
            "status": "UNDEFINED_EMPTY_REFERENCE", "reference_units": 0,
            "prediction_units": len(pred_units), "rate": None,
            "substitutions": None, "deletions": None, "insertions": None,
        }
    edits = _levenshtein_alignment(ref_units, pred_units)
    return {
        "status": "MEASURED",
        "reference_units": len(ref_units),
        "prediction_units": len(pred_units),
        "substitutions": edits["substitutions"],
        "deletions": edits["deletions"],
        "insertions": edits["insertions"],
        "edit_distance": edits["distance"],
        "rate": edits["distance"] / len(ref_units),
    }


def evaluate_prediction(reference_raw: str, prediction_raw: str) -> dict[str, Any]:
    reference = normalize_text(reference_raw)
    prediction = normalize_text(prediction_raw)
    cer = _metric(reference, prediction, "character")
    wer = _metric(reference, prediction, "word")
    normalized_accuracy = None if cer["rate"] is None else 1.0 - cer["rate"]
    return {
        "normalization_policy": "Unicode NFKC; lowercase; collapse whitespace; retain punctuation and numbers",
        "normalized_reference_length": len(reference),
        "normalized_prediction_length": len(prediction),
        "cer": cer,
        "wer": wer,
        "character_level_normalized_accuracy": normalized_accuracy,
    }


def _token_error_state(reference: list[str], prediction: list[str]) -> tuple[set[tuple[Any, ...]], dict[int, str | None]]:
    alignment = _levenshtein_alignment(reference, prediction)["alignment"]
    errors: set[tuple[Any, ...]] = set()
    mapped: dict[int, str | None] = {index: None for index in range(len(reference))}
    insertion_counts: Counter[tuple[int, str]] = Counter()
    for step in alignment:
        operation = step["operation"]
        if operation in ("equal", "substitute"):
            ref_index = int(step["reference_index"])
            mapped[ref_index] = step["prediction"]
            if operation == "substitute":
                errors.add(("reference_position", ref_index))
        elif operation == "delete":
            errors.add(("reference_position", int(step["reference_index"])))
        elif operation == "insert":
            anchor = int(step.get("anchor", 0))
            token = str(step["prediction"])
            insertion_counts[(anchor, token)] += 1
            errors.add(("insertion", anchor, token, insertion_counts[(anchor, token)]))
    return errors, mapped


def correction_analysis(reference_raw: str, baseline_raw: str, corrected_raw: str) -> dict[str, Any]:
    reference = normalize_text(reference_raw).split()
    baseline = normalize_text(baseline_raw).split()
    corrected = normalize_text(corrected_raw).split()
    baseline_errors, baseline_map = _token_error_state(reference, baseline)
    corrected_errors, corrected_map = _token_error_state(reference, corrected)
    errors_corrected = len(baseline_errors - corrected_errors)
    errors_introduced = len(corrected_errors - baseline_errors)
    errors_remaining = len(baseline_errors & corrected_errors)

    changed_reference_positions = {
        index for index in range(len(reference)) if baseline_map.get(index) != corrected_map.get(index)
    }
    changed_insertions = {
        error for error in baseline_errors.symmetric_difference(corrected_errors) if error[0] == "insertion"
    }
    correction_attempts = len(changed_reference_positions) + len(changed_insertions)
    successful_reference_corrections = sum(
        baseline_map.get(index) != reference[index] and corrected_map.get(index) == reference[index]
        for index in changed_reference_positions
    )
    removed_insertions = len({error for error in baseline_errors - corrected_errors if error[0] == "insertion"})
    successful_corrections = successful_reference_corrections + removed_insertions
    token_accuracy = successful_corrections / correction_attempts if correction_attempts else None
    return {
        "unit": "word-level aligned error events",
        "errors_corrected": errors_corrected,
        "errors_introduced": errors_introduced,
        "errors_remaining": errors_remaining,
        "correction_attempts": correction_attempts,
        "successful_corrections": successful_corrections,
        "token_level_correction_accuracy": token_accuracy,
        "token_accuracy_note": "Only changed aligned word positions and changed insertions count as attempts; unchanged correct words are excluded.",
    }


def evaluate_document(reference_raw: str, baseline_raw: str, corrected_raw: str) -> dict[str, Any]:
    baseline = evaluate_prediction(reference_raw, baseline_raw)
    corrected = evaluate_prediction(reference_raw, corrected_raw)
    baseline_cer = baseline["cer"]["rate"]
    corrected_cer = corrected["cer"]["rate"]
    if baseline_cer is None or corrected_cer is None:
        absolute_reduction = relative_reduction = None
        outcome = "NOT_YET_MEASURED"
    else:
        absolute_reduction = baseline_cer - corrected_cer
        relative_reduction = absolute_reduction / baseline_cer if baseline_cer > 0 else None
        if corrected_cer < baseline_cer:
            outcome = "IMPROVED"
        elif corrected_cer > baseline_cer:
            outcome = "WORSENED"
        else:
            outcome = "UNCHANGED"
    baseline_clean = baseline_cer == 0 if baseline_cer is not None else None
    return {
        "status": "MEASURED",
        "baseline_ocr": baseline,
        "post_correction": corrected,
        "absolute_cer_reduction": absolute_reduction,
        "relative_cer_reduction": relative_reduction,
        "outcome": outcome,
        "correction_performance": correction_analysis(reference_raw, baseline_raw, corrected_raw),
        "baseline_is_clean": baseline_clean,
        "clean_document_preserved": (corrected_cer == 0) if baseline_clean else None,
    }


def _read_optional(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def evaluate_saved_runs(manifest_path: Path, actual_runs: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents: list[dict[str, Any]] = []
    for entry in manifest:
        document_id = entry["document_id"]
        run_dir = actual_runs / document_id
        reference_path = _resolve_project_path(entry["ground_truth_reference_path"])
        baseline_path = run_dir / "ocr_raw.txt"
        corrected_path = run_dir / "corrected.txt"
        metadata_path = run_dir / "run_metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                documents.append({**entry, "metric_status": "INVALID_METADATA", "reason": str(exc)})
                continue
        if str(metadata.get("status", "")).upper() == "FAILED":
            documents.append({**entry, "metric_status": "FAILED_RUN", "reason": metadata.get("reason", "Run marked failed")})
            continue
        if not reference_path.is_file():
            documents.append({**entry, "metric_status": "MISSING_REFERENCE", "reason": str(reference_path)})
            continue
        reference_raw = reference_path.read_text(encoding="utf-8")
        if not normalize_text(reference_raw):
            documents.append({**entry, "metric_status": "EMPTY_REFERENCE", "reason": str(reference_path)})
            continue
        baseline_raw = _read_optional(baseline_path)
        corrected_raw = _read_optional(corrected_path)
        if baseline_raw is None or corrected_raw is None:
            missing = [name for name, value in (("ocr_raw.txt", baseline_raw), ("corrected.txt", corrected_raw)) if value is None]
            documents.append({
                **entry,
                "metric_status": "NOT_YET_MEASURED",
                "reason": f"Missing genuine output: {', '.join(missing)}",
                "available_partial_metrics": {
                    "baseline_ocr": evaluate_prediction(reference_raw, baseline_raw) if baseline_raw is not None else None,
                    "post_correction": evaluate_prediction(reference_raw, corrected_raw) if corrected_raw is not None else None,
                },
            })
            continue
        result = evaluate_document(reference_raw, baseline_raw, corrected_raw)
        documents.append({**entry, "metric_status": "MEASURED", "metrics": result, "run_metadata": metadata})

    measured = [doc for doc in documents if doc["metric_status"] == "MEASURED"]
    outcomes = Counter(doc["metrics"]["outcome"] for doc in measured)
    clean_docs = [doc for doc in measured if doc["metrics"]["baseline_is_clean"]]
    clean_preserved = sum(doc["metrics"]["clean_document_preserved"] is True for doc in clean_docs)
    summary: dict[str, Any] = {
        "status": "MEASURED" if measured else "NOT_YET_MEASURED",
        "documents_in_manifest": len(documents),
        "documents_measured": len(measured),
        "documents_not_yet_measured": len(documents) - len(measured),
        "documents_improved": outcomes["IMPROVED"],
        "documents_unchanged": outcomes["UNCHANGED"],
        "documents_worsened": outcomes["WORSENED"],
        "clean_baseline_documents_measured": len(clean_docs),
        "clean_documents_preserved": clean_preserved,
        "clean_document_preservation_rate": clean_preserved / len(clean_docs) if clean_docs else None,
    }
    if measured:
        summary.update({
            "macro_baseline_cer": sum(doc["metrics"]["baseline_ocr"]["cer"]["rate"] for doc in measured) / len(measured),
            "macro_post_correction_cer": sum(doc["metrics"]["post_correction"]["cer"]["rate"] for doc in measured) / len(measured),
            "macro_baseline_wer": sum(doc["metrics"]["baseline_ocr"]["wer"]["rate"] for doc in measured) / len(measured),
            "macro_post_correction_wer": sum(doc["metrics"]["post_correction"]["wer"]["rate"] for doc in measured) / len(measured),
        })
        summary["macro_baseline_normalized_character_accuracy"] = 1 - summary["macro_baseline_cer"]
        summary["macro_post_correction_normalized_character_accuracy"] = 1 - summary["macro_post_correction_cer"]
    return {
        "evaluation_status": summary["status"],
        "normalization_policy": "Unicode NFKC; lowercase; collapse whitespace; retain punctuation and numbers",
        "summary": summary,
        "documents": documents,
    }


def write_results(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = [
        "document_id", "input_filename", "metric_status", "outcome", "baseline_cer", "post_correction_cer",
        "baseline_wer", "post_correction_wer", "absolute_cer_reduction", "relative_cer_reduction",
        "baseline_normalized_character_accuracy", "post_correction_normalized_character_accuracy",
        "errors_corrected", "errors_introduced", "errors_remaining", "token_level_correction_accuracy", "reason",
    ]
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for document in report["documents"]:
            row = {key: "" for key in fields}
            row.update({
                "document_id": document["document_id"],
                "input_filename": document["input_filename"],
                "metric_status": document["metric_status"],
                "reason": document.get("reason", ""),
            })
            metrics = document.get("metrics")
            if metrics:
                correction = metrics["correction_performance"]
                row.update({
                    "outcome": metrics["outcome"],
                    "baseline_cer": metrics["baseline_ocr"]["cer"]["rate"],
                    "post_correction_cer": metrics["post_correction"]["cer"]["rate"],
                    "baseline_wer": metrics["baseline_ocr"]["wer"]["rate"],
                    "post_correction_wer": metrics["post_correction"]["wer"]["rate"],
                    "absolute_cer_reduction": metrics["absolute_cer_reduction"],
                    "relative_cer_reduction": metrics["relative_cer_reduction"],
                    "baseline_normalized_character_accuracy": metrics["baseline_ocr"]["character_level_normalized_accuracy"],
                    "post_correction_normalized_character_accuracy": metrics["post_correction"]["character_level_normalized_accuracy"],
                    "errors_corrected": correction["errors_corrected"],
                    "errors_introduced": correction["errors_introduced"],
                    "errors_remaining": correction["errors_remaining"],
                    "token_level_correction_accuracy": correction["token_level_correction_accuracy"],
                })
            writer.writerow(row)


def self_test() -> None:
    reference = "The patient received a prescription after review."
    baseline = "The patent recieved a prescripton after review."
    corrected = reference
    result = evaluate_document(reference, baseline, corrected)
    assert result["baseline_ocr"]["cer"]["rate"] > 0
    assert result["post_correction"]["cer"]["rate"] == 0
    assert result["post_correction"]["wer"]["rate"] == 0
    assert result["absolute_cer_reduction"] > 0
    assert result["relative_cer_reduction"] == 1.0
    assert result["correction_performance"]["errors_corrected"] == 3
    assert result["correction_performance"]["errors_introduced"] == 0
    assert result["correction_performance"]["token_level_correction_accuracy"] == 1.0
    clean = evaluate_document(reference, reference, reference)
    assert clean["clean_document_preserved"] is True
    assert clean["correction_performance"]["token_level_correction_accuracy"] is None
    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        (temporary_path / "result.json").write_text(json.dumps(result), encoding="utf-8")
        assert json.loads((temporary_path / "result.json").read_text(encoding="utf-8"))["outcome"] == "IMPROVED"
    print("Metric self-test passed: CER, WER, reductions, correction events, and clean preservation.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate OCR metrics from genuine manually saved outputs.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--actual-runs", type=Path, default=DEFAULT_ACTUAL_RUNS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    report = evaluate_saved_runs(args.manifest.resolve(), args.actual_runs.resolve())
    write_results(report, args.output.resolve())
    print(f"Evaluation status: {report['evaluation_status']}")
    print(f"Measured documents: {report['summary']['documents_measured']}")
    print(f"Results: {(args.output.resolve() / 'metrics.json')}")


if __name__ == "__main__":
    main()
