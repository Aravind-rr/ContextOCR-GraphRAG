from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.store import store

from .charts import generate_charts
from .config import load_evaluation_config
from .metrics import aggregate_documents, evaluate_document


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


class EvaluationService:
    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path

    def _ground_truth(self, run: dict[str, Any], document: dict[str, Any], supplied: dict[str, str], config: dict[str, Any]) -> tuple[str | None, str | None]:
        for key in (run["id"], document["id"], document["name"], Path(document["name"]).stem):
            if supplied.get(key, "").strip():
                return supplied[key], "request"
        directory = Path(config["ground_truth_directory"])
        if not directory.is_absolute():
            directory = Path.cwd() / directory
        for name in (f"{document['id']}.txt", f"{Path(document['name']).stem}.txt"):
            candidate = directory / name
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8"), str(candidate.resolve())
        return None, None

    def _select_runs(self, run_ids: list[str] | None) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        if run_ids:
            unique_ids = list(dict.fromkeys(run_ids))
            rows = [store.one("SELECT * FROM runs WHERE id=?", (run_id,)) for run_id in unique_ids]
            missing = [run_id for run_id, row in zip(unique_ids, rows) if row is None]
            if missing:
                raise ValueError(f"Unknown processing run(s): {', '.join(missing)}")
            runs = [store.decode(row) for row in rows if row]
        else:
            runs = [store.decode(row) for row in store.all("SELECT * FROM runs WHERE status='COMPLETED' ORDER BY updated_at DESC")]
        skipped: list[dict[str, str]] = []
        latest: dict[str, dict[str, Any]] = {}
        for run in sorted(runs, key=lambda item: item["updated_at"], reverse=True):
            if run["status"] != "COMPLETED":
                skipped.append({"run_id": run["id"], "reason": f"processing run status is {run['status']}"})
            elif not run.get("result", {}).get("original_text"):
                skipped.append({"run_id": run["id"], "reason": "processing output text is missing"})
            elif run["document_id"] in latest:
                skipped.append({"run_id": run["id"], "reason": "older duplicate processing run for the same document"})
            else:
                latest[run["document_id"]] = run
        return list(latest.values()), skipped

    def run(self, run_ids: list[str] | None = None, ground_truths: dict[str, str] | None = None, quality_labels: dict[str, str] | None = None) -> dict[str, Any]:
        started = utcnow(); evaluation_id = str(uuid.uuid4())
        config = load_evaluation_config(self.config_path)
        runs, skipped = self._select_runs(run_ids)
        if not runs:
            raise ValueError("No valid completed processing runs are available for evaluation")
        supplied_truth = ground_truths or {}; supplied_labels = quality_labels or {}
        documents: list[dict[str, Any]] = []
        signatures: list[dict[str, str]] = []
        gt_sources: dict[str, str | None] = {}
        for run in runs:
            row = store.one("SELECT * FROM documents WHERE id=?", (run["document_id"],))
            if not row:
                skipped.append({"run_id": run["id"], "reason": "document database record is missing"}); continue
            document = store.decode(row)
            truth, source = self._ground_truth(run, document, supplied_truth, config)
            evaluation = evaluate_document(run, document, truth, config)
            evaluation["evaluation_id"] = evaluation_id
            evaluation["evaluation_timestamp"] = started
            evaluation["ground_truth_source"] = source
            evaluation["quality_label"] = supplied_labels.get(run["id"]) or supplied_labels.get(document["id"])
            file_hash = hashlib.sha256(Path(document["path"]).read_bytes()).hexdigest() if Path(document["path"]).is_file() else "missing"
            output_hash = canonical_hash({key: run["result"].get(key) for key in ("tokens", "original_text", "corrected_text", "preprocessing", "correction_records")})
            signature = canonical_hash({"file_sha256": file_hash, "pipeline_output_sha256": output_hash, "config": config, "ground_truth": truth})
            evaluation["input_signature"] = signature
            signatures.append({"run_id": run["id"], "document_id": document["id"], "file_sha256": file_hash, "pipeline_output_sha256": output_hash, "evaluation_input_sha256": signature})
            gt_sources[document["id"]] = source
            documents.append(evaluation)
        if not documents:
            raise ValueError("All selected documents were invalid or missing")
        summary = aggregate_documents(documents)
        output_root = Path(config["output_directory"])
        if not output_root.is_absolute(): output_root = Path.cwd() / output_root
        output_dir = output_root / evaluation_id
        charts_dir = output_dir / "charts"; output_dir.mkdir(parents=True, exist_ok=False)
        classification = self._classification_metrics(documents)
        charts, graph_source_data = generate_charts(documents, charts_dir, config, evaluation_id, classification)
        dependencies = {}
        for name in ("fastapi", "numpy", "opencv-python-headless", "langdetect", "wordfreq"):
            try: dependencies[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: dependencies[name] = None
        report = {
            "evaluation_id": evaluation_id, "status": "COMPLETED", "started_at": started, "completed_at": utcnow(),
            "mode": "WITH_GROUND_TRUTH" if summary["documents_with_ground_truth"] == len(documents) else ("WITHOUT_GROUND_TRUTH" if summary["documents_with_ground_truth"] == 0 else "MIXED"),
            "input_signature": canonical_hash(signatures), "documents": documents, "summary": summary,
            "configuration": config, "input_signatures": signatures, "ground_truth_sources": gt_sources,
            "skipped_documents": skipped, "charts": charts, "graph_source_data": graph_source_data,
            "classification_evaluation": classification,
            "reproducibility": {"python": platform.python_version(), "platform": platform.platform(), "dependencies": dependencies},
            "output_directory": str(output_dir.resolve()),
        }
        (output_dir/"report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        (output_dir/"metrics.json").write_text(json.dumps(documents, indent=2, ensure_ascii=False), encoding="utf-8")
        (output_dir/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (output_dir/"graph_source_data.json").write_text(json.dumps(graph_source_data, indent=2, ensure_ascii=False), encoding="utf-8")
        self._csv(output_dir/"document_results.csv", documents)
        log_lines = [f"evaluation_id={evaluation_id}", f"started_at={started}", f"completed_at={report['completed_at']}", f"documents_found={len(runs)}", f"documents_processed={len(documents)}", f"documents_failed_or_skipped={len(skipped)}", f"documents_without_ground_truth={len(documents)-summary['documents_with_ground_truth']}", f"input_signature={report['input_signature']}", f"configuration={json.dumps(config, sort_keys=True)}", f"output_directory={output_dir.resolve()}"]
        log_lines.extend(f"skipped run_id={item['run_id']} reason={item['reason']}" for item in skipped)
        (output_dir/"evaluation.log").write_text("\n".join(log_lines)+"\n", encoding="utf-8")
        store.execute("INSERT INTO evaluations VALUES (?,?,?,?)", (evaluation_id, documents[0]["run_id"], started, json.dumps(report)))
        return report

    @staticmethod
    def _csv(path: Path, documents: list[dict[str, Any]]) -> None:
        fields = ["evaluation_id", "evaluation_timestamp", "document_id", "run_id", "filename", "file_type", "number_of_pages", "number_of_images_pages_processed", "ocr_engine", "extracted_character_count", "extracted_word_count", "ocr_confidence", "cer", "wer", "noise_ratio", "valid_word_ratio", "dictionary_match_ratio", "non_alphanumeric_ratio", "repeated_character_artifact_ratio", "average_word_length", "vocabulary_richness", "language_consistency", "detected_language", "text_density_characters_per_megapixel", "structural_consistency", "final_quality_score", "quality_class", "ground_truth_available", "input_signature"]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(documents)

    @staticmethod
    def _classification_metrics(documents: list[dict[str, Any]]) -> dict[str, Any] | None:
        labeled = [document for document in documents if document.get("quality_label") in {"High Quality", "Medium Quality", "Low Quality"}]
        labels = sorted({document["quality_label"] for document in labeled} | {document["quality_class"] for document in labeled})
        if len(labeled) < 2 or len(labels) < 2:
            return None
        matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
        for document in labeled: matrix[document["quality_label"]][document["quality_class"]] += 1
        per_class = {}
        total = len(labeled)
        for label in labels:
            tp = matrix[label][label]; fp = sum(matrix[actual][label] for actual in labels if actual != label)
            fn = sum(matrix[label][predicted] for predicted in labels if predicted != label); tn = total-tp-fp-fn
            precision = tp/(tp+fp) if tp+fp else None; recall = tp/(tp+fn) if tp+fn else None
            per_class[label] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": 2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else None, "specificity": tn/(tn+fp) if tn+fp else None, "sample_size": sum(matrix[label].values())}
        macro = {key: sum(row[key] for row in per_class.values() if row[key] is not None)/sum(row[key] is not None for row in per_class.values()) for key in ("precision", "recall", "f1", "specificity")}
        return {"sample_size": total, "labels": labels, "accuracy": sum(matrix[label][label] for label in labels)/total, "macro": macro, "per_class": per_class, "confusion_matrix": matrix, "roc_auc": None, "roc_note": "ROC/AUC omitted because threshold classes do not provide independent probabilistic class scores."}
