from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


EVALUATION_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATION_ROOT.parent
MANIFEST_JSON = EVALUATION_ROOT / "dataset" / "manifests" / "dataset_manifest.json"
MANIFEST_CSV = EVALUATION_ROOT / "dataset" / "manifests" / "dataset_manifest.csv"
ACTUAL_RUNS = EVALUATION_ROOT / "actual_runs"
DEFAULT_IDS = [*(f"IMG{i:03d}" for i in range(1, 6)), *(f"PDF{i:03d}" for i in range(1, 6))]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def update_manifest_status(document_id: str) -> None:
    entries = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    matched = False
    for entry in entries:
        if entry["document_id"] == document_id:
            entry["execution_status"] = "ACTUALLY_RUN"
            matched = True
            break
    if not matched:
        raise KeyError(f"Manifest entry not found: {document_id}")
    write_json(MANIFEST_JSON, entries)

    with MANIFEST_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for row in rows:
        if row.get("document_id") == document_id:
            row["execution_status"] = "ACTUALLY_RUN"
    with MANIFEST_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def existing_success(run_dir: Path) -> bool:
    required = ("run_metadata.json", "ocr_raw.txt", "corrected.txt", "corrected.pdf", "pipeline.json")
    if not all((run_dir / name).is_file() for name in required):
        return False
    try:
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("status") == "ACTUALLY_RUN"


def domain_for(entry: dict[str, Any]) -> str:
    category = str(entry.get("document_category", "general")).lower()
    if category == "medical":
        return "medical"
    if any(label in category for label in ("invoice", "receipt", "financial")):
        return "financial"
    return "general"


def stage_progress(run: dict[str, Any]) -> str:
    stages = run.get("stages", [])
    completed = sum(stage.get("status") == "COMPLETED" for stage in stages)
    active = next((stage.get("name") for stage in stages if stage.get("status") == "RUNNING"), None)
    warnings = sum(stage.get("status") == "WARNING" for stage in stages)
    return f"{completed}/{len(stages)} completed; active={active or 'none'}; warnings={warnings}"


def process_document(
    client: httpx.Client,
    api_url: str,
    entry: dict[str, Any],
    threshold: float,
    timeout_seconds: int,
    force: bool,
    resume_run_id: str | None = None,
) -> bool:
    document_id = entry["document_id"]
    run_dir = ACTUAL_RUNS / document_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if existing_success(run_dir) and not force:
        print(f"[{document_id}] already completed; skipping", flush=True)
        update_manifest_status(document_id)
        return True

    input_path = PROJECT_ROOT / entry["input_path"]
    reference_path = PROJECT_ROOT / entry["ground_truth_reference_path"]
    if not input_path.is_file() or not reference_path.is_file():
        raise FileNotFoundError(f"Missing input or reference for {document_id}")

    domain = domain_for(entry)
    started_at = utc_now()
    start = time.monotonic()
    if resume_run_id:
        run_id = resume_run_id
        response = client.get(f"{api_url}/api/runs/{run_id}", timeout=120)
        response.raise_for_status()
        run = response.json()
        api_document_id = run["document_id"]
        print(f"[{document_id}] resuming pipeline run {run_id}", flush=True)
    else:
        print(f"[{document_id}] uploading {input_path.name} (domain={domain}, threshold={threshold:.2f})", flush=True)
        mime_type = mimetypes.guess_type(input_path.name)[0] or "application/octet-stream"
        with input_path.open("rb") as handle:
            response = client.post(
                f"{api_url}/api/documents/upload",
                files={"file": (input_path.name, handle, mime_type)},
                timeout=120,
            )
        response.raise_for_status()
        uploaded = response.json()
        api_document_id = uploaded.get("id") or uploaded.get("document_id")
        if not api_document_id:
            raise RuntimeError(f"Upload response has no document ID: {uploaded}")

        response = client.post(
            f"{api_url}/api/runs",
            json={
                "document_id": api_document_id,
                "confidence_threshold": threshold,
                "domain": domain,
                "start_stage": 1,
            },
            timeout=120,
        )
        response.raise_for_status()
        run = response.json()
        run_id = run["id"]
        write_json(run_dir / "active_run.json", {"run_id": run_id, "started_at": started_at})
        print(f"[{document_id}] pipeline run {run_id} started", flush=True)

    last_progress = ""
    while True:
        try:
            response = client.get(f"{api_url}/api/runs/{run_id}", timeout=120)
            response.raise_for_status()
            run = response.json()
        except httpx.HTTPError as exc:
            if time.monotonic() - start > timeout_seconds:
                raise TimeoutError(f"Pipeline exceeded {timeout_seconds} seconds") from exc
            print(f"[{document_id}] transient polling error ({type(exc).__name__}); retrying", flush=True)
            time.sleep(5)
            continue
        progress = stage_progress(run)
        if progress != last_progress:
            print(f"[{document_id}] {progress}", flush=True)
            last_progress = progress
        status = str(run.get("status", "")).upper()
        if status == "COMPLETED":
            break
        if status == "FAILED":
            raise RuntimeError(f"Pipeline failed: {json.dumps(run.get('stages', []), ensure_ascii=False)}")
        if time.monotonic() - start > timeout_seconds:
            raise TimeoutError(f"Pipeline exceeded {timeout_seconds} seconds")
        time.sleep(5)

    result = run.get("result", {})
    original_text = result.get("original_text")
    corrected_text = result.get("corrected_text")
    if not isinstance(original_text, str) or not original_text.strip():
        raise RuntimeError("Completed run did not contain non-empty original_text")
    if not isinstance(corrected_text, str) or not corrected_text.strip():
        raise RuntimeError("Completed run did not contain non-empty corrected_text")

    pdf_response = client.get(f"{api_url}/api/runs/{run_id}/export/pdf", timeout=180)
    pdf_response.raise_for_status()
    if not pdf_response.content.startswith(b"%PDF"):
        raise RuntimeError("Export endpoint did not return a PDF")

    elapsed = round(time.monotonic() - start, 3)
    artifacts = {
        "pipeline.json": run,
        "result.json": result,
        "corrections.json": result.get("traces", []),
        "evidence.json": {
            "semantic": result.get("semantic_evidence", []),
            "graph": result.get("graph_evidence", []),
            "combined": result.get("combined_evidence", []),
        },
        "graph.json": result.get("graph", {"nodes": [], "edges": []}),
    }
    for filename, value in artifacts.items():
        write_json(run_dir / filename, value)
    (run_dir / "ocr_raw.txt").write_text(original_text, encoding="utf-8")
    (run_dir / "corrected.txt").write_text(corrected_text, encoding="utf-8")
    (run_dir / "corrected.pdf").write_bytes(pdf_response.content)

    metadata = {
        "status": "ACTUALLY_RUN",
        "document_id": document_id,
        "api_document_id": api_document_id,
        "run_id": run_id,
        "input_filename": input_path.name,
        "input_sha256": sha256(input_path),
        "ground_truth_sha256": sha256(reference_path),
        "api_url": api_url,
        "domain": domain,
        "confidence_threshold": threshold,
        "started_at": started_at,
        "completed_at": utc_now(),
        "wall_runtime_seconds": elapsed,
        "pipeline_status": run.get("status"),
        "pipeline_stats": result.get("stats", {}),
        "stage_timings_ms": {
            stage.get("key", str(stage.get("number"))): stage.get("runtime_ms")
            for stage in run.get("stages", [])
        },
        "stage_warnings": [
            {
                "stage": stage.get("key"),
                "messages": stage.get("warnings", []),
                "error": stage.get("error"),
            }
            for stage in run.get("stages", [])
            if stage.get("status") == "WARNING"
        ],
        "artifacts": sorted([*artifacts, "ocr_raw.txt", "corrected.txt", "corrected.pdf"]),
    }
    write_json(run_dir / "run_metadata.json", metadata)
    (run_dir / "active_run.json").unlink(missing_ok=True)
    (run_dir / "run_failure.json").unlink(missing_ok=True)
    update_manifest_status(document_id)
    print(
        f"[{document_id}] completed in {elapsed:.1f}s; "
        f"tokens={result.get('stats', {}).get('ocr_tokens')}, "
        f"low-confidence={result.get('stats', {}).get('low_confidence_tokens')}, "
        f"corrections={result.get('stats', {}).get('corrections_applied')}",
        flush=True,
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the authorized 10-document OCR evaluation subset.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--ids", nargs="+", default=DEFAULT_IDS)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after a failed/timed-out document")
    parser.add_argument("--resume-run-id", help="Resume one existing API run; requires exactly one --ids value")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    entries = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    by_id = {entry["document_id"]: entry for entry in entries}
    unknown = [document_id for document_id in args.ids if document_id not in by_id]
    if unknown:
        parser.error(f"Unknown document IDs: {', '.join(unknown)}")
    unauthorized = [document_id for document_id in args.ids if document_id not in DEFAULT_IDS]
    if unauthorized:
        parser.error(f"This runner is restricted to the authorized subset: {', '.join(unauthorized)}")
    if args.resume_run_id and len(args.ids) != 1:
        parser.error("--resume-run-id requires exactly one document ID")

    health_error: Exception | None = None
    for _attempt in range(30):
        try:
            health = httpx.get(f"{args.api_url.rstrip('/')}/health", timeout=5)
            health.raise_for_status()
            print(f"API healthy: {health.json()}", flush=True)
            break
        except httpx.HTTPError as exc:
            health_error = exc
            time.sleep(2)
    else:
        raise RuntimeError(f"API did not become healthy within 60 seconds: {health_error}")
    failures: list[str] = []
    with httpx.Client(limits=httpx.Limits(max_keepalive_connections=0)) as client:
        for document_id in args.ids:
            try:
                process_document(
                    client,
                    args.api_url.rstrip("/"),
                    by_id[document_id],
                    args.threshold,
                    args.timeout_seconds,
                    args.force,
                    args.resume_run_id,
                )
            except Exception as exc:
                failures.append(document_id)
                run_dir = ACTUAL_RUNS / document_id
                run_dir.mkdir(parents=True, exist_ok=True)
                write_json(run_dir / "run_failure.json", {
                    "status": "FAILED",
                    "document_id": document_id,
                    "failed_at": utc_now(),
                    "reason": f"{type(exc).__name__}: {exc}",
                })
                print(f"[{document_id}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                if not args.continue_on_error:
                    break
    if failures:
        print(f"Failed documents: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"All requested documents completed: {', '.join(args.ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
