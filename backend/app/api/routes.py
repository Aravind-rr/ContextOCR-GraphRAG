from __future__ import annotations

import asyncio
import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from app.core.config import get_settings
from app.core.store import store
from app.pipeline.processing import dependency_available, evaluate_text
from app.pipeline.runner import initial_stages, pipeline_runner
from app.schemas.api import DatasetEvaluationRequest, EvaluationRequest, ExperimentRequest, RunCreate
from app.evaluation import EvaluationService
from app.services.local_models import llama_server

router = APIRouter(prefix="/api")
settings = get_settings()
evaluation_service = EvaluationService()
ALLOWED = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".tif": "image/tiff", ".tiff": "image/tiff"}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def document_response(row: dict[str, Any]) -> dict[str, Any]:
    doc = store.decode(row)
    doc["preview_url"] = f"/files/input/{doc['safe_name']}"
    return doc


def get_document(document_id: str) -> dict[str, Any]:
    row = store.one("SELECT * FROM documents WHERE id=?", (document_id,))
    if not row:
        raise HTTPException(404, "Document not found")
    return document_response(row)


def register_document(path: Path, original_name: str, media_type: str, is_demo: bool = False) -> dict[str, Any]:
    doc_id = str(uuid.uuid4())
    pages = 1
    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as pdf:
            pages = len(pdf)
    else:
        with Image.open(path) as image:
            pages = getattr(image, "n_frames", 1)
    created = utcnow()
    store.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?)", (doc_id, original_name, path.name, str(path.resolve()), media_type, path.stat().st_size, pages, created, int(is_demo)))
    return get_document(doc_id)


@router.post("/documents/upload", status_code=201)
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    original = Path(file.filename or "document").name
    suffix = Path(original).suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(415, f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED))}")
    doc_id = str(uuid.uuid4())
    safe_name = f"{doc_id}{suffix}"
    target = settings.data_dir / "input" / safe_name
    size = 0
    try:
        with target.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_mb * 1024 * 1024:
                    raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")
                output.write(chunk)
        # Decode to verify the extension is not merely renamed.
        if suffix == ".pdf":
            with fitz.open(target) as pdf: pages = len(pdf)
        else:
            with Image.open(target) as image:
                image.verify()
            with Image.open(target) as image:
                pages = getattr(image, "n_frames", 1)
    except HTTPException:
        target.unlink(missing_ok=True); raise
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(422, f"Invalid or corrupt document: {exc}") from exc
    created = utcnow()
    store.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?)", (doc_id, original, safe_name, str(target.resolve()), ALLOWED[suffix], size, pages, created, 0))
    return get_document(doc_id)


@router.post("/documents/demo", status_code=201)
def demo_document() -> dict[str, Any]:
    demo_id = str(uuid.uuid4())
    target = settings.data_dir / "input" / f"{demo_id}.png"
    image = Image.new("RGB", (1400, 1000), "white")
    draw = ImageDraw.Draw(image)
    font_candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    bold_candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    )
    font_path = next((path for path in font_candidates if path.is_file()), None)
    bold_path = next((path for path in bold_candidates if path.is_file()), font_path)
    font = ImageFont.truetype(str(font_path), 44) if font_path else ImageFont.load_default()
    bold = ImageFont.truetype(str(bold_path), 54) if bold_path else font
    draw.text((80, 70), "DEMO OCR RESEARCH FIXTURE", font=bold, fill="black")
    draw.text((80, 180), "Government Regulation Research Bulletin", font=font, fill="black")
    draw.text((80, 270), "University Department of Computer Science", font=font, fill="black")
    draw.text((80, 360), "Institutional policy and public administration", font=font, fill="black")

    # This deliberately noisy line is real raster input (not an injected OCR result).
    # It makes the confidence gate and constrained correction path reviewable.
    noisy = Image.new("L", (1150, 70), 255)
    noisy_draw = ImageDraw.Draw(noisy)
    noisy_draw.text((0, 3), "The govemment issued a revised regulation.", font=font, fill=0)
    noisy = ImageEnhance.Contrast(noisy.filter(ImageFilter.GaussianBlur(1))).enhance(.65)
    image.paste(Image.merge("RGB", (noisy, noisy, noisy)), (80, 500))
    draw.text((80, 650), "Reference number: OCR-2026-0917", font=font, fill="black")
    draw.text((80, 760), "Synthetic review document - not research evidence", font=font, fill=(80, 80, 80))
    image.save(target, format="PNG")
    return register_document(target, "demo-scanned-review-fixture.png", "image/png", True)


@router.get("/documents/{document_id}")
def document(document_id: str) -> dict[str, Any]:
    return get_document(document_id)


@router.post("/runs", status_code=202)
async def create_run(body: RunCreate, background: BackgroundTasks) -> dict[str, Any]:
    get_document(body.document_id)
    run_id, created = str(uuid.uuid4()), utcnow()
    config = body.model_dump()
    store.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)", (run_id, body.document_id, "QUEUED", created, created, json.dumps(config), "{}", json.dumps(initial_stages())))
    background.add_task(pipeline_runner.run, run_id)
    return pipeline_runner.load_run(run_id)


@router.get("/runs")
def list_runs() -> list[dict[str, Any]]:
    rows = store.all("SELECT r.*,d.name AS document_name,d.pages,d.is_demo FROM runs r JOIN documents d ON d.id=r.document_id ORDER BY r.created_at DESC")
    return [store.decode(row) for row in rows]


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    try: return pipeline_runner.load_run(run_id)
    except KeyError: raise HTTPException(404, "Run not found")


@router.get("/runs/{run_id}/pipeline")
def pipeline(run_id: str) -> list[dict[str, Any]]:
    return get_run(run_id)["stages"]


@router.get("/runs/{run_id}/stages/{stage}")
def stage(run_id: str, stage: str) -> dict[str, Any]:
    match = next((s for s in get_run(run_id)["stages"] if s["key"] == stage or str(s["number"]) == stage), None)
    if not match: raise HTTPException(404, "Stage not found")
    return match


def result_part(run_id: str, key: str, default: Any) -> Any:
    return get_run(run_id)["result"].get(key, default)


@router.get("/runs/{run_id}/ocr")
def ocr(run_id: str) -> Any: return result_part(run_id, "tokens", [])
@router.get("/runs/{run_id}/tokens")
def tokens(run_id: str) -> Any: return result_part(run_id, "tokens", [])
@router.get("/runs/{run_id}/corrections")
def corrections(run_id: str) -> Any: return result_part(run_id, "traces", [])
@router.get("/runs/{run_id}/evidence")
def evidence(run_id: str) -> Any: return {"semantic": result_part(run_id, "semantic_evidence", []), "graph": result_part(run_id, "graph_evidence", []), "combined": result_part(run_id, "combined_evidence", [])}
@router.get("/runs/{run_id}/graph")
def graph(run_id: str) -> Any: return result_part(run_id, "graph", {"nodes": [], "edges": []})
@router.get("/runs/{run_id}/results")
def results(run_id: str) -> Any: return get_run(run_id)["result"]


@router.get("/runs/{run_id}/export/pdf")
def export_pdf(run_id: str) -> FileResponse:
    run = get_run(run_id)
    raw_path = run["result"].get("export_pdf_path")
    if not raw_path:
        raise HTTPException(409, "Searchable PDF output is not available yet")
    path = Path(raw_path).resolve()
    output_dir = (settings.data_dir / "output").resolve()
    if path.parent != output_dir or not path.is_file():
        raise HTTPException(404, "Searchable PDF output was not found")
    source_name = Path(run["result"].get("document", {}).get("name", "document")).stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", source_name).strip("-._") or "document"
    return FileResponse(path, media_type="application/pdf", filename=f"{safe_stem}-corrected.pdf")


@router.get("/runs/{run_id}/events")
async def events(run_id: str) -> StreamingResponse:
    get_run(run_id)
    queue = pipeline_runner.subscribe(run_id)
    async def stream():
        yield f"data: {json.dumps({'type':'snapshot','run':get_run(run_id)})}\n\n"
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=20)
                yield f"data: {json.dumps(event)}\n\n"
                if event["type"] == "complete": break
            except asyncio.TimeoutError:
                latest = get_run(run_id)
                yield ": keep-alive\n\n"
                if latest["status"] in ("COMPLETED", "FAILED"): break
    return StreamingResponse(stream(), media_type="text/event-stream")


@router.post("/evaluation/run", status_code=201)
def evaluate(body: EvaluationRequest) -> dict[str, Any]:
    try:
        return evaluation_service.run(
            [body.run_id],
            {body.run_id: body.ground_truth} if body.ground_truth and body.ground_truth.strip() else {},
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/evaluation/dataset", status_code=201)
def evaluate_dataset(body: DatasetEvaluationRequest) -> dict[str, Any]:
    try:
        return evaluation_service.run(body.run_ids or None, body.ground_truths, body.quality_labels)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/evaluation/{evaluation_id}")
def evaluation(evaluation_id: str) -> dict[str, Any]:
    row = store.one("SELECT * FROM evaluations WHERE id=?", (evaluation_id,))
    if not row: raise HTTPException(404, "Evaluation not found")
    decoded = store.decode(row)
    return decoded.get("metrics", decoded)


@router.get("/evaluation/{evaluation_id}/charts/{chart_name}")
def evaluation_chart(evaluation_id: str, chart_name: str) -> FileResponse:
    report = evaluation(evaluation_id)
    chart = next((item for item in report.get("charts", []) if item["name"] == chart_name), None)
    if not chart:
        raise HTTPException(404, "Chart is unavailable for this evaluation run")
    path = (Path(report["output_directory"]) / "charts" / chart["file"]).resolve()
    expected = (Path(report["output_directory"]) / "charts").resolve()
    if path.parent != expected or not path.is_file():
        raise HTTPException(404, "Chart file was not found")
    return FileResponse(path, media_type="image/svg+xml")


@router.post("/experiments/run")
def experiments(body: ExperimentRequest) -> dict[str, Any]:
    run = get_run(body.run_id); result = run["result"]
    if not result.get("original_text"): raise HTTPException(409, "Pipeline output is not available")
    baseline = evaluate_text(result["original_text"], result["original_text"], body.ground_truth, [])
    full = evaluate_text(result["original_text"], result["corrected_text"], body.ground_truth, result.get("correction_records", []))
    unavailable = "Not executed: this strategy requires a separately configured implementation/model."
    names = ["Traditional Correction", "Unrestricted SLM", "Confidence Gate Only", "Confidence + SLM", "Confidence + FAISS", "Confidence + GraphRAG"]
    return {"run_id": body.run_id, "experiments": [{"name": "OCR Baseline", "status": "COMPLETED", "metrics": baseline}, *[{"name": n, "status": "UNAVAILABLE", "reason": unavailable, "metrics": None} for n in names], {"name": "Full Proposed", "status": "COMPLETED", "metrics": full}]}


def status_item(ready: bool, detail: str) -> dict[str, str]:
    return {"status": "READY" if ready else "UNAVAILABLE", "detail": detail}


@router.get("/system/status")
def system_status() -> dict[str, Any]:
    tesseract = shutil.which("tesseract")
    paddle = dependency_available("paddleocr") and dependency_available("paddle")
    embedding_path = Path(settings.embedding_model)
    embedding_gguf_path = Path(settings.embedding_gguf_model_path or "")
    llama_executable = Path(settings.llama_server_executable or "")
    bge_weights = any(
        (embedding_path / name).is_file()
        for name in ("pytorch_model.bin", "model.safetensors", "model.safetensors.index.json")
    )
    bge_gguf_ready = embedding_gguf_path.is_file() and llama_executable.is_file()
    faiss_ready = dependency_available("faiss")
    bge_ready = faiss_ready
    bge_ready = bge_ready and (bge_gguf_ready or (dependency_available("sentence_transformers") and bge_weights))
    layout_path = Path(settings.doc_layout_model_path or "")
    layout_ready = dependency_available("doclayout_yolo") and layout_path.is_file()
    neo_ready, neo_detail = False, "NEO4J_URI is not configured"
    if settings.neo4j_uri:
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)); driver.verify_connectivity(); driver.close()
            neo_ready, neo_detail = True, "Connected and verified"
        except Exception as exc: neo_detail = f"Connection unavailable: {type(exc).__name__}"
    slm_ready, slm_detail = llama_server.configured(settings)
    slm_status = "RUNNING" if llama_server.is_running(settings) else ("READY" if slm_ready else "UNAVAILABLE")
    runs = [store.decode(r) for r in store.all("SELECT result_json FROM runs WHERE status='COMPLETED'")]
    totals = {"documents_processed": len(runs), "ocr_tokens": sum(r["result"].get("stats", {}).get("ocr_tokens", 0) for r in runs), "low_confidence_tokens": sum(r["result"].get("stats", {}).get("low_confidence_tokens", 0) for r in runs), "corrections_applied": sum(r["result"].get("stats", {}).get("corrections_applied", 0) for r in runs), "corrections_rejected": sum(r["result"].get("stats", {}).get("corrections_rejected", 0) for r in runs), "slm_calls": sum(r["result"].get("stats", {}).get("slm_calls", 0) for r in runs)}
    return {"components": {
        "backend": status_item(True, "FastAPI process responding"),
        "ocr_engine": status_item(paddle or bool(tesseract), f"PaddleOCR {settings.paddle_ocr_version} mobile models on {settings.paddle_device}" if paddle else (f"Tesseract fallback: {tesseract}" if tesseract else "Install PaddleOCR or Tesseract")),
        "layout_model": status_item(layout_ready, f"DocLayout-YOLO: {layout_path.name}" if layout_ready else "DocLayout-YOLO checkpoint/package unavailable; OpenCV fallback enabled"),
        "embedding_model": status_item(bge_ready, f"BGE-M3 Q4_K_M via local llama.cpp: {embedding_gguf_path.name}" if bge_gguf_ready else (f"BGE-M3 local checkpoint: {settings.embedding_model}" if bge_ready else "BGE-M3 checkpoint/dependencies unavailable; TF-IDF fallback enabled")),
        "faiss": status_item(faiss_ready, "FAISS IndexFlatIP available" if faiss_ready else "faiss module is not installed"),
        "neo4j": status_item(neo_ready, neo_detail),
        "qwen": {"status": slm_status, "detail": slm_detail},
    }, "totals": totals, "privacy": "Core OCR correction pipeline runs locally."}
