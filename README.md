# ContextOCR GraphRAG Workbench

Production-oriented final-year CSE capstone implementation for **Context-Aware Zero-Shot Prompt Engineering for Small Language Model Post-Correction Using GraphRAG**.

The application runs a document through visible, inspectable stages: ingestion → conditional preprocessing → layout → reading order → OCR → confidence gate → candidates → semantic evidence → knowledge graph → hybrid GraphRAG → dynamic prompt → local Qwen selection → validation → output. It never sends document contents to a cloud LLM.

## What works without heavyweight models

- PDF, PNG, JPEG and TIFF upload with type, size and path-traversal protection.
- PDF rendering and embedded digital-PDF text extraction with PyMuPDF.
- Conditional OpenCV preprocessing with stored intermediates.
- Real OpenCV layout-region fallback and geometric reading order.
- Tesseract OCR fallback when the executable is installed.
- Confidence distribution/gating, lexical candidates, TF-IDF retrieval fallback, in-run graph extraction, dynamic prompts, strict validation, SQLite run history, SSE updates, JSON export and ground-truth evaluation.

Fallbacks are labeled in the pipeline. Unavailable components never emit fabricated status, metrics, model selections, or evidence.

## Architecture

`frontend` is a strict TypeScript React/Vite client. `backend` is FastAPI with a 15-stage asynchronous pipeline and SQLite metadata. Intermediate images and JSON exports live under `data`. Neo4j, PaddleOCR, DocLayout-YOLO, BGE-M3/FAISS, and Qwen run locally in the review configuration. Heavy inference stages use disposable worker processes so the full pipeline fits on a 16 GB review machine. See [architecture](docs/architecture.md) and [pipeline](docs/pipeline.md).

## Local setup

Prerequisites: Python 3.11, Node 20+, and optionally Tesseract. Copy `.env.example` to `.env` and adjust paths.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
$env:PYTHONPATH="backend"
uvicorn app.main:app --reload --port 8000
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. API documentation is at `http://localhost:8000/docs`.

## Heavy/local model setup

- **PaddleOCR:** install a compatible local `paddlepaddle` and `paddleocr`; set `OCR_ENGINE=paddle`. Without it, the backend uses an installed Tesseract executable or reports OCR unavailable.
- **BGE-M3 + FAISS:** the review configuration uses the local Q4_K_M BGE-M3 GGUF through `llama.cpp` and FAISS (`EMBEDDING_GGUF_MODEL_PATH`). The short-lived embedding server is stopped before Qwen starts. The PyTorch checkpoint remains a secondary path; TF-IDF is used only as an explicitly labelled fallback.
- **Neo4j:** set `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD`. Connectivity is verified on every status request. The extracted graph remains inspectable in-run if persistence is unavailable.
- **DocLayout-YOLO:** install `doclayout-yolo` with `python -m pip install doclayout-yolo==0.0.4 --no-deps` after the main requirements, download the DocStructBench checkpoint, and set `DOC_LAYOUT_MODEL_PATH`.
- **Qwen2.5-3B:** download the official Q4_K_M GGUF plus a local `llama-server` binary, then set `SLM_MODEL_PATH` and `LLAMA_SERVER_EXECUTABLE`. The backend starts and stops the local server automatically. No cloud API is called.
- **Memory-efficient BGE-M3:** after downloading the official checkpoint to `models/bge-m3`, run `python backend/scripts/convert_bge_m3_fp16.py`. This creates the same BGE-M3 weights as FP16 safetensor shards at `models/bge-m3-fp16` for the 16 GB review machine.

## Demo flow

1. Open Overview and inspect verified component status.
2. Choose **Process document**, upload a real file, set domain and confidence threshold, then run.
3. Open stages as SSE events arrive. Inspect OCR tokens, confidence histogram, correction trace, evidence, graph and output tabs.
4. For evaluation, open Evaluation and run reference-free quality analysis, or paste verified ground truth to additionally calculate CER/WER. Current run metadata, tables, dynamic charts and output paths are displayed.
5. Use **demo data** only for dependency testing. It is visibly tagged and never mixed with research evaluation.

## Faculty demonstration documents

Six polished, synthetic scanned PDFs are available in `output/pdf`, one for every domain in the UI. Four contain intentional OCR-like spelling errors and two are clean controls. None contains an embedded text layer, so the real preprocessing and PaddleOCR path is exercised. Open `output/pdf/DEMO_GUIDE.md` for the expected corrections and thresholds verified on this machine.

## Tests and builds

```powershell
$env:PYTHONPATH="backend"
pytest
cd frontend
npm run build
```

The integration test creates its own deterministic PDF fixture and runs all 15 stages without downloading models. See [evaluation](docs/evaluation.md) for metric definitions.

## Docker

Copy `.env.example` to `.env`, set a non-default `NEO4J_PASSWORD`, then run:

```powershell
docker compose up --build
```

The UI is on port 5173, API on 8000, and Neo4j Browser on 7474. Model directories are mounted read-only at `/models`; set `SLM_MODEL_PATH=/models/<directory>`. Large ML models are intentionally not baked into images.

## Troubleshooting

The System Status and per-stage Diagnostics panels are the source of truth. See [troubleshooting](docs/troubleshooting.md) for common local-model and OCR issues. Configuration is documented in [models](docs/models.md), and endpoints in [API](docs/api.md).
