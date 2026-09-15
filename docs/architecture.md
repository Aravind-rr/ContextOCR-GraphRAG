# Architecture

The browser calls a FastAPI service; no document content leaves the local deployment. SQLite stores document/run/evaluation metadata. Page images and immutable exports live under `data`. The runner persists every stage transition before publishing it over Server-Sent Events, allowing completed stages to be inspected while later work continues.

Optional adapters are capability checked: PaddleOCR/Tesseract for OCR, BGE-M3/FAISS or TF-IDF for retrieval, Neo4j for graph persistence, and a local Transformers directory for Qwen2.5-3B. A missing adapter yields a warning or explicit unavailable state rather than synthetic output.

