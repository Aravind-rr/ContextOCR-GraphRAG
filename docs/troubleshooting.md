# Troubleshooting

- **OCR unavailable:** install PaddleOCR or ensure `tesseract --version` works in the same terminal.
- **BGE/FAISS unavailable:** install `sentence-transformers` and `faiss-cpu`; otherwise the UI accurately labels TF-IDF fallback retrieval.
- **Neo4j unavailable:** verify the URI, credentials, server certificate/network, and `docker compose ps`.
- **Qwen unavailable:** set `SLM_MODEL_PATH` to readable local files and install compatible Torch/Transformers. The pipeline preserves originals on failure.
- **No low-confidence tokens:** digital PDF extraction reports source certainty 1.0, so use a scanned image to demonstrate OCR confidence gating.
- **Evaluation unavailable:** complete a run and paste manually verified ground truth.
