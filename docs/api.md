# API

OpenAPI is served at `/docs`. Core endpoints include `POST /api/documents/upload`, `POST /api/documents/demo`, `POST /api/runs`, `GET /api/runs`, run stage/OCR/correction/evidence/graph/results resources, `GET /api/runs/{id}/events` (SSE), `POST /api/evaluation/run`, `POST /api/evaluation/dataset`, `GET /api/evaluation/{id}`, `GET /api/evaluation/{id}/charts/{name}`, `POST /api/experiments/run`, and `GET /api/system/status`. Errors use HTTP status codes and a `detail` string.
