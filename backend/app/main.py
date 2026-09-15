from __future__ import annotations

from contextlib import asynccontextmanager
import json

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import get_settings
from app.core.store import store
from app.services.local_models import llama_server

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # A process termination cannot finish an in-flight background task. Make
    # that state explicit on restart so history never claims it is still live.
    for row in store.all("SELECT id,stages_json FROM runs WHERE status IN ('RUNNING','QUEUED')"):
        stages = json.loads(row["stages_json"])
        for stage in stages:
            if stage["status"] == "RUNNING":
                stage["status"] = "FAILED"
                stage["error"] = "Backend process stopped before this stage completed."
        store.execute(
            "UPDATE runs SET status='FAILED',stages_json=? WHERE id=?",
            (json.dumps(stages), row["id"]),
        )
    yield
    llama_server.stop()


app = FastAPI(title=settings.app_name, version="1.0.0", description="Local, confidence-aware OCR post-correction research system", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(router)
app.mount("/files/input", StaticFiles(directory=settings.data_dir / "input"), name="input-files")
app.mount("/files/processed", StaticFiles(directory=settings.data_dir / "processed"), name="processed-files")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": "Internal processing error", "type": type(exc).__name__})
