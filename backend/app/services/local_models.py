from __future__ import annotations

import json
import gc
import os
import subprocess
import sys
import tempfile
import threading
import time
import socket
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings


@lru_cache(maxsize=2)
def get_embedding_model(model_path: str):
    import torch
    from sentence_transformers import SentenceTransformer

    fp16_checkpoint = (Path(model_path) / "model.safetensors.index.json").is_file()
    model_kwargs = {"dtype": torch.float16, "low_cpu_mem_usage": False} if fp16_checkpoint else None
    return SentenceTransformer(model_path, device="cpu", model_kwargs=model_kwargs, local_files_only=Path(model_path).exists())


@lru_cache(maxsize=2)
def get_paddle_ocr(lang: str, version: str, device: str):
    from paddleocr import PaddleOCR

    return PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        return_word_box=True,
        device=device,
        enable_mkldnn=False,
    )


@lru_cache(maxsize=2)
def get_layout_model(model_path: str):
    from doclayout_yolo import YOLOv10

    return YOLOv10(model_path)


def release_layout_model() -> None:
    get_layout_model.cache_clear()
    gc.collect()


def release_paddle_ocr() -> None:
    get_paddle_ocr.cache_clear()
    gc.collect()


def release_embedding_model() -> None:
    get_embedding_model.cache_clear()
    gc.collect()


def run_model_worker(payload: dict[str, Any], timeout: int = 240) -> Any:
    """Run a memory-heavy local model in a disposable child process."""
    descriptor, temporary_name = tempfile.mkstemp(prefix="contextocr-worker-", suffix=".json")
    os.close(descriptor)
    output_path = Path(temporary_name)
    try:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        completed = subprocess.run(
            [sys.executable, "-m", "app.services.model_worker", str(output_path)],
            input=json.dumps(payload),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            creationflags=creationflags,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
            raise RuntimeError(f"Isolated {payload.get('action')} worker failed with exit code {completed.returncode}: {detail[-1600:]}")
        return json.loads(output_path.read_text(encoding="utf-8"))
    finally:
        output_path.unlink(missing_ok=True)


def embed_with_llama_cpp(
    texts: list[str],
    model_path: Path,
    executable: Path,
    timeout: int = 90,
) -> list[list[float]]:
    """Create normalized BGE-M3 embeddings with a short-lived GGUF server.

    The quantized model avoids the large transient memory allocation made by
    PyTorch while loading BGE-M3 on Windows. A free ephemeral port is selected
    so this worker cannot conflict with the API or Qwen server ports.
    """
    if not texts:
        return []
    model_path = model_path.resolve()
    executable = executable.resolve()
    if not model_path.is_file():
        raise RuntimeError(f"Embedding GGUF is missing: {model_path}")
    if not executable.is_file():
        raise RuntimeError(f"llama.cpp server is missing: {executable}")

    # Keep the worker in the application's known local port range. On some
    # Windows installations a randomly assigned high port can fall inside a
    # reserved/excluded range and llama.cpp then exits while binding it.
    port = None
    for candidate_port in range(8092, 8102):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.bind(("127.0.0.1", candidate_port))
            port = candidate_port
            break
        except OSError:
            continue
    if port is None:
        raise RuntimeError("No free local port is available for BGE-M3 (8092-8101)")

    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        [
            str(executable), "-m", str(model_path), "--embedding",
            "--pooling", "cls", "--host", "127.0.0.1", "--port", str(port),
            "-ngl", "0", "-c", "512", "-np", "1", "-b", "256", "-ub", "256",
            "--no-webui", "--log-disable",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    try:
        deadline = time.monotonic() + timeout
        health_url = f"http://127.0.0.1:{port}/health"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"BGE-M3 llama.cpp server exited with code {process.returncode}")
            try:
                with urllib.request.urlopen(health_url, timeout=1) as response:
                    if json.loads(response.read().decode("utf-8")).get("status") == "ok":
                        break
            except (OSError, ValueError, urllib.error.URLError):
                time.sleep(0.2)
        else:
            raise RuntimeError("Timed out while loading the BGE-M3 GGUF model")

        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/embeddings",
            data=json.dumps({"model": "bge-m3", "input": texts}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        ordered = sorted(payload["data"], key=lambda item: int(item["index"]))
        return [[float(value) for value in item["embedding"]] for item in ordered]
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()


class LlamaServerManager:
    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.lock = threading.Lock()
        self.last_error: str | None = None

    @staticmethod
    def _request(url: str, *, payload: dict[str, Any] | None = None, timeout: float = 3, api_key: str | None = None) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Local llama.cpp HTTP {exc.code}: {body[:800]}") from exc

    def is_running(self, settings: Settings | None = None) -> bool:
        config = settings or get_settings()
        try:
            return self._request(f"{config.llama_server_url.rstrip('/')}/health", timeout=1).get("status") == "ok"
        except Exception:
            return False

    def configured(self, settings: Settings | None = None) -> tuple[bool, str]:
        config = settings or get_settings()
        model = Path(config.slm_model_path or "")
        executable = Path(config.llama_server_executable or "")
        if not model.is_file():
            return False, "SLM_MODEL_PATH does not point to a GGUF model"
        if not executable.is_file():
            return False, "LLAMA_SERVER_EXECUTABLE is unavailable"
        return True, f"{model.name} via local llama.cpp"

    def ensure_running(self, settings: Settings | None = None, timeout: int = 90) -> None:
        config = settings or get_settings()
        if self.is_running(config):
            return
        ready, detail = self.configured(config)
        if not ready:
            raise RuntimeError(detail)
        with self.lock:
            if self.is_running(config):
                return
            executable = str(Path(config.llama_server_executable or "").resolve())
            model = str(Path(config.slm_model_path or "").resolve())
            port = config.llama_server_url.rstrip("/").rsplit(":", 1)[-1]
            args = [executable, "-m", model, "--host", "127.0.0.1", "--port", port,
                    "-ngl", str(config.slm_gpu_layers), "-c", str(config.slm_context_size),
                    "--jinja", "--no-webui"]
            api_key = os.getenv("LLAMA_SERVER_API_KEY")
            if api_key:
                args.extend(["--api-key", api_key])
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self.process = subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                creationflags=creationflags,
            )
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    self.last_error = f"llama-server exited with code {self.process.returncode}"
                    raise RuntimeError(self.last_error)
                if self.is_running(config):
                    self.last_error = None
                    return
                time.sleep(0.5)
            self.last_error = "Timed out waiting for local Qwen server"
            raise RuntimeError(self.last_error)

    def complete(self, prompt: str, settings: Settings | None = None) -> tuple[str, dict[str, Any]]:
        config = settings or get_settings()
        self.ensure_running(config)
        payload = {
            "model": "qwen2.5-3b-instruct",
            "messages": [
                {"role": "system", "content": "You are a constrained OCR correction selector. Return valid JSON only; never add candidates."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": config.slm_max_new_tokens,
            "response_format": {"type": "json_object"},
        }
        started = time.perf_counter()
        response = self._request(
            f"{config.llama_server_url.rstrip('/')}/v1/chat/completions",
            payload=payload,
            timeout=180,
            api_key=os.getenv("LLAMA_SERVER_API_KEY"),
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        content = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})
        return content, {"latency_ms": latency_ms, "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None


llama_server = LlamaServerManager()
