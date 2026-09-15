from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .config import get_settings


class Store:
    def __init__(self) -> None:
        url = get_settings().database_url
        path = Path(url.removeprefix("sqlite:///"))
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = threading.Lock()
        self._init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
              id TEXT PRIMARY KEY, name TEXT NOT NULL, safe_name TEXT NOT NULL,
              path TEXT NOT NULL, media_type TEXT NOT NULL, size INTEGER NOT NULL,
              pages INTEGER NOT NULL, created_at TEXT NOT NULL, is_demo INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS runs (
              id TEXT PRIMARY KEY, document_id TEXT NOT NULL, status TEXT NOT NULL,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL, config_json TEXT NOT NULL,
              result_json TEXT NOT NULL DEFAULT '{}', stages_json TEXT NOT NULL,
              FOREIGN KEY(document_id) REFERENCES documents(id)
            );
            CREATE TABLE IF NOT EXISTS evaluations (
              id TEXT PRIMARY KEY, run_id TEXT NOT NULL, created_at TEXT NOT NULL,
              metrics_json TEXT NOT NULL, FOREIGN KEY(run_id) REFERENCES runs(id)
            );
            """)

    def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        with self.lock, self.connect() as db:
            db.execute(sql, params)
            db.commit()

    def one(self, sql: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(sql, params).fetchone()
            return dict(row) if row else None

    def all(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [dict(row) for row in db.execute(sql, params).fetchall()]

    @staticmethod
    def decode(row: dict[str, Any]) -> dict[str, Any]:
        for key in ("config_json", "result_json", "stages_json", "metrics_json"):
            if key in row:
                row[key.removesuffix("_json")] = json.loads(row.pop(key) or "{}")
        if "is_demo" in row:
            row["is_demo"] = bool(row["is_demo"])
        return row


store = Store()
