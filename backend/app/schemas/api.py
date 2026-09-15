from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StageStatus(str, Enum):
    waiting = "WAITING"
    running = "RUNNING"
    completed = "COMPLETED"
    warning = "WARNING"
    failed = "FAILED"
    skipped = "SKIPPED"


class StageRecord(BaseModel):
    key: str
    number: int
    name: str
    status: StageStatus = StageStatus.waiting
    input: Any = None
    method: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    next_step: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    runtime_ms: float | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class RunCreate(BaseModel):
    document_id: str
    confidence_threshold: float = Field(default=0.90, ge=0, le=1)
    domain: str = "general"
    start_stage: int = Field(default=1, ge=1, le=15)


class EvaluationRequest(BaseModel):
    run_id: str
    ground_truth: str | None = None


class DatasetEvaluationRequest(BaseModel):
    run_ids: list[str] = Field(default_factory=list)
    ground_truths: dict[str, str] = Field(default_factory=dict)
    quality_labels: dict[str, str] = Field(default_factory=dict)


class ExperimentRequest(BaseModel):
    run_id: str
    ground_truth: str = Field(min_length=1)
