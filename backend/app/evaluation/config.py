from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(__file__).resolve().parents[3] / "backend" / "evaluation_config.json"


def load_evaluation_config(path: Path | None = None) -> dict[str, Any]:
    source = (path or CONFIG_PATH).resolve()
    config = json.loads(source.read_text(encoding="utf-8"))
    weights = config.get("quality_weights", {})
    if not weights or any(float(value) < 0 for value in weights.values()):
        raise ValueError("Evaluation quality weights must be present and non-negative")
    if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
        raise ValueError("Evaluation quality weights must sum to 1.0")
    thresholds = config.get("quality_thresholds", {})
    high, medium = float(thresholds.get("high", 80)), float(thresholds.get("medium", 60))
    if not 0 <= medium < high <= 100:
        raise ValueError("Quality thresholds must satisfy 0 <= medium < high <= 100")
    config["config_path"] = str(source)
    return deepcopy(config)
