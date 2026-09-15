"""Verify one complete constrained correction against the configured local Qwen model."""

from app.core.config import get_settings
from app.pipeline.processing import (
    build_prompts,
    extract_graph,
    generate_candidates,
    graph_evidence,
    run_slm,
    validate_decisions,
)


tokens = [
    {"id": "t1", "page": 1, "text": "Government", "confidence": .99, "confidence_source": "smoke", "bbox": [0, 0, 10, 10]},
    {"id": "t2", "page": 1, "text": "govemment", "confidence": .45, "confidence_source": "smoke", "bbox": [11, 0, 30, 10]},
    {"id": "t3", "page": 1, "text": "regulation", "confidence": .99, "confidence_source": "smoke", "bbox": [31, 0, 50, 10]},
]
candidates = generate_candidates(tokens, .70, "government", 5)
entities, relationships = extract_graph(tokens, "qwen-smoke")
graph = graph_evidence(candidates, entities, relationships)
prompts = build_prompts(tokens, candidates, [], graph)
decisions, method, warnings = run_slm(prompts, get_settings())
validated = validate_decisions(tokens, candidates, decisions)
print({"method": method, "warnings": warnings, "decision": decisions[0], "validation": validated[0]})
if decisions[0]["status"] != "COMPLETED":
    raise SystemExit("Local Qwen call did not complete")
if validated[0]["final"] != "government":
    raise SystemExit("Local Qwen did not select the expected constrained candidate")
