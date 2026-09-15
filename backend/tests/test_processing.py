from pathlib import Path

import cv2
import numpy as np

from app.core.config import Settings
from app.pipeline.processing import (analyze_confidence, build_prompts,
    calibrated_paddle_confidence, corrected_text, detect_layout, evaluate_text,
    extract_graph, generate_candidates, graph_evidence, make_chunks, preprocess_page, reading_order,
    semantic_retrieve, validate_decisions)


def tokens():
    return [
        {"id":"t1","page":1,"text":"The","confidence":.99,"confidence_source":"fixture","bbox":[0,0,10,10]},
        {"id":"t2","page":1,"text":"govemment","confidence":.31,"confidence_source":"fixture","bbox":[11,0,40,10]},
        {"id":"t3","page":1,"text":"regulation","confidence":.98,"confidence_source":"fixture","bbox":[41,0,80,10]},
    ]


def test_preprocessing_and_layout(tmp_path: Path):
    source, target = tmp_path/"source.png", tmp_path/"out.png"
    image = np.full((300, 500, 3), 255, np.uint8)
    cv2.putText(image, "Research document", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.imwrite(str(source), image)
    detail = preprocess_page(source, target)
    assert target.exists() and "grayscale" in detail["operations"]
    regions, _, _ = detect_layout(target)
    assert regions and reading_order(regions)[0]["order"] == 1


def test_confidence_gate_and_candidates():
    values = tokens()
    analysis = analyze_confidence(values, .7)
    assert analysis["low_count"] == 1 and analysis["high_count"] == 2
    generated = generate_candidates(values, .7, "government", 5)
    assert len(generated) == 1
    assert "government" in [c["value"] for c in generated[0]["candidates"]]
    assert generated[0]["candidates"][-1]["value"] == "KEEP_ORIGINAL"


def test_paddle_line_score_is_calibrated_for_token_level_lexical_anomaly():
    effective, evidence = calibrated_paddle_confidence("govemment", .980749)
    assert effective < .9
    assert evidence and evidence["candidate"] == "government"
    unchanged, evidence = calibrated_paddle_confidence("government", .980749)
    assert unchanged == .980749 and evidence is None


def test_government_graph_has_meaningful_typed_relationships():
    values = [
        {"id": f"t{i}", "page": 1, "text": word, "confidence": .98, "bbox": [i, 0, i + 1, 1]}
        for i, word in enumerate("Public Administration Circular government ministry department authority issued policy public offices regulation implementation report GOV-2026-451 08 September 2026".split(), 1)
    ]
    nodes, edges = extract_graph(values, "doc1", "government", "circular.png", .9)
    assert {node["type"] for node in nodes} >= {"Document", "GovernmentBody", "Policy", "ReferenceIdentifier", "Date"}
    relation_types = {edge["type"] for edge in edges}
    assert {"ISSUED", "HAS_REFERENCE", "DATED", "GOVERNED_BY"} <= relation_types
    assert all(edge["type"] != "CO_OCCURS_IN_DOCUMENT" for edge in edges)


def test_retrieval_returns_real_chunk(monkeypatch):
    # Unit tests exercise the deterministic TF-IDF path; the downloaded BGE-M3
    # checkpoint is covered by the opt-in full integration smoke test.
    monkeypatch.setattr("app.pipeline.processing.dependency_available", lambda name: False)
    values=tokens(); candidates=generate_candidates(values,.7,"government",5)
    chunks=make_chunks(values,80)
    evidence,method,warnings=semantic_retrieve(values,candidates,chunks,Settings(retrieval_top_k=1))
    assert evidence[0]["results"][0]["text"] == "The regulation"
    assert evidence[0]["results"][0]["similarity"] >= 0
    assert method


def test_retrieval_prefers_low_memory_bge_gguf(monkeypatch, tmp_path: Path):
    model_path = tmp_path / "bge-m3-Q4_K_M.gguf"
    executable = tmp_path / "llama-server.exe"
    model_path.touch(); executable.touch()
    monkeypatch.setattr("app.pipeline.processing.dependency_available", lambda name: name == "faiss")
    monkeypatch.setattr(
        "app.pipeline.processing.embed_with_llama_cpp",
        lambda texts, model, server: [[1.0, 0.0] for _ in texts],
    )
    values = tokens(); candidates = generate_candidates(values, .7, "government", 5)
    chunks = make_chunks(values, 80)
    evidence, method, warnings = semantic_retrieve(
        values, candidates, chunks,
        Settings(
            retrieval_top_k=1,
            embedding_gguf_model_path=model_path,
            llama_server_executable=executable,
        ),
    )
    assert evidence and "BGE-M3 Q4_K_M" in method
    assert warnings == []


def test_prompt_validation_and_output():
    values=tokens(); candidates=generate_candidates(values,.7,"government",5)
    prompts=build_prompts(values,candidates,[],graph_evidence(candidates,[],[]))
    assert "Select exactly one" in prompts[0]["rendered"]
    decisions=[{"token_id":"t2","status":"COMPLETED","selection":"government","reason":"context","confidence":.9}]
    checked=validate_decisions(values,candidates,decisions)
    assert checked[0]["status"] == "ACCEPTED"
    original,corrected,records=corrected_text(values,checked)
    assert original == "The govemment regulation"
    assert corrected == "The government regulation" and records[0]["changed"]


def test_invalid_model_output_is_rejected():
    values=tokens(); candidates=generate_candidates(values,.7,"government",5)
    checked=validate_decisions(values,candidates,[{"token_id":"t2","status":"COMPLETED","selection":"invented"}])
    assert checked[0]["status"] == "REJECTED" and checked[0]["final"] == "govemment"


def test_metrics_are_deterministic():
    metrics=evaluate_text("helo world","hello world","hello world",[{"changed":True,"final":"hello"}])
    assert metrics["cer"] == 0 and metrics["wer"] == 0
    assert metrics["baseline_cer"] > 0
