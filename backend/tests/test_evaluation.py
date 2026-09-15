import json
import uuid
from pathlib import Path

from app.core.store import store
from app.evaluation.config import load_evaluation_config
from app.evaluation.metrics import (
    classify_quality,
    edit_counts,
    eligible_word_data,
    error_rate,
    evaluate_document,
    noise_metrics,
    normalize_text,
    quality_score,
)
from app.evaluation.service import EvaluationService


def config():
    return load_evaluation_config()


def fixture_run(text: str, confidence: float = .95):
    token_values = text.split()
    tokens = [
        {"id": f"t{i}", "page": 1, "text": value, "confidence": confidence,
         "ocr_confidence": confidence, "confidence_source": "PaddleOCR line score",
         "bbox": [i * 10, 0, i * 10 + 8, 10], "line_index": 0}
        for i, value in enumerate(token_values)
    ]
    return {
        "id": "run", "status": "COMPLETED", "stages": [{"key": "ocr", "method": "PaddleOCR test fixture"}],
        "result": {"original_text": text, "corrected_text": text, "tokens": tokens,
                   "preprocessing": [{"width": 1000, "height": 1000}], "correction_records": []},
    }


def fixture_document(tmp_path: Path):
    path = tmp_path / "document.png"; path.write_bytes(b"current-input")
    return {"id": "doc", "name": "document.png", "media_type": "image/png", "pages": 1, "path": str(path)}


def test_cer_and_wer_use_reference_denominators():
    cer, char_counts = error_rate("kitten", "sitting")
    wer, word_counts = error_rate(["the", "cat"], ["the", "dog"])
    assert char_counts["distance"] == 3 and cer == 3 / 7
    assert word_counts["substitutions"] == 1 and wer == .5


def test_noise_and_artifact_ratios_are_actual_character_counts():
    values = noise_metrics("clean #### coooool \ufffd", config())
    assert values["noise_character_count"] >= 9
    assert values["artifact_character_count"] >= 8
    assert 0 < values["noise_ratio"] <= 1


def test_valid_words_dictionary_and_vocabulary_richness():
    values = eligible_word_data("government policy policy xqzv", config())
    assert values["valid_word_ratio"] == .75
    assert values["dictionary_match_ratio"] == .75
    assert values["vocabulary_richness"] == .75


def test_normalization_and_special_characters():
    assert normalize_text(" cafe\u0301  \r\n policy ") == "café \n policy"
    values = noise_metrics("café — policy", config())
    assert values["noise_ratio"] == 0


def test_weighted_score_renormalizes_missing_components():
    score, active = quality_score({"valid_word_ratio": 1.0, "noise_cleanliness": .5, "cer": None}, config())
    assert round(sum(active.values()), 10) == 1
    assert 50 < score < 100


def test_quality_threshold_classification():
    thresholds = config()["quality_thresholds"]
    assert classify_quality(80, thresholds) == "High Quality"
    assert classify_quality(60, thresholds) == "Medium Quality"
    assert classify_quality(59.999, thresholds) == "Low Quality"


def test_missing_ground_truth_never_fabricates_cer_or_wer(tmp_path: Path):
    result = evaluate_document(fixture_run("A clear government policy document"), fixture_document(tmp_path), None, config())
    assert result["ground_truth_available"] is False
    assert result["cer"] is None and result["wer"] is None
    assert 0 <= result["final_quality_score"] <= 100


def test_ground_truth_metrics_and_improvement(tmp_path: Path):
    run = fixture_run("The government policy")
    run["result"]["original_text"] = "The govemment policy"
    result = evaluate_document(run, fixture_document(tmp_path), "The government policy", config())
    assert result["cer"] == 0 and result["wer"] == 0
    assert result["baseline_cer"] > 0 and result["correction_improvement"]["cer_absolute_reduction"] > 0


def test_empty_document_is_low_quality_without_fake_missing_metrics(tmp_path: Path):
    result = evaluate_document(fixture_run(""), fixture_document(tmp_path), None, config())
    assert result["extracted_character_count"] == 0
    assert result["ocr_confidence"] is None and result["valid_word_ratio"] is None
    assert result["final_quality_score"] == 0 and result["quality_class"] == "Low Quality"


def test_multilingual_text_is_handled_deterministically(tmp_path: Path):
    cfg = config(); cfg["expected_language"] = "fr"
    text = "Le gouvernement publie une nouvelle politique administrative pour les citoyens."
    first = evaluate_document(fixture_run(text), fixture_document(tmp_path), None, cfg)
    second = evaluate_document(fixture_run(text), fixture_document(tmp_path), None, cfg)
    assert first["language_consistency"] == second["language_consistency"]
    assert first["detected_language"] == "fr"


def test_high_quality_scores_above_ocr_garbage(tmp_path: Path):
    high = evaluate_document(fixture_run("The government published a clear administrative policy for all public offices."), fixture_document(tmp_path), None, config())
    low = evaluate_document(fixture_run("#### zzzz \ufffd @@ xx qqqq", .2), fixture_document(tmp_path), None, config())
    assert high["final_quality_score"] > low["final_quality_score"]


def write_service_config(tmp_path: Path) -> Path:
    cfg = config(); cfg.pop("config_path", None)
    cfg["output_directory"] = str(tmp_path / "evaluations")
    cfg["ground_truth_directory"] = str(tmp_path / "ground_truth")
    path = tmp_path / "evaluation_config.json"; path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def insert_completed_run(tmp_path: Path, document_id: str, run_id: str, text: str, updated: str):
    source = tmp_path / f"{document_id}.png"; source.write_bytes(text.encode())
    store.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?)", (document_id, f"{document_id}.png", source.name, str(source), "image/png", source.stat().st_size, 1, updated, 0))
    run = fixture_run(text); run["id"] = run_id
    store.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)", (run_id, document_id, "COMPLETED", updated, updated, "{}", json.dumps(run["result"]), json.dumps(run["stages"])))


def test_evaluation_run_outputs_and_stale_prevention(tmp_path: Path):
    document_id, run_id = str(uuid.uuid4()), str(uuid.uuid4())
    insert_completed_run(tmp_path, document_id, run_id, "The government policy is clear and reproducible.", "2026-01-01T00:00:00+00:00")
    service = EvaluationService(write_service_config(tmp_path))
    first = service.run([run_id]); second = service.run([run_id])
    assert first["evaluation_id"] != second["evaluation_id"]
    assert first["input_signature"] == second["input_signature"]
    assert Path(first["output_directory"], "document_results.csv").is_file()
    assert Path(first["output_directory"], "charts", first["charts"][0]["file"]).is_file()
    changed = store.decode(store.one("SELECT * FROM runs WHERE id=?", (run_id,)))
    changed["result"]["corrected_text"] += " Updated"
    store.execute("UPDATE runs SET result_json=? WHERE id=?", (json.dumps(changed["result"]), run_id))
    third = service.run([run_id])
    assert third["input_signature"] != first["input_signature"]


def test_duplicate_document_runs_keep_latest_only(tmp_path: Path):
    document_id, older, newer = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
    insert_completed_run(tmp_path, document_id, older, "Older valid document text", "2026-01-01T00:00:00+00:00")
    old = store.one("SELECT * FROM runs WHERE id=?", (older,))
    store.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)", (newer, document_id, "COMPLETED", "2026-01-02T00:00:00+00:00", "2026-01-02T00:00:00+00:00", old["config_json"], old["result_json"], old["stages_json"]))
    report = EvaluationService(write_service_config(tmp_path)).run([older, newer])
    assert len(report["documents"]) == 1 and report["documents"][0]["run_id"] == newer
    assert any("duplicate" in item["reason"] for item in report["skipped_documents"])
