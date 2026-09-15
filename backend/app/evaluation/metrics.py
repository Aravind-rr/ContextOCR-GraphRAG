from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections import Counter
from typing import Any, Iterable

from langdetect import DetectorFactory, LangDetectException, detect_langs
from wordfreq import zipf_frequency

DetectorFactory.seed = 0


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_text(text: str | None) -> str:
    value = unicodedata.normalize("NFC", text or "").replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+", " ", value).strip()


def words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", normalize_text(text), flags=re.UNICODE)


def edit_counts(hypothesis: list[str] | str, reference: list[str] | str) -> dict[str, int]:
    """Return deterministic Levenshtein S/D/I counts (reference is denominator)."""
    hyp, ref = list(hypothesis), list(reference)
    rows: list[list[tuple[int, int, int, int]]] = [[(0, 0, 0, 0)] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(1, len(ref) + 1):
        rows[i][0] = (i, 0, i, 0)
    for j in range(1, len(hyp) + 1):
        rows[0][j] = (j, 0, 0, j)
    for i, ref_item in enumerate(ref, 1):
        for j, hyp_item in enumerate(hyp, 1):
            if ref_item == hyp_item:
                rows[i][j] = rows[i - 1][j - 1]
                continue
            sub = rows[i - 1][j - 1]
            delete = rows[i - 1][j]
            insert = rows[i][j - 1]
            choices = [
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            ]
            rows[i][j] = min(choices, key=lambda item: (item[0], item[2] + item[3], item[1], item[2]))
    distance, substitutions, deletions, insertions = rows[-1][-1]
    return {"distance": distance, "substitutions": substitutions, "deletions": deletions, "insertions": insertions, "reference_length": len(ref)}


def error_rate(hypothesis: list[str] | str, reference: list[str] | str) -> tuple[float | None, dict[str, int]]:
    counts = edit_counts(hypothesis, reference)
    if counts["reference_length"] == 0:
        return None, counts
    return counts["distance"] / counts["reference_length"], counts


def eligible_word_data(text: str, config: dict[str, Any]) -> dict[str, Any]:
    tokens = words(text)
    maximum = int(config["maximum_word_length"])
    repeated = int(config["repeated_character_minimum"])
    expected = str(config["expected_language"])
    threshold = float(config["dictionary_zipf_threshold"])
    normalized = [token.casefold() for token in tokens]
    valid = [
        token for token in tokens
        if 1 <= len(token) <= maximum
        and not re.search(rf"(.)\1{{{repeated - 1},}}", token.casefold())
        and (expected != "en" or len(token) == 1 or bool(re.search(r"[aeiouy]", token, re.I)))
    ]
    dictionary = [token for token in tokens if zipf_frequency(token.casefold(), expected) >= threshold]
    return {
        "tokens": tokens,
        "normalized": normalized,
        "valid_count": len(valid),
        "dictionary_count": len(dictionary),
        "valid_word_ratio": len(valid) / len(tokens) if tokens else None,
        "dictionary_match_ratio": len(dictionary) / len(tokens) if tokens else None,
        "average_word_length": sum(sum(char.isalpha() for char in token) for token in valid) / len(valid) if valid else None,
        "vocabulary_richness": len(set(normalized)) / len(normalized) if normalized else None,
    }


def noise_metrics(text: str, config: dict[str, Any]) -> dict[str, Any]:
    value = normalize_text(text)
    total = len(value)
    noise: set[int] = set()
    artifacts: set[int] = set()
    for index, char in enumerate(value):
        category = unicodedata.category(char)
        if char == "\ufffd" or category in {"Co", "Cs"} or (category.startswith("C") and char not in "\n\t"):
            noise.add(index)
    symbol_min = int(config["symbol_sequence_minimum"])
    repeat_min = int(config["repeated_character_minimum"])
    for match in re.finditer(rf"[^\w\s]{{{symbol_min},}}", value, flags=re.UNICODE):
        noise.update(range(match.start(), match.end()))
    for match in re.finditer(rf"(.)\1{{{repeat_min - 1},}}", value, flags=re.I):
        artifacts.update(range(match.start(), match.end()))
    for match in re.finditer(r"(?:\ufffd|Ã.|Â.)", value):
        artifacts.update(range(match.start(), match.end()))
    noise.update(artifacts)
    non_alphanumeric = sum(not char.isalnum() for char in value)
    return {
        "noise_character_count": len(noise),
        "noise_ratio": len(noise) / total if total else None,
        "non_alphanumeric_ratio": non_alphanumeric / total if total else None,
        "artifact_character_count": len(artifacts),
        "repeated_character_artifact_ratio": len(artifacts) / total if total else None,
    }


def language_consistency(text: str, config: dict[str, Any]) -> tuple[float | None, str | None]:
    letters = "".join(char for char in normalize_text(text) if char.isalpha() or char.isspace())
    if len(letters.replace(" ", "")) < int(config["minimum_language_characters"]):
        return None, None
    try:
        probabilities = detect_langs(letters)
    except LangDetectException:
        return None, None
    expected = str(config["expected_language"])
    probability = next((float(item.prob) for item in probabilities if item.lang == expected), 0.0)
    return clamp(probability), probabilities[0].lang if probabilities else None


def weighted_confidence(tokens: list[dict[str, Any]]) -> tuple[float | None, int]:
    actual: list[tuple[float, int]] = []
    for token in tokens:
        source = str(token.get("confidence_source", "")).lower()
        if token.get("ocr_confidence") is not None:
            value = float(token["ocr_confidence"])
        elif "tesseract" in source or "paddleocr" in source:
            value = float(token.get("confidence", 0))
        else:
            continue
        actual.append((clamp(value), max(1, sum(char.isalnum() for char in str(token.get("text", ""))))))
    denominator = sum(weight for _, weight in actual)
    return (sum(value * weight for value, weight in actual) / denominator if denominator else None, len(actual))


def text_density(text: str, preprocessing: list[dict[str, Any]]) -> float | None:
    pixels = sum(max(0.0, float(page.get("width", 0))) * max(0.0, float(page.get("height", 0))) for page in preprocessing)
    if pixels <= 0:
        return None
    return sum(char.isalnum() for char in text) / (pixels / 1_000_000)


def structural_consistency(tokens: list[dict[str, Any]]) -> tuple[float | None, dict[str, float]]:
    if not tokens:
        return 0.0, {"fragmentation_quality": 0.0, "bbox_quality": 0.0, "page_balance": 0.0}
    alpha = [str(token.get("text", "")) for token in tokens if any(char.isalpha() for char in str(token.get("text", "")))]
    fragments = sum(len(re.sub(r"\W", "", token)) == 1 and token.casefold() not in {"a", "i"} for token in alpha)
    fragmentation_quality = 1 - fragments / len(alpha) if alpha else 0.0
    bbox_valid = sum(
        isinstance(token.get("bbox"), list) and len(token["bbox"]) == 4
        and float(token["bbox"][2]) > float(token["bbox"][0])
        and float(token["bbox"][3]) > float(token["bbox"][1])
        for token in tokens
    ) / len(tokens)
    counts = list(Counter(int(token.get("page", 1)) for token in tokens).values())
    page_balance = 1.0 if len(counts) == 1 else 1 / (1 + statistics.pstdev(counts) / max(statistics.mean(counts), 1))
    components = {"fragmentation_quality": clamp(fragmentation_quality), "bbox_quality": clamp(bbox_valid), "page_balance": clamp(page_balance)}
    line_tokens = [token for token in tokens if token.get("line_index") is not None]
    if line_tokens:
        groups = Counter((token.get("page"), token.get("line_index")) for token in line_tokens)
        components["line_coherence"] = sum(count >= 2 for count in groups.values()) / len(groups)
    return statistics.mean(components.values()), components


def classify_quality(score: float, thresholds: dict[str, Any]) -> str:
    if score >= float(thresholds["high"]):
        return "High Quality"
    if score >= float(thresholds["medium"]):
        return "Medium Quality"
    return "Low Quality"


def quality_score(components: dict[str, float | None], config: dict[str, Any], empty_text: bool = False) -> tuple[float, dict[str, float]]:
    if empty_text:
        return 0.0, {}
    weights = {key: float(value) for key, value in config["quality_weights"].items()}
    active = {key: weights[key] for key, value in components.items() if value is not None and key in weights}
    total = sum(active.values())
    if total == 0:
        return 0.0, {}
    normalized_weights = {key: value / total for key, value in active.items()}
    score = 100 * sum(clamp(float(components[key])) * weight for key, weight in normalized_weights.items())
    return round(score, 4), normalized_weights


def evaluate_document(run: dict[str, Any], document: dict[str, Any], ground_truth: str | None, config: dict[str, Any]) -> dict[str, Any]:
    result = run.get("result", {})
    text = normalize_text(result.get("corrected_text") or result.get("original_text") or "")
    original = normalize_text(result.get("original_text") or "")
    tokens = list(result.get("tokens") or [])
    lexical = eligible_word_data(text, config)
    noise = noise_metrics(text, config)
    language_score, detected_language = language_consistency(text, config)
    ocr_confidence, confidence_tokens = weighted_confidence(tokens)
    structural_score, structural_parts = structural_consistency(tokens)
    density = text_density(text, list(result.get("preprocessing") or []))
    gt = normalize_text(ground_truth)
    cer = wer = baseline_cer = baseline_wer = None
    char_counts = word_counts = baseline_char_counts = baseline_word_counts = None
    if gt:
        cer, char_counts = error_rate(list(text), list(gt))
        wer, word_counts = error_rate(words(text.casefold()), words(gt.casefold()))
        baseline_cer, baseline_char_counts = error_rate(list(original), list(gt))
        baseline_wer, baseline_word_counts = error_rate(words(original.casefold()), words(gt.casefold()))
    density_quality = clamp(density / float(config["density_target_characters_per_megapixel"])) if density is not None else None
    components: dict[str, float | None] = {
        "ocr_confidence": ocr_confidence,
        "character_accuracy": None if cer is None else clamp(1 - cer),
        "word_accuracy": None if wer is None else clamp(1 - wer),
        "valid_word_ratio": lexical["valid_word_ratio"],
        "dictionary_match_ratio": lexical["dictionary_match_ratio"],
        "noise_cleanliness": None if noise["noise_ratio"] is None else 1 - noise["noise_ratio"],
        "artifact_cleanliness": None if noise["repeated_character_artifact_ratio"] is None else 1 - noise["repeated_character_artifact_ratio"],
        "language_consistency": language_score,
        "structural_consistency": structural_score,
        "non_alphanumeric_cleanliness": None if noise["non_alphanumeric_ratio"] is None else 1 - noise["non_alphanumeric_ratio"],
        "density_quality": density_quality,
    }
    score, active_weights = quality_score(components, config, empty_text=not text)
    method = next((stage.get("method") for stage in run.get("stages", []) if stage.get("key") == "ocr"), None)
    return {
        "document_id": document["id"], "run_id": run["id"], "filename": document["name"],
        "file_type": document["media_type"], "number_of_pages": document["pages"],
        "number_of_images_pages_processed": len(result.get("preprocessing") or result.get("page_images") or []),
        "ocr_engine": method, "extracted_character_count": len(text), "extracted_word_count": len(lexical["tokens"]),
        "ocr_confidence": ocr_confidence, "ocr_confidence_token_count": confidence_tokens,
        "ground_truth_available": bool(gt), "ground_truth_character_count": len(gt) if gt else None,
        "ground_truth_word_count": len(words(gt)) if gt else None,
        "cer": cer, "wer": wer, "cer_counts": char_counts, "wer_counts": word_counts,
        "baseline_cer": baseline_cer, "baseline_wer": baseline_wer,
        "baseline_cer_counts": baseline_char_counts, "baseline_wer_counts": baseline_word_counts,
        **noise,
        "valid_word_ratio": lexical["valid_word_ratio"], "dictionary_match_ratio": lexical["dictionary_match_ratio"],
        "average_word_length": lexical["average_word_length"], "vocabulary_richness": lexical["vocabulary_richness"],
        "language_consistency": language_score, "detected_language": detected_language,
        "text_density_characters_per_megapixel": density,
        "structural_consistency": structural_score, "structural_components": structural_parts,
        "quality_components": components, "active_quality_weights": active_weights,
        "final_quality_score": score, "quality_class": classify_quality(score, config["quality_thresholds"]),
        "correction_improvement": {
            "cer_absolute_reduction": None if cer is None else baseline_cer - cer,
            "wer_absolute_reduction": None if wer is None else baseline_wer - wer,
        },
        "before_preprocessing_available": False,
        "before_preprocessing_note": "The pipeline does not persist a separate OCR pass on the unprocessed page; no before-preprocessing metric is fabricated.",
    }


def descriptive(values: Iterable[float | None]) -> dict[str, float | int | None]:
    usable = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not usable:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None, "standard_deviation": None, "p25": None, "p75": None}
    def percentile(fraction: float) -> float:
        position = (len(usable) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        return usable[low] if low == high else usable[low] + (usable[high] - usable[low]) * (position - low)
    return {
        "count": len(usable), "mean": statistics.mean(usable), "median": statistics.median(usable),
        "minimum": min(usable), "maximum": max(usable),
        "standard_deviation": statistics.pstdev(usable), "p25": percentile(.25), "p75": percentile(.75),
    }


def aggregate_documents(documents: list[dict[str, Any]]) -> dict[str, Any]:
    classes = Counter(document["quality_class"] for document in documents)
    total = len(documents)
    metric_names = ["final_quality_score", "ocr_confidence", "cer", "wer", "noise_ratio", "valid_word_ratio", "dictionary_match_ratio", "language_consistency", "structural_consistency", "text_density_characters_per_megapixel"]
    ground_truth_count = sum(document["ground_truth_available"] for document in documents)
    aggregate_errors: dict[str, Any] = {}
    for metric, key in (("cer", "cer_counts"), ("wer", "wer_counts")):
        rows = [document[key] for document in documents if document.get(key)]
        reference = sum(row["reference_length"] for row in rows)
        distance = sum(row["distance"] for row in rows)
        aggregate_errors[metric] = distance / reference if reference else None
        aggregate_errors[f"{metric}_reference_units"] = reference
        aggregate_errors[f"{metric}_errors"] = distance
    return {
        "number_of_documents": total, "documents_with_ground_truth": ground_truth_count,
        "ground_truth_coverage": ground_truth_count / total if total else 0.0,
        "quality_classes": {name: {"count": classes.get(name, 0), "percentage": classes.get(name, 0) / total if total else 0.0} for name in ("High Quality", "Medium Quality", "Low Quality")},
        "metrics": {name: descriptive(document.get(name) for document in documents) for name in metric_names},
        "aggregate_error_rates": aggregate_errors,
    }
