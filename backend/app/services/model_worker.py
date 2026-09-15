"""Short-lived worker entry point for memory-heavy local inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def layout(payload: dict[str, Any]) -> dict[str, Any]:
    from doclayout_yolo import YOLOv10

    prediction = YOLOv10(payload["model_path"]).predict(
        payload["image_path"],
        imgsz=1024,
        conf=payload["confidence"],
        device=payload["device"],
        verbose=False,
    )[0]
    return {
        "boxes": prediction.boxes.xyxy.cpu().tolist(),
        "scores": prediction.boxes.conf.cpu().tolist(),
        "classes": prediction.boxes.cls.cpu().tolist(),
        "names": prediction.names,
    }


def ocr(payload: dict[str, Any]) -> list[dict[str, Any]]:
    from paddleocr import PaddleOCR

    engine = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        return_word_box=True,
        device=payload["device"],
        enable_mkldnn=False,
    )
    pages = []
    for image_path in payload["image_paths"]:
        lines = []
        for page_result in engine.predict(image_path, return_word_box=True):
            raw = page_result.json.get("res", page_result.json)
            texts = raw.get("rec_texts", [])
            scores = raw.get("rec_scores", [])
            boxes = raw.get("rec_boxes", [])
            word_groups = raw.get("text_word", [])
            word_box_groups = raw.get("text_word_boxes", [])
            for index, text in enumerate(texts):
                lines.append({
                    "text": str(text),
                    "score": float(scores[index]) if index < len(scores) else 0.0,
                    "box": boxes[index].tolist() if index < len(boxes) and hasattr(boxes[index], "tolist") else (boxes[index] if index < len(boxes) else []),
                    "words": [str(value) for value in word_groups[index]] if index < len(word_groups) else [],
                    "word_boxes": word_box_groups[index].tolist() if index < len(word_box_groups) and hasattr(word_box_groups[index], "tolist") else (word_box_groups[index] if index < len(word_box_groups) else []),
                })
        pages.append({"lines": lines})
    return pages


def semantic(payload: dict[str, Any]) -> list[list[dict[str, float | int]]]:
    import faiss
    import numpy as np

    from app.services.local_models import get_embedding_model

    model = get_embedding_model(payload["model_path"])
    corpus = payload["corpus"]
    queries = payload["queries"]
    vectors = np.asarray(model.encode(corpus, normalize_embeddings=True), dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    query_vectors = np.asarray(model.encode(queries, normalize_embeddings=True), dtype="float32")
    scores, ids = index.search(query_vectors, min(int(payload["top_k"]), len(corpus)))
    return [
        [{"index": int(item_id), "score": float(score)} for item_id, score in zip(row_ids, row_scores)]
        for row_ids, row_scores in zip(ids, scores)
    ]


def main() -> None:
    output_path = Path(sys.argv[1])
    payload = json.loads(sys.stdin.read())
    actions = {"layout": layout, "ocr": ocr, "semantic": semantic}
    result = actions[payload["action"]](payload)
    output_path.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
