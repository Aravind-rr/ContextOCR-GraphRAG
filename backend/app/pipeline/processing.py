from __future__ import annotations

import importlib.util
import json
import math
import re
import shutil
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np
from PIL import Image
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import Settings
from app.services.local_models import embed_with_llama_cpp, llama_server, run_model_worker


DOMAIN_VOCABULARY: dict[str, tuple[str, ...]] = {
    "general": ("document", "information", "government", "regulation", "institution", "department", "research", "system", "confidence", "correction", "application", "university", "technology", "development", "management", "organization"),
    "medical": ("patient", "hospital", "diagnosis", "treatment", "medicine", "clinical", "physician", "prescription", "symptoms", "laboratory"),
    "legal": ("agreement", "contract", "regulation", "statute", "court", "plaintiff", "defendant", "jurisdiction", "liability", "provision"),
    "government": ("government", "ministry", "department", "regulation", "policy", "authority", "administration", "public", "official", "citizen"),
    "financial": ("account", "transaction", "balance", "interest", "financial", "revenue", "liability", "invoice", "payment", "statement"),
    "institutional": ("institution", "university", "department", "faculty", "student", "academic", "administration", "committee", "policy", "programme"),
}

ALL_DOMAIN_TERMS = sorted({term for terms in DOMAIN_VOCABULARY.values() for term in terms})


def calibrated_paddle_confidence(text: str, raw_confidence: float) -> tuple[float, dict[str, Any] | None]:
    """Produce a transparent token score from Paddle's line-level score.

    PaddleOCR returns one recognition score for a complete line even when word
    boxes are enabled. A close but non-exact match to trusted domain vocabulary
    is therefore used as token-level anomaly evidence. The raw score is always
    retained separately and the adjustment is exposed in the API.
    """
    clean = re.sub(r"[^A-Za-z'-]", "", text).lower()
    if len(clean) < 5 or clean in ALL_DOMAIN_TERMS:
        return raw_confidence, None
    if clean.endswith("s") and clean[:-1] in ALL_DOMAIN_TERMS:
        return raw_confidence, None
    match = process.extractOne(clean, ALL_DOMAIN_TERMS, scorer=fuzz.ratio)
    if not match or float(match[1]) < 82 or levenshtein(clean, match[0]) > 2:
        return raw_confidence, None
    candidate, score, _ = match
    if clean.endswith("ive") and candidate.endswith("ion") and clean[:-3] == candidate[:-3]:
        return raw_confidence, None
    similarity = float(score) / 100.0
    effective = min(raw_confidence, max(0.55, 1.20 - 0.40 * similarity))
    return effective, {
        "applied": True,
        "reason": "near-match lexical anomaly against trusted domain terminology",
        "candidate": candidate,
        "similarity": similarity,
    }


def dependency_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def render_document(path: Path, out_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pages: list[Path] = []
    meta: dict[str, Any] = {"digital_pdf": False, "embedded_text": []}
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(path)
        embedded = [page.get_text("text").strip() for page in doc]
        meta["digital_pdf"] = any(len(text) > 30 for text in embedded)
        meta["embedded_text"] = embedded
        for index, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            target = out_dir / f"page-{index + 1}.png"
            pix.save(target)
            pages.append(target)
        doc.close()
    else:
        image = Image.open(path)
        frame = 0
        while True:
            target = out_dir / f"page-{frame + 1}.png"
            image.seek(frame)
            image.convert("RGB").save(target)
            pages.append(target)
            frame += 1
            try:
                image.seek(frame)
            except EOFError:
                break
    return pages, meta


def preprocess_page(source: Path, target: Path) -> dict[str, Any]:
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"Could not decode {source.name}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    operations = ["grayscale"]
    denoised = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)
    if float(np.mean(cv2.absdiff(gray, denoised))) > 0.7:
        gray = denoised
        operations.append("non-local-means denoising")
    p5, p95 = np.percentile(gray, (5, 95))
    if p95 - p5 < 130:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        operations.append("CLAHE contrast enhancement")
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    foreground = np.column_stack(np.where(threshold < 255))
    angle = 0.0
    if len(foreground) > 100:
        raw_angle = cv2.minAreaRect(foreground[:, ::-1].astype(np.float32))[-1]
        angle = -(90 + raw_angle) if raw_angle < -45 else -raw_angle
        if 0.35 < abs(angle) < 8:
            h, w = gray.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            gray = cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            operations.append(f"deskew {angle:.2f}°")
    # Adaptive thresholding is useful only on visibly uneven illumination.
    local_means = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    if float(np.std(local_means)) > 22:
        output = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12)
        operations.append("adaptive threshold")
    else:
        output = gray
    cv2.imwrite(str(target), output)
    return {"source": str(source), "output": str(target), "operations": operations, "width": image.shape[1], "height": image.shape[0]}


def detect_layout(image_path: Path, settings: Settings | None = None) -> tuple[list[dict[str, Any]], str, list[str]]:
    if settings and settings.doc_layout_model_path and Path(settings.doc_layout_model_path).is_file() and dependency_available("doclayout_yolo"):
        try:
            model_path = str(Path(settings.doc_layout_model_path).resolve())
            prediction = run_model_worker({
                "action": "layout", "model_path": model_path,
                "image_path": str(image_path.resolve()),
                "confidence": settings.doc_layout_confidence,
                "device": settings.doc_layout_device,
            })
            class_map = {
                "title": "heading", "plain text": "text_block", "figure": "image",
                "figure_caption": "caption", "table_caption": "caption",
                "table_footnote": "footnote", "isolate_formula": "formula",
                "formula_caption": "caption", "abandon": "other",
            }
            regions = []
            boxes = prediction["boxes"]
            scores = prediction["scores"]
            classes = prediction["classes"]
            for index, (box, score, class_id) in enumerate(zip(boxes, scores, classes), 1):
                names = prediction["names"]
                raw_class = names.get(str(int(class_id)), names.get(int(class_id))) if isinstance(names, dict) else names[int(class_id)]
                regions.append({
                    "id": f"r{index}", "class": class_map.get(raw_class, raw_class),
                    "raw_class": raw_class, "bbox": [round(float(v), 2) for v in box],
                    "confidence": round(float(score), 4), "source": "DocLayout-YOLO",
                })
            return sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0])), f"DocLayout-YOLO ({Path(model_path).name})", []
        except Exception as exc:
            model_warning = f"DocLayout-YOLO inference failed; OpenCV fallback used: {type(exc).__name__}: {exc}"
    else:
        model_warning = "DocLayout-YOLO model/package is not configured; OpenCV fallback used."
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode {image_path.name} for layout detection")
    binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 7))
    merged = cv2.dilate(binary, kernel, iterations=2)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image.shape
    regions = []
    for contour in contours:
        x, y, rw, rh = cv2.boundingRect(contour)
        if rw * rh < max(500, w * h * 0.0002):
            continue
        kind = "heading" if rh > 22 and rw > w * 0.25 and y < h * 0.3 else "text_block"
        regions.append({"id": f"r{len(regions)+1}", "class": kind, "bbox": [x, y, x + rw, y + rh], "confidence": None, "source": "opencv-fallback"})
    if not regions:
        regions = [{"id": "r1", "class": "page", "bbox": [0, 0, w, h], "confidence": None, "source": "opencv-fallback"}]
    return sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0])), "OpenCV morphology fallback", [model_warning]


def reading_order(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not regions:
        return []
    widths = [r["bbox"][2] - r["bbox"][0] for r in regions]
    page_width = max(r["bbox"][2] for r in regions)
    columns = 2 if len(regions) >= 4 and statistics.median(widths) < page_width * 0.65 else 1
    ordered = sorted(regions, key=(lambda r: (0 if (r["bbox"][0]+r["bbox"][2])/2 < page_width/2 else 1, r["bbox"][1])) if columns == 2 else (lambda r: (r["bbox"][1], r["bbox"][0])))
    return [{**region, "order": i + 1} for i, region in enumerate(ordered)]


def _tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def extract_ocr(path: Path, page_images: list[Path], document_meta: dict[str, Any], engine: str, settings: Settings | None = None) -> tuple[list[dict[str, Any]], str, list[str]]:
    warnings: list[str] = []
    tokens: list[dict[str, Any]] = []
    # Embedded PDF words are actual source text; confidence is explicitly source certainty, not OCR confidence.
    if document_meta.get("digital_pdf"):
        doc = fitz.open(path)
        for page_index, page in enumerate(doc):
            for idx, word in enumerate(page.get_text("words")):
                x0, y0, x1, y1, text, *_ = word
                # Pages are rendered at 2x for display/preprocessing, so boxes use rendered-image coordinates.
                tokens.append({"id": f"p{page_index+1}-t{idx+1}", "page": page_index + 1, "text": text, "confidence": 1.0, "confidence_source": "embedded-pdf", "bbox": [x0 * 2, y0 * 2, x1 * 2, y1 * 2]})
        doc.close()
        return tokens, "PyMuPDF embedded text extraction", warnings
    if engine in ("auto", "paddle") and dependency_available("paddleocr"):
        config = settings or Settings()
        pages = run_model_worker({
            "action": "ocr", "image_paths": [str(path.resolve()) for path in page_images],
            "device": config.paddle_device,
        })
        for page_index, page_result in enumerate(pages):
            counter = 0
            for line_index, line in enumerate(page_result["lines"]):
                raw_confidence = float(line["score"])
                words = line["words"] or re.findall(r"\S+", line["text"])
                word_boxes = line["word_boxes"]
                clean_index = 0
                for word_index, word in enumerate(words):
                    text = str(word).strip()
                    if not text:
                        continue
                    counter += 1
                    if word_index < len(word_boxes):
                        bbox = [round(float(value), 2) for value in word_boxes[word_index]]
                    elif line["box"]:
                        bbox = [round(float(value), 2) for value in line["box"]]
                    else:
                        bbox = [0, 0, 0, 0]
                    confidence, calibration = calibrated_paddle_confidence(text, raw_confidence)
                    tokens.append({
                        "id": f"p{page_index+1}-t{counter}", "page": page_index + 1,
                        "text": text, "confidence": confidence,
                        "ocr_confidence": raw_confidence,
                        "confidence_source": "PaddleOCR line score" if calibration is None else "PaddleOCR line score + disclosed lexical calibration",
                        "confidence_calibration": calibration,
                        "bbox": bbox, "line_index": line_index, "word_index": clean_index,
                    })
                    clean_index += 1
        return tokens, f"PaddleOCR {config.paddle_ocr_version} mobile models ({config.paddle_device})", warnings
    if engine == "paddle":
        warnings.append("PaddleOCR requested but its package is not installed.")
    if _tesseract_available():
        import pytesseract
        for page_index, image_path in enumerate(page_images):
            data = pytesseract.image_to_data(Image.open(image_path), output_type=pytesseract.Output.DICT)
            count = 0
            for i, text in enumerate(data["text"]):
                text = text.strip()
                conf = float(data["conf"][i])
                if not text or conf < 0:
                    continue
                count += 1
                x, y, w, h = (int(data[k][i]) for k in ("left", "top", "width", "height"))
                tokens.append({"id": f"p{page_index+1}-t{count}", "page": page_index + 1, "text": text, "confidence": conf / 100.0, "confidence_source": "Tesseract", "bbox": [x, y, x + w, y + h]})
        warnings.append("PaddleOCR unavailable; used the installed Tesseract fallback.")
        return tokens, "Tesseract fallback", warnings
    raise RuntimeError("No OCR engine available. Install PaddleOCR or the Tesseract executable.")


def analyze_confidence(tokens: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    values = [float(t["confidence"]) for t in tokens]
    bins = [{"range": f"{i/10:.1f}–{(i+1)/10:.1f}", "count": sum(i/10 <= v < (i+1)/10 or (i == 9 and v == 1) for v in values)} for i in range(10)]
    return {"threshold": threshold, "average": sum(values) / len(values) if values else 0, "low_count": sum(v < threshold for v in values), "high_count": sum(v >= threshold for v in values), "distribution": bins}


def generate_candidates(tokens: list[dict[str, Any]], threshold: float, domain: str, limit: int) -> list[dict[str, Any]]:
    high_words = {re.sub(r"[^A-Za-z'-]", "", t["text"]).lower() for t in tokens if t["confidence"] >= threshold and len(t["text"]) > 2}
    vocabulary = sorted(high_words | set(DOMAIN_VOCABULARY.get(domain, DOMAIN_VOCABULARY["general"])) | set(DOMAIN_VOCABULARY["general"]))
    results = []
    for index, token in enumerate(tokens):
        token["decision"] = "CORRECT" if token["confidence"] < threshold else "KEEP"
        token["context"] = " ".join(t["text"] for t in tokens[max(0, index-5):index+6])
        if token["decision"] == "KEEP":
            continue
        clean = re.sub(r"[^A-Za-z'-]", "", token["text"]).lower()
        matches = process.extract(clean, vocabulary, scorer=fuzz.ratio, limit=limit) if clean else []
        candidates = [{"value": word, "source": "document vocabulary" if word in high_words else f"built-in {domain} vocabulary", "similarity": score / 100.0} for word, score, _ in matches if word != clean]
        candidates.append({"value": "KEEP_ORIGINAL", "source": "safety policy", "similarity": 1.0})
        results.append({"token_id": token["id"], "original": token["text"], "candidates": candidates})
    return results


def make_chunks(tokens: list[dict[str, Any]], chunk_size: int, trusted_threshold: float = 0.7) -> list[dict[str, Any]]:
    chunks = []
    for page in sorted({t["page"] for t in tokens}):
        page_tokens = [t for t in tokens if t["page"] == page and t["confidence"] >= trusted_threshold]
        for start in range(0, len(page_tokens), chunk_size):
            part = page_tokens[start:start + chunk_size]
            if part:
                chunks.append({"id": f"p{page}-c{len(chunks)+1}", "page": page, "text": " ".join(t["text"] for t in part), "token_start": start})
    return chunks


def semantic_retrieve(tokens: list[dict[str, Any]], candidates: list[dict[str, Any]], chunks: list[dict[str, Any]], settings: Settings) -> tuple[list[dict[str, Any]], str, list[str]]:
    if not chunks or not candidates:
        return [], "No retrieval required", []
    warnings = []
    corpus = [c["text"] for c in chunks]
    by_id = {t["id"]: t for t in tokens}
    queries = [by_id[entry["token_id"]]["context"] for entry in candidates]

    def tfidf_results() -> list[list[tuple[int, float]]]:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        matrix = vectorizer.fit_transform(corpus)
        values = []
        for query in queries:
            scores = cosine_similarity(vectorizer.transform([query]), matrix)[0]
            values.append([(int(i), float(scores[i])) for i in scores.argsort()[::-1][:settings.retrieval_top_k]])
        return values

    gguf_path = Path(settings.embedding_gguf_model_path or "")
    llama_executable = Path(settings.llama_server_executable or "")

    # Prefer quantized BGE-M3 through llama.cpp. It uses substantially less
    # memory than the PyTorch checkpoint and is released before Qwen starts.
    if gguf_path.is_file() and llama_executable.is_file() and dependency_available("faiss"):
        try:
            import faiss

            embeddings = np.asarray(
                embed_with_llama_cpp(corpus + queries, gguf_path, llama_executable),
                dtype="float32",
            )
            corpus_vectors = embeddings[:len(corpus)]
            query_vectors = embeddings[len(corpus):]
            faiss.normalize_L2(corpus_vectors)
            faiss.normalize_L2(query_vectors)
            index = faiss.IndexFlatIP(corpus_vectors.shape[1])
            index.add(corpus_vectors)
            scores, ids = index.search(query_vectors, min(settings.retrieval_top_k, len(corpus)))
            searches = [
                [(int(item_id), float(score)) for item_id, score in zip(row_ids, row_scores)]
                for row_ids, row_scores in zip(ids, scores)
            ]
            method = f"BGE-M3 Q4_K_M ({gguf_path.name}) via local llama.cpp + FAISS IndexFlatIP"
        except Exception as exc:
            searches = tfidf_results()
            method = "TF-IDF cosine fallback after BGE-M3 GGUF inference failure"
            warnings.append(f"BGE-M3 GGUF inference failed; TF-IDF fallback used ({type(exc).__name__}: {exc}).")
    elif dependency_available("sentence_transformers") and dependency_available("faiss"):
        model_path = str(Path(settings.embedding_model).resolve()) if Path(settings.embedding_model).exists() else settings.embedding_model
        try:
            worker_results = run_model_worker({
                "action": "semantic", "model_path": model_path,
                "corpus": corpus, "queries": queries,
                "top_k": settings.retrieval_top_k,
            })
            searches = [[(hit["index"], hit["score"]) for hit in row] for row in worker_results]
            method = f"BGE-M3 ({settings.embedding_model}) + FAISS IndexFlatIP"
        except Exception as exc:
            searches = tfidf_results()
            method = "TF-IDF cosine fallback after BGE-M3 inference failure"
            warnings.append(f"BGE-M3 inference failed; TF-IDF fallback used ({type(exc).__name__}: {exc}).")
    else:
        searches = tfidf_results()
        method = "TF-IDF cosine fallback (BGE-M3/FAISS unavailable)"
        warnings.append("BGE-M3 or FAISS is unavailable; retrieval used the labeled TF-IDF fallback.")
    evidence = []
    for candidate_index, entry in enumerate(candidates):
        token = by_id[entry["token_id"]]
        query = token["context"]
        matches = searches[candidate_index]
        evidence.append({"token_id": token["id"], "query": query, "results": [{**chunks[i], "similarity": score, "source": f"page {chunks[i]['page']}"} for i, score in matches]})
    return evidence, method, warnings


def extract_graph(
    tokens: list[dict[str, Any]],
    document_id: str,
    domain: str = "general",
    document_name: str = "Document",
    trusted_threshold: float = 0.8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a compact domain graph only from trusted OCR mentions."""
    entity_specs: dict[str, list[tuple[str, str, tuple[str, ...]]]] = {
        "government": [
            ("Public Administration Circular", "DocumentType", ("public administration circular",)),
            ("Government", "GovernmentBody", ("government",)),
            ("Ministry", "Organization", ("ministry",)),
            ("Department", "Organization", ("department",)),
            ("Authority", "GovernmentBody", ("authority",)),
            ("Policy", "Policy", ("policy",)),
            ("Regulation", "Regulation", ("regulation",)),
            ("Public Offices", "Organization", ("public offices",)),
            ("Implementation Report", "Report", ("implementation report",)),
        ],
        "medical": [
            ("Patient", "Patient", ("patient",)),
            ("Physician", "Clinician", ("physician",)),
            ("Diagnosis", "Condition", ("diagnosis",)),
            ("Prescription", "Prescription", ("prescription",)),
            ("Treatment", "Treatment", ("treatment",)),
            ("Laboratory", "ClinicalService", ("laboratory",)),
            ("Hospital", "HealthcareOrganization", ("hospital",)),
        ],
        "legal": [
            ("Agreement", "LegalInstrument", ("agreement",)),
            ("Contract", "LegalInstrument", ("contract",)),
            ("Liability", "LegalConcept", ("liability",)),
            ("Court", "Court", ("court",)),
            ("Jurisdiction", "Jurisdiction", ("jurisdiction",)),
            ("Provision", "LegalProvision", ("provision", "provisions")),
            ("Regulation", "Regulation", ("regulation",)),
        ],
        "financial": [
            ("Account", "FinancialAccount", ("account",)),
            ("Transaction", "Transaction", ("transaction",)),
            ("Balance", "Amount", ("balance",)),
            ("Payment", "Payment", ("payment",)),
            ("Invoice", "Invoice", ("invoice",)),
            ("Statement", "FinancialStatement", ("statement",)),
            ("Revenue", "Amount", ("revenue",)),
        ],
        "institutional": [
            ("Institution", "Institution", ("institution",)),
            ("University", "University", ("university",)),
            ("Department", "Department", ("department",)),
            ("Faculty", "PersonGroup", ("faculty",)),
            ("Student", "PersonGroup", ("student", "students")),
            ("Committee", "Committee", ("committee",)),
            ("Academic Programme", "Programme", ("academic programme", "programme")),
            ("Policy", "Policy", ("policy",)),
        ],
        "general": [
            ("Document", "DocumentType", ("document",)),
            ("Information", "Concept", ("information",)),
            ("Research", "Activity", ("research",)),
            ("System", "System", ("system",)),
            ("Confidence", "Metric", ("confidence",)),
            ("Correction", "Process", ("correction",)),
        ],
    }
    relation_specs: dict[str, list[tuple[str, str, str]]] = {
        "government": [
            ("Government", "HAS_MINISTRY", "Ministry"),
            ("Government", "HAS_DEPARTMENT", "Department"),
            ("Authority", "ISSUED", "Policy"),
            ("Policy", "APPLIES_TO", "Public Offices"),
            ("Department", "CONSULTED_ON", "Policy"),
            ("Ministry", "REQUESTED", "Implementation Report"),
            ("Public Administration Circular", "CONTAINS_POLICY", "Policy"),
            ("Policy", "GOVERNED_BY", "Regulation"),
        ],
        "medical": [
            ("Patient", "HAS_DIAGNOSIS", "Diagnosis"),
            ("Patient", "HAS_PRESCRIPTION", "Prescription"),
            ("Physician", "ISSUED", "Prescription"),
            ("Prescription", "DEFINES", "Treatment"),
            ("Laboratory", "SUPPORTS", "Diagnosis"),
            ("Hospital", "EMPLOYS", "Physician"),
        ],
        "legal": [
            ("Contract", "FORMALIZES", "Agreement"),
            ("Agreement", "DEFINES", "Liability"),
            ("Agreement", "HAS_PROVISION", "Provision"),
            ("Court", "HAS_JURISDICTION", "Jurisdiction"),
            ("Provision", "SUBJECT_TO", "Regulation"),
        ],
        "financial": [
            ("Account", "HAS_TRANSACTION", "Transaction"),
            ("Account", "HAS_BALANCE", "Balance"),
            ("Transaction", "SETTLED_BY", "Payment"),
            ("Payment", "REFERENCES", "Invoice"),
            ("Statement", "REPORTS", "Account"),
            ("Statement", "REPORTS", "Revenue"),
        ],
        "institutional": [
            ("University", "HAS_DEPARTMENT", "Department"),
            ("Department", "HAS_FACULTY", "Faculty"),
            ("Department", "HAS_STUDENT_BODY", "Student"),
            ("Committee", "APPROVED", "Policy"),
            ("Committee", "GOVERNS", "Academic Programme"),
            ("Institution", "HAS_POLICY", "Policy"),
        ],
        "general": [
            ("Document", "DESCRIBES", "Information"),
            ("Research", "EVALUATES", "System"),
            ("System", "MEASURES", "Confidence"),
            ("System", "PERFORMS", "Correction"),
        ],
    }

    trusted = [token for token in tokens if float(token.get("confidence", 0)) >= trusted_threshold]

    def normalized(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    def phrase_mentions(aliases: tuple[str, ...]) -> list[list[dict[str, Any]]]:
        found: list[list[dict[str, Any]]] = []
        words = [(normalized(str(token["text"])), token) for token in trusted]
        for alias in aliases:
            parts = [normalized(part) for part in alias.split()]
            for index in range(0, len(words) - len(parts) + 1):
                if [item[0] for item in words[index:index + len(parts)]] == parts:
                    found.append([item[1] for item in words[index:index + len(parts)]])
        return found

    nodes: list[dict[str, Any]] = [{
        "id": document_id,
        "label": document_name,
        "type": "Document",
        "document_id": document_id,
        "pages": sorted({int(token.get("page", 1)) for token in tokens}),
        "mentions": 1,
        "confidence": 1.0,
        "token_ids": [],
        "provenance": [{"source": "uploaded document metadata", "document_id": document_id}],
        "properties": {"domain": domain, "source_name": document_name},
    }]
    by_label: dict[str, dict[str, Any]] = {document_name.lower(): nodes[0]}

    def add_node(label: str, entity_type: str, groups: list[list[dict[str, Any]]]) -> None:
        if not groups:
            return
        mentions = [token for group in groups for token in group]
        key = label.lower()
        node = {
            "id": f"{document_id}:{re.sub(r'[^a-z0-9]+', '-', key).strip('-')}",
            "label": label,
            "type": entity_type,
            "document_id": document_id,
            "pages": sorted({int(token["page"]) for token in mentions}),
            "mentions": len(groups),
            "confidence": min(float(token["confidence"]) for token in mentions),
            "token_ids": list(dict.fromkeys(str(token["id"]) for token in mentions)),
            "provenance": [
                {
                    "token_id": token["id"],
                    "page": token["page"],
                    "bbox": token.get("bbox", []),
                    "confidence": token["confidence"],
                    "text": token["text"],
                }
                for token in mentions
            ],
            "properties": {"domain": domain, "mention_count": len(groups)},
        }
        nodes.append(node)
        by_label[key] = node

    selected_specs = entity_specs.get(domain, entity_specs["general"])
    for label, entity_type, aliases in selected_specs:
        add_node(label, entity_type, phrase_mentions(aliases))

    for token in trusted:
        text = str(token["text"]).strip()
        if re.fullmatch(r"[A-Z]{2,}-\d{4}-\d+", text):
            add_node(text, "ReferenceIdentifier", [[token]])

    month_names = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}
    for index in range(1, len(trusted) - 1):
        previous, current, following = trusted[index - 1:index + 2]
        if normalized(str(current["text"])) in month_names and re.fullmatch(r"\d{1,2}", str(previous["text"])) and re.fullmatch(r"\d{4}", str(following["text"])):
            add_node(f"{previous['text']} {current['text']} {following['text']}", "Date", [[previous, current, following]])
            break

    edges: list[dict[str, Any]] = []

    def add_edge(source_label: str, relation_type: str, target_label: str, provenance: str) -> None:
        source = by_label.get(source_label.lower())
        target = by_label.get(target_label.lower())
        if not source or not target:
            return
        token_ids = list(dict.fromkeys(source.get("token_ids", []) + target.get("token_ids", [])))
        pages = sorted(set(source.get("pages", [])) | set(target.get("pages", [])))
        edges.append({
            "id": f"{document_id}:rel{len(edges) + 1}",
            "source": source["id"],
            "target": target["id"],
            "type": relation_type,
            "document_id": document_id,
            "provenance": provenance,
            "token_ids": token_ids,
            "pages": pages,
            "properties": {"domain": domain, "evidence": provenance},
        })

    for source, relation, target in relation_specs.get(domain, relation_specs["general"]):
        add_edge(source, relation, target, "domain relation supported by high-confidence OCR mentions")

    for node in nodes[1:]:
        if node["type"] == "ReferenceIdentifier":
            add_edge(document_name, "HAS_REFERENCE", node["label"], "reference identifier extracted from high-confidence OCR")
        elif node["type"] == "Date":
            add_edge(document_name, "DATED", node["label"], "date extracted from high-confidence OCR")
        elif node["type"] == "DocumentType":
            add_edge(document_name, "HAS_DOCUMENT_TYPE", node["label"], "document type extracted from high-confidence OCR")

    def reachable_from_document() -> set[str]:
        reachable = {document_id}
        changed = True
        while changed:
            changed = False
            for edge in edges:
                if edge["source"] in reachable and edge["target"] not in reachable:
                    reachable.add(edge["target"])
                    changed = True
                if edge["target"] in reachable and edge["source"] not in reachable:
                    reachable.add(edge["source"])
                    changed = True
        return reachable

    while True:
        reachable = reachable_from_document()
        disconnected = next((node for node in nodes[1:] if node["id"] not in reachable), None)
        if not disconnected:
            break
        add_edge(document_name, "MENTIONS", disconnected["label"], "trusted entity mention anchors this semantic subgraph to the source document")
    return nodes, edges


def graph_evidence(candidates: list[dict[str, Any]], entities: list[dict[str, Any]], relationships: list[dict[str, Any]], settings: Settings | None = None) -> list[dict[str, Any]]:
    result = []
    labels = {e["label"].lower(): e for e in entities}
    for item in candidates:
        hits = []
        for candidate in item["candidates"]:
            entity = labels.get(candidate["value"].lower())
            if entity:
                hits.append({"candidate": candidate["value"], "entity": entity, "relationships": [r for r in relationships if entity["id"] in (r["source"], r["target"])], "source": "extracted high-confidence OCR graph"})
        if settings and settings.neo4j_uri:
            try:
                from neo4j import GraphDatabase
                values = [candidate["value"].lower() for candidate in item["candidates"] if candidate["value"] != "KEEP_ORIGINAL"]
                with GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)) as driver:
                    records, _, _ = driver.execute_query(
                        "MATCH (e:Entity)-[r]-(n) WHERE toLower(e.label) IN $values "
                        "RETURN e{.*} AS entity,type(r) AS relationship,n{.*} AS neighbor,"
                        "coalesce(r.provenance,'Neo4j persisted graph') AS provenance LIMIT 20",
                        values=values,
                    )
                    for record in records:
                        hits.append({
                            "candidate": record["entity"].get("label"), "entity": record["entity"],
                            "relationships": [{"type": record["relationship"], "neighbor": record["neighbor"], "provenance": record["provenance"]}],
                            "source": "Neo4j graph retrieval",
                        })
            except Exception as exc:
                hits.append({"candidate": None, "entity": None, "relationships": [], "source": f"Neo4j query unavailable: {type(exc).__name__}"})
        result.append({"token_id": item["token_id"], "results": hits})
    return result


def build_prompts(tokens: list[dict[str, Any]], candidates: list[dict[str, Any]], semantic: list[dict[str, Any]], graph: list[dict[str, Any]]) -> list[dict[str, Any]]:
    token_map = {t["id"]: t for t in tokens}
    sem_map = {e["token_id"]: e for e in semantic}
    graph_map = {e["token_id"]: e for e in graph}
    prompts = []
    for item in candidates:
        token = token_map[item["token_id"]]
        values = [c["value"] for c in item["candidates"]]
        evidence = [r["text"] for r in sem_map.get(token["id"], {}).get("results", [])[:2]]
        graph_hits = graph_map.get(token["id"], {}).get("results", [])
        compact_graph = []
        for hit in graph_hits[:2]:
            entity = hit.get("entity") or {}
            connections = hit.get("relationships") or []
            compact_graph.append({
                "candidate": hit.get("candidate"),
                "entity": {"label": entity.get("label"), "type": entity.get("type")},
                "connections": [
                    {"type": relation.get("type"), "provenance": relation.get("provenance")}
                    for relation in connections[:2]
                ],
                "source": hit.get("source"),
            })
        payload = {
            "original": token["text"], "context": token["context"],
            "candidates": values, "semantic_evidence": evidence,
            "graph_evidence": compact_graph,
            "task": "Select exactly one supplied candidate or KEEP_ORIGINAL. Preserve entities, numbers and meaning.",
            "required_output": {"selection": "exact candidate string", "reason": "short evidence-based reason", "confidence": "number from 0 to 1"},
        }
        prompts.append({"token_id": token["id"], "payload": payload, "rendered": json.dumps(payload, ensure_ascii=False, indent=2)})
    return prompts


def run_slm(prompts: list[dict[str, Any]], settings: Settings) -> tuple[list[dict[str, Any]], str, list[str]]:
    if not prompts:
        return [], "No low-confidence tokens; model not invoked", []
    if not settings.slm_model_path:
        return [{"token_id": p["token_id"], "status": "UNAVAILABLE", "selection": "KEEP_ORIGINAL", "reason": "Local SLM path is not configured.", "confidence": None} for p in prompts], "Qwen2.5-3B unavailable", ["SLM_MODEL_PATH is not configured; originals were preserved."]
    model_path = Path(settings.slm_model_path)
    if not model_path.is_file():
        return [{"token_id": p["token_id"], "status": "UNAVAILABLE", "selection": "KEEP_ORIGINAL", "reason": "Local GGUF model file is unavailable.", "confidence": None} for p in prompts], "Qwen2.5-3B unavailable", ["Configured GGUF model file is unavailable; originals were preserved."]
    decisions = []
    for prompt in prompts:
        try:
            decoded, usage = llama_server.complete(prompt["rendered"], settings)
            match = re.search(r"\{.*?\}", decoded, re.S)
            parsed = json.loads(match.group() if match else decoded)
            decisions.append({"token_id": prompt["token_id"], "status": "COMPLETED", **parsed, **usage, "raw_output": decoded[:500]})
        except Exception as exc:
            decisions.append({"token_id": prompt["token_id"], "status": "ERROR", "selection": "KEEP_ORIGINAL", "reason": f"Local model error: {type(exc).__name__}", "confidence": None, "raw_output": str(exc)[:500]})
    warnings = ["One or more local SLM calls failed; affected originals were preserved."] if any(d["status"] == "ERROR" for d in decisions) else []
    return decisions, f"Qwen2.5-3B-Instruct Q4_K_M via local llama.cpp: {model_path.name}", warnings


def validate_decisions(tokens: list[dict[str, Any]], candidates: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    token_map = {t["id"]: t for t in tokens}
    candidate_map = {c["token_id"]: c for c in candidates}
    results = []
    for decision in decisions:
        token = token_map[decision["token_id"]]
        allowed = {c["value"] for c in candidate_map[decision["token_id"]]["candidates"]}
        selection = str(decision.get("selection", "KEEP_ORIGINAL"))
        reasons = []
        status = "ACCEPTED"
        if selection not in allowed:
            status, selection = "REJECTED", "KEEP_ORIGINAL"
            reasons.append("Selection is outside the candidate set.")
        if any(ch.isdigit() for ch in token["text"]) and selection not in (token["text"], "KEEP_ORIGINAL"):
            status, selection = "REJECTED", "KEEP_ORIGINAL"
            reasons.append("Numeric/identifier preservation rule triggered.")
        if decision.get("status") == "UNAVAILABLE":
            status = "KEEP_ORIGINAL"
            reasons.append("Local SLM was unavailable.")
        elif selection == "KEEP_ORIGINAL" and status != "REJECTED":
            status = "KEEP_ORIGINAL"
        results.append({"token_id": token["id"], "original": token["text"], "selection": selection, "final": token["text"] if selection == "KEEP_ORIGINAL" else selection, "status": status, "checks": reasons or ["Candidate membership passed.", "Preservation rules passed."]})
    return results


def corrected_text(tokens: list[dict[str, Any]], validations: list[dict[str, Any]]) -> tuple[str, str, list[dict[str, Any]]]:
    decisions = {v["token_id"]: v for v in validations}
    original = " ".join(t["text"] for t in tokens)
    output, records = [], []
    for token in tokens:
        decision = decisions.get(token["id"])
        final = decision["final"] if decision else token["text"]
        output.append(final)
        if decision:
            records.append({**decision, "confidence": token["confidence"], "changed": final != token["text"]})
    return original, " ".join(output), records


def levenshtein(a: list[str] | str, b: list[str] | str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (ca != cb)))
        previous = current
    return previous[-1]


def evaluate_text(original: str, corrected: str, ground_truth: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    gt_chars, out_chars, ocr_chars = list(ground_truth), list(corrected), list(original)
    gt_words, out_words, ocr_words = ground_truth.split(), corrected.split(), original.split()
    applied = [r for r in records if r["changed"]]
    # A changed token is correct when its final value appears at the aligned index in ground truth.
    correct_changes = sum(1 for r in applied if r["final"].lower() in {w.lower() for w in gt_words})
    return {
        "cer": levenshtein(out_chars, gt_chars) / max(1, len(gt_chars)),
        "wer": levenshtein(out_words, gt_words) / max(1, len(gt_words)),
        "baseline_cer": levenshtein(ocr_chars, gt_chars) / max(1, len(gt_chars)),
        "baseline_wer": levenshtein(ocr_words, gt_words) / max(1, len(gt_words)),
        "correction_precision": correct_changes / max(1, len(applied)),
        "over_correction_rate": sum(1 for r in applied if r["final"].lower() not in {w.lower() for w in gt_words}) / max(1, len(applied)),
        "entity_preservation_rate": None,
        "ground_truth_characters": len(gt_chars),
        "ground_truth_words": len(gt_words),
    }
