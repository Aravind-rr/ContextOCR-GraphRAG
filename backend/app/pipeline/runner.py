from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.store import store
from app.schemas.api import StageRecord, StageStatus
from app.services.local_models import (
    release_embedding_model,
    release_layout_model,
    release_paddle_ocr,
)

from .processing import (analyze_confidence, build_prompts, corrected_text,
    detect_layout, extract_graph, extract_ocr, generate_candidates,
    graph_evidence, make_chunks, preprocess_page, reading_order,
    render_document, run_slm, semantic_retrieve, validate_decisions)
from .pdf_export import export_corrected_pdf


PIPELINE_STAGES = [
    ("document_input", "Document Input"), ("preprocessing", "Preprocessing"),
    ("layout", "Layout Detection"), ("reading_order", "Reading Order"),
    ("ocr", "OCR Extraction"), ("confidence", "Confidence Analysis"),
    ("low_confidence", "Low-Confidence Detection"), ("candidates", "Candidate Generation"),
    ("semantic", "Semantic Retrieval"), ("graph", "Knowledge Graph"),
    ("graphrag", "Hybrid GraphRAG"), ("prompt", "Dynamic Prompt Construction"),
    ("slm", "Qwen2.5-3B Selection"), ("validation", "Validation"),
    ("output", "Final Output"),
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def initial_stages() -> list[dict[str, Any]]:
    return [StageRecord(key=key, number=i + 1, name=name, next_step=PIPELINE_STAGES[i+1][1] if i+1 < len(PIPELINE_STAGES) else None).model_dump(mode="json") for i, (key, name) in enumerate(PIPELINE_STAGES)]


class PipelineRunner:
    def __init__(self) -> None:
        self.queues: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}

    def subscribe(self, run_id: str) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.queues.setdefault(run_id, []).append(queue)
        return queue

    async def emit(self, run_id: str, event: dict[str, Any]) -> None:
        for queue in self.queues.get(run_id, []):
            await queue.put(event)

    def load_run(self, run_id: str) -> dict[str, Any]:
        row = store.one("SELECT * FROM runs WHERE id=?", (run_id,))
        if not row:
            raise KeyError(run_id)
        return store.decode(row)

    def save(self, run: dict[str, Any]) -> None:
        run["updated_at"] = now()
        store.execute("UPDATE runs SET status=?,updated_at=?,result_json=?,stages_json=? WHERE id=?", (run["status"], run["updated_at"], json.dumps(run["result"]), json.dumps(run["stages"]), run["id"]))

    async def _stage(self, run: dict[str, Any], index: int, method: str, function: Any) -> Any:
        stage = run["stages"][index]
        stage.update(status=StageStatus.running.value, started_at=now(), method=method)
        self.save(run)
        await self.emit(run["id"], {"type": "stage", "stage": stage})
        started = time.perf_counter()
        try:
            value = await asyncio.to_thread(function)
            stage["runtime_ms"] = round((time.perf_counter() - started) * 1000, 2)
            stage["completed_at"] = now()
            stage["status"] = StageStatus.warning.value if stage["warnings"] else StageStatus.completed.value
            self.save(run)
            await self.emit(run["id"], {"type": "stage", "stage": stage})
            return value
        except Exception as exc:
            stage.update(status=StageStatus.failed.value, completed_at=now(), runtime_ms=round((time.perf_counter()-started)*1000, 2), error=str(exc))
            run["status"] = "FAILED"
            self.save(run)
            await self.emit(run["id"], {"type": "stage", "stage": stage})
            raise

    async def run(self, run_id: str) -> None:
        run = self.load_run(run_id)
        doc = store.decode(store.one("SELECT * FROM documents WHERE id=?", (run["document_id"],)))
        settings = get_settings()
        config = run["config"]
        threshold = float(config["confidence_threshold"])
        work = settings.data_dir / "processed" / run_id
        work.mkdir(parents=True, exist_ok=True)
        result = run["result"]
        run["status"] = "RUNNING"
        self.save(run)
        try:
            def s1():
                pages, meta = render_document(Path(doc["path"]), work / "original")
                result.update(document=doc, page_images=[str(p) for p in pages], document_meta=meta)
                run["stages"][0].update(input={"filename": doc["name"], "size": doc["size"]}, output={"pages": len(pages), "digital_pdf": meta["digital_pdf"]}, diagnostics={"media_type": doc["media_type"]})
                return pages
            pages = await self._stage(run, 0, "Validated file ingestion and page rendering", s1)

            def s2():
                output_dir = work / "preprocessed"; output_dir.mkdir(exist_ok=True)
                details = [preprocess_page(p, output_dir / p.name) for p in pages]
                result["preprocessing"] = details
                run["stages"][1].update(input={"pages": len(pages)}, parameters={"conditional_operations": True}, output=details, diagnostics={"operations": sorted({op for d in details for op in d["operations"]})})
                return [Path(d["output"]) for d in details]
            processed = await self._stage(run, 1, "OpenCV conditional preprocessing", s2)

            def s3():
                detected = [detect_layout(p, settings) for p in processed]
                layouts = [{"page": i+1, "regions": item[0], "method": item[1]} for i, item in enumerate(detected)]
                result["layout"] = layouts
                warnings = [warning for item in detected for warning in item[2]]
                methods = sorted({item[1] for item in detected})
                run["stages"][2].update(input={"pages": len(processed)}, method="; ".join(methods), parameters={"confidence": settings.doc_layout_confidence, "device": settings.doc_layout_device}, output=layouts, diagnostics={"region_count": sum(len(x["regions"]) for x in layouts), "methods": methods}, warnings=warnings)
                return layouts
            try:
                layouts = await self._stage(run, 2, "Layout analysis", s3)
            finally:
                release_layout_model()

            def s4():
                orders = [{"page": p["page"], "regions": reading_order(p["regions"])} for p in layouts]
                result["reading_order"] = orders
                run["stages"][3].update(input={"regions": sum(len(p["regions"]) for p in layouts)}, method="Column-aware geometric ordering", output=orders, diagnostics={"ordered_regions": sum(len(p["regions"]) for p in orders)})
                return orders
            await self._stage(run, 3, "Geometric reading-order reconstruction", s4)

            def s5():
                tokens, method, warnings = extract_ocr(Path(doc["path"]), processed, result["document_meta"], settings.ocr_engine, settings)
                result["tokens"] = tokens
                calibrated = [t for t in tokens if t.get("confidence_calibration")]
                run["stages"][4].update(input={"pages": len(processed)}, method=method, output={"token_count": len(tokens), "preview": " ".join(t["text"] for t in tokens[:40]), "calibrated_tokens": [{"text": t["text"], "raw_ocr_confidence": t.get("ocr_confidence"), "effective_confidence": t["confidence"], "evidence": t["confidence_calibration"]} for t in calibrated]}, diagnostics={"engine": method, "raw_score_scope": "line", "effective_score_scope": "token", "calibrated_token_count": len(calibrated)}, warnings=warnings)
                if not tokens: raise RuntimeError("OCR completed but returned no tokens.")
                return tokens
            try:
                tokens = await self._stage(run, 4, "Configured OCR engine", s5)
            finally:
                release_paddle_ocr()

            def s6():
                analysis = analyze_confidence(tokens, threshold); result["confidence"] = analysis
                run["stages"][5].update(input={"tokens": len(tokens)}, method="Token confidence distribution and configurable threshold", parameters={"threshold": threshold}, output=analysis, diagnostics={"confidence_source": sorted({t["confidence_source"] for t in tokens})})
                return analysis
            analysis = await self._stage(run, 5, "Confidence analysis", s6)

            def s7():
                selected = []
                for token in tokens:
                    if token["confidence"] >= threshold:
                        continue
                    raw = float(token.get("ocr_confidence", token["confidence"]))
                    calibration = token.get("confidence_calibration") or {}
                    reason = f"effective confidence {token['confidence']:.3f} < threshold {threshold:.3f}"
                    if calibration:
                        reason += f"; raw PaddleOCR line score {raw:.3f}, lexical near-match {calibration.get('candidate')} ({float(calibration.get('similarity', 0)):.3f})"
                    selected.append({
                        "id": token["id"], "text": token["text"],
                        "confidence": token["confidence"], "ocr_confidence": raw,
                        "confidence_calibration": calibration, "page": token["page"],
                        "reason": reason,
                    })
                result["low_confidence"] = selected
                run["stages"][6].update(input={"tokens": len(tokens)}, method="Selective confidence gate", parameters={"threshold": threshold}, output=selected, diagnostics={"selected": len(selected), "kept": len(tokens)-len(selected)})
                return selected
            await self._stage(run, 6, "Confidence gate", s7)

            def s8():
                values = generate_candidates(tokens, threshold, config["domain"], settings.candidate_count); result["candidates"] = values
                run["stages"][7].update(input={"selected_tokens": len(values)}, method="RapidFuzz edit similarity + domain/document vocabulary", parameters={"domain": config["domain"], "limit": settings.candidate_count}, output=values, diagnostics={"candidate_sets": len(values)})
                return values
            candidates = await self._stage(run, 7, "Provenance-aware lexical candidate generation", s8)

            def s9():
                chunks = make_chunks(tokens, settings.chunk_size, threshold)
                evidence, method, warnings = semantic_retrieve(tokens, candidates, chunks, settings)
                result.update(chunks=chunks, semantic_evidence=evidence)
                run["stages"][8].update(input={"chunks": len(chunks), "queries": len(candidates)}, method=method, parameters={"top_k": settings.retrieval_top_k, "chunk_size_tokens": settings.chunk_size}, output=evidence, diagnostics={"evidence_sets": len(evidence)}, warnings=warnings)
                return evidence
            try:
                semantic = await self._stage(run, 8, "Semantic evidence retrieval", s9)
            finally:
                release_embedding_model()

            def s10():
                entities, relationships = extract_graph(tokens, doc["id"], config["domain"], doc["name"], threshold)
                result["graph"] = {"schema_version": "2.0", "domain": config["domain"], "nodes": entities, "edges": relationships, "persisted_to_neo4j": False}
                warnings = []
                if settings.neo4j_uri:
                    try:
                        from neo4j import GraphDatabase
                        with GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)) as driver:
                            with driver.session() as session:
                                session.run("MATCH (e:Entity {document_id:$doc}) DETACH DELETE e", doc=doc["id"])
                                session.run("MERGE (d:Document {id:$id}) SET d.name=$name,d.label=$name,d.type='Document',d.document_id=$id,d.domain=$domain", id=doc["id"], name=doc["name"], domain=config["domain"])
                                for entity in entities:
                                    if entity["type"] == "Document":
                                        continue
                                    safe_label = re.sub(r"[^A-Za-z0-9_]", "", entity["type"]) or "KnowledgeEntity"
                                    session.run(
                                        f"MERGE (e:Entity:{safe_label} {{id:$id}}) "
                                        "SET e.label=$label,e.type=$type,e.document_id=$doc,e.pages=$pages,"
                                        "e.confidence=$confidence,e.token_ids=$token_ids,e.provenance_json=$provenance,"
                                        "e.properties_json=$properties",
                                        id=entity["id"], label=entity["label"], type=entity["type"],
                                        doc=doc["id"], pages=entity["pages"], confidence=entity["confidence"],
                                        token_ids=entity.get("token_ids", []),
                                        provenance=json.dumps(entity.get("provenance", [])),
                                        properties=json.dumps(entity.get("properties", {})),
                                    )
                                for relationship in relationships:
                                    safe_type = re.sub(r"[^A-Z0-9_]", "", relationship["type"].upper()) or "RELATED_TO"
                                    session.run(
                                        f"MATCH (a {{id:$source}}),(b {{id:$target}}) "
                                        f"MERGE (a)-[r:{safe_type} {{id:$id}}]->(b) "
                                        "SET r.document_id=$doc,r.provenance=$provenance,"
                                        "r.token_ids=$token_ids,r.pages=$pages,r.properties_json=$properties",
                                        id=relationship["id"], source=relationship["source"],
                                        target=relationship["target"], doc=doc["id"],
                                        provenance=relationship["provenance"],
                                        token_ids=relationship.get("token_ids", []),
                                        pages=relationship.get("pages", []),
                                        properties=json.dumps(relationship.get("properties", {})),
                                    )
                        result["graph"]["persisted_to_neo4j"] = True
                    except Exception as exc: warnings.append(f"Neo4j persistence failed: {exc}")
                else: warnings.append("Neo4j is not configured; the actual extracted in-run graph remains inspectable but was not persisted.")
                run["stages"][9].update(input={"high_confidence_tokens": analysis["high_count"]}, method="Domain-aware entity and relationship extraction with token provenance + Neo4j persistence", output=result["graph"], diagnostics={"schema_version": "2.0", "nodes": len(entities), "edges": len(relationships), "relationship_types": sorted({edge["type"] for edge in relationships}), "persisted_to_neo4j": result["graph"]["persisted_to_neo4j"]}, warnings=warnings)
                return entities, relationships
            entities, relationships = await self._stage(run, 9, "Knowledge graph extraction", s10)

            def s11():
                graph_ev = graph_evidence(candidates, entities, relationships, settings)
                combined = [{"token_id": c["token_id"], "semantic": next((x["results"] for x in semantic if x["token_id"] == c["token_id"]), []), "graph": next((x["results"] for x in graph_ev if x["token_id"] == c["token_id"]), [])} for c in candidates]
                result.update(graph_evidence=graph_ev, combined_evidence=combined)
                run["stages"][10].update(input={"semantic_sets": len(semantic), "graph_nodes": len(entities)}, method="Provenance-preserving semantic + graph evidence union", output=combined, diagnostics={"combined_sets": len(combined)})
                return graph_ev
            graph_ev = await self._stage(run, 10, "Hybrid GraphRAG evidence assembly", s11)

            def s12():
                prompts = build_prompts(tokens, candidates, semantic, graph_ev); result["prompts"] = prompts
                run["stages"][11].update(input={"candidate_sets": len(candidates)}, method="Per-token constrained JSON prompt template", output=prompts, diagnostics={"prompt_count": len(prompts)})
                return prompts
            prompts = await self._stage(run, 11, "Dynamic prompt construction", s12)

            def s13():
                decisions, method, warnings = run_slm(prompts, settings); result["slm_decisions"] = decisions
                completed = [d for d in decisions if d["status"] == "COMPLETED"]
                run["stages"][12].update(input={"prompts": len(prompts)}, method=method, output=decisions, diagnostics={"calls": len(completed), "unavailable": sum(d["status"] == "UNAVAILABLE" for d in decisions), "errors": sum(d["status"] == "ERROR" for d in decisions), "total_model_latency_ms": round(sum(float(d.get("latency_ms", 0)) for d in completed), 2)}, warnings=warnings)
                return decisions
            decisions = await self._stage(run, 12, "Candidate-constrained local inference", s13)

            def s14():
                validations = validate_decisions(tokens, candidates, decisions); result["validations"] = validations
                run["stages"][13].update(input={"decisions": len(decisions)}, method="Candidate membership + identifier preservation policy", output=validations, diagnostics=dict(Counter(v["status"] for v in validations)))
                return validations
            from collections import Counter
            validations = await self._stage(run, 13, "Independent policy validation", s14)

            def s15():
                original, corrected, records = corrected_text(tokens, validations)
                traces = []
                for record in records:
                    token_id = record["token_id"]
                    traces.append({**record, "why_selected": next((x["reason"] for x in result["low_confidence"] if x["id"] == token_id), ""), "candidates": next((x["candidates"] for x in candidates if x["token_id"] == token_id), []), "semantic_evidence": next((x["results"] for x in semantic if x["token_id"] == token_id), []), "graph_evidence": next((x["results"] for x in graph_ev if x["token_id"] == token_id), []), "prompt": next((x["rendered"] for x in prompts if x["token_id"] == token_id), ""), "slm_decision": next((x for x in decisions if x["token_id"] == token_id), None)})
                pdf_out = settings.data_dir / "output" / f"{run_id}-corrected.pdf"
                export_corrected_pdf(pdf_out, tokens=tokens, correction_records=records, source_name=doc["name"], run_id=run_id, domain=config["domain"], threshold=threshold)
                json_out = settings.data_dir / "output" / f"{run_id}.json"
                result.update(original_text=original, corrected_text=corrected, correction_records=records, traces=traces, export_pdf_url=f"/api/runs/{run_id}/export/pdf", export_pdf_path=str(pdf_out), provenance_json_path=str(json_out), stats={"ocr_tokens": len(tokens), "low_confidence_tokens": len(candidates), "corrections_applied": sum(r["changed"] for r in records), "corrections_rejected": sum(r["status"] == "REJECTED" for r in records), "slm_calls": sum(d["status"] == "COMPLETED" for d in decisions)})
                json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
                run["stages"][14].update(input={"validated_records": len(validations)}, method="Immutable original + validated token substitution and searchable PDF generation", output={"original_text": original, "corrected_text": corrected, "pdf_export": str(pdf_out), "download_url": result["export_pdf_url"], "provenance_json": str(json_out)}, diagnostics={**result["stats"], "pdf_text_selectable": True})
            await self._stage(run, 14, "Final output assembly and searchable PDF export", s15)
            run["status"] = "COMPLETED"; self.save(run)
            await self.emit(run_id, {"type": "complete", "run_id": run_id})
        except Exception:
            return


pipeline_runner = PipelineRunner()
