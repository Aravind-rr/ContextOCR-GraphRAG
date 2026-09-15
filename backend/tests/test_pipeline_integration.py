import asyncio
import json
import uuid
from pathlib import Path

import fitz

from app.core.config import get_settings
from app.core.store import store
from app.pipeline.runner import initial_stages, pipeline_runner


def test_digital_pdf_complete_pipeline(tmp_path: Path):
    pdf_path=tmp_path/"fixture.pdf"
    pdf=fitz.open();page=pdf.new_page();page.insert_text((72,72),"University Government regulation research document");pdf.save(pdf_path);pdf.close()
    doc_id,run_id=str(uuid.uuid4()),str(uuid.uuid4())
    now="2026-01-01T00:00:00+00:00"
    store.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?)",(doc_id,"fixture.pdf",pdf_path.name,str(pdf_path),"application/pdf",pdf_path.stat().st_size,1,now,1))
    config={"document_id":doc_id,"confidence_threshold":.7,"domain":"government","start_stage":1}
    store.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",(run_id,doc_id,"QUEUED",now,now,json.dumps(config),"{}",json.dumps(initial_stages())))
    asyncio.run(pipeline_runner.run(run_id))
    run=pipeline_runner.load_run(run_id)
    assert run["status"] == "COMPLETED"
    assert run["result"]["stats"]["ocr_tokens"] == 5
    assert run["result"]["stats"]["low_confidence_tokens"] == 0
    assert run["result"]["corrected_text"] == run["result"]["original_text"]
    output_pdf = Path(run["result"]["export_pdf_path"])
    assert output_pdf.is_file()
    with fitz.open(output_pdf) as exported:
        assert "University Government regulation research document" in "".join(page.get_text() for page in exported)
    assert all(s["status"] in ("COMPLETED","WARNING") for s in run["stages"])
