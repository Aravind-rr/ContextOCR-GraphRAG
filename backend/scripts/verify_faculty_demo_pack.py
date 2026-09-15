"""Run the real preprocessing and PaddleOCR worker over every faculty fixture."""

from pathlib import Path

import fitz

from app.pipeline.processing import preprocess_page
from app.services.local_models import run_model_worker


source_dir = Path("output/pdf")
work_dir = Path("tmp/pdfs/ocr-verification")
work_dir.mkdir(parents=True, exist_ok=True)
inputs: list[Path] = []
names: list[str] = []
for pdf_path in sorted(source_dir.glob("*.pdf")):
    rendered = work_dir / f"{pdf_path.stem}-rendered.png"
    processed = work_dir / f"{pdf_path.stem}-processed.png"
    with fitz.open(pdf_path) as document:
        document[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(rendered)
    preprocess_page(rendered, processed)
    inputs.append(processed)
    names.append(pdf_path.name)

pages = run_model_worker({"action": "ocr", "image_paths": [str(path.resolve()) for path in inputs], "device": "cpu"}, timeout=360)
for name, page in zip(names, pages):
    print(f"\n{name}")
    for line in page["lines"]:
        print(f"  {line['score']:.3f}  {line['text']}")
