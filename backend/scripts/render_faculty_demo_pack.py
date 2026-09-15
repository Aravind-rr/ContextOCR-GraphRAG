"""Render the faculty PDF pack to PNG files for visual QA."""

from pathlib import Path

import fitz


source_dir = Path("output/pdf")
target_dir = Path("tmp/pdfs")
target_dir.mkdir(parents=True, exist_ok=True)
for pdf_path in sorted(source_dir.glob("*.pdf")):
    with fitz.open(pdf_path) as document:
        if len(document) != 1:
            raise ValueError(f"Expected one page in {pdf_path}")
        pixmap = document[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        target = target_dir / f"{pdf_path.stem}.png"
        pixmap.save(target)
        print(target)
