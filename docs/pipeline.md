# Pipeline

The 15 numbered stages match the UI. Preprocessing decisions depend on image statistics. Embedded PDFs use source text; scanned pages use PaddleOCR when installed or Tesseract. Only confidence values below configurable τ proceed. Candidates retain vocabulary provenance and always include `KEEP_ORIGINAL`. Evidence keeps source page and similarity. Graph entities are conservatively extracted from high-confidence text. Prompts are per-token JSON payloads. Model responses are parsed and then independently checked for candidate membership and identifier preservation. Originals are permanently retained.

