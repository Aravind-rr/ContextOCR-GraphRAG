# Genuine Manual Run Evidence

The ten document folders here are intentionally empty except for `.gitkeep`. Do not place synthetic or predicted results in them.

After manually processing the matching input through the existing website, save the genuine evidence in its folder:

| Filename | Required | Contents |
|---|---|---|
| `ocr_raw.txt` | Yes | Exact unedited OCR text produced before correction |
| `corrected.txt` | Yes | Exact final corrected text produced by the system |
| `corrected.pdf` | Recommended | Actual selectable-text PDF exported by the system |
| `pipeline.json` | Recommended | Complete stage outputs, evidence, correction trace, and provenance |
| `run_metadata.json` | Yes | Run ID, timestamp, status, threshold, domain, hardware, versions, and measured runtime |
| `notes.md` | Optional | Operator observations and failures; clearly distinguish observation from interpretation |
| `screenshots/` | Optional | Genuine interface screenshots from the recorded run |

Example `run_metadata.json` structure (illustrative only):

```json
{
  "document_id": "IMG001",
  "status": "ACTUALLY_RUN",
  "run_id": "replace-with-real-run-id",
  "timestamp": "replace-with-real-timestamp",
  "confidence_threshold": "replace-with-real-value",
  "domain": "replace-with-selected-domain",
  "runtime_seconds": "replace-with-measured-value",
  "hardware": "replace-with-actual-hardware",
  "notes": "replace-with-observed-facts"
}
```

Do not copy the illustrative values as experimental evidence. If a run fails, set `status` to `FAILED` and record the factual failure reason. Preserve raw files exactly as exported.

