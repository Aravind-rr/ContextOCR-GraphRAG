# Actual OCR Evaluation Results

These results were calculated from genuine pipeline runs of `IMG001–IMG005` and `PDF001–PDF005` at a confidence threshold of **0.90**. The remaining 20 dataset documents were not processed.

## Experimental setup

- Images used PaddleOCR; PDFs used PyMuPDF embedded-text extraction because the generated PDFs contain a text layer.
- Medical documents used the `medical` domain; invoice/receipt documents used `financial`.
- The full image pipeline included DocLayout-YOLO, BGE-M3/FAISS retrieval, Neo4j GraphRAG, Qwen2.5-3B selection, validation, and searchable-PDF export.
- Normalization: Unicode NFKC, lowercase, collapsed whitespace, with punctuation and numbers retained.

## Aggregate metrics

| Group | N | Baseline CER | Corrected CER | Baseline WER | Corrected WER | Improved / unchanged / worsened |
|---|---:|---:|---:|---:|---:|---:|
| Images | 5 | 17.91% | 18.30% | 46.38% | 46.66% | 0 / 3 / 2 |
| PDFs | 5 | 8.48% | 8.48% | 22.99% | 22.99% | 0 / 5 / 0 |
| Overall | 10 | 13.19% | 13.39% | 34.69% | 34.83% | 0 / 8 / 2 |

The ten runs examined **9** low-confidence tokens, made **8** successful Qwen calls, and applied **4** token substitutions. Total summed pipeline-stage time was **566.6 seconds**.

## Per-document results

| ID | Format | Outcome | Baseline CER | Corrected CER | Baseline WER | Corrected WER | Low-confidence | Applied | Runtime (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| IMG001 | image | UNCHANGED | 15.59% | 15.59% | 41.25% | 41.25% | 2 | 0 | 188.7 |
| IMG002 | image | WORSENED | 16.53% | 16.73% | 46.48% | 47.89% | 3 | 2 | 83.8 |
| IMG003 | image | UNCHANGED | 20.88% | 20.88% | 48.78% | 48.78% | 1 | 1 | 75.0 |
| IMG004 | image | UNCHANGED | 19.11% | 19.11% | 47.95% | 47.95% | 2 | 0 | 72.6 |
| IMG005 | image | WORSENED | 17.42% | 19.18% | 47.44% | 47.44% | 1 | 1 | 100.3 |
| PDF001 | pdf | UNCHANGED | 7.60% | 7.60% | 20.00% | 20.00% | 0 | 0 | 9.2 |
| PDF002 | pdf | UNCHANGED | 8.37% | 8.37% | 23.94% | 23.94% | 0 | 0 | 9.2 |
| PDF003 | pdf | UNCHANGED | 7.85% | 7.85% | 20.73% | 20.73% | 0 | 0 | 10.6 |
| PDF004 | pdf | UNCHANGED | 9.56% | 9.56% | 24.66% | 24.66% | 0 | 0 | 8.7 |
| PDF005 | pdf | UNCHANGED | 9.00% | 9.00% | 25.64% | 25.64% | 0 | 0 | 8.6 |

## Applied correction trace

| ID | Original | Final | Validation status | Model confidence |
|---|---|---|---|---:|
| IMG002 | Clinic | clinical | ACCEPTED | 0.857 |
| IMG002 | PATLENT | patient | ACCEPTED | 0.857 |
| IMG003 | PATLENT | tablet | ACCEPTED | 0.857 |
| IMG005 | contact | demonstration | ACCEPTED | 0.827 |

## Pipeline warnings

| ID | Stage | Warning |
|---|---|---|
| IMG002 | Qwen2.5-3B Selection | One or more local SLM calls failed; affected originals were preserved. |

## Evidence-based interpretation

- Macro CER changed from **13.19%** to **13.39%**; macro WER changed from **34.69%** to **34.83%**.
- Macro character-level normalized accuracy (`1 − CER`) changed from **86.81%** to **86.61%**.
- Outcomes were **0 improved**, **8 unchanged**, and **2 worsened**. These are reported as observed, without replacing or hand-editing model output.
- The five PDFs produced no low-confidence selections because embedded-text extraction assigned no calibrated low-confidence tokens. Their injected spelling corruptions therefore passed through unchanged.
- The image runs demonstrate that the full confidence-aware/Qwen route executed, but three accepted substitutions were not supported by the intended reference (`Clinic → clinical`, `PATLENT → tablet`, and `contact → demonstration`). This exposes a candidate/validation precision limitation rather than a runtime failure.
- Baseline error rates include template/header/footer and OCR punctuation differences because the strict normalization policy intentionally retains punctuation and numbers.
- None of the measured baselines had CER exactly zero, so clean-document preservation is not defined for this ten-document run under the strict metric policy.

## Artifact locations

- Per-run raw OCR, corrected text, searchable PDF, trace, evidence, graph, timings, and full pipeline record: `evaluation/actual_runs/<ID>/`
- Machine-readable complete metrics: `evaluation/results/metrics.json`
- Flat per-document metrics: `evaluation/results/metrics.csv`

## Reproduce

```powershell
.\.venv\Scripts\python.exe evaluation\run_evaluation_subset.py
.\.venv\Scripts\python.exe evaluation\calculate_metrics.py
.\.venv\Scripts\python.exe evaluation\generate_results_report.py
.\.venv\Scripts\python.exe evaluation\validate_dataset.py
```
