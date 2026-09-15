# Dynamic document-quality evaluation

Every evaluation is computed from completed pipeline outputs at request time. It receives a unique UUID and UTC timestamp and writes an immutable run directory under `data/evaluations/<evaluation-id>/`. No metric or chart contains example values.

## Metrics

| Metric | Exact definition | Range | Better | Ground truth |
|---|---|---:|---|---|
| CER | `(character substitutions + deletions + insertions) / reference characters` | `0..infinity` | Lower | Required |
| WER | `(word substitutions + deletions + insertions) / reference words` | `0..infinity` | Lower | Required |
| OCR confidence | Character-count-weighted mean of actual PaddleOCR/Tesseract token scores | `0..1` | Higher | No |
| Noise ratio | Unique invalid control/private/surrogate/replacement, excessive-symbol, repeated-artifact, or mojibake character positions divided by extracted characters | `0..1` | Lower | No |
| Valid-word ratio | Linguistically valid alphabetic tokens divided by eligible alphabetic tokens; configured length/repetition rules and an English vowel rule are applied | `0..1` | Higher | No |
| Dictionary match ratio | Eligible tokens at or above the configured `wordfreq` Zipf threshold divided by eligible tokens | `0..1` | Higher | No |
| Non-alphanumeric ratio | Characters for which Unicode `isalnum()` is false divided by extracted characters | `0..1` | Lower | No |
| Repeated/artifact ratio | Repeated-run and replacement/mojibake character positions divided by extracted characters | `0..1` | Lower | No |
| Average word length | Alphabetic characters in valid words divided by valid words | `0..infinity` | Contextual | No |
| Vocabulary richness | Unique case-folded eligible words divided by eligible words | `0..1` | Contextual | No |
| Language consistency | Deterministic `langdetect` probability assigned to the configured expected language | `0..1` | Higher | No |
| Text density | Alphanumeric extracted characters divided by processed page area in megapixels | `0..infinity` | Contextual | No |
| Structural consistency | Mean of fragmentation quality, valid bounding-box ratio, page token-count balance, and line coherence when available | `0..1` | Higher | No |

Missing denominators produce `null`, never zero. Empty extracted text receives a quality score of zero and `Low Quality`, while unavailable individual metrics remain `null`.

## Quality score

The fixed configuration is in `backend/evaluation_config.json`. Each available component is clamped to `0..1`; unavailable components are removed and the remaining weights are renormalized to sum to one.

`quality score = 100 * sum(normalized active weight_i * component_i)`

| Component | Weight |
|---|---:|
| OCR confidence | 0.15 |
| Character accuracy (`max(0, 1-CER)`) | 0.20 |
| Word accuracy (`max(0, 1-WER)`) | 0.15 |
| Valid-word ratio | 0.10 |
| Dictionary match ratio | 0.08 |
| Noise cleanliness (`1-noise ratio`) | 0.10 |
| Artifact cleanliness (`1-artifact ratio`) | 0.05 |
| Language consistency | 0.07 |
| Structural consistency | 0.05 |
| Non-alphanumeric cleanliness (`1-non-alphanumeric ratio`) | 0.03 |
| Density quality (`min(1, density/configured target)`) | 0.02 |

With ground truth, CER and WER components participate. Without it they are `null` and their combined weight is redistributed proportionally over valid reference-free components. Ground truth is accepted by API/UI or discovered as `<document-id>.txt` or `<filename-stem>.txt` in the configured directory.

Default classes are fixed per run configuration: `80-100 High Quality`, `60-79.9999 Medium Quality`, and `0-59.9999 Low Quality`. They are configurable but never silently optimized.

## Aggregation, charts, and labelled evaluation

Dataset summaries include count, mean, median, minimum, maximum, population standard deviation, P25, and P75. CER/WER expose macro document statistics and micro aggregate errors divided by total reference units. Class counts and percentages are recomputed each run.

For one document, a quality-component chart is generated. With two or more documents, valid score/class distributions, metric distributions, and scatter plots are generated. CER/WER charts require ground truth. Correction before/after charts compare original OCR and corrected output only when ground truth exists. Pre-preprocessing OCR comparison is explicitly unavailable because the pipeline does not persist a separate OCR pass over unprocessed pages; it is never fabricated.

Confusion matrices and accuracy, precision, recall, F1, and specificity are returned only when at least two labelled examples spanning at least two classes are supplied. ROC/AUC is omitted because threshold classes do not provide independent probabilistic class scores.

## Freshness and reproducibility

Each document records SHA-256 hashes of its source file, relevant pipeline outputs, configuration, and ground truth. Every request recalculates and writes a new directory; chart paths are resolved only through the current manifest. Identical inputs yield the same signature and values but a different evaluation ID. Any source, OCR output, corrected text, ground truth, or configuration change changes the signature.

Each directory contains `report.json`, `metrics.json`, `summary.json`, `document_results.csv`, `graph_source_data.json`, `evaluation.log`, and `charts/`.

## Run

In the Evaluation page, leave ground truth blank for reference-free analysis or paste verified text to enable CER/WER. Select **Evaluate current document** or **Evaluate latest completed dataset**.

```powershell
Invoke-RestMethod http://localhost:8000/api/evaluation/run -Method Post -ContentType application/json -Body '{"run_id":"PROCESSING_RUN_ID","ground_truth":null}'
Invoke-RestMethod http://localhost:8000/api/evaluation/dataset -Method Post -ContentType application/json -Body '{"run_ids":[],"ground_truths":{},"quality_labels":{}}'
```
