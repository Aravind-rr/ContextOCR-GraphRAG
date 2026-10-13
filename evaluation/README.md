# ContextOCR Evaluation Dataset

This directory contains a reproducible synthetic evaluation dataset and offline metric tooling. Dataset generation does **not** import or invoke PaddleOCR, DocLayout-YOLO, FAISS, BGE-M3, Neo4j, GraphRAG, Qwen, or the application pipeline.

## Dataset composition

| Property | Images | PDFs | Total |
|---|---:|---:|---:|
| Documents | 15 | 15 | 30 |
| Clean documents | 5 | 5 | 10 |
| Intentionally challenging documents | 10 | 10 | 20 |
| Manual-execution split | 5 | 5 | 10 |
| Reserved unprocessed split | 10 | 10 | 20 |

Each category has six unique documents: three PNGs and three one-page PDFs.

| Category | Count |
|---|---:|
| Medical reports and medication records | 6 |
| Invoices and receipts | 6 |
| Certificates | 6 |
| Application forms | 6 |
| Financial summaries | 6 |

Every document is fictional and prominently marked as synthetic. Names, identifiers, medical observations, transactions, and financial figures are invented.

## Controlled variations

Intentional textual corruption and actual OCR errors are different concepts:

- **Injected corruption** is visible text deliberately altered by the generator. Its correct intended form remains in the separate ground-truth file and its exact transformation is recorded in both manifests.
- **Observed OCR error** can only be established after a genuine manual pipeline run. No observed OCR errors are claimed by this dataset.

The 20 challenging documents contain two controlled textual corruptions each (40 events total). Counts include both formats.

| Injected error type | Events |
|---|---:|
| Missing character | 10 |
| Duplicated character | 8 |
| l/1 substitution | 8 |
| i/l substitution | 8 |
| O/0 substitution | 2 |
| Misspelling | 2 |
| c/e substitution | 2 |

Visual variations include mild sparse noise, sub-degree rotation, low contrast, mild blur, varied font sizes, and tighter line spacing. Some documents have more than one visual tag. Clean documents have no injected textual corruption and no deliberate quality degradation.

## Ground-truth procedure

The generator constructs the correct intended text first. That canonical text is saved as UTF-8 under `dataset/ground_truth/`. A separate rendered copy is then created and the declared corruptions are applied only to the rendered input. Consequently:

1. The ground truth never contains an injected corruption.
2. The manifest records every intended-to-rendered transformation.
3. Ground truth is not represented as OCR output or as a successful correction.
4. Raw manually captured OCR and corrected outputs remain separate and auditable.

## Directory structure

```text
evaluation/
  generate_dataset.py
  validate_dataset.py
  calculate_metrics.py
  README.md
  REPORT_TEMPLATE.md
  dataset/
    inputs/images/                 # IMG001-IMG015 PNG inputs
    inputs/pdfs/                   # PDF001-PDF015 one-page PDF inputs
    ground_truth/images/           # IMG001-IMG015 canonical UTF-8 text
    ground_truth/pdfs/             # PDF001-PDF015 canonical UTF-8 text
    manifests/dataset_manifest.csv
    manifests/dataset_manifest.json
  actual_runs/
    README.md
    IMG001/ ... IMG005/
    PDF001/ ... PDF005/
  results/                         # Created by calculate_metrics.py
```

## Reproducible commands

Run these from the repository root with the existing virtual environment. They perform no inference.

```powershell
.\.venv\Scripts\python.exe evaluation\generate_dataset.py
.\.venv\Scripts\python.exe evaluation\validate_dataset.py
.\.venv\Scripts\python.exe evaluation\calculate_metrics.py --self-test
```

After genuine outputs have been manually saved:

```powershell
.\.venv\Scripts\python.exe evaluation\calculate_metrics.py
```

That command writes `evaluation/results/metrics.json` and `evaluation/results/metrics.csv`. Missing output files are explicitly reported as `NOT_YET_MEASURED`; no values are invented.

## Manual execution protocol

Only these ten inputs are assigned to manual execution:

- `IMG001` through `IMG005`
- `PDF001` through `PDF005`

Run one document at a time through the website. Record the threshold and system configuration. Save genuine outputs and evidence using the filenames described in `actual_runs/README.md`. Then manually change that document's manifest status from `NOT_RUN` to `ACTUALLY_RUN` in both manifest files. Do not change status if the run was not completed and recorded.

`IMG006-IMG015` and `PDF006-PDF015` remain reserved and unprocessed.

## Metric definitions

For reference sequence length `N`:

- `CER = (character substitutions + deletions + insertions) / reference characters`
- `WER = (word substitutions + deletions + insertions) / reference words`
- `Character-level normalized accuracy = 1 - CER`
- `Absolute CER reduction = baseline CER - post-correction CER`
- `Relative CER reduction = absolute CER reduction / baseline CER`, undefined when baseline CER is zero

Text normalization uses Unicode NFKC, lowercasing, and whitespace collapsing. Punctuation and numbers are retained. Raw files are never modified.

Correction events are derived from aligned word-level errors. `errors_corrected` are baseline error events absent after correction; `errors_introduced` are new post-correction error events; and `errors_remaining` occur in both. Token-level correction accuracy is successful corrections divided by actual changed positions/insertion events. Unchanged correct words are excluded, preventing inflated results.

Empty references are invalid. Missing outputs are `NOT_YET_MEASURED`. Failed runs can be marked in `run_metadata.json` and are reported as `FAILED_RUN`.

## Limitations and future plan

- Synthetic documents cannot reproduce the full distribution of real scans, handwriting, cameras, or institutional forms.
- Injected corruption does not guarantee the OCR engine will make the corresponding error.
- Thirty documents are sufficient for a controlled capstone demonstration, not for broad statistical generalization.
- The reserved split should remain untouched until the evaluation protocol is frozen.
- Future work should add ethically sourced, de-identified real documents, repeated runs, confidence intervals, hardware-normalized runtime measurements, and independent annotation review.

