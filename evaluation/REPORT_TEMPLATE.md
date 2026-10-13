# Context-Aware OCR Post-Correction Evaluation Report

> Evidence status: **NOT_YET_MEASURED**. This template contains dataset facts and metric definitions only. Replace pending fields exclusively with genuine manually recorded results. Any example explicitly marked illustrative is not an experimental result.

## 1. Dataset description

The evaluation dataset contains 30 unique, synthetic, single-page documents: 15 PNG images and 15 PDFs. It covers medical, invoice/receipt, certificate, application-form, and financial-summary categories. Ten documents are clean; twenty contain controlled textual corruptions and visual variations. All records are fictional.

Manual evaluation subset: `IMG001-IMG005` and `PDF001-PDF005` (10 documents).

Reserved unprocessed subset: `IMG006-IMG015` and `PDF006-PDF015` (20 documents).

## 2. Experimental setup

| Item | Recorded value |
|---|---|
| Review date | NOT_YET_RECORDED |
| Hardware and RAM | NOT_YET_RECORDED |
| Operating system | NOT_YET_RECORDED |
| Git commit | NOT_YET_RECORDED |
| Docker/image versions | NOT_YET_RECORDED |
| PaddleOCR version | NOT_YET_RECORDED |
| DocLayout-YOLO checkpoint | NOT_YET_RECORDED |
| BGE-M3 checkpoint | NOT_YET_RECORDED |
| Qwen checkpoint | NOT_YET_RECORDED |
| Confidence threshold | NOT_YET_RECORDED |
| Retrieval and generation parameters | NOT_YET_RECORDED |

Execution protocol: process one assigned document at a time, preserve the raw OCR output, export the corrected output and PDF, save stage evidence, and record the run metadata before calculating metrics.

## 3. Baseline OCR results

Status: **NOT_YET_MEASURED**

| Document | Baseline CER | Baseline WER | Character-level normalized accuracy | Notes |
|---|---:|---:|---:|---|
| IMG001 | Pending | Pending | Pending | |
| IMG002 | Pending | Pending | Pending | |
| IMG003 | Pending | Pending | Pending | |
| IMG004 | Pending | Pending | Pending | |
| IMG005 | Pending | Pending | Pending | |
| PDF001 | Pending | Pending | Pending | |
| PDF002 | Pending | Pending | Pending | |
| PDF003 | Pending | Pending | Pending | |
| PDF004 | Pending | Pending | Pending | |
| PDF005 | Pending | Pending | Pending | |

## 4. Post-correction results

Status: **NOT_YET_MEASURED**

| Document | Post-correction CER | Post-correction WER | Character-level normalized accuracy | Outcome |
|---|---:|---:|---:|---|
| IMG001-IMG005 | Pending | Pending | Pending | Pending |
| PDF001-PDF005 | Pending | Pending | Pending | Pending |

## 5. CER and WER comparison

Status: **NOT_YET_MEASURED**

`CER = (S + D + I) / reference characters`

`WER = (S + D + I) / reference words`

`Character-level normalized accuracy = 1 - CER`

`Absolute CER reduction = baseline CER - post-correction CER`

`Relative CER reduction = absolute CER reduction / baseline CER` when baseline CER is non-zero.

| Aggregate | Baseline | Post-correction | Absolute change | Relative change |
|---|---:|---:|---:|---:|
| Macro CER | Pending | Pending | Pending | Pending |
| Macro WER | Pending | Pending | Pending | Pending |

## 6. Correction and error-introduction analysis

Status: **NOT_YET_MEASURED**

| Measure | Result |
|---|---:|
| Errors corrected | Pending |
| Errors introduced | Pending |
| Errors remaining | Pending |
| Documents improved | Pending |
| Documents unchanged | Pending |
| Documents worsened | Pending |
| Token-level correction accuracy | Pending |

Token-level correction accuracy excludes unchanged correct words. It measures successful correction events only among positions or insertions actually changed by the correction stage.

## 7. Clean-document preservation

Status: **NOT_YET_MEASURED**

Report only documents whose genuine baseline OCR exactly matches normalized ground truth. For those documents, record whether post-correction output remained exact. Do not infer preservation from the generator's clean/challenging label.

| Clean baseline documents | Preserved | Altered | Preservation rate |
|---:|---:|---:|---:|
| Pending | Pending | Pending | Pending |

## 8. Runtime and resource usage

Status: **NOT_YET_MEASURED**

| Document | Total runtime | OCR | Layout | Retrieval | Qwen | Peak RAM | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Genuine runs only | Pending | Pending | Pending | Pending | Pending | Pending | |

State whether timings are cold-start or warm-cache measurements and record hardware consistently.

## 9. Failure analysis

For each failed or worsened document, record factual evidence before interpretation.

| Document | Stage | Observed failure | Input characteristic | Evidence path | Likely cause | Follow-up test |
|---|---|---|---|---|---|---|
| Pending | Pending | Pending | Pending | Pending | Pending | Pending |

Separate injected dataset corruption from errors actually observed in OCR. Include rejected corrections and unsupported model suggestions.

## 10. Limitations and future work

- The dataset is synthetic and small; conclusions cannot be generalized to all document domains.
- Injected textual corruptions are controlled challenges, not measured OCR errors.
- Manual execution can introduce operator variance; preserve exact exports and metadata.
- A future study should add de-identified real-world data, independent ground-truth review, repeated runs, uncertainty intervals, and hardware-normalized timing.
- Reserved documents should be used only after evaluation settings are frozen to reduce tuning bias.

## Reproducibility record

```powershell
.\.venv\Scripts\python.exe evaluation\validate_dataset.py
.\.venv\Scripts\python.exe evaluation\calculate_metrics.py --self-test
.\.venv\Scripts\python.exe evaluation\calculate_metrics.py
```

Attach the manifest, raw outputs, corrected outputs, metric JSON/CSV, and Git commit used for the final results.

