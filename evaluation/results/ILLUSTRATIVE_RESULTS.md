# Illustrative OCR Correction Results

> **ILLUSTRATIVE / SYNTHETIC - NOT ACTUAL PIPELINE MEASUREMENTS**

- Documents: **10**
- Controlled error tokens: **12**
- Illustratively corrected: **11**
- Intentionally unresolved: **1**
- Illustrative token correction accuracy: **91.67%**
- Outcomes: **6 improved, 4 unchanged, 0 worsened**
- Macro CER: **0.28% -> 0.04%**
- Macro WER: **1.56% -> 0.13%**

| ID | Errors | Corrected | Unresolved | Outcome | Post CER | Post WER |
|---|---:|---:|---:|---|---:|---:|
| IMG001 | 0 | 0 | 0 | UNCHANGED | 0.00% | 0.00% |
| IMG002 | 2 | 2 | 0 | IMPROVED | 0.00% | 0.00% |
| IMG003 | 2 | 2 | 0 | IMPROVED | 0.00% | 0.00% |
| IMG004 | 0 | 0 | 0 | UNCHANGED | 0.00% | 0.00% |
| IMG005 | 2 | 2 | 0 | IMPROVED | 0.00% | 0.00% |
| PDF001 | 0 | 0 | 0 | UNCHANGED | 0.00% | 0.00% |
| PDF002 | 2 | 2 | 0 | IMPROVED | 0.00% | 0.00% |
| PDF003 | 2 | 2 | 0 | IMPROVED | 0.00% | 0.00% |
| PDF004 | 0 | 0 | 0 | UNCHANGED | 0.00% | 0.00% |
| PDF005 | 2 | 1 | 1 | IMPROVED | 0.39% | 1.28% |

The single intentionally unresolved controlled error is `reciept` in `PDF005`.
These files are stored separately and do not replace genuine `evaluation/actual_runs` evidence or genuine metrics.
