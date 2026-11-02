# Existing-record checks for v8

`make_review_numbers.py` regenerates the example update-record values from the checked ledgers, and evaluates the original residual curves at the selected KL candidate's nominal cost. Reference pages and DOIs were checked against Crossref and the author-hosted papers; the checked metadata is cached so reproduction does not require network access.

The file also audits two suggestions that are not used as new manuscript evidence: second-server points compared with original-server curves, and a post hoc ±3 pp TOST calculation for the 75% contrast. The former is not a validation of interpolation bias; the latter is not added to the original nine tests or used to claim equivalence. Original numerical records and training scripts remain unchanged.

```bash
python extensions/cscwd2027_1102/review_v8/make_review_numbers.py
```
