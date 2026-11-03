# CSCWD manuscript v10

V10 foregrounds compact output correction and removes the overbroad acceptance conclusion in v9. The update record keeps each result with its evaluation batch and frozen reference. Residual GRPO's crossing of a five-point mean-loss boundary is reported with uncertainty, alongside constant controls from the initial batch. RFJ3, object-list weighting and the directional tests are explained. Original numerical tables I-V and all nine tests are unchanged.

```bash
pip install -r manuscript/cscwd2027_v10/requirements.txt
python extensions/cscwd2027_1102/review_v10/make_review_numbers.py
python manuscript/cscwd2027_v10/build_cscwd.py
```

The editable Word paper is saved to `output/`. Export with Word or LibreOffice. IEEE template and existing simulation/posture graphics are reused from `../cscwd2027_v4/`; the original curve graphic is retained in `figures/`. No supplementary-file dependency or new physical evaluation is introduced. No evaluated collaborative benefit, coordinate causality, or cross-batch superiority is claimed.
