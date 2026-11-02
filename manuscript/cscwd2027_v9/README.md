# CSCWD manuscript v9 — selection and acceptance

V9 clarifies fixed-checkpoint evaluation without per-grasp health routing, separates development selection from test-loss review, and uses one same-server source in the review table. It removes the sparse-curve statistic from the abstract while preserving its descriptive record in the body and scripts. The first five scientific tables and the original nine tests are unchanged. RFA is positioned as a distinct closed-loop runtime-fault problem. No new training or simulation is used.

```bash
pip install -r manuscript/cscwd2027_v9/requirements.txt
python extensions/cscwd2027_1102/review_v9/make_review_numbers.py
python manuscript/cscwd2027_v9/build_cscwd.py
```

The builder saves one editable Word paper and table/citation records to `output/`. Export PDF with Word or LibreOffice. The IEEE template, simulation image and posture graphic are reused from `../cscwd2027_v4/`; the original curve graphic remains in `figures/`. Previous versions and all original records remain available. The record is an illustrative report and does not claim evaluated collaborative benefit or certified deployment performance.
