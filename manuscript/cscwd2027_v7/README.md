# CSCWD manuscript v7 — single self-contained paper

Current source for *Recovery–Retention Trade-offs in Adapting a Pretrained Dexterous Grasp Planner*. V7 does not depend on a supplementary submission. It restores the important method details, original nine-test table and retention controls to the main paper while preserving the original results, run-level uncertainty and curve analysis.

```bash
pip install -r manuscript/cscwd2027_v7/requirements.txt
python manuscript/cscwd2027_v7/build_cscwd.py
```

The Word manuscript and generated table/citation records are saved to `output/`. Export to PDF with Word or LibreOffice. The checked paper uses IEEE Letter two-column format and 10 pt Times New Roman body text. All original data, the completed extension and earlier versions remain available. No new training or simulation is used.

Numerical inputs are the original ledger, the completed same-server extension and `extensions/cscwd2027_1102/review_v5/numbers.json`. The IEEE template, simulation image and posture graphic are reused from `../cscwd2027_v4/`. The curve PNG/PDF is included in `figures/`.

The public CSCWD 2027 call specifies a maximum of six pages including figures and references, but does not establish acceptance of supplementary files. Therefore v7 contains all essential evidence in the main paper and removes every supplementary cross-reference. The earlier v6 supplement is a previous working arrangement, not a required part of the v7 submission.
