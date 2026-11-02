# CSCWD manuscript v8 — self-contained paper

V8 retains the original numerical results and single-paper arrangement. It adds a concrete example update record inspired by model reporting, one concise limitations paragraph, two checked documentation/engineering references, and more precise interpretations of critic sharing, coordinates and interpolated curves. The example records retain their separate evaluation sources and the specific postures tested; they are not a practitioner evaluation.

```bash
pip install -r manuscript/cscwd2027_v8/requirements.txt
python extensions/cscwd2027_1102/review_v8/make_review_numbers.py
python manuscript/cscwd2027_v8/build_cscwd.py
```

The Word paper and table/citation records are saved to `output/`. Use Word or LibreOffice to export PDF. The IEEE template, simulation image and posture plot are reused from `../cscwd2027_v4/`; the checked curve graphic is in `figures/`. The original curve bootstrap is reused from `review_v5/`, without resampling or changing its values. No new training or simulation is used.
