# CSCWD manuscript v5

Source for *Recovery–Retention Trade-offs in Adapting a Pretrained Dexterous Grasp Planner*, with a short supplementary document.

```bash
pip install -r manuscript/cscwd2027_v5/requirements.txt
python extensions/cscwd2027_1102/review_v5/make_numbers_and_figs.py
python extensions/cscwd2027_1102/review_v5/make_tradeoff_figure.py
python manuscript/cscwd2027_v5/build_cscwd.py
python manuscript/cscwd2027_v5/build_cscwd.py --supplement
```

Word files and generated table/citation records are saved to `output/`. The author's checked main manuscript has five IEEE Letter pages, with 10 pt Times New Roman body text; the supplement has three single-column pages. Use Word or LibreOffice to export PDF. The shared IEEE template, simulation image and posture plot remain in `../cscwd2027_v4/`; no repeated asset download is needed. V4 stays available.

The v5 update adds run-level uncertainty for nominal cost and constants, exposes the 17.7 pp descriptive separation of interpolated original test curves, and moves secondary checks to the supplement. It preserves the original eight-condition table, nine Holm tests and completed extension. Figure 1 retains the selected-object and rerender provenance disclosures. No new simulation or training data are used.
