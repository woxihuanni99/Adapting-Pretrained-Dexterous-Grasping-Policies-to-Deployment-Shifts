# CSCWD manuscript v6

Source for *Recovery–Retention Trade-offs in Adapting a Pretrained Dexterous Grasp Planner* and its short supplement. V6 incorporates the author-supplied six-page v5 PDF and the existing-data review revision. The earlier repository v5 is an intermediate revision; v6 is the current merged manuscript. Previous versions and all original results remain available.

```bash
pip install -r manuscript/cscwd2027_v6/requirements.txt
python extensions/cscwd2027_1102/review_v5/make_numbers_and_figs.py
python manuscript/cscwd2027_v6/build_cscwd.py
python manuscript/cscwd2027_v6/build_cscwd.py --supplement
```

The existing-data analysis and checked curve graphic are reused unchanged from `review_v5/`. To regenerate the graphic, run its `make_tradeoff_figure.py` and copy the generated `fig1_tradeoff.png` and `.pdf` from `manuscript/cscwd2027_v5/figures/` into this version's `figures/` folder. Shared IEEE template, simulation image and posture graphic remain in `../cscwd2027_v4/`.

Builders save Word files and table/citation records to `output/`. The checked main manuscript has five IEEE Letter pages with 10 pt Times New Roman body text; the supplement has three single-column pages. Export PDF with Word or LibreOffice.

This revision foregrounds the recovery–retention comparison, adds run-level intervals for nominal loss and constants, restores deliberate-example and rerender provenance, and moves secondary checks to the supplement. The 17.7 pp interpolated curve statistic is descriptive; original development-selected outcomes and the nine-test family are unchanged. No new training or simulation is used.
