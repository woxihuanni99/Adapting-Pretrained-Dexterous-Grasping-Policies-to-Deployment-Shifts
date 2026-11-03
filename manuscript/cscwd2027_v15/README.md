# CSCWD manuscript v15

One abstract clause is changed from v14. It now reports the rank correlation between RFJ3 target change and recovery across 51 analyzed runs spanning seven update variants: rho = 0.97. The existing diagnostic reconstruction gives 0.9654298642533936. No other paragraph, figure or table is changed.

```bash
pip install -r manuscript/cscwd2027_v14/requirements.txt
python manuscript/cscwd2027_v14/patch_v14.py
python manuscript/cscwd2027_v15/patch_v15.py
```

Export the generated Word in `output/` with Times New Roman. The final local document is six IEEE Letter pages, with a 155-word abstract, four figures, six numerical tables and 31 cited references. All non-abstract XML nodes and all other ZIP parts are preserved. [Diagnostic records](../../extensions/cscwd2027_1102/review_v12/) and [v14 source checks](../../extensions/cscwd2027_1102/review_v14/) are unchanged.
