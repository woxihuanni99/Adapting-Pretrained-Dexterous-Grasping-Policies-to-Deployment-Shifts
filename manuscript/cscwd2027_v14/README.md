# CSCWD manuscript v14

V14 edits the supplied v13 in place. Its thesis is that a constant 24-dimensional correction suffices for strong recovery at the reference little-finger lock; added state-dependent adaptation capacity should be assessed against this baseline. The prose directly reports the findings and concentrates scope information in one paragraph. Tables, figures, equations, author block, fonts and page settings are preserved.

```bash
pip install -r manuscript/cscwd2027_v14/requirements.txt
python manuscript/cscwd2027_v14/patch_v14.py
```

The retained `reference_v13.docx` is hash-pinned. `original_selection.json` is the original development ledger; selected losses are recomputed rather than typed into the patch. The generated Word and change record are saved in `output/`. Export using Times New Roman in Word or LibreOffice. The verified local export has six IEEE Letter pages, four figures, six numerical tables and 31 cited references. All Word package parts except the targeted body XML are byte-preserved.

The [source checks and visualization provenance](../../extensions/cscwd2027_1102/review_v14/) accompany the paper. No new training, simulation, practitioner study or causal intervention was added.
