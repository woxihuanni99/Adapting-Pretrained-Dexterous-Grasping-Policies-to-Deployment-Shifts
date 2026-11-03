# CSCWD manuscript v12

V12 retains the supplied v11 title, figures, tables and repaired OOXML table properties. Targeted prose changes replace causal and equivalence claims with the observed compact-correction results and pooled checkpoint diagnostics. Original values and tests remain unchanged.

```bash
pip install -r manuscript/cscwd2027_v12/requirements.txt
python manuscript/cscwd2027_v12/patch_v12.py
python extensions/cscwd2027_1102/review_v12/check_diagnostics.py
```

The retained `reference_v11.docx` is the supplied source artifact, hash-pinned by the patch. All package parts except body XML are preserved. Output Word and a change record are saved in `output/`. Export with installed Times New Roman using Word or LibreOffice. The manuscript is six pages including four figures, six tables and 31 references. The supplied artwork is preserved; numerical diagnostic data can be independently reconstructed by the linked script. No new physical evaluations, user studies or mechanism controls are claimed.
