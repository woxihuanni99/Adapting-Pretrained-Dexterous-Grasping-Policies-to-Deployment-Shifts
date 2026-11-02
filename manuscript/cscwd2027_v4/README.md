# CSCWD manuscript v4 source

This directory generates the Word manuscript *Recovery–Retention Trade-offs in Dexterous Grasping Policy Adaptation*. Scientific inputs are the original `analysis/numbers.json` and `extensions/cscwd2027_1102/results/numbers.json`.

```bash
pip install -r manuscript/cscwd2027_v4/requirements.txt
python manuscript/cscwd2027_v4/make_figures.py
python manuscript/cscwd2027_v4/build_word.py
```

The Word file and table/citation records are written under `output/`. Open the generated Word file in Word or LibreOffice and export it to PDF. The author's checked PDF is six IEEE Letter pages, Times New Roman 10 pt body; pagination can vary with fonts and office software. The repository does not install fonts or modify an existing environment.

Figure 1 is a workflow schematic. Figure 4 is rebuilt from the two immutable number ledgers. `make_tradeoff.py` rebuilds Figure 3 from the original candidate means and run-level values. Figure 2 uses the archived Isaac Gym simulation image `fig3_sim_examples.png`; no generated imagery or real-robot photographs are used. The supplied PNG/PDF assets are the checked v4 figures. Figure 3 covers 14 of 16 original candidates, as stated in its caption.

`build_word.py` is the v4 builder with repository-relative inputs and a local output directory. It does not update project state or scheduled tasks. `release_checks.json` records hashes of the checked source artifacts and the original number ledger. Table values are generated from the ledgers; `table_audit.json` and `references.json` record the v4 output.

The retained IEEE conference template (`ieee_template_transitional.docx`) was obtained from https://2027.ieee-aicas.org/conference-template-letter.docx and converted from Strict to Transitional OOXML for python-docx compatibility. It is an IEEE template resource, separate from this repository's MIT-licensed research code.
