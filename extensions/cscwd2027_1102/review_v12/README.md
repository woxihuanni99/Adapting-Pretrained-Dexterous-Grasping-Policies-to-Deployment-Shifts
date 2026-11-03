# Checkpoint diagnostics for CSCWD v12

```bash
python extensions/cscwd2027_1102/review_v12/check_diagnostics.py
```

Pure standard-library reconstruction of the 51 available reference-lock diagnostic checkpoints, the pooled Spearman correlations and the 12 retrained-75% points. Coverage is explicit in `numbers.json`; it is not every original learning rate. `reference_runs.csv` gives figure-level coordinates and identifiers. No simulation, training or causal intervention is used. The source ledger and original tests are unchanged.
