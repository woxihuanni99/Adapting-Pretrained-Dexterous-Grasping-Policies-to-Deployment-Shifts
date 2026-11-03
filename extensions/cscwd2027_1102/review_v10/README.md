# Compact-correction and repeat-reference review for v10

This module reports existing constant controls beside residual GRPO in their own evaluation batches. It does not rank policies across batches or change development selection or the original tests. It also decomposes the observed change in nominal loss into the change in frozen reference and adapted nominal success; no hardware cause is inferred.

```bash
python extensions/cscwd2027_1102/review_v10/make_review_numbers.py
```

The repeat batch has no evaluations of the compact controls. Original records are preserved. No new training, simulation or evaluation is used.
