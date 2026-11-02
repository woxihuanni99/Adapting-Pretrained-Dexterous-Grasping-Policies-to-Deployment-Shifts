# Existing-data analyses for manuscript v5

`make_numbers_and_figs.py` reads the original number ledger and the completed same-server extension. It computes Student intervals across training runs for achieved nominal losses and constant corrections. It also computes the equal-optimizer mean separation between piecewise-linear original test curves over their common nominal-loss support (2.25266–5.41986 pp).

The curve functional is 17.731731 pp. A percentile bootstrap resamples the paired fault and nominal score of each independent training run within each rate, recomputing common support on every draw. With fixed seed 110205 and 10,000 draws, the interval is [14.883492, 19.990974] pp. Identical candidate costs are pooled by mean success before interpolation. No extrapolation is used; all 10,000 draws have nonempty support. This is a descriptive curve statistic, not a trained matched-retention model comparison or an additional member of the original nine-test family.

`make_tradeoff_figure.py` builds the low-loss detail and full-range inset. The original per-attempt records, extension selections and primary statistics remain separate and unchanged. All numerical inputs are already in this repository.
