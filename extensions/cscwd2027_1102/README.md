# CSCWD extension dated 2 November 2026

Reproduction code and official per-attempt success flags for the completed recovery–retention extension used by the CSCWD manuscript v4.

## Recompute on CPU

From the repository root, in a Python environment of your choice:

```bash
pip install -r extensions/cscwd2027_1102/requirements.txt
python extensions/cscwd2027_1102/recompute.py
```

Outputs go to `reproduced/` by default. Use `--out <directory>` to change it. The script rebuilds development scores and selection before computing test contrasts, using exact fractions for per-run rates and Welch–Satterthwaite intervals over independent training runs. It checks the results against `results/`.

The released record contains all 543 formal evaluation units (361 main, 146 fixed references, 36 selected full-weight references), totaling 565,072 attempts. It also includes the separate 2,048-attempt frozen development calibration used as the nominal baseline. `data/attempt_outcomes.json.gz` retains the ordered official binary flags and the SHA-256 of each source `result.json`; it omits unrelated simulator logs and machine paths. `data/chains.json` maps runs, rates, conditions and evaluation IDs. `data/original_development.json` contains exact scores for the 40 archived checkpoints used in the environment-drift analysis; their earlier per-attempt records are in the original repository data archives.

Verified on the released data: 150 run-condition scores, expanded-grid selection, 40-checkpoint nominal-loss drift, frozen scores and all 16 descriptive contrasts. Reference-LF contrast: 28.0 [23.1, 33.0] percentage points; additional nominal loss: 3.1 points. The original eight-condition analysis and nine Holm tests remain in `analysis/` and are not overwritten.

## Run the training/evaluation programs

`training/` contains the eight programs used in this extension, with the original science code preserved. Only the output-directory guard is made portable; `source_manifest.json` records the exact staged and released hashes and this change. Copy these programs into the DemoGrasp checkout described in the root README. Keep the matching Isaac Gym Preview 4/PyTorch environment. Copy `training/object_lists/union_ycb_unidex/` under DemoGrasp's `assets/union_ycb_unidex/`.

`training/commands.json` records each enabled phase and evaluation command, with storage prefixes replaced by `${EXTENSION_ROOT}`, `${PROJECT_ROOT}`, `${WORK_ROOT}` and `${ARCHIVE_ROOT}`. Substitute your local paths, pretrained checkpoint and archived adapted checkpoints, and execute the phases in order from the DemoGrasp root. The `program` and `arguments` fields form the command after `python`. This is a command manifest, not an automatic cluster launcher. Optional `DEXGRASP_ROOT` restricts script output paths to a chosen directory. GPU training needs the licensed simulator, upstream assets and model checkpoints; CPU recomputation above needs none of them.

Completed training: 22 runs, 426,496 interactions. PPO/GRPO add 2×10⁻⁵; matched residual GRPO uses 3×10⁻⁴, fixed max(σ₀, 0.05), frozen scale parameters and two epochs. The original one-time boundary rule governs extra runs. Protocol, phase budgets, seeds, object lists and actual selected rates are recorded under `data/` and `training/`. No new experiments are launched by the CPU reproduction script.

## Figures and manuscript

See [`../../manuscript/cscwd2027_v4/`](../../manuscript/cscwd2027_v4/) for v4 plotting and Word generation. [`MANIFEST.sha256`](MANIFEST.sha256) lists this release's files using paths relative to the repository root.
