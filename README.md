# Adapting Pretrained Dexterous Grasping Policies to Deployment Shifts

Code and data for the manuscript

> **Adapting Pretrained Dexterous Grasping Policies to Deployment Shifts: An Empirical Study of Update Parameterization and Recovery–Retention Trade-offs**
> Haoze Lan, 2026 (manuscript in preparation)

The study post-trains the one-step [DemoGrasp](https://github.com/BeingBeyond/DemoGrasp) Shadow-hand planner in Isaac Gym under deployment shifts such as finger locks, point-cloud noise and calibration offsets. It compares four schemes: full-weight fine-tuning or a residual latent-noise head, each trained with PPO or GRPO. Each scheme uses one learning rate, selected under a shared five-point nominal-loss limit, and is then evaluated after the protocol was locked (evaluation seed 4242).

This repository contains:

- the training and evaluation scripts;
- the experiment-management code that generated every plan;
- the per-attempt evaluation records behind all reported numbers;
- the saved inputs and action means of the checkpoint output analysis;
- the scripts that rebuild every number and figure of the paper from these records.

## Reproduce the numbers and figures

The analysis needs Python ≥ 3.8 with NumPy, SciPy and Matplotlib. It does not need a GPU or a simulator.

```bash
pip install -r requirements.txt
python data/unpack.py                       # verifies SHA-256 of all archives, then extracts them under data/
cd analysis
python scripts/make_numbers_and_figs.py     # rebuilds numbers.json and figures/ (also runs the 7 Oct and 8 Oct audits)
python scripts/audit_recompute_primary.py   # independent rebuild of selection, the nine tests and Holm correction
python scripts/audit_collision_split.py     # collision-mask grouping of the post-lock evaluation
```

`analysis/numbers.json` and `analysis/figures/` are the versions used in the manuscript. In our check, rebuilding from the released data reproduced all 7,076 numeric fields of `numbers.json`; the largest difference was 5 × 10⁻¹², a SciPy t-quantile rounding difference. The primary audit reported `passed: 9 primary tests; 521 raw result files`. Figures may differ at the byte level across Matplotlib versions.

The statistical unit is the independent training run. Each run's score is Y = (S_seen + S_unseen)/2, where S_seen comes from 1,128 attempts and S_unseen from 800 attempts. Contrasts use Welch–Satterthwaite intervals; the nine pre-specified one-sided tests use Holm correction. `audit_*` scripts rebuild every score from binary per-attempt outcomes (`episodes[*].official_ppo_print_successes` in each `result.json`) and do not import the server-side analysis code.

## Repository layout

| Path | Contents |
|---|---|
| `training/` | Training, evaluation and diagnostic scripts (`research_*.py`). They run inside a DemoGrasp checkout (see below). |
| `experiment_tools/` | Plan generation, development-set selection, seed top-up, statistics and the job daemon used on the training server, exactly as run. Paths inside refer to our server layout. |
| `analysis/` | `scripts/`: rebuild of all reported numbers, tables and figures; `numbers.json`: the numerical record of the manuscript; `figures/`: Figures 2–4 and S1. |
| `data/meta/temp/` | Locked manifests and statistics (selected configurations, run-to-tag mapping, final statistics). |
| `data/meta/tools/` | The locked statistics core (`final_stats_core.py`) and selection script, as frozen at the lock on 29 Sep 2026. |
| `data/archives/` | Compressed records; `data/unpack.py` verifies and extracts them. |

The archives are:

| Archive | Extracts to | Contents |
|---|---|---|
| `locked_runs.tar.gz.part00/01` | `data/runs/` | 10,691 run directories of the main study. Every development and post-lock evaluation (`result.json` with per-attempt outcomes) and every training run (`train_log.json`, `curve.json`). |
| `locked_plans.tar.gz` | `data/meta/reports/plans/` | The 2,217 plan files (exact command lines, seeds and object lists) of the main study. |
| `supp_20261007.tar.gz` | `data/supp_20261007/` | Setting-matched GRPO control (18 new runs), reproduction check and re-evaluation with all robot–table collisions enabled (Supplementary Section S8). Includes initial observations of the collision comparison. |
| `supp_20261008.tar.gz` | `data/supp_20261008/` | Output-shift coordinate control: 36 runs, 72 development and 216 test evaluations, and the learned 24-dimensional shifts (`shift.pt`) of every phase (Supplementary Section S9.2). |
| `output_analysis_20261008.tar.gz` | `data/output_analysis_20261008/` | The 2,048 saved nominal states, the frozen mean and the adapted means of 72 checkpoints, and their summary (Section 6.1, Supplementary Section S9.1). |

Model checkpoints, several gigabytes in total, are not included. They are available from the author on request.

## Training and evaluation code

The scripts in `training/` extend DemoGrasp without modifying it. To run them:

1. Check out [BeingBeyond/DemoGrasp](https://github.com/BeingBeyond/DemoGrasp) at commit `1255dc5384485682d04e1dbebe330bf2d234d20b`.
2. Install Isaac Gym Preview 4 and the DemoGrasp requirements.
3. Copy `training/research_*.py` into the DemoGrasp root.

The DemoFunGrasp case study uses [BeingBeyond/DemoFunGrasp](https://github.com/BeingBeyond/DemoFunGrasp) at commit `116b74d13ab59aadf7a2edde7f6ba80823b7ba73` (`research_dfg_grpo.py`).

Each plan in `data/meta/reports/plans/` records the exact command line of one training or evaluation job. `experiment_tools/plangen.py` and the `*_gen.py` scripts generated these plans.

Scripts used for the reported results:

| Script | Role |
|---|---|
| `research_shift_run.py` | Applies a deployment shift (finger lock, point-cloud noise, calibration offset, combinations) to any training or evaluation script. |
| `research_ppo_finetune.py`, `research_ppo_pcl.py` | Full-weight PPO. |
| `research_grpo_finetune_v2.py` | Full-weight GRPO (state-paired groups of eight actions). |
| `research_plugin_rl.py` | Residual latent-noise head on the frozen planner (PPO and GRPO). |
| `research_grpo_fm.py` | Development and post-lock evaluation. |
| `research_grpo_fm_coll.py` | Evaluation with robot–table collisions enabled in all environments. |
| `research_grpo_finetune_v2_kl.py` | Full-weight GRPO with a KL penalty to the pretrained mean. |
| `research_plugin_rl_ms.py` | Residual head with block-wise mean scaling. |
| `research_plugin_rl_cst.py` | Directly trained constant latent offset. |
| `research_delta_diag.py` | Post-hoc constant extracted from trained residual heads. |
| `research_plugin_rl_fs.py` | Setting-matched residual GRPO control. |
| `research_grpo_finetune_v2_bias.py` | Output-shift coordinate control (raw or σ0-scaled 24-dimensional shift). |
| `research_output_diag.py` | Output analysis of saved checkpoints on a common state set. |
| `research_dfgshift_run.py`, `research_dfg_grpo.py` | DemoFunGrasp functional-grasping case study. |

The remaining scripts come from earlier exploratory work before the protocol lock: frozen-planner probes, data collection, oracles and a sample-then-verify route. None of the paper's results depend on them.

## Changes made for this release

- **Output-path guard (`training/`).** The original scripts refused to write outside the authors' storage volumes. The guard now accepts any path, or only paths under `$DEXGRASP_ROOT` if that variable is set.
- **Data paths (`analysis/scripts/make_numbers_and_figs.py`).** These now point to the extracted archives under `data/` instead of our original directory layout.

No other code was changed.

## License

The code in this repository is released under the MIT License (see `LICENSE`). DemoGrasp and DemoFunGrasp are MIT-licensed projects of BeingBeyond and are not redistributed here. Isaac Gym is distributed by NVIDIA under its own license.

## Citation

See `CITATION.cff`. Please cite the manuscript once it is published.
