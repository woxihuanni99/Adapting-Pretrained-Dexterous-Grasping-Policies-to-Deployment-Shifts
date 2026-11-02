"""A schematic workflow and a larger-type comparison from immutable ledgers."""
import hashlib
import json
import os
import argparse
from pathlib import Path

R = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out-dir', type=Path, default=R/'figures')
args = parser.parse_args()
F = args.out_dir
F.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(R / "mplconfig")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.48, 1.32))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
boxes = [(0.015, .60, "Pretrained DemoGrasp\none-step grasp planner"),
         (.555, .60, "Four update procedures\nfull/residual × PPO/GRPO"),
         (.555, .045, "Development selection\nnominal loss ≤5 pp"),
         (.015, .045, "Nominal + shifted tests\nrecovery and retention")]
for x, y, label in boxes:
    ax.add_patch(FancyBboxPatch((x, y), .425, .35, boxstyle="round,pad=.009,rounding_size=.025",
                               linewidth=.8, edgecolor="#405f75", facecolor="#f4f7f9"))
    ax.text(x+.2125, y+.175, label, ha="center", va="center", fontsize=8.6, linespacing=1.25)
for a, b in [((.45, .775), (.545, .775)), ((.7675, .575), (.7675, .415)), ((.545, .22), (.45, .22))]:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, linewidth=.9, color="#405f75"))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
for ext in ("png", "pdf"):
    fig.savefig(F / ("fig0_pipeline."+ext), dpi=600)
plt.close(fig)

old_path = (R.parents[1]/"analysis/numbers.json")
new_path = R.parents[1]/"extensions/cscwd2027_1102/results/numbers.json"
old = json.loads(old_path.read_text(encoding="utf8"))
new = json.loads(new_path.read_text(encoding="utf8"))
items = [("LF_ref", "Reference", 0), ("LF25", "25% lock", .9), ("LF75_eval", "75% lock", 1.8),
         ("LF75_retrain", "75% retrained", 3.0), ("RFLF", "Ring + little", 3.9)]
fig, ax = plt.subplots(figsize=(3.48, 2.65))
provenance = []
for i, (key, label, y) in enumerate(items):
    v = old["boundary"][key]
    effect, lo, hi = 100*v["fault"]["est"], 100*v["fault"]["ci"][0], 100*v["fault"]["ci"][1]
    yy = y-.13 if i < 3 else y
    ax.errorbar(effect, yy, xerr=[[effect-lo], [hi-effect]], fmt="o", ms=4, lw=1.05, capsize=2.1, color="#b44920")
    ax.text(1.19, y, f'{100*v["frozen"]:.1f}', transform=ax.get_yaxis_transform(), fontsize=9.2, ha="center", va="center", clip_on=False)
    provenance.append({"protocol": "original", "condition": key, "effect_pp": effect, "CI95_pp": [lo, hi], "frozen_original_pct": 100*v["frozen"]})
for key, y in (("LF", 0), ("LF25", .9), ("LF75", 1.8)):
    v = next(x for x in new["contrasts"] if x["comparison"] == "equal_optimizer_average_selected_residual_minus_expanded_selected_FW" and x["test_condition"] == key)
    effect = v["difference_pp"]; lo, hi = v["CI95_unadjusted_two_sided_pp"]
    ax.errorbar(effect, y+.13, xerr=[[effect-lo], [hi-effect]], fmt="s", ms=3.9, lw=1.05, capsize=2.1, color="#245f8f")
    provenance.append({"protocol": "same_server_extension", "condition": key, "effect_pp": effect, "CI95_pp": [lo, hi]})
ax.axvline(0, color=".5", ls="--", lw=.7)
ax.axhline(2.45, xmin=-.7, xmax=1.3, color=".82", lw=.5)
ax.set_yticks([x[2] for x in items], [x[1] for x in items], fontsize=9.2)
ax.tick_params(axis="y", length=0, pad=4)
ax.set_ylim(4.4, -.65); ax.set_xlim(-8, 45); ax.set_xticks([0, 10, 20, 30, 40])
ax.tick_params(axis="x", labelsize=9)
ax.set_xlabel("Residual − full-weight success (pp)", fontsize=8.9)
ax.spines["left"].set_visible(False)
ax.text(1.19, -.54, "Frozen\noriginal (%)", transform=ax.get_yaxis_transform(), fontsize=8.3, ha="center", va="center", clip_on=False, linespacing=1.05)
fig.legend(handles=[Line2D([], [], color="#b44920", marker="o", linestyle="None", markersize=4, label="Original"),
                    Line2D([], [], color="#245f8f", marker="s", linestyle="None", markersize=4, label="Expanded grid")],
           loc="upper center", bbox_to_anchor=(.55, 1.015), ncol=2, frameon=False, fontsize=8.7, handletextpad=.25, columnspacing=.9)
fig.subplots_adjust(left=.40, right=.80, top=.84, bottom=.21)
for ext in ("png", "pdf"):
    fig.savefig(F / ("fig2_posture."+ext), dpi=600)
plt.close(fig)
(F / "figures_provenance_v4.json").write_text(json.dumps({"inputs": [{"path": p.relative_to(R.parents[1]).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in (old_path, new_path)], "posture_rows": provenance, "posture_font_pt": 9.2, "workflow_is_schematic": True, "new_simulation": False}, indent=2), encoding="utf8")
