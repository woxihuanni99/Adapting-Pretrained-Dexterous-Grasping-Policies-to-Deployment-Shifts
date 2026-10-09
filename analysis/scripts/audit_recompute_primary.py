#!/usr/bin/env python3
"""Independently reconstruct the locked primary analysis and rate selection.

Reads archived binary attempt outcomes; never trains, evaluates, edits input
data, imports the historical statistical core, or changes manuscript numbers.
SciPy supplies Student-t probabilities/quantiles only. Run, for example:
  py -3.11 -B scripts/audit_recompute_primary.py
The default data directory is ../data relative to the manuscript version.
"""
import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

from scipy.stats import t as student_t


METHODS = ("res_ppo", "res_grpo", "full_ppo", "full_grpo")
WEIGHTS = (0.5, 0.5, -0.5, -0.5)
H4 = {
    "H4_LFpn_srcLF": ("LF", "LF", "LFpn"),
    "H4_LFpn_srcPCL": ("PCL", "pn10", "LFpn"),
    "H4_THzr_srcTH": ("TH", "TH", "THzr"),
    "H4_THzr_srcZR": ("ZR", "zr", "THzr"),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, label):
    if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-11):
        raise AssertionError(f"{label}: recomputed {a!r}, archived {b!r}")


class Archive:
    def __init__(self, root):
        self.root = root
        self.records = {}
        self.slot_identities = {}

    def score(self, name, expected_attempts=None, final_split=None):
        if name not in self.records:
            path = self.root / "runs" / name / "result.json"
            raw = path.read_bytes()
            payload = json.loads(raw)
            if payload["status"] != "completed":
                raise AssertionError(f"Unfinished result: {name}")
            successes = []
            episodes = payload["episodes"]
            for episode in episodes:
                flags = episode["official_ppo_print_successes"]
                assets = episode["object_assets"]
                if len(flags) != len(assets) or any(x not in (0, 1) for x in flags):
                    raise AssertionError(f"Invalid binary outcomes/assets: {name}")
                if final_split:
                    if len(assets) != {"seen": 141, "unseen": 100}[final_split]:
                        raise AssertionError(f"Wrong slot count: {name}")
                    old = self.slot_identities.setdefault(final_split, assets)
                    if old != assets:
                        raise AssertionError(f"Changed slot identities/order: {name}")
                successes.extend(flags)
            if final_split and len(episodes) != 8:
                raise AssertionError(f"Wrong final episode count: {name}")
            if not successes:
                raise AssertionError(f"Empty outcomes: {name}")
            score = sum(successes) / len(successes)
            close(score, payload["totals"]["success_rate"], name + ": total")
            if payload["totals"].get("episodes", len(successes)) != len(successes):
                raise AssertionError(f"Wrong recorded attempts: {name}")
            self.records[name] = {
                "file": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                "attempts": len(successes), "successes": sum(successes),
                "score": score, "episode_records": len(episodes),
            }
        record = self.records[name]
        if expected_attempts is not None and record["attempts"] != expected_attempts:
            raise AssertionError(f"Expected {expected_attempts} attempts: {name}")
        return record["score"]


def summarize(groups):
    """Weighted independent-group estimate; variance comes from training runs."""
    estimate = 0.0
    components = []
    for group in groups:
        values = group["values"]
        n = len(values)
        if n < 2:
            raise AssertionError("Need at least two independent training runs")
        group["n"] = n
        group["mean"] = statistics.mean(values)
        group["sd"] = statistics.stdev(values)
        estimate += group["weight"] * group["mean"]
        components.append(group["weight"] ** 2 * statistics.variance(values) / n)
    variance = sum(components)
    denominator = sum(v * v / (g["n"] - 1) for v, g in zip(components, groups))
    df = variance * variance / denominator if denominator else float("inf")
    se = math.sqrt(variance)
    statistic = estimate / se if se else math.copysign(float("inf"), estimate)
    q = float(student_t.ppf(0.975, df))
    return {
        "estimate": estimate, "se": se, "df": df, "t": statistic,
        "ci95_two_sided": [estimate - q * se, estimate + q * se],
        "p_one_sided": float(student_t.sf(statistic, df)),
        "groups": groups,
    }


def adjust_holm(tests):
    ordered = sorted(tests, key=lambda key: tests[key]["p_one_sided"])
    previous = 0.0
    for rank, key in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - rank) * tests[key]["p_one_sided"]))
        tests[key]["p_holm"] = previous
        estimate = tests[key]["estimate"]
        if estimate > 0 and previous < 0.05:
            outcome = "direction supported (Holm)"
        elif estimate < 0 and tests[key]["ci95_two_sided"][1] < 0:
            outcome = "evidence for the opposite direction"
        else:
            outcome = ("mean direction consistent, evidence insufficient" if estimate > 0
                       else "mean direction opposite, evidence insufficient")
        tests[key]["outcome"] = outcome


def reconstruct_primary(archive, manifest):
    uses = defaultdict(list)
    checkpoints = {}

    def score(tag, condition):
        sources = manifest["evals"][tag][condition]
        parts = {split: archive.score(sources[split], count, split)
                 for split, count in (("seen", 1128), ("unseen", 800))}
        return (parts["seen"] + parts["unseen"]) / 2, parts, sources

    def frozen(condition):
        return sum(archive.score(manifest["frozen"][condition][split], count, split)
                   for split, count in (("seen", 1128), ("unseen", 800))) / 2

    def group(test, method, model_set, weight, condition, combined=None):
        entries = manifest["methods"][method]["sets"][model_set]
        tags = [entry["tag"] for entry in entries]
        if len(set(tags)) != len(tags):
            raise AssertionError(f"Repeated tag within {test}/{method}/{model_set}")
        run_records = []
        for entry in entries:
            tag = entry["tag"]
            prior = checkpoints.setdefault(tag, entry["ck"])
            if prior != entry["ck"]:
                raise AssertionError(f"Conflicting declared checkpoint for {tag}")
            primary, split_scores, sources = score(tag, condition)
            record = {"tag": tag, "checkpoint": entry["ck"], "source_condition": condition,
                      "source_score": primary, "split_scores": split_scores, "result_names": sources}
            if combined:
                secondary, split_secondary, sources_secondary = score(tag, combined)
                a, b = frozen(condition), frozen(combined)
                value = (primary - a) - (secondary - b)
                record.update({"combined_condition": combined, "combined_score": secondary,
                               "combined_split_scores": split_secondary,
                               "combined_result_names": sources_secondary,
                               "frozen_source": a, "frozen_combined": b})
            else:
                value = primary
            record["value"] = value
            run_records.append(record)
            uses[tag].append({"test": test, "method": method, "model_set": model_set,
                              "condition": condition, "combined_condition": combined})
        return {"method": method, "model_set": model_set, "weight": weight,
                "tags": tags, "values": [r["value"] for r in run_records], "runs": run_records}

    tests = {}
    for name, condition in (("H1", "LF"), ("H3a", "LF25"), ("H3b", "LF75")):
        tests[name] = summarize([group(name, m, "LF", w, condition)
                                 for m, w in zip(METHODS, WEIGHTS)])
    tests["H2"] = summarize(
        [group("H2", m, "LF", w, "LF") for m, w in zip(METHODS, WEIGHTS)] +
        [group("H2", m, "PCL", -w, "pn10") for m, w in zip(METHODS, WEIGHTS)])
    for name, (model_set, condition, combined) in H4.items():
        tests[name] = summarize([group(name, m, model_set, 0.25, condition, combined) for m in METHODS])
    h5 = []
    for method, weight in zip(METHODS, WEIGHTS):
        model_set = "LF_H5" if manifest["methods"][method]["sets"].get("LF_H5") else "LF"
        h5.append(group("H5", method, model_set, weight, "LF"))
    tests["H5"] = summarize(h5)
    tests = {k: tests[k] for k in ("H1", "H2", "H3a", "H3b", *H4, "H5")}
    for name, test in tests.items():
        tags = [tag for g in test["groups"] for tag in g["tags"]]
        if len(tags) != len(set(tags)):
            raise AssertionError(f"Duplicated training run inside independent groups for {name}")
    adjust_holm(tests)
    reuse = {tag: {"declared_checkpoint": checkpoints[tag], "uses": value}
             for tag, value in uses.items()}
    return tests, reuse


def selection_identities(script):
    """Read only literal identity dictionaries; do not execute archived code."""
    wanted = {"C", "FAULT", "FROZEN_FAULT"}
    extracted = {}
    for node in ast.parse(script.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    extracted[target.id] = ast.literal_eval(node.value)
    if set(extracted) != wanted:
        raise AssertionError("Missing selection identity dictionary")
    return extracted


def reconstruct_selection(archive, source):
    identities = selection_identities(source)
    candidates = identities["C"]
    baseline = archive.score("fmdet2k_munom_frozen", 2048)
    frozen_fault = {c: archive.score(name, 512) for c, name in identities["FROZEN_FAULT"].items()}
    first = {}
    triggers = []
    for method, conditions in candidates.items():
        for condition, rates in conditions.items():
            for rate, tags in rates.items():
                if len(tags) != 3 or len(set(tags)) != 3:
                    raise AssertionError("Initial selection must use three distinct runs")
                values = [archive.score("fmdet2k_munom_" + tag, 2048) for tag in tags]
                change = statistics.mean(values) - baseline
                first.setdefault(method, {}).setdefault(rate, {})[condition] = {
                    "tags": tags, "nominal_runs": values, "nominal_change": change,
                    "distance_from_five_point_boundary": abs(change + 0.05)}
                if abs(change + 0.05) <= 0.01 and [method, rate] not in triggers:
                    triggers.append([method, rate])
    output = {"status": "reconstructed_from_attempt_outcomes", "N0_frozen_2048": baseline,
              "F_frozen_fault": frozen_fault, "trigger_list": triggers,
              "initial_three_run_trigger_check": first, "methods": {}}
    for method, conditions in candidates.items():
        table = {}
        for condition, rates in conditions.items():
            for rate, initial in rates.items():
                tags = list(initial)
                if [method, rate] in triggers:
                    if not tags[-1].endswith("s3"):
                        raise AssertionError("Cannot derive declared top-up identities")
                    tags += [tags[-1][:-1] + str(seed) for seed in (4, 5)]
                nom_names = ["fmdet2k_munom_" + tag for tag in tags]
                fault_names = [identities["FAULT"][condition] % tag for tag in tags]
                nom = [archive.score(name, 2048) for name in nom_names]
                fault = [archive.score(name, 512) for name in fault_names]
                change = statistics.mean(nom) - baseline
                table.setdefault(rate, {})[condition] = {
                    "seeds": tags, "N_seeds": nom, "S_seeds": fault,
                    "N": statistics.mean(nom), "S": statistics.mean(fault), "dN": change,
                    "feasible_d05": change >= -0.05, "feasible_d10": change >= -0.10,
                    "nominal_result_names": nom_names, "fault_result_names": fault_names,
                    "distance_from_d05_feasibility": change + 0.05,
                }
        result = {"table": table}
        for delta in (0.05, 0.10):
            feasible, recovery_ratio = [], []
            per_condition = {"LF": [], "PCL": []}
            for rate, row in table.items():
                changes = [row[c]["dN"] for c in ("LF", "PCL")]
                for c in per_condition:
                    if row[c]["dN"] >= -delta:
                        per_condition[c].append((row[c]["S"], -float(rate), rate))
                if min(changes) >= -delta:
                    objective = (row["LF"]["S"] + row["PCL"]["S"]) / 2
                    feasible.append((objective, min(changes), -float(rate), rate))
                    ratio = statistics.mean([(row[c]["S"] - frozen_fault[c]) /
                                             (baseline - frozen_fault[c]) for c in ("LF", "PCL")])
                    recovery_ratio.append((ratio, min(changes), -float(rate), rate))
            best = max(feasible) if feasible else None
            result[f"d{round(delta * 100):02d}"] = {
                "shared_lr": best[3] if best else "no feasible shared learning rate",
                "objective": best[0] if best else None,
                "sensitivity_recovery_ratio_lr": max(recovery_ratio)[3] if recovery_ratio else None,
                "per_condition_reference": {c: max(rows)[2] if rows else "no feasible"
                                            for c, rows in per_condition.items()},
            }
        result["switch_LF"] = {
            "keep_main_config": result["d05"]["shared_lr"],
            "unconstrained_dev_best": max((row["LF"]["S"], -float(rate), rate)
                                          for rate, row in table.items())[2],
        }
        output["methods"][method] = result
    return output


def compare_primary(tests, locked, numbers):
    differences = []
    for name, value in tests.items():
        ref = locked["primary"][name]
        for key in ("estimate", "se", "df", "t", "p_one_sided", "p_holm"):
            close(value[key], ref[key], f"final_stats/{name}/{key}")
            differences.append(abs(value[key] - ref[key]))
        for i, bound in enumerate(value["ci95_two_sided"]):
            close(bound, ref["ci95_two_sided"][i], f"final_stats/{name}/CI/{i}")
            differences.append(abs(bound - ref["ci95_two_sided"][i]))
        if value["outcome"] != ref["outcome"]:
            raise AssertionError(f"Changed decision: {name}")
        if len(value["groups"]) != len(ref["groups"]):
            raise AssertionError(f"Changed groups: {name}")
        for g, r in zip(value["groups"], ref["groups"]):
            if g["n"] != r["n"] or g["weight"] != r["weight"]:
                raise AssertionError(f"Changed group n/weight: {name}")
            for x, y in zip(g["values"], r["values"]):
                close(x, y, f"final_stats/{name}/run_value")
        manuscript = numbers["tests"][name]
        for target, key in (("est", "estimate"), ("df", "df"),
                            ("p", "p_one_sided"), ("p_holm", "p_holm")):
            close(value[key], manuscript[target], f"numbers/{name}/{target}")
        for x, y in zip(value["ci95_two_sided"], manuscript["ci"]):
            close(x, y, f"numbers/{name}/CI")
    return {"status": "passed", "tests_checked": len(tests),
            "largest_absolute_numeric_difference_vs_final_stats": max(differences)}


def compare_selection(rebuilt, reference, manifest):
    if sorted(rebuilt["trigger_list"]) != sorted(reference["trigger_list"]):
        raise AssertionError("Changed initial top-up trigger list")
    close(rebuilt["N0_frozen_2048"], reference["N0_frozen_2048"], "selection/frozen")
    for condition, value in rebuilt["F_frozen_fault"].items():
        close(value, reference["F_frozen_fault"][condition], "selection/frozen_fault")
    for method, output in rebuilt["methods"].items():
        archived = reference["methods"][method]
        for rate, conditions in output["table"].items():
            for condition, row in conditions.items():
                ref = archived["table"][rate][condition]
                if row["seeds"] != ref["seeds"]:
                    raise AssertionError("Selection seed identity mismatch")
                for key in ("N", "S", "dN"):
                    close(row[key], ref[key], f"selection/{method}/{rate}/{condition}/{key}")
                for key in ("N_seeds", "S_seeds"):
                    for x, y in zip(row[key], ref[key]):
                        close(x, y, "selection/run_score")
        for delta in ("d05", "d10"):
            a, b = output[delta], archived[delta]
            for key in ("shared_lr", "sensitivity_recovery_ratio_lr", "per_condition_reference"):
                if a[key] != b[key]:
                    raise AssertionError(f"Selection mismatch {method}/{delta}/{key}")
            close(a["objective"], b["objective"], "selection/objective")
        if output["switch_LF"] != archived["switch_LF"]:
            raise AssertionError("Changed ideal-switching rate")
        if output["d05"]["shared_lr"] != manifest["methods"][method]["lr_main"]:
            raise AssertionError("Selection and locked main manifest disagree")
        if output["switch_LF"]["unconstrained_dev_best"] != manifest["methods"][method]["lr_H5"]:
            raise AssertionError("Selection and H5 manifest disagree")
    return {"status": "passed", "trigger_stage": "recomputed from original three runs",
            "final_stage": "recomputed from three/five runs without rounding"}


def write_summary(path, report):
    lines = ["# Independent primary-analysis reconstruction", "",
             f"Status: **{report['status']}**. No training, simulation, reselection of the manuscript, or input-file changes.", "",
             "The calculations use binary attempt outcomes; each training run contributes one score. "
             "The original statistical core and analysis scripts are not imported or executed.", "",
             "| Test | Estimate [95% CI], pp | df | One-sided p | Holm p | Group n |",
             "|---|---:|---:|---:|---:|---|"]
    for key, value in report["primary"].items():
        lo, hi = value["ci95_two_sided"]
        sizes = "/".join(str(g["n"]) for g in value["groups"])
        lines.append(f"| {key} | {100*value['estimate']:+.4f} [{100*lo:.4f}, {100*hi:.4f}] "
                     f"| {value['df']:.4f} | {value['p_one_sided']:.10g} | {value['p_holm']:.10g} | {sizes} |")
    lines += ["", "Group order is residual PPO, residual GRPO, full-weight PPO, full-weight GRPO; "
              "H2 lists the LF groups followed by the PCL groups. H4 subtracts combined-shift recovery "
              "from source-shift recovery within each trained model before averaging four schemes.", "",
              f"Read {len(report['raw_result_records'])} distinct result files; "
              f"the nine tests involve {len(report['model_reuse'])} distinct declared training-model tags. "
              "Shared tags across tests are explicitly retained in the JSON; they are not additional replications.", "",
              "## Development selection", ""]
    selection = report["selection"]
    if selection["status"] == "reconstructed_from_attempt_outcomes":
        lines += ["Initial three-run trigger decisions, subsequent three/five-run means, five/ten-point "
                  "selection, recovery-ratio sensitivity, per-condition rates, and H5 rates all reproduce the archive.", "",
                  "| Method | Five-point shared rate | Ten-point shared rate | H5 LF rate |",
                  "|---|---|---|---|"]
        for method, item in selection["methods"].items():
            lines.append(f"| {method} | {item['d05']['shared_lr']} | {item['d10']['shared_lr']} "
                         f"| {item['switch_LF']['unconstrained_dev_best']} |")
        lines += ["", "Selected unrounded feasibility boundaries (N−N0; proportions):"]
        for method, rate, condition in (("full_ppo", "3e-5", "PCL"),
                                        ("res_grpo", "3e-3", "PCL"),
                                        ("res_grpo", "3e-3", "LF")):
            row = selection["methods"][method]["table"][rate][condition]
            lines.append(f"- {method}, {rate}, {condition}: {row['dN']:.17g}; "
                         f"signed distance to −0.05: {row['distance_from_d05_feasibility']:.17g}.")
    else:
        lines.append("Not fully reconstructed: " + selection["reason"])
    lines += ["", "## Scope", "",
              "This verifies recorded outcomes through the specified selection and nine-test analysis. "
              "It does not rerun simulation, verify checkpoint bytes or physical lock trajectories, establish "
              "independent replication, repair development selection effects, or reconstruct every diagnostic "
              "and figure in the paper. Frozen references are fixed conditional on the recorded evaluations; "
              "their episodes are not extra training replications.", "",
              "The companion JSON contains every group, weight, tag, checkpoint path declared by the manifest, "
              "run score, reuse relationship, source-file hash, unrounded selection boundary and comparison result."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    version = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=version.parent / "data")
    parser.add_argument("--numbers", type=Path, default=version / "numbers.json")
    parser.add_argument("--output", type=Path, default=version / "audit" / "primary_recomputed.json")
    args = parser.parse_args()
    root = args.data_root.resolve()
    meta = root / "meta" / "temp"
    manifest_path = meta / "final_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    archive = Archive(root)
    primary, reuse = reconstruct_primary(archive, manifest)
    primary_records = len(archive.records)
    selection_source = root / "meta" / "tools" / "select_shared_lr.py"
    try:
        selection = reconstruct_selection(archive, selection_source)
    except (FileNotFoundError, KeyError) as error:
        selection = {"status": "not_fully_reconstructed", "reason": str(error)}
    # Archived summaries are read only after the independent values exist.
    locked_path = meta / "final_stats.json"
    locked = json.loads(locked_path.read_text())
    numbers = json.loads(args.numbers.read_text())
    comparison = compare_primary(primary, locked, numbers)
    selection_path = meta / "shared_lr_select.json"
    selection_comparison = ({"status": "not_available"} if selection["status"] != "reconstructed_from_attempt_outcomes"
                            else compare_selection(selection, json.loads(selection_path.read_text()), manifest))
    report = {
        "status": "passed" if selection_comparison["status"] == "passed" else "primary_passed_selection_incomplete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "eval_seed": manifest["eval_seed"],
        "primary": primary, "comparison": comparison, "model_reuse": reuse,
        "selection": selection, "selection_comparison": selection_comparison,
        "primary_result_files_read": primary_records, "raw_result_records": archive.records,
        "provenance": {str(path): sha256(path) for path in
                       (Path(__file__), manifest_path, locked_path, args.numbers, selection_source, selection_path)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_summary(args.output.with_suffix(".md"), report)
    print(f"{report['status']}: {len(primary)} primary tests; {len(archive.records)} raw result files; "
          f"{len(reuse)} distinct primary-model tags")
    print("Output:", args.output)


if __name__ == "__main__":
    main()
