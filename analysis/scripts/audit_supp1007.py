#!/usr/bin/env python3
"""Recompute the supplementary experiments of 7 October 2026 from binary attempt outcomes.

Inputs
  --new-root   backup of the 7 October server outputs (runs/<name>/result.json, episode_XXX/initial_state.npz,
               temp/fs_manifest.json, temp/fs_select.json, temp/fs_stats.json, temp/coll_repro.json, temp/coll_stats.json)
  --data-root  paper data archive (runs/<name>/result.json of the locked evaluations, meta/temp/final_manifest.json)

The script does not import the server-side analysis code. It rebuilds every score from the per-attempt success flags,
recomputes the selection record, the setting-matched contrasts, the reproduction check (flags) and the collision
re-evaluation with its own Welch-Satterthwaite implementation, and requires agreement with the archived server records
to 1e-12 (interval bounds 1e-9, which absorbs SciPy-version differences in t quantiles). Output: audit/supp1007_recomputed.json (input to numbers.json through build_supp1007_record.py).
Independent training runs are the units; everything here is descriptive and outside the nine-test family."""
import argparse, hashlib, json, math, statistics
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

TOL = 1e-12
TOL_CI = 1e-9
SLOTS = {"seen": 141, "unseen": 100}
ATTEMPTS = {"seen": 1128, "unseen": 800}


class Runs:
    def __init__(self, root):
        self.root = Path(root)
        self.cache = {}
        self.files = {}

    def flags(self, name):
        if name not in self.cache:
            path = self.root / "runs" / name / "result.json"
            raw = path.read_bytes()
            rec = json.loads(raw)
            if rec.get("status") != "completed":
                raise AssertionError("unfinished result: " + name)
            eps = []
            for ep in rec["episodes"]:
                f = ep["official_ppo_print_successes"]
                if any(x not in (0, 1) for x in f):
                    raise AssertionError("non-binary outcome: " + name)
                eps.append((list(f), list(ep.get("object_assets", []))))
            n = sum(len(f) for f, _ in eps)
            s = sum(sum(f) for f, _ in eps)
            if abs(s / n - rec["totals"]["success_rate"]) > TOL:
                raise AssertionError("total success mismatch: " + name)
            self.cache[name] = eps
            self.files[name] = hashlib.sha256(raw).hexdigest()
        return self.cache[name]

    def score(self, name, attempts=None, split=None):
        eps = self.flags(name)
        n = sum(len(f) for f, _ in eps)
        if attempts is not None and n != attempts:
            raise AssertionError("%s: %d attempts, expected %d" % (name, n, attempts))
        if split is not None:
            if len(eps) != 8 or any(len(f) != SLOTS[split] for f, _ in eps):
                raise AssertionError("final protocol violated: " + name)
        return sum(sum(f) for f, _ in eps) / n


def welch(groups):
    est = sum(w * statistics.mean(v) for w, v in groups)
    comps = [w * w * statistics.variance(v) / len(v) for w, v in groups]
    var = sum(comps)
    den = sum(c * c / (len(v) - 1) for c, (_, v) in zip(comps, groups))
    df = var * var / den if den > 0 else math.inf
    q = float(student_t.ppf(0.975, df)) if math.isfinite(df) else 1.959963984540054
    se = math.sqrt(var)
    return {"est": est, "se": se, "df": df, "ci": [est - q * se, est + q * se], "n": [len(v) for _, v in groups],
            "means": [statistics.mean(v) for _, v in groups]}


def same(a, b, label):
    if isinstance(a, dict):
        for k in a:
            if k not in b:
                raise AssertionError(label + ": missing key " + k)
            same(a[k], b[k], label + "." + k)
        return
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(label + ": length")
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, "%s[%d]" % (label, i))
        return
    if isinstance(a, float) or isinstance(b, float):
        # interval bounds depend on the Student-t quantile at fractional df; SciPy versions differ in the last digits
        tol = TOL_CI if ".ci[" in label else TOL
        if a is None or b is None or not (abs(a - b) <= tol or (math.isinf(a) and math.isinf(b))):
            raise AssertionError("%s: recomputed %r, archived %r" % (label, a, b))
        return
    if a != b:
        raise AssertionError("%s: recomputed %r, archived %r" % (label, a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="audit/supp1007_recomputed.json")
    ap.add_argument("--skip-coll", action="store_true", help="only the setting-matched control and reproduction check")
    a = ap.parse_args()
    new, old = Runs(a.new_root), Runs(a.data_root)
    tmp = Path(a.new_root) / "temp"
    fm = json.loads((Path(a.data_root) / "meta/temp/final_manifest.json").read_text())
    man = json.loads((tmp / "fs_manifest.json").read_text())
    out = {"inputs": {}, "checks": []}

    def Y(runs, src, pre=""):
        return (runs.score(pre + src["seen"], ATTEMPTS["seen"], "seen") +
                runs.score(pre + src["unseen"], ATTEMPTS["unseen"], "unseen")) / 2

    # ---- A. development selection -------------------------------------------------------------------------
    label = {"LF": "shlockLF", "PCL": "pn10"}
    N0 = new.score("fs_fmdet2k_munom_frozen", 2048)

    def dev(entries, c):
        S = [new.score("fmdet_munom_%s_%s_64x8" % (label[c], e["short"]), 512) for e in entries]
        N = [new.score("fmdet2k_munom_%s" % e["short"], 2048) for e in entries]
        return {"S_runs": S, "N_runs": N, "S": statistics.mean(S), "N": statistics.mean(N),
                "dN": statistics.mean(N) - N0, "tags": [e["tag"] for e in entries]}

    sel = {"N0_this_machine": N0, "rates": {}, "fullweight_new": {}}
    for lr in ("1e-3", "3e-3"):
        st = {c: dev(man["fsr"]["%s|%s" % (c, lr)], c) for c in ("LF", "PCL")}
        sel["rates"][lr] = {"per_condition": st, "feasible": all(st[c]["dN"] >= -0.05 for c in st),
                            "mean_S": statistics.mean(st[c]["S"] for c in st), "worst_dN": min(st[c]["dN"] for c in st)}
    for c in ("LF", "PCL"):
        sel["fullweight_new"][c] = dev(man["fsf"][c], c)
    feas = [lr for lr in ("1e-3", "3e-3") if sel["rates"][lr]["feasible"]]
    feas.sort(key=lambda lr: (-sel["rates"][lr]["mean_S"], -sel["rates"][lr]["worst_dN"], float(lr)))
    sel["selected"] = feas[0] if feas else None
    arch = json.loads((tmp / "fs_select.json").read_text())
    sel["N0_original"] = old.score("fmdet2k_munom_frozen", 2048)
    same(sel, arch, "fs_select")
    out["selection"] = sel
    out["checks"].append("fs_select.json reproduced")

    # ---- B. test-set contrasts of the setting-matched control ---------------------------------------------
    conds = {"LF": ["LF", "nom", "LF25", "LF75"], "PCL": ["pn10", "nom"]}

    def yname(cond, tag):
        return {"seen": "final_seen_%s_%s" % (cond, tag), "unseen": "final_unseen_%s_%s" % (cond, tag)}

    fs = {"selected_rate": sel["selected"], "frozen_archived": {}, "conditions": {}}
    for cond in ("LF", "LF25", "LF75", "pn10", "nom"):
        fs["frozen_archived"][cond] = Y(old, fm["frozen"][cond])
    for c, cl in conds.items():
        f_new = [e["tag"] for e in man["fsf"][c]]
        r_old = [e["tag"] for e in fm["methods"]["res_grpo"]["sets"][c]]
        f_old = [e["tag"] for e in fm["methods"]["full_grpo"]["sets"][c]]
        for cond in cl:
            fn = [Y(new, yname(cond, t)) for t in f_new]
            ro = [Y(old, fm["evals"][t][cond]) for t in r_old]
            fo = [Y(old, fm["evals"][t][cond]) for t in f_old]
            rec = {"fullweight_new": fn, "residual_old": ro, "fullweight_old": fo,
                   "G_old": welch([(1, ro), (-1, fo)]), "fullweight_new_minus_old": welch([(1, fn), (-1, fo)]),
                   "rates": {}}
            for lr in ("1e-3", "3e-3"):
                rn = [Y(new, yname(cond, e["tag"])) for e in man["fsr"]["%s|%s" % (c, lr)]]
                rec["rates"][lr] = {"residual_matched": rn, "G_new": welch([(1, rn), (-1, fn)]),
                                    "Delta": welch([(1, rn), (-1, fn), (-1, ro), (1, fo)]),
                                    "residual_matched_minus_old": welch([(1, rn), (-1, ro)])}
            fs["conditions"]["%s|%s" % (c, cond)] = rec
    arch = json.loads((tmp / "fs_stats.json").read_text())
    same(fs, arch, "fs_stats")
    if arch["selected_rate"] is not None or any("category" in r for v in arch["conditions"].values() for r in v["rates"].values()):
        raise AssertionError("unexpected category in fs_stats.json")
    out["setting_matched"] = fs
    out["checks"].append("fs_stats.json reproduced; no category assigned (no feasible rate)")

    # ---- C. reproduction check (success flags) ------------------------------------------------------------
    rep_arch = json.loads((tmp / "coll_repro.json").read_text())
    rep = {}
    for nm, v in rep_arch["runs"].items():
        base = nm.split("_", 1)[1]
        e_new, e_old = new.flags(nm), old.flags(base)
        same_eps = [f1 == f2 for (f1, _), (f2, _) in zip(e_new, e_old)]
        rep[nm] = {"flags_identical_all": all(same_eps) and len(e_new) == len(e_old), "episodes_identical": sum(same_eps),
                   "success_new": new.score(nm), "success_archived": old.score(base),
                   "initial_state_max_abs_diff_server": v["initial_state_max_abs_diff"]}
        same({k: rep[nm][k] for k in ("flags_identical_all", "episodes_identical", "success_new", "success_archived")},
             v, "coll_repro." + nm)
    out["reproduction"] = {"verdict": rep_arch["verdict"], "runs": rep}
    out["checks"].append("coll_repro.json flags and success rates reproduced (initial states taken from the server record)")

    # ---- D. full collision re-evaluation -------------------------------------------------------------------
    if not a.skip_coll:
        cs_arch = json.loads((tmp / "coll_stats.json").read_text())
        if cs_arch.get("arm_original") != "collref":
            raise AssertionError("expected the same-session original-setting arm (collref)")
        M = ("res_ppo", "res_grpo", "full_ppo", "full_grpo")
        PW, OW = (0.5, 0.5, -0.5, -0.5), (-0.5, 0.5, -0.5, 0.5)
        cs = {"arm_original": "collref", "frozen": {}, "conditions": {}}
        for cond in ("LF", "pn10", "nom"):
            cs["frozen"][cond] = {"orig": Y(new, fm["frozen"][cond], "collref_"), "coll": Y(new, fm["frozen"][cond], "coll_")}
        for cset, fc in (("LF", "LF"), ("PCL", "pn10")):
            for cond in (fc, "nom"):
                g = {}
                for m in M:
                    tags = [e["tag"] for e in fm["methods"][m]["sets"][cset]]
                    yo = [Y(new, fm["evals"][t][cond], "collref_") for t in tags]
                    yc = [Y(new, fm["evals"][t][cond], "coll_") for t in tags]
                    g[m] = {"orig": yo, "coll": yc, "diff": [x - y for x, y in zip(yc, yo)]}
                cs["conditions"]["%s|%s" % (cset, cond)] = {
                    "per_method": g,
                    "param_orig": welch([(w, g[m]["orig"]) for m, w in zip(M, PW)]),
                    "param_coll": welch([(w, g[m]["coll"]) for m, w in zip(M, PW)]),
                    "param_change": welch([(w, g[m]["diff"]) for m, w in zip(M, PW)]),
                    "opt_coll": welch([(w, g[m]["coll"]) for m, w in zip(M, OW)]),
                    "opt_orig": welch([(w, g[m]["orig"]) for m, w in zip(M, OW)])}
        # opt_orig is independently derived here; it is not present in the server summary.
        comparable = {**cs, "conditions": {k: {q: x for q, x in v.items() if q != "opt_orig"}
                                                for k, v in cs["conditions"].items()}}
        same(comparable, cs_arch, "coll_stats")
        # initial observations: collision run versus same-session original-setting run
        init = {}
        expected = set(cs_arch["initial_state_check"])
        actual = {p.name for p in (Path(a.new_root) / "runs").glob("coll_final_*") if p.is_dir()}
        if actual != expected:
            raise AssertionError("collision initial-state evaluation inventory mismatch")
        for name in sorted(expected):
            d = Path(a.new_root) / "runs" / name
            ref = Path(a.new_root) / "runs" / ("collref_" + d.name.split("_", 1)[1])
            envs, eps = set(), 0
            states = sorted(d.glob("episode_*/initial_state.npz"))
            ref_states = sorted(ref.glob("episode_*/initial_state.npz"))
            if len(states) != 8 or [p.parent.name for p in states] != [p.parent.name for p in ref_states]:
                raise AssertionError("missing initial-state episodes: " + d.name)
            for p1, p2 in zip(states, ref_states):
                with np.load(p1, allow_pickle=False) as one, np.load(p2, allow_pickle=False) as two:
                    x, y = one["policy_observation"], two["policy_observation"]
                split = d.name.split("_")[2]
                if x.shape != y.shape or x.ndim != 2 or x.shape[0] != SLOTS[split] or not (np.isfinite(x).all() and np.isfinite(y).all()):
                    raise AssertionError("invalid initial observations: " + str(p1))
                eps += 1
                envs |= set(np.nonzero(np.abs(x - y).max(1) > 1e-6)[0].tolist())
            init[d.name] = {"episodes_compared": eps, "envs_with_any_difference": sorted(envs)}
        same(init, cs_arch["initial_state_check"], "coll_stats.initial_state_check")
        cs["initial_state_check"] = init
        # flags also identical where nothing could differ? record how many evaluations changed at all
        changed = 0
        for nm in init:
            if [f for f, _ in new.flags(nm)] != [f for f, _ in new.flags("collref_" + nm.split("_", 1)[1])]:
                changed += 1
        cs["evaluations_with_any_changed_outcome"] = changed
        slots = sorted({(nm.split("_")[2], e) for nm, v in init.items() for e in v["envs_with_any_difference"]})
        cs["initial_state_slots"] = {"n_slots": len(slots), "slots": [list(x) for x in slots]}
        # repeated original-setting evaluations on the new machine: reproduction-check copy versus same-session re-run
        rep2 = {}
        for nm in rep:
            if not nm.startswith("repro_"):
                continue
            base = nm[len("repro_"):]
            a1, a2 = [f for f, _ in new.flags(nm)], [f for f, _ in new.flags("collref_" + base)]
            rep2[base] = {"identical": a1 == a2, "success_repro": new.score(nm), "success_collref": new.score("collref_" + base)}
        cs["repeat_original_setting"] = {"n": len(rep2), "n_identical": sum(v["identical"] for v in rep2.values()),
                                         "max_abs_diff": max(abs(v["success_repro"] - v["success_collref"]) for v in rep2.values()),
                                         "runs": rep2}
        out["collision_full"] = cs
        out["checks"].append("coll_stats.json reproduced, including the initial-state check")

    out["inputs"] = {"result_files_new": len(new.files), "result_files_archived": len(old.files),
                     "sha256_new": hashlib.sha256("".join(sorted(new.files.values())).encode()).hexdigest(),
                     "sha256_archived": hashlib.sha256("".join(sorted(old.files.values())).encode()).hexdigest()}
    for f in ("fs_select.json", "fs_stats.json", "coll_repro.json", "coll_stats.json"):
        p = tmp / f
        if p.exists():
            out["inputs"][f] = hashlib.sha256(p.read_bytes()).hexdigest()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("\n".join(out["checks"]))
    print("result files: new %d, archived %d -> %s" % (len(new.files), len(old.files), a.out))


if __name__ == "__main__":
    main()
