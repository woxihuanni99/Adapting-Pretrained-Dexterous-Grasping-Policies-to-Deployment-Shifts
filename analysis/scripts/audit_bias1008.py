#!/usr/bin/env python3
"""Recompute the output-shift coordinate control of 8 October 2026 from binary attempt outcomes, and add the context comparison
with the locked residual GRPO and full-weight GRPO groups (pre-specified in the plan, section 4: computed off the server because
the locked runs were evaluated on other machines).

Inputs
  --new-root   backup of the 8 October server outputs (runs/<name>/result.json, temp/bx_manifest.json, temp/bx_select.json,
               temp/bx_stats.json)
  --data-root  paper data archive (runs/<name>/result.json of the locked evaluations, meta/temp/final_manifest.json)

The script does not import the server-side analysis code. It rebuilds every score from the per-attempt success flags,
re-derives the development selection and the test-set contrasts with its own Welch-Satterthwaite implementation and requires
agreement with the server records (estimates 1e-12; interval bounds 1e-9 for SciPy-version differences in t quantiles).
Independent training runs are the units; everything is descriptive and outside the nine-test family."""
import argparse, hashlib, json, math, statistics
from pathlib import Path
from scipy.stats import t as student_t

TOL, TOL_CI = 1e-12, 1e-9
SLOTS = {"seen": 141, "unseen": 100}
ATTEMPTS = {"seen": 1128, "unseen": 800}
LRS = {"raw": ("3e-3", "1e-2", "3e-2"), "sigma": ("1e-2", "3e-2", "1e-1")}
CONDS = {"LF": ["LF", "nom", "LF25", "LF75"], "PCL": ["pn10", "nom"]}
LABEL = {"LF": "shlockLF", "PCL": "pn10"}


class Runs:
    def __init__(self, root):
        self.root, self.cache, self.files = Path(root), {}, {}

    def flags(self, name):
        if name not in self.cache:
            raw = (self.root / "runs" / name / "result.json").read_bytes()
            rec = json.loads(raw)
            if rec.get("status") != "completed":
                raise AssertionError("unfinished result: " + name)
            eps = []
            for ep in rec["episodes"]:
                f = ep["official_ppo_print_successes"]
                if any(x not in (0, 1) for x in f):
                    raise AssertionError("non-binary outcome: " + name)
                eps.append(list(f))
            n, s = sum(map(len, eps)), sum(map(sum, eps))
            if abs(s / n - rec["totals"]["success_rate"]) > TOL:
                raise AssertionError("total success mismatch: " + name)
            self.cache[name], self.files[name] = eps, hashlib.sha256(raw).hexdigest()
        return self.cache[name]

    def score(self, name, attempts=None, split=None):
        eps = self.flags(name)
        n = sum(map(len, eps))
        if attempts is not None and n != attempts:
            raise AssertionError("%s: %d attempts, expected %d" % (name, n, attempts))
        if split is not None and (len(eps) != 8 or any(len(f) != SLOTS[split] for f in eps)):
            raise AssertionError("final protocol violated: " + name)
        return sum(map(sum, eps)) / n


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
                raise AssertionError(label + ": key missing in archive: " + k)
            same(a[k], b[k], label + "." + k)
        for k in b:
            if k not in a:
                raise AssertionError(label + ": archived key not recomputed: " + k)
        return
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(label + ": length")
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, "%s[%d]" % (label, i))
        return
    if isinstance(a, float) or isinstance(b, float):
        tol = TOL_CI if ".ci[" in label else TOL
        if a is None or b is None or not (abs(a - b) <= tol or (math.isinf(a) and math.isinf(b))):
            raise AssertionError("%s: recomputed %r, archived %r" % (label, a, b))
        return
    if a != b:
        raise AssertionError("%s: recomputed %r, archived %r" % (label, a, b))


def word(w):
    return ("the coordinate changed this outcome at the selected operating points" if (w["ci"][0] > 0 or w["ci"][1] < 0)
            else "no difference detected at the selected operating points (not evidence of equivalence)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="audit/bias1008_recomputed.json")
    a = ap.parse_args()
    new, old = Runs(a.new_root), Runs(a.data_root)
    tmp = Path(a.new_root) / "temp"
    man = json.loads((tmp / "bx_manifest.json").read_text())
    fm = json.loads((Path(a.data_root) / "meta/temp/final_manifest.json").read_text())
    out = {"checks": []}

    # ---- A. development selection (per arm, shared over LF and PCL) ---------------------------------------------
    N0 = new.score("bx_fmdet2k_munom_frozen", 2048)

    def dev(entries, c):
        S = [new.score("fmdet_munom_%s_%s_64x8" % (LABEL[c], e["short"]), 512) for e in entries]
        N = [new.score("fmdet2k_munom_%s" % e["short"], 2048) for e in entries]
        return {"S_runs": S, "N_runs": N, "S": statistics.mean(S), "N": statistics.mean(N), "dN": statistics.mean(N) - N0,
                "tags": [e["tag"] for e in entries]}

    sel = {"N0_this_machine": N0, "arms": {}}
    for arm, lrs in LRS.items():
        rec = {"rates": {}}
        for lr in lrs:
            st = {c: dev(man["arms"][arm]["%s|%s" % (c, lr)], c) for c in ("LF", "PCL")}
            rec["rates"][lr] = {"per_condition": st, "feasible": all(st[c]["dN"] >= -0.05 for c in st),
                                "mean_S": statistics.mean(st[c]["S"] for c in st), "worst_dN": min(st[c]["dN"] for c in st)}
        feas = [lr for lr in lrs if rec["rates"][lr]["feasible"]]
        feas.sort(key=lambda lr: (-rec["rates"][lr]["mean_S"], -rec["rates"][lr]["worst_dN"], float(lr)))
        rec["selected"] = feas[0] if feas else None
        rec["selected_at_grid_edge"] = (feas[0] in (lrs[0], lrs[-1])) if feas else None
        if not feas:
            rec["note"] = "no feasible configuration"
        sel["arms"][arm] = rec
    arch = json.loads((tmp / "bx_select.json").read_text())
    same({"N0_this_machine": sel["N0_this_machine"], "arms": sel["arms"]},
         {"N0_this_machine": arch["N0_this_machine"], "arms": arch["arms"]}, "bx_select")
    out["selection"] = sel
    out["checks"].append("bx_select.json reproduced from development flags")

    # ---- B. test-set contrasts (server record) + C. context vs locked groups ------------------------------------
    def Y(runs, src):
        return (runs.score(src["seen"], ATTEMPTS["seen"], "seen") + runs.score(src["unseen"], ATTEMPTS["unseen"], "unseen")) / 2

    def yname(cond, tag):
        return {"seen": "final_seen_%s_%s" % (cond, tag), "unseen": "final_unseen_%s_%s" % (cond, tag)}

    chosen = {arm: sel["arms"][arm]["selected"] for arm in LRS}
    st = {"selected": chosen, "conditions": {}}
    ctx = {"frozen_locked": {}, "conditions": {}}
    for cond in ("LF", "LF25", "LF75", "pn10", "nom"):
        ctx["frozen_locked"][cond] = Y(old, fm["frozen"][cond])
    for c, conds in CONDS.items():
        r_old = [e["tag"] for e in fm["methods"]["res_grpo"]["sets"][c]]
        f_old = [e["tag"] for e in fm["methods"]["full_grpo"]["sets"][c]]
        for cond in conds:
            rec = {"runs": {}, "D_idx": {}, "context": {}}
            for arm in LRS:
                for lr in LRS[arm]:
                    rec["runs"]["%s|%s" % (arm, lr)] = [Y(new, yname(cond, e["tag"])) for e in man["arms"][arm]["%s|%s" % (c, lr)]]
            for i in range(3):
                lr_r, lr_s = LRS["raw"][i], LRS["sigma"][i]
                rec["D_idx"]["%s_vs_%s" % (lr_s, lr_r)] = welch([(1, rec["runs"]["sigma|" + lr_s]), (-1, rec["runs"]["raw|" + lr_r])])
            if all(chosen.values()):
                d = welch([(1, rec["runs"]["sigma|" + chosen["sigma"]]), (-1, rec["runs"]["raw|" + chosen["raw"]])])
                rec["D_sel"] = d
                if c == "LF" and cond in ("LF", "nom"):
                    rec["D_sel_wording"] = word(d)
            st["conditions"]["%s|%s" % (c, cond)] = rec
            # context (not in the server record): locked groups were evaluated on another machine
            R = [Y(old, fm["evals"][t][cond]) for t in r_old]
            Fw = [Y(old, fm["evals"][t][cond]) for t in f_old]
            cx = {"residual_grpo_locked": R, "fullweight_grpo_locked": Fw, "arms": {}}
            for arm in LRS:
                for lr in ([chosen[arm]] if chosen[arm] else LRS[arm]):
                    v = rec["runs"]["%s|%s" % (arm, lr)]
                    cx["arms"]["%s|%s" % (arm, lr)] = {"minus_residual_grpo_locked": welch([(1, v), (-1, R)]),
                                                       "minus_fullweight_grpo_locked": welch([(1, v), (-1, Fw)])}
            ctx["conditions"]["%s|%s" % (c, cond)] = cx
    arch = json.loads((tmp / "bx_stats.json").read_text())
    same(st, arch, "bx_stats")
    out["stats"] = st
    out["context"] = ctx
    out["checks"].append("bx_stats.json reproduced from test flags; context vs locked residual/full-weight GRPO added")
    out["inputs"] = {"result_files_new": len(new.files), "result_files_archived": len(old.files),
                     "sha256_new": hashlib.sha256("".join(sorted(new.files.values())).encode()).hexdigest(),
                     "sha256_archived": hashlib.sha256("".join(sorted(old.files.values())).encode()).hexdigest()}
    for f in ("bx_manifest.json", "bx_select.json", "bx_stats.json"):
        out["inputs"][f] = hashlib.sha256((tmp / f).read_bytes()).hexdigest()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("\n".join(out["checks"]))
    print("result files: new %d, archived %d -> %s" % (len(new.files), len(old.files), a.out))

    def pp(w):
        return "%+.1f [%.1f, %.1f]" % (100 * w["est"], 100 * w["ci"][0], 100 * w["ci"][1])
    print("selected", chosen)
    for k, rec in st["conditions"].items():
        print(k, "|", " ".join("%s %.1f" % (kk, 100 * statistics.mean(v)) for kk, v in rec["runs"].items()))
        if "D_sel" in rec:
            print("    D_sel", pp(rec["D_sel"]), rec.get("D_sel_wording", ""))
        for kk, w in rec["D_idx"].items():
            print("    D_idx", kk, pp(w))
        cx = ctx["conditions"][k]
        print("    locked res GRPO %.1f  full GRPO %.1f" % (100 * statistics.mean(cx["residual_grpo_locked"]), 100 * statistics.mean(cx["fullweight_grpo_locked"])))
        for kk, v in cx["arms"].items():
            print("    %s - res %s | - full %s" % (kk, pp(v["minus_residual_grpo_locked"]), pp(v["minus_fullweight_grpo_locked"])))


if __name__ == "__main__":
    main()
