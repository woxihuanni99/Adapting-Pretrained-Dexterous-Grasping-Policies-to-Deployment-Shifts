"""Train the route-2 grasp-success verifier (research_vnet.py) on candidate rollouts from research_vsel.py --mode collect.
Held-out split is by OBJECT (--val-frac of objects), so the offline numbers measure generalisation to new shapes.
Offline selection metrics on held-out groups (K candidates, same state and friction): success of the verifier's
top-1 candidate vs candidate 0 (planner mean) vs group mean vs pass@K (oracle). Writes verifier.pt + result.json."""
import argparse
import os
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))


def require_e(path):
    path = Path(path).expanduser().resolve()
    # Public release: the original guard restricted outputs to the authors' storage volumes.
    # Set DEXGRASP_ROOT to restrict outputs to one directory tree; unset, any path is accepted.
    root = os.environ.get("DEXGRASP_ROOT")
    if root and not str(path).startswith(str(Path(root).expanduser().resolve())):
        raise ValueError("Research files must resolve under DEXGRASP_ROOT: " + str(path))
    return path


def auc(s, y):
    import numpy as np
    o = np.argsort(s, kind="stable")
    rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
    npos, nneg = y.sum(), len(y) - y.sum()
    if npos == 0 or nneg == 0:
        return None
    return float((rk[y > 0].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--data", required=True, help="comma list of run dirs (relative to the research root) with rollouts.npz")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--ensemble", type=int, default=5)
    p.add_argument("--rank-weight", type=float, default=0.0, help="extra in-group listwise loss (softmax over candidates)")
    p.add_argument("--full", action="store_true", help="after validation, refit on all objects for the saved verifier")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--mu-input", action="store_true", help="ORACLE diagnostic: append [log mu, is_nominal] to x (offline metrics only)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from research_vnet import Verifier, save_verifier
    out = require_e(args.output)
    out.mkdir(parents=True, exist_ok=False)
    base = ROOT.parents[1]
    parts = []
    for d in args.data.split(","):
        f = require_e(base / d.strip() / "rollouts.npz")
        z = np.load(f, allow_pickle=False)
        names = z["obj_names"]
        parts.append(dict(x=z["x"].astype(np.float32), m=z["m"], u=z["u"], r=z["r"], mu=z["mu"],
                          obj=np.array([str(names[i]) for i in z["obj"]]), K=int(z["group_size"]), src=d))
    if args.mu_input:
        for q in parts:
            nom = (q["mu"] < 0).astype(np.float32)
            lm = np.where(q["mu"] > 0, np.log(np.clip(q["mu"], 1e-3, None)), 0.0).astype(np.float32)
            q["x"] = np.concatenate([q["x"], lm[:, None], nom[:, None]], 1)
    K = parts[0]["K"]
    assert all(q["K"] == K for q in parts)
    cat = lambda k: np.concatenate([q[k] for q in parts])
    X, M, U, R, MU, OBJ = cat("x"), cat("m"), cat("u"), cat("r"), cat("mu"), cat("obj")
    # rows are stored group-contiguous (K consecutive rows = one state) in every part
    ng = len(R) // K
    objs = sorted(set(OBJ.tolist()))
    rng = np.random.RandomState(args.seed)
    val_objs = set(rng.choice(objs, max(1, int(round(args.val_frac * len(objs)))), replace=False).tolist())
    gobj = OBJ.reshape(ng, K)[:, 0]
    gval = np.array([o in val_objs for o in gobj])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    def fit(gmask, tag):
        rows = np.repeat(gmask, K)
        xm, xs = X[rows].mean(0), X[rows].std(0) + 1e-4
        v = Verifier(X.shape[1], M.shape[1], args.hidden, args.ensemble, args.layers, args.dropout).to(dev)
        v.x_mean.copy_(torch.tensor(xm)); v.x_std.copy_(torch.tensor(xs))
        opt = torch.optim.AdamW(v.parameters(), lr=args.lr, weight_decay=args.wd)
        gidx = np.nonzero(gmask)[0]
        Xt, Mt, Ut, Rt = (torch.tensor(a).to(dev) for a in (X, M, U, R))
        E = args.ensemble
        boot = [torch.tensor(rng.choice(gidx, len(gidx), replace=True)).to(dev) for _ in range(E)]
        steps = max(1, len(gidx) * K // args.batch)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps)
        gb = max(1, args.batch // K)
        t0 = time.time()
        for ep in range(args.epochs):
            v.train()
            tot = 0.0
            for s in range(steps):
                loss = 0.0
                for e in range(E):
                    g = boot[e][torch.randint(0, len(gidx), (gb,), device=dev)]
                    r_ = (g.unsqueeze(1) * K + torch.arange(K, device=dev)).reshape(-1)
                    logit = v.forward(Xt[r_], Mt[r_], Ut[r_], members=[e])[0]
                    y = Rt[r_]
                    le = F.binary_cross_entropy_with_logits(logit, y)
                    if args.rank_weight > 0:
                        lg, yg = logit.view(-1, K), y.view(-1, K)
                        has = (yg.sum(1) > 0) & (yg.sum(1) < K)
                        if has.any():
                            tgt = yg[has] / yg[has].sum(1, keepdim=True)
                            le = le + args.rank_weight * -(tgt * F.log_softmax(lg[has], 1)).sum(1).mean()
                    loss = loss + le
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                tot += float(loss) / E
            if ep % 10 == 0 or ep == args.epochs - 1:
                print(json.dumps({"fit": tag, "epoch": ep, "loss": tot / steps, "seconds": time.time() - t0}), flush=True)
        return v

    def evaluate(v, gmask):
        rows = np.nonzero(np.repeat(gmask, K))[0]
        v.eval()
        with torch.no_grad():
            sc = []
            for i in range(0, len(rows), 65536):
                b = rows[i:i + 65536]
                sc.append(v.score(torch.tensor(X[b]).to(dev), torch.tensor(M[b]).to(dev), torch.tensor(U[b]).to(dev)).cpu().numpy())
            sc = np.concatenate(sc)
        r = R[rows].reshape(-1, K)
        s = sc.reshape(-1, K)
        mu = MU[rows].reshape(-1, K)[:, 0]
        top = r[np.arange(len(r)), s.argmax(1)]
        inf = (r.max(1) > r.min(1))
        wg = [auc(s[i], r[i]) for i in np.nonzero(inf)[0]]
        wg = [w for w in wg if w is not None]
        res = {"groups": int(len(r)), "auc": auc(sc, R[rows]), "sel_success": float(top.mean()),
               "cand0_success": float(r[:, 0].mean()), "group_mean": float(r.mean()), "pass_at_k": float(r.max(1).mean()),
               "sel_minus_cand0": float(top.mean() - r[:, 0].mean()), "within_group_auc": float(np.mean(wg)) if wg else None,
               "informative_groups": int(inf.sum()), "per_mu": {}}
        for m_ in sorted(set(mu.tolist())):
            k_ = mu == m_
            res["per_mu"]["nom" if m_ < 0 else "%.2f" % m_] = {"sel": float(top[k_].mean()), "cand0": float(r[k_, 0].mean()),
                                                              "mean": float(r[k_].mean()), "pass": float(r[k_].max(1).mean())}
        return res

    v = fit(~gval, "train")
    val = evaluate(v, gval)
    trn = evaluate(v, ~gval)
    print(json.dumps({"val": val, "train": {k: trn[k] for k in ("auc", "sel_success", "cand0_success", "pass_at_k")}}), flush=True)
    if args.full:
        v = fit(np.ones(ng, dtype=bool), "full")
    meta = {"in_dim": int(X.shape[1]), "act_dim": int(M.shape[1]), "hidden": args.hidden, "ensemble": args.ensemble,
            "dropout": args.dropout, "mu_input": args.mu_input,
            "layers": args.layers, "group_size": K, "data": args.data, "val_objects": sorted(val_objs), "full_refit": args.full,
            "samples": int(len(R)), "args": vars(args)}
    save_verifier(v, out / "verifier.pt", meta)
    totals = {"val_sel_success": val["sel_success"], "val_cand0_success": val["cand0_success"], "val_pass_at_k": val["pass_at_k"],
              "val_auc": val["auc"], "val_within_group_auc": val["within_group_auc"], "val_sel_minus_cand0": val["sel_minus_cand0"],
              "samples": int(len(R))}
    (out / "result.json").write_text(json.dumps({"status": "completed", "totals": totals, "val": val, "train": trn, "meta": meta}, indent=1))
    print("VTRAIN_DONE", json.dumps(totals), flush=True)


if __name__ == "__main__":
    main()
