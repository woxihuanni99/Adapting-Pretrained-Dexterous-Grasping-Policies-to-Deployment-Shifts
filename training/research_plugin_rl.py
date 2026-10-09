"""Plug-in RL on a FROZEN DemoGrasp planner: a small head steers the planner's latent noise.

The frozen planner samples u = m0(s) + sigma0 * eps, eps ~ N(0, I) (pre-tanh). The plug-in replaces eps by
eps' ~ N(delta(s), diag(exp(rho))^2): delta(s) is a residual/steering MLP on the planner's own features
(zero-initialised, so training starts exactly at the frozen planner) and rho a learned noise scale. For a one-step
Gaussian planner a residual on the action mean and a shift of the latent noise coincide, so one head covers
  --algo ppo : RFS-style (ICLR 2026, residual + latent steering), PPO with a learned value baseline, independent states
  --algo sac : DSRL-style (CoRL 2025, RL in the latent-noise space), SAC-style Q(s, eps') regression on the terminal
               reward + entropy-regularised actor, replay buffer carried across phases
  --algo grpo: ours in plug-in form, state-paired groups + group-relative advantages (no value function)
Same budget/phases as research_grpo_fm.py. --mode eval uses the identical random-number sequence as
research_grpo_fm.py eval, so results pair with fmsel_* runs (frozen planner = fmsel_*_frozen_*).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def require_e(path):
    path = Path(path).expanduser().resolve()
    # Public release: the original guard restricted outputs to the authors' storage volumes.
    # Set DEXGRASP_ROOT to restrict outputs to one directory tree; unset, any path is accepted.
    root = os.environ.get("DEXGRASP_ROOT")
    if root and not str(path).startswith(str(Path(root).expanduser().resolve())):
        raise ValueError("Research files must resolve under DEXGRASP_ROOT: " + str(path))
    return path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()




def with_normals(pc, np, k=20):
    pc = np.asarray(pc, dtype=np.float64)
    d = ((pc[:, None, :] - pc[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d, axis=1, kind="stable")[:, 1:k + 1]
    nb = pc[idx]
    c = nb - nb.mean(1, keepdims=True)
    w, v = np.linalg.eigh(np.einsum("nki,nkj->nij", c, c))
    nrm = v[np.arange(len(pc)), :, np.argmin(w, axis=1)]
    flip = (nrm * (pc - pc.mean(0))).sum(1) < 0
    nrm[flip] *= -1
    return nrm.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("train", "eval"), default="train")
    p.add_argument("--algo", choices=("ppo", "sac", "grpo"), default="grpo")
    p.add_argument("--output", required=True)
    p.add_argument("--init-checkpoint", default=str(ROOT / "ckpt/shadow.pt"), help="frozen planner weights")
    p.add_argument("--init-head", default=None, help="head.pt from the previous phase (or for eval)")
    p.add_argument("--init-optimizer", default=None)
    p.add_argument("--init-buffer", default=None, help="sac: replay buffer from the previous phase")
    p.add_argument("--object-list", required=True)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--mus", default="0.1,0.2,0.3")
    p.add_argument("--iterations", type=int, default=0)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--sac-updates", type=int, default=100)
    p.add_argument("--sac-batch", type=int, default=512)
    p.add_argument("--sac-alpha", type=float, default=0.01)
    p.add_argument("--deterministic", action="store_true", help="eval: eps' = delta(s) (no noise), i.e. tanh(m0 + sigma0*delta)")
    p.add_argument("--pcl-noise", type=float, default=0.0)
    p.add_argument("--mu-input", action="store_true", help="oracle: give the head log(mu) of its env (friction-aware upper bound)")
    p.add_argument("--unpaired", action="store_true", help="grpo: independent states inside a group")
    p.add_argument("--adverse-scale", type=float, default=0.0)
    p.add_argument("--value-init", choices=("new", "official"), default="new",
                   help="ppo: 'official' starts the value baseline from the planner checkpoint's own critic (trainable copy)")
    p.add_argument("--no-std-norm", action="store_true", help="grpo: advantage = r - group mean (Dr.GRPO style, no per-group std)")
    p.add_argument("--ppo-paired", action="store_true", help="ppo: state-paired groups of K (same as grpo) with the value baseline")
    p.add_argument("--diag-iters", type=int, default=0,
                   help="diagnostic only (no update): N paired rollouts; compare the value baseline with the K-sample Monte-Carlo group mean")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--disable-gpu-jit-fusion", action="store_true")
    args = p.parse_args()
    mus = None if args.mus == "none" else [float(x) for x in args.mus.split(",")]
    output = require_e(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cache = require_e(ROOT.parents[1] / "cache" / "shadow_probe")
    for key, sub in {"TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp", "XDG_CACHE_HOME": "xdg",
                     "TORCH_EXTENSIONS_DIR": "torch_extensions", "CUDA_CACHE_PATH": "cuda"}.items():
        (cache / sub).mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(cache / sub)
    os.chdir(str(ROOT))
    from isaacgym import gymapi, gymtorch  # noqa: F401
    from isaacgym.torch_utils import quat_from_angle_axis, quat_apply
    import isaacgymenvs
    import numpy as np
    import torch
    import yaml
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from isaacgymenvs.utils.utils import set_seed
    from run_rl_grasp import build_runner
    train = args.mode == "train"
    K = args.group_size if train else 1
    rows = yaml.safe_load((ROOT / "assets" / args.object_list).read_text())
    num_envs = len(rows)
    if train and (num_envs % K or any(rows[i] != rows[i - i % K] for i in range(num_envs)) or sorted(rows) != rows):
        raise ValueError("object list must be sorted with each object repeated K times consecutively")
    G = num_envs // K
    init = require_e(args.init_checkpoint)
    overrides = ["task=grasp", "train=PPOOneStep", "hand=shadow_simple", "test=True",
                 "num_envs=%d" % num_envs, "seed=%d" % args.seed, "headless=True",
                 "force_render=False", "capture_video=False", "graphics_device_id=-1",
                 "task.env.observationType=eefpose+objinitpose+objpcl", "task.env.armController=pose",
                 "task.env.asset.multiObjectList=" + args.object_list,
                 "task.env.randomizeTrackingReference=True", "task.env.randomizeGraspPose=True",
                 "task.env.resetDofPosRandomInterval=0", "task.env.episodeLength=50",
                 "task.env.enablePointCloud=True", "train.params.is_vision=True",
                 "task.env.trackingReferenceFile=tasks/grasp_ref_shadow.pkl",
                 "task.env.trackingReferenceLiftTimestep=11", "task.env.render.enable=False",
                 "task.env.enableRobotTableCollision=False", "task.sim.physx.contact_collection=1"]
    with initialize_config_dir(config_dir=str(ROOT / "tasks"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint = str(init)
    cfg.task.env.asset.assetRoot = str(require_e(ROOT / "assets"))
    cfg.task.env.trackingReferenceFile = str(require_e(ROOT / cfg.task.env.trackingReferenceFile))
    cfg.seed = set_seed(args.seed, torch_deterministic=cfg.torch_deterministic, rank=0)
    if args.disable_gpu_jit_fusion:
        torch._C._jit_override_can_fuse_on_gpu(False)
        if hasattr(torch._C, "_jit_set_nvfuser_enabled"):
            torch._C._jit_set_nvfuser_enabled(False)
    from isaacgymenvs.tasks import isaacgym_task_map
    base_cls = isaacgym_task_map[cfg.task_name]
    mu_env = []

    class GroupFrictionGrasp(base_cls):
        def create_sim(self):
            result = super().create_sim()
            for i, env_ptr in enumerate(self.envs):
                if mus is None:
                    mu_env.append(float(self.cfg["env"]["objectFriction"]))
                    continue
                mu = mus[(i // K) % len(mus)]
                if train and args.adverse_scale > 0 and (i % K) % 2 == 1:
                    mu = mu * args.adverse_scale
                mu_env.append(mu)
                for name in ("robot", "object"):
                    h = self.gym.find_actor_handle(env_ptr, name)
                    shapes = self.gym.get_actor_rigid_shape_properties(env_ptr, h)
                    for s in shapes:
                        s.friction = mu
                    if not self.gym.set_actor_rigid_shape_properties(env_ptr, h, shapes):
                        raise RuntimeError("friction setter failed")
                    got = [float(s.friction) for s in self.gym.get_actor_rigid_shape_properties(env_ptr, h)]
                    if any(abs(g - mu) > 1e-6 for g in got):
                        raise RuntimeError("friction readback mismatch")
            return result

    isaacgym_task_map[cfg.task_name] = GroupFrictionGrasp
    try:
        env = isaacgymenvs.make(cfg.seed, cfg.task_name, cfg.task.env.numEnvs, cfg.sim_device, cfg.rl_device,
                                cfg.graphics_device_id, cfg.headless, cfg.multi_gpu, cfg.capture_video,
                                cfg.force_render, cfg)
    finally:
        isaacgym_task_map[cfg.task_name] = base_cls
    if len(env.object_fns) != num_envs or any(Path(env.object_fns[i]).name != rows[i] for i in range(num_envs)):
        raise RuntimeError("env->object mapping does not follow the object list")
    dev = env.device
    mu_t = torch.tensor(mu_env, device=dev)
    ids = torch.arange(num_envs, device=dev)
    env.reset_idx(ids)
    runner = build_runner(cfg, env)
    ac = runner.actor_critic
    import torch.nn as nn
    for q in ac.parameters():
        q.requires_grad_(False)
    ac.eval()
    feat_dim = int(ac.pc_start_idx + ac.pc_emb_dim) + (1 if args.mu_input else 0)
    log_mu = torch.log(mu_t).unsqueeze(1)
    act_dim = int(ac.log_std.shape[0])
    sigma0 = torch.exp(ac.log_std).detach()

    def features(o):
        with torch.no_grad():
            pc = o[:, ac.pc_start_idx:].reshape(-1, *ac.pc_shape)
            f = ac.backbone(pc).reshape(-1, ac.pc_emb_dim)
            x = torch.cat([o[:, :ac.pc_start_idx], f], 1)
            m0 = ac.actor_mean(x)
            if args.mu_input:
                x = torch.cat([x, log_mu], 1)
            return x, m0

    def mlp(i, o):
        return nn.Sequential(nn.LayerNorm(i), nn.Linear(i, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, o))

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.delta = mlp(feat_dim, act_dim)
            nn.init.zeros_(self.delta[-1].weight)
            nn.init.zeros_(self.delta[-1].bias)
            self.rho = nn.Parameter(torch.zeros(act_dim))
            if args.value_init == "official":
                if args.mu_input or getattr(ac, "asymmetric", False) or feat_dim != int(ac.num_state_based_obs):
                    raise ValueError("--value-init official needs the critic's own (symmetric) input")
                import copy
                self.value = copy.deepcopy(ac.critic)  # the checkpoint's critic, trained on the same binary reward
                for q_ in self.value.parameters():
                    q_.requires_grad_(True)
            else:
                self.value = mlp(feat_dim, 1)         # ppo baseline
            self.q = mlp(feat_dim + act_dim, 1)       # sac critic

        def dist(self, x):
            return self.delta(x), self.rho.expand(x.shape[0], -1)

    head = Head().to(dev)
    if args.init_head:
        head.load_state_dict(torch.load(str(require_e(args.init_head)), map_location=dev))
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    if args.init_optimizer:
        opt.load_state_dict(torch.load(str(require_e(args.init_optimizer)), map_location=dev))
        for g_ in opt.param_groups:
            g_["lr"] = args.lr
    lo, hi = env.reset_position_range[:, 0], env.reset_position_range[:, 1]
    gen = torch.Generator(device="cpu").manual_seed(args.seed * 7919 + 1)

    def sample_poses(n):
        pos = lo.cpu() + (hi - lo).cpu() * torch.rand(n, 3, generator=gen)
        pos[:, 2] += float(env.table_heights[0])
        axis = torch.randn(n, 3, generator=gen)
        axis = axis / axis.norm(dim=1, keepdim=True)
        angle = (torch.rand(n, generator=gen) * 2 - 1) * math.pi
        return torch.cat([pos, quat_from_angle_axis(angle, axis)], 1).to(dev)

    def logp(e, d, rho):
        return (-0.5 * ((e - d) / torch.exp(rho)) ** 2 - rho).sum(-1)

    def rollout(paired):
        poses = sample_poses(G).repeat_interleave(K, 0) if paired else sample_poses(num_envs)
        obs = env.reset_idx(ids, object_init_pose=poses)["obs"].clone()
        if args.pcl_noise > 0:
            npc = obs.shape[1] - ac.pc_start_idx
            zz = torch.randn(G if paired else num_envs, npc, generator=gen)
            if paired:
                zz = zz.repeat_interleave(K, 0)
            obs[:, ac.pc_start_idx:] += args.pcl_noise * zz.to(dev)
        x, m0 = features(obs)
        with torch.no_grad():
            d, rho = head.dist(x)
            z = torch.randn((num_envs, act_dim), generator=gen).to(dev)
            e = d + torch.exp(rho) * z
            if not train and args.deterministic:
                e = d.clone()
            env.generate_reaching_plan_idx(ids, actions=torch.tanh(m0 + sigma0 * e))
            for t in range(env.max_episode_length):
                env.step(env.compute_reference_actions())
                if t == env.max_episode_length - 2:
                    r = env.successes.clone().float()
                    r = torch.where(env.has_hit_table.bool(), torch.zeros_like(r), r)
                    break
        return obs, x, e, d, rho, r

    (output / "manifest.json").write_text(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "args": vars(args),
        "overrides": overrides, "groups": G, "group_size": K, "planner_sha256": digest(init), "script_sha256": digest(Path(__file__))}, indent=2))
    t0 = time.time()
    if not train:
        episodes = []
        for ep_i in range(args.episodes):
            obs, x, e, d, rho, r = rollout(paired=False)
            dd = output / ("episode_%03d" % ep_i)
            dd.mkdir()
            np.savez_compressed(dd / "initial_state.npz", policy_observation=obs[:, :14].cpu().numpy())
            episodes.append({"episode": ep_i, "official_ppo_print_successes": r.cpu().tolist(), "object_assets": list(rows), "mu": mu_env})
            print(json.dumps({"episode": ep_i, "success": float(r.mean())}), flush=True)
        S = np.array([q["official_ppo_print_successes"] for q in episodes])
        totals = {"success_rate": float(S.mean()), "episodes": int(S.size)}
        (output / "result.json").write_text(json.dumps({"status": "completed", "totals": totals, "evaluated_environment_episodes": int(S.size), "episodes": episodes}))
        print("PLUGIN_EVAL_DONE", json.dumps(totals), flush=True)
        env.gym.destroy_sim(env.sim)
        return
    def value_diag(r, v):
        """value baseline v(s) vs the K-sample Monte-Carlo mean of the same (paired) state"""
        rg, vg = r.view(G, K), v.view(G, K).mean(1)
        mg = rg.mean(1)
        mixed = (rg.max(1).values - rg.min(1).values) > 0
        ac_, ag_ = (r - v).view(G, K), rg - mg.unsqueeze(1)

        def corr(a, b):
            a, b = a - a.mean(), b - b.mean()
            den = float(a.norm() * b.norm())
            return float((a * b).sum()) / den if den > 0 else float("nan")
        out = {"mixed_groups": float(mixed.float().mean()), "all_zero_groups": float((rg.max(1).values == 0).float().mean()),
               "v_mean": float(vg.mean()), "mc_mean": float(mg.mean()), "v_bias": float((vg - mg).mean()),
               "v_mse": float(((vg - mg) ** 2).mean()), "v_mc_corr": corr(vg, mg),
               "v_ev": float(1 - (mg - vg).var() / mg.var()) if float(mg.var()) > 0 else float("nan"),
               "adv_corr": corr(ac_.reshape(-1), ag_.reshape(-1))}
        if bool(mixed.any()):
            a1, a2 = ac_[mixed].reshape(-1), ag_[mixed].reshape(-1)
            out["adv_corr_mixed"] = corr(a1, a2)
            out["adv_sign_agree_mixed"] = float(((a1 > 0) == (a2 > 0)).float().mean())
        return out

    if args.diag_iters > 0:
        rows_ = []
        for it in range(args.diag_iters):
            obs, x, e, d_old, rho_old, r = rollout(paired=True)
            with torch.no_grad():
                v = head.value(x).squeeze(-1)
            rows_.append({"it": it, "success": float(r.mean()), **value_diag(r, v)})
            print(json.dumps(rows_[-1]), flush=True)
        keys = [k for k in rows_[0] if k != "it"]
        summ = {k: float(np.nanmean([q.get(k, float("nan")) for q in rows_])) for k in keys}
        (output / "diag.json").write_text(json.dumps({"summary": summ, "iterations": rows_}, indent=1))
        (output / "result.json").write_text(json.dumps({"status": "completed", "totals": summ}))
        print("PLUGIN_DIAG_DONE", json.dumps(summ), flush=True)
        env.gym.destroy_sim(env.sim)
        return
    buf = None
    if args.algo == "sac" and args.init_buffer:
        buf = torch.load(str(require_e(args.init_buffer)), map_location=dev)
    curve = []
    for it in range(args.iterations):
        obs, x, e, d_old, rho_old, r = rollout(paired=(args.algo == "grpo" and not args.unpaired) or (args.algo == "ppo" and args.ppo_paired))
        lp_old = logp(e, d_old, rho_old).detach()
        stats = {}
        if args.algo in ("grpo", "ppo"):
            if args.algo == "grpo":
                rg = r.view(G, K)
                if args.no_std_norm:
                    adv = (rg - rg.mean(1, keepdim=True)).reshape(-1)
                else:
                    adv = ((rg - rg.mean(1, keepdim=True)) / (rg.std(1, keepdim=True) + 1e-6)).reshape(-1)
            else:
                with torch.no_grad():
                    v = head.value(x).squeeze(-1)
                vstats = {"v_mean": float(v.mean()),
                          "v_ev_batch": float(1 - (r - v).var() / r.var()) if float(r.var()) > 0 else float("nan")}
                if args.ppo_paired:
                    vstats.update({"diag_" + k: val for k, val in value_diag(r, v).items()})
                adv = r - v
                adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            order = torch.randperm(num_envs, device=dev)
            mb = num_envs // args.minibatches
            cf, kl = [], []
            for _ in range(args.epochs):
                for k in range(args.minibatches):
                    b = order[k * mb:(k + 1) * mb]
                    d, rho = head.dist(x[b])
                    logr = (logp(e[b], d, rho) - lp_old[b]).clamp(-20, 20)
                    ratio = torch.exp(logr)
                    loss = -torch.min(ratio * adv[b], torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv[b]).mean()
                    if args.algo == "ppo":
                        loss = loss + 0.5 * ((head.value(x[b]).squeeze(-1) - r[b]) ** 2).mean()
                    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
                    cf.append(float(((ratio - 1).abs() > args.clip).float().mean()))
                    kl.append(float((ratio - 1 - logr).mean()))
            stats = {"clipfrac": float(np.mean(cf)), "approx_kl": float(np.mean(kl))}
            if args.algo == "ppo":
                stats.update(vstats)
        else:
            new = torch.cat([x, e, r.unsqueeze(-1)], 1).detach()
            buf = new if buf is None else torch.cat([buf, new], 0)
            ql, al = [], []
            for _ in range(args.sac_updates):
                b = buf[torch.randint(0, buf.shape[0], (args.sac_batch,), device=dev)]
                bx, be, br = b[:, :feat_dim], b[:, feat_dim:feat_dim + act_dim], b[:, -1]
                qloss = ((head.q(torch.cat([bx, be], 1)).squeeze(-1) - br) ** 2).mean()
                d, rho = head.dist(bx)
                en = d + torch.exp(rho) * torch.randn_like(d)
                aloss = (args.sac_alpha * logp(en, d, rho) - head.q(torch.cat([bx, en], 1)).squeeze(-1)).mean()
                opt.zero_grad(); (qloss + aloss).backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
                ql.append(float(qloss)); al.append(float(aloss))
            stats = {"q_loss": float(np.mean(ql)), "actor_loss": float(np.mean(al)), "buffer": int(buf.shape[0])}
        with torch.no_grad():
            dn, rn = head.dist(x)
        curve.append({"it": it, "success": float(r.mean()), "delta_norm": float(dn.norm(dim=1).mean()),
                      "noise_scale": float(torch.exp(rn).mean()), "seconds": time.time() - t0, **stats})
        print(json.dumps(curve[-1]), flush=True)
    torch.save(head.state_dict(), output / "head.pt")
    torch.save(opt.state_dict(), output / "optimizer.pt")
    if buf is not None:
        torch.save(buf, output / "buffer.pt")
    (output / "curve.json").write_text(json.dumps(curve, indent=1))
    (output / "train_log.json").write_text(json.dumps({"seconds": time.time() - t0, "episodes": args.iterations * num_envs,
                                                       "head_sha256": digest(output / "head.pt")}, indent=2))
    print("PLUGIN_DONE", output, flush=True)
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
