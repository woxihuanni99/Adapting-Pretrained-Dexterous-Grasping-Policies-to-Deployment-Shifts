"""Route 2: sample-then-verify test-time selection for the one-step DemoGrasp planner (frozen or post-trained).

--mode collect : paired groups of K candidates per state (same object, pose and friction inside a group; candidate 0 is
                 the planner mean, candidates 1..K-1 = mean + std * z). All candidates are executed. Saves the planner's
                 actor input x, mean m, candidate u, success r, friction mu, object id and group id (rollouts.npz) and
                 logs mean success, mean-candidate success and pass@K (the oracle best-of-K ceiling) per friction.
                 Object list must be sorted with every object repeated K times consecutively.
--mode select  : the official deterministic evaluation of research_grpo_fm.py (--mode eval --deterministic): identical
                 environment set-up, pose stream and global RNG use, so runs are paired episode-by-episode with fmdet_*.
                 Instead of tanh(mean) it executes the best of K candidates under a learned verifier (research_vnet.py).
                 --select mean reproduces fmdet; --select random picks a random candidate (control).
Friction list --mus: floats or 'nom' (asset default) per group, cycled with --mu-offset.
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
sys.path.insert(0, str(ROOT))


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("collect", "select"), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--init-checkpoint", default=str(ROOT / "ckpt/shadow.pt"))
    p.add_argument("--object-list", required=True)
    p.add_argument("--group-size", type=int, default=8, help="collect: candidates executed per state; select: candidates scored")
    p.add_argument("--mus", default="0.1,0.2,0.3", help="comma list of floats or 'nom'; 'none' = all asset default")
    p.add_argument("--mu-offset", type=int, default=0)
    p.add_argument("--iterations", type=int, default=5, help="collect: rollouts")
    p.add_argument("--episodes", type=int, default=4, help="select: episodes per env")
    p.add_argument("--std-floor", type=float, default=0.05)
    p.add_argument("--cand-scale", type=float, default=1.0, help="candidate std = scale * max(policy std, floor)")
    p.add_argument("--verifier", default=None)
    p.add_argument("--select", choices=("verifier", "random", "mean"), default="verifier")
    p.add_argument("--beta", type=float, default=0.0, help="pessimism: score = mean - beta * ensemble std")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--disable-gpu-jit-fusion", action="store_true")
    args = p.parse_args()
    if args.mus == "none":
        mus = None
    else:
        mus = [None if x.strip() == "nom" else float(x) for x in args.mus.split(",")]
    output = require_e(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cache = require_e(ROOT.parents[1] / "cache" / "shadow_probe")
    for key, sub in {"TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp", "XDG_CACHE_HOME": "xdg",
                     "TORCH_EXTENSIONS_DIR": "torch_extensions", "CUDA_CACHE_PATH": "cuda"}.items():
        (cache / sub).mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(cache / sub)
    os.chdir(str(ROOT))
    from isaacgym import gymapi, gymtorch  # noqa: F401
    from isaacgym.torch_utils import quat_from_angle_axis
    import isaacgymenvs
    import numpy as np
    import torch
    import yaml
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from isaacgymenvs.utils.utils import set_seed
    from run_rl_grasp import build_runner
    collect = args.mode == "collect"
    K = args.group_size if collect else 1
    rows = yaml.safe_load((ROOT / "assets" / args.object_list).read_text())
    num_envs = len(rows)
    if collect and (num_envs % K or any(rows[i] != rows[i - i % K] for i in range(num_envs)) or sorted(rows) != rows):
        raise ValueError("object list must be sorted with each object repeated K times consecutively")
    if not collect and args.select == "verifier" and not args.verifier:
        raise ValueError("--select verifier needs --verifier")
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
                mu = None if mus is None else mus[(i // K + args.mu_offset) % len(mus)]
                if mu is None:
                    mu_env.append(-1.0)  # asset default ('nom')
                    continue
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
    ids = torch.arange(num_envs, device=dev)
    env.reset_idx(ids)
    runner = build_runner(cfg, env)
    ac = runner.actor_critic
    ac.eval()
    for q in ac.parameters():
        q.requires_grad_(False)
    verifier, vmeta = None, None
    if not collect and args.select == "verifier":
        from research_vnet import load_verifier
        with torch.random.fork_rng(devices=[dev] if str(dev).startswith("cuda") else []):
            verifier, vmeta = load_verifier(require_e(args.verifier), dev)
    lo, hi = env.reset_position_range[:, 0], env.reset_position_range[:, 1]
    gen = torch.Generator(device="cpu").manual_seed(args.seed * 7919 + 1)
    gen2 = torch.Generator(device="cpu").manual_seed(args.seed * 7919 + 2)

    def sample_poses(n):
        pos = lo.cpu() + (hi - lo).cpu() * torch.rand(n, 3, generator=gen)
        pos[:, 2] += float(env.table_heights[0])
        axis = torch.randn(n, 3, generator=gen)
        axis = axis / axis.norm(dim=1, keepdim=True)
        angle = (torch.rand(n, generator=gen) * 2 - 1) * math.pi
        return torch.cat([pos, quat_from_angle_axis(angle, axis)], 1).to(dev)

    def actor_input(o):
        pc = o[:, ac.pc_start_idx:].reshape(-1, *ac.pc_shape)
        f = ac.backbone(pc).reshape(-1, ac.pc_emb_dim)
        return torch.cat([o[:, :ac.pc_start_idx], f], 1)

    def execute(u):
        env.generate_reaching_plan_idx(ids, actions=torch.tanh(u))
        for t in range(env.max_episode_length):
            env.step(env.compute_reference_actions())
            if t == env.max_episode_length - 2:
                r = env.successes.clone().float()
                return torch.where(env.has_hit_table.bool(), torch.zeros_like(r), r)

    base_std = torch.exp(ac.log_std).detach().clamp(min=args.std_floor)
    cand_std = args.cand_scale * base_std
    manifest = {"utc": datetime.now(timezone.utc).isoformat(), "args": vars(args), "overrides": overrides,
                "groups": G, "group_size": K, "init_checkpoint_sha256": digest(init), "script_sha256": digest(Path(__file__)),
                "policy_std": torch.exp(ac.log_std).tolist(), "cand_std": cand_std.tolist(), "verifier_meta": vmeta}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    t0 = time.time()
    mu_t = torch.tensor(mu_env, device=dev)
    if collect:
        X, M, U, R, MU, OB, GR = [], [], [], [], [], [], []
        obj_ids = {n: i for i, n in enumerate(sorted(set(rows)))}
        ob = torch.tensor([obj_ids[n] for n in rows])
        for it in range(args.iterations):
            poses = sample_poses(G).repeat_interleave(K, 0)
            obs = env.reset_idx(ids, object_init_pose=poses)["obs"].clone()
            with torch.no_grad():
                x = actor_input(obs)
                m = ac.actor_mean(x)
                z = torch.randn(m.shape, generator=gen).to(dev)
                z[0::K] = 0.0  # candidate 0 of every group = planner mean
                u = m + cand_std.expand_as(m) * z
                r = execute(u)
            rg = r.view(G, K)
            rec = {"it": it, "success": float(r.mean()), "cand0_success": float(rg[:, 0].mean()),
                   "pass_at_k": float(rg.max(1).values.mean()), "informative": float((rg.max(1).values > rg.min(1).values).float().mean()),
                   "seconds": time.time() - t0}
            print(json.dumps(rec), flush=True)
            X.append(x.half().cpu()); M.append(m.cpu()); U.append(u.cpu()); R.append(r.cpu()); MU.append(mu_t.cpu())
            OB.append(ob.clone()); GR.append(torch.arange(num_envs) // K + it * G)
        cat = lambda L: torch.cat(L).numpy()
        Rn, MUn = cat(R), cat(MU)
        np.savez_compressed(output / "rollouts.npz", x=cat(X), m=cat(M), u=cat(U), r=Rn, mu=MUn, obj=cat(OB), group=cat(GR),
                            obj_names=np.array(sorted(set(rows))), group_size=K)
        Rg = Rn.reshape(-1, K)
        mug = MUn.reshape(-1, K)[:, 0]
        per_mu = {}
        for v in sorted(set(mug.tolist())):
            s = Rg[mug == v]
            per_mu["nom" if v < 0 else "%.2f" % v] = {"success": float(s.mean()), "cand0_success": float(s[:, 0].mean()),
                                                      "pass_at_k": float(s.max(1).mean()), "groups": int(len(s))}
        totals = {"success_rate": float(Rn.mean()), "cand0_success": float(Rg[:, 0].mean()), "pass_at_k": float(Rg.max(1).mean()),
                  "episodes": int(Rn.size), "per_mu": per_mu}
    else:
        episodes = []
        for e in range(args.episodes):
            poses = sample_poses(num_envs)
            obs = env.reset_idx(ids, object_init_pose=poses)["obs"].clone()
            with torch.no_grad():
                x = actor_input(obs)
                m = ac.actor_mean(x)
                _ = torch.randn(m.shape, generator=gen)  # keep the fmdet pose/noise stream aligned
                Kc = args.group_size
                zc = torch.randn(Kc, *m.shape, generator=gen2).to(dev)
                zc[0] = 0.0
                cands = m.unsqueeze(0) + cand_std.view(1, 1, -1) * zc  # (Kc, B, A)
                if args.select == "mean":
                    pick = torch.zeros(num_envs, dtype=torch.long, device=dev)
                    sc = None
                elif args.select == "random":
                    pick = torch.randint(0, Kc, (num_envs,), generator=gen2).to(dev)
                    sc = None
                else:
                    sc = verifier.score(x.unsqueeze(0).expand(Kc, -1, -1).reshape(Kc * num_envs, -1),
                                        m.unsqueeze(0).expand(Kc, -1, -1).reshape(Kc * num_envs, -1),
                                        cands.reshape(Kc * num_envs, -1), beta=args.beta).view(Kc, num_envs)
                    pick = sc.argmax(0)
                u = cands[pick, torch.arange(num_envs, device=dev)]
                r = execute(u)
            episodes.append({"episode": e, "official_ppo_print_successes": r.cpu().tolist(), "object_assets": list(rows),
                             "mu": mu_env, "pick": pick.cpu().tolist(),
                             "score_pick": None if sc is None else sc.max(0).values.cpu().tolist(),
                             "score_mean_cand": None if sc is None else sc[0].cpu().tolist()})
            print(json.dumps({"episode": e, "success": float(r.mean()), "picked_mean_frac": float((pick == 0).float().mean())}), flush=True)
        S = np.array([ep["official_ppo_print_successes"] for ep in episodes])
        P = np.array([ep["pick"] for ep in episodes])
        totals = {"success_rate": float(S.mean()), "episodes": int(S.size), "picked_mean_frac": float((P == 0).mean())}
        (output / "result_episodes.json").write_text(json.dumps({"episodes": episodes}))
        res = {"status": "completed", "totals": totals, "evaluated_environment_episodes": int(S.size), "episodes": episodes}
        (output / "result.json").write_text(json.dumps(res))
        (output / "timing.json").write_text(json.dumps({"wall_seconds": time.time() - t0}))
        print("VSEL_DONE", json.dumps(totals), flush=True)
        env.gym.destroy_sim(env.sim)
        return
    (output / "result.json").write_text(json.dumps({"status": "completed", "totals": totals}))
    (output / "timing.json").write_text(json.dumps({"wall_seconds": time.time() - t0}))
    print("VCOLLECT_DONE", json.dumps(totals), flush=True)
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
