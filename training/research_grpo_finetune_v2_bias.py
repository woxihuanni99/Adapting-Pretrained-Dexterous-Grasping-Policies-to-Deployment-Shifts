"""BIAS-ONLY / CONSTANT-SHIFT VARIANT (2026-10-08; copy of research_grpo_finetune_v2.py, original unchanged).
With --bias-coord the whole planner is frozen and only a 24-dim output shift is trained:
  raw  : m(s) = m0(s) + b              (b trained directly; equals training the output bias of actor_mean)
  sigma: m(s) = m0(s) + sigma0 * c     (c trained in sigma0-scaled coordinates; sigma0 = pretrained exp(log_std))
Sampling and scoring keep the full-weight GRPO scale max(sigma0, --std-floor). The cumulative shift is carried between phases
in shift.pt (--init-shift); --save-full folds it into actor_mean[-1].bias and writes final.pt for the unchanged eval script.
"""
"""DemoGrasp (A) + group-relative policy optimization (B) for low-friction adaptation. (v2)

v2 = v1 + --std-floor: exploration AND scoring use sigma_b = max(sigma, floor) per action dim.
Why: shadow.pt has near-zero std in some (wrist) dims; with sigma~1e-3 the score gradient eps/sigma is
huge and noisy, v1 clip fractions were 0.7-0.9 and grad-norm clipping starved the informative dims.
Only the mean (and backbone) is learned; the deployed policy keeps shadow.pt's own std.

B follows the GRPO family used for RL post-training of pretrained robot policies with binary
outcome rewards (SimpleVLA-RL, ICLR 2026; off-policy/on-policy GRPO analysis, ICLR 2026):
no learned value function; advantages are normalized within a group of rollouts that share
the same task instance. Our adaptation to demonstration-editing grasp planners:
  * STATE-PAIRED GROUPS: the K members of a group share the same object, the same friction and
    the SAME initial object pose (passed via Grasp.reset_idx(object_init_pose=...)), so outcome
    differences inside a group come only from the sampled edit (and simulator sensitivity),
    not from pose or object difficulty. (Diagnostics: 85% of outcome variance is within
    (object, friction); a learned value/critic explains only ~2% of it.)
  * One planner decision per episode (terminal reward = official success at step max-2, zero if
    the table is hit, identical to DemoGrasp's own PPO), clipped surrogate on pre-tanh Gaussian
    log-ratios (tanh Jacobian cancels), fixed exploration std from shadow.pt unless --learn-std.
One process = one phase: G objects, each listed K times consecutively in the object list, so env
i uses object i // K. Friction mu = MUS[g % len(MUS)] on all robot and object shapes of group g.
Writes OUTPUT/{manifest.json, friction_readback.json, curve.json, final.pt, train_log.json}.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--init-checkpoint", default=str(ROOT / "ckpt/shadow.pt"))
    p.add_argument("--object-list", required=True, help="yaml with each object repeated K times")
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--mus", default="0.1,0.2,0.3")
    p.add_argument("--iterations", type=int, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--minibatches", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--adv", choices=("grpo", "rloo", "drgrpo"), default="grpo",
                   help="grpo: (r-mean)/std; rloo: r - leave-one-out mean; drgrpo: r-mean (no std)")
    p.add_argument("--unpaired", action="store_true", help="ablation: independent random poses inside a group")
    p.add_argument("--learn-std", action="store_true")
    p.add_argument("--std-floor", type=float, default=0.0, help="per-dim lower bound on behavior/scoring std (pre-tanh)")
    p.add_argument("--pcl-noise", type=float, default=0.0,
                   help="Gaussian noise (m) on the observed object point cloud obs[:, pc_start:], shared inside a group; physics unchanged")
    p.add_argument("--init-optimizer", default=None, help="optimizer.pt from the previous phase (keeps Adam moments across phases)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--disable-gpu-jit-fusion", action="store_true")
    p.add_argument("--bias-coord", choices=("raw", "sigma"), required=True, help="train only an output shift in raw or sigma0 coordinates")
    p.add_argument("--init-shift", default=None, help="shift.pt of the previous phase (cumulative output shift)")
    p.add_argument("--save-full", action="store_true", help="also write final.pt with the shift folded into actor_mean[-1].bias")
    p.add_argument("--selfcheck", action="store_true", help="run checks after the first update and exit")
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
    from isaacgym import gymapi  # noqa: F401
    from isaacgym.torch_utils import quat_from_angle_axis
    import isaacgymenvs
    import numpy as np
    import torch
    import yaml
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from isaacgymenvs.utils.utils import set_seed
    from run_rl_grasp import build_runner
    K = args.group_size
    rows = yaml.safe_load((ROOT / "assets" / args.object_list).read_text())
    num_envs = len(rows)
    if num_envs % K or any(rows[i] != rows[i - i % K] for i in range(num_envs)) or sorted(rows) != rows:
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
                 "task.env.enableRobotTableCollision=False"]
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
    readback = []
    from isaacgymenvs.tasks import isaacgym_task_map
    base_cls = isaacgym_task_map[cfg.task_name]

    class GroupFrictionGrasp(base_cls):
        def create_sim(self):
            result = super().create_sim()
            if mus is None:  # asset-default friction
                return result
            for i, env_ptr in enumerate(self.envs):
                mu = mus[(i // K) % len(mus)]
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
                readback.append({"env": i, "group": i // K, "mu": mu, "object": self.object_fns[i % len(self.object_fns)]})
            return result

    isaacgym_task_map[cfg.task_name] = GroupFrictionGrasp
    try:
        env = isaacgymenvs.make(cfg.seed, cfg.task_name, cfg.task.env.numEnvs, cfg.sim_device, cfg.rl_device,
                                cfg.graphics_device_id, cfg.headless, cfg.multi_gpu, cfg.capture_video,
                                cfg.force_render, cfg)
    finally:
        isaacgym_task_map[cfg.task_name] = base_cls
    if any(env.object_fns[i] != env.object_fns[i - i % K] for i in range(num_envs)):
        raise RuntimeError("env->object mapping does not follow groups")
    (output / "friction_readback.json").write_text(json.dumps(readback, indent=1))
    (output / "manifest.json").write_text(json.dumps({
        "utc": datetime.now(timezone.utc).isoformat(), "args": vars(args), "overrides": overrides,
        "groups": G, "group_size": K, "budget_episodes": args.iterations * num_envs,
        "init_checkpoint_sha256": digest(init), "script_sha256": digest(Path(__file__))}, indent=2))
    ids = torch.arange(num_envs, device=env.device)
    env.reset_idx(ids)
    runner = build_runner(cfg, env)  # test mode: loads shadow.pt weights
    ac = runner.actor_critic
    ac.train()
    for q in ac.parameters():
        q.requires_grad_(False)
    assert not args.learn_std, "bias-only variant keeps the sampling scale fixed"
    sigma0 = torch.exp(ac.log_std).detach().clone()
    scale = torch.ones_like(sigma0) if args.bias_coord == "raw" else sigma0.clone()
    shift0 = torch.zeros_like(sigma0)
    if args.init_shift:
        rec = torch.load(str(require_e(args.init_shift)), map_location=env.device)
        assert rec["coord"] == args.bias_coord, "coordinate changed between phases"
        assert torch.allclose(rec["sigma0"].to(env.device), sigma0), "sigma0 differs from the previous phase"
        shift0 = rec["shift"].to(env.device).clone()
    vshift = torch.zeros_like(sigma0, requires_grad=True)
    params = [vshift]
    frozen_ref = {n: q.detach().clone() for n, q in ac.named_parameters()}
    opt = torch.optim.Adam(params, lr=args.lr)
    if args.init_optimizer:
        opt.load_state_dict(torch.load(str(require_e(args.init_optimizer)), map_location=env.device))
        for group in opt.param_groups:
            group["lr"] = args.lr
    dev = env.device
    lo, hi = env.reset_position_range[:, 0], env.reset_position_range[:, 1]
    gen = torch.Generator(device="cpu").manual_seed(args.seed * 7919 + 1)

    def sample_poses(n):
        pos = lo.cpu() + (hi - lo).cpu() * torch.rand(n, 3, generator=gen)
        pos[:, 2] += env.table_heights[:n].cpu() if env.table_heights.shape[0] >= n else float(env.table_heights[0])
        axis = torch.randn(n, 3, generator=gen)
        axis = axis / axis.norm(dim=1, keepdim=True)
        angle = (torch.rand(n, generator=gen) * 2 - 1) * math.pi
        quat = quat_from_angle_axis(angle, axis)
        return torch.cat([pos, quat], 1).to(dev)

    def pre_mean(o):
        pc = o[:, ac.pc_start_idx:].reshape(-1, *ac.pc_shape)
        f = ac.backbone(pc).reshape(-1, ac.pc_emb_dim)
        return ac.actor_mean(torch.cat([o[:, :ac.pc_start_idx], f], 1)) + shift0 + scale * vshift

    def logp(u, m, std):
        return (-0.5 * ((u - m) / std) ** 2 - torch.log(std)).sum(-1)

    (output / "policy_std.json").write_text(json.dumps({"std": torch.exp(ac.log_std).tolist(), "floor": args.std_floor}))
    curve, t0 = [], time.time()
    for it in range(args.iterations):
        if args.unpaired:
            poses = sample_poses(num_envs)
        else:
            poses = sample_poses(G).repeat_interleave(K, 0)
        obs = env.reset_idx(ids, object_init_pose=poses)["obs"].clone()
        if args.pcl_noise > 0:
            npc = obs.shape[1] - ac.pc_start_idx
            z = torch.randn(num_envs if args.unpaired else G, npc, generator=gen)
            if not args.unpaired:
                z = z.repeat_interleave(K, 0)
            obs[:, ac.pc_start_idx:] += args.pcl_noise * z.to(dev)
        with torch.no_grad():
            m_old = pre_mean(obs)
            std = torch.exp(ac.log_std).detach().clamp(min=args.std_floor).expand_as(m_old)
            u = m_old + std * torch.randn_like(m_old)
            env.generate_reaching_plan_idx(ids, actions=torch.tanh(u))
            for t in range(env.max_episode_length):
                env.step(env.compute_reference_actions())
                if t == env.max_episode_length - 2:
                    r = env.successes.clone().float()
                    r = torch.where(env.has_hit_table.bool(), torch.zeros_like(r), r)
                    break
            rg = r.view(G, K)
            mean_g = rg.mean(1, keepdim=True)
            if args.adv == "grpo":
                adv = (rg - mean_g) / (rg.std(1, keepdim=True) + 1e-6)
            elif args.adv == "rloo":
                adv = rg - (rg.sum(1, keepdim=True) - rg) / (K - 1)
            else:
                adv = rg - mean_g
            adv = adv.reshape(-1)
            informative = float(((rg.max(1).values - rg.min(1).values) > 0).float().mean())
        order = torch.randperm(num_envs, device=dev)
        mb = num_envs // args.minibatches
        clipfrac, kl = [], []
        for ep in range(args.epochs):
            for k in range(args.minibatches):
                b = order[k * mb:(k + 1) * mb]
                m = pre_mean(obs[b])
                s = torch.exp(ac.log_std).clamp(min=args.std_floor).expand_as(m)
                logr = (logp(u[b], m, s) - logp(u[b], m_old[b], std[b])).clamp(-20, 20)
                ratio = torch.exp(logr)
                s1, s2 = ratio * adv[b], torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv[b]
                loss = -torch.min(s1, s2).mean()
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                clipfrac.append(float(((ratio - 1).abs() > args.clip).float().mean()))
                kl.append(float((ratio - 1 - logr).mean()))
        curve.append({"it": it, "success": float(r.mean()), "informative_groups": informative,
                      "clipfrac": float(np.mean(clipfrac)), "approx_kl": float(np.mean(kl)), "seconds": time.time() - t0})
        print(json.dumps(curve[-1]), flush=True)
        if args.selfcheck:
            chk = {"optimizer_tensors": sum(len(g["params"]) for g in opt.param_groups),
                   "optimizer_shapes": [list(q.shape) for g in opt.param_groups for q in g["params"]],
                   "planner_params_requiring_grad": [n for n, q in ac.named_parameters() if q.requires_grad],
                   "planner_max_change": max(float((q.detach() - frozen_ref[n]).abs().max()) for n, q in ac.named_parameters()),
                   "scale_minus_target": float((scale - (torch.ones_like(sigma0) if args.bias_coord == "raw" else sigma0)).abs().max()),
                   "sampling_std_minus_floor": float((torch.exp(ac.log_std).clamp(min=args.std_floor) - sigma0.clamp(min=args.std_floor)).abs().max()),
                   "vshift_norm_after_first_iteration": float(vshift.detach().norm())}
            with torch.no_grad():
                m_shift = pre_mean(obs[:64])
                bias_backup = ac.actor_mean[-1].bias.detach().clone()
                ac.actor_mean[-1].bias += shift0 + scale * vshift
                pc = obs[:64, ac.pc_start_idx:].reshape(-1, *ac.pc_shape)
                f_ = ac.backbone(pc).reshape(-1, ac.pc_emb_dim)
                m_fold = ac.actor_mean(torch.cat([obs[:64, :ac.pc_start_idx], f_], 1))
                ac.actor_mean[-1].bias.copy_(bias_backup)
            chk["fold_max_abs_diff"] = float((m_shift - m_fold).abs().max())
            (output / "selfcheck.json").write_text(json.dumps(chk, indent=1))
            print("SELFCHECK", json.dumps(chk), flush=True)
            env.gym.destroy_sim(env.sim)
            return
    shift = (shift0 + scale * vshift).detach()
    torch.save({"coord": args.bias_coord, "shift": shift.cpu(), "sigma0": sigma0.cpu(), "phase_increment": (scale * vshift).detach().cpu(),
                "lr": args.lr}, output / "shift.pt")
    torch.save(opt.state_dict(), output / "optimizer.pt")
    assert max(float((q.detach() - frozen_ref[n]).abs().max()) for n, q in ac.named_parameters()) == 0.0, "planner changed"
    if args.save_full:
        with torch.no_grad():
            ac.actor_mean[-1].bias += shift
        torch.save(ac.state_dict(), output / "final.pt")
    (output / "curve.json").write_text(json.dumps(curve, indent=1))
    (output / "train_log.json").write_text(json.dumps({"seconds": time.time() - t0, "shift_sha256": digest(output / "shift.pt"),
                                                       "final_sha256": digest(output / "final.pt") if args.save_full else None,
                                                       "shift": shift.cpu().tolist(), "coord": args.bias_coord,
                                                       "episodes": args.iterations * num_envs}, indent=2))
    print("GRPO_DONE", output, flush=True)
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
