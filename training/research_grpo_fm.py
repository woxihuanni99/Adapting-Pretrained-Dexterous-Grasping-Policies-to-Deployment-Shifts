"""DemoGrasp (A) + state-paired GRPO (B) + friction-risk margin reward (C). Train and paired evaluation.

C follows the idea of FirmGrasp (arXiv 2607.25049, 2026): judge a grasp by its force-closure margin under an
ADVERSE friction tail instead of the nominal coefficient. Here it is measured on the simulated grasp itself:
during the lift window, the net contact force F_i on each fingertip body is split along the object's outward
surface normal n_i (nearest object point-cloud point, kNN-PCA normals) into f_n = <F_i, n_i>_+ and
f_t = |F_i - <F_i,n_i> n_i|. The grasp's required friction is mu_req = sum f_t / sum f_n (force weighted).
Margin per step = clip(1 - mu_req / (s * mu), -1, 1), with s * mu the CVaR (adverse-tail mean) of a friction
belief uniform on [0.5 mu, mu] (s = 0.625 by default); steps without object contact count as -1.
Episode margin m in [-1, 1] = mean over the lift window. Training reward r = success + lambda * m
(lambda = --margin-weight; 0 gives plain GRPO). All else is research_grpo_finetune_v2.py (std floor,
Adam state across phases, pcl noise). --mode eval runs paired evaluation on any object list (num_envs = len(list),
conditions drawn from a dedicated generator) and logs success, margin, mu_req and contact fraction per episode.
Requires contact collection: task.sim.physx.contact_collection=1 (reporting only, dynamics unchanged).
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
    p.add_argument("--output", required=True)
    p.add_argument("--init-checkpoint", default=str(ROOT / "ckpt/shadow.pt"))
    p.add_argument("--object-list", required=True)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--mus", default="0.1,0.2,0.3", help="friction per group (cycled); 'none' = asset default")
    p.add_argument("--iterations", type=int, default=0)
    p.add_argument("--episodes", type=int, default=4, help="eval: episodes per env")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--adv", choices=("grpo", "rloo", "drgrpo"), default="grpo")
    p.add_argument("--unpaired", action="store_true")
    p.add_argument("--std-floor", type=float, default=0.05)
    p.add_argument("--pcl-noise", type=float, default=0.0)
    p.add_argument("--margin-weight", type=float, default=0.0)
    p.add_argument("--margin-scale", type=float, default=0.625, help="CVaR adverse-tail friction as a fraction of mu")
    p.add_argument("--contact-threshold", type=float, default=0.05, help="N; fingertip counts as in contact above this")
    p.add_argument("--reward-mode", choices=("margin", "hold"), default="margin",
                   help="margin: FirmGrasp-style required-friction margin; hold: lift-window contact (hold) fraction in [-1,1]")
    p.add_argument("--adverse-scale", type=float, default=0.0,
                   help=">0: every action is executed twice in its group, at mu and at scale*mu (adverse friction tail)")
    p.add_argument("--adverse-weight", type=float, default=0.5, help="weight of the adverse-friction success in the action return")
    p.add_argument("--deterministic", action="store_true", help="eval: execute tanh(mean) like DemoGrasp's official test (noise is still drawn so runs stay paired)")
    p.add_argument("--init-optimizer", default=None)
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
    # --- contact forces and object normals for the friction-risk margin ---
    ncf = gymtorch.wrap_tensor(env.gym.acquire_net_contact_force_tensor(env.sim)).view(num_envs, -1, 3)
    tips = env.fingertip_handles
    pcl_obj = env.obj_pcl_buf.clone()  # (B, N, 3) object frame
    cache_n = {}
    nrm = torch.zeros_like(pcl_obj)
    for i in range(num_envs):
        key = rows[i]
        if key not in cache_n:
            cache_n[key] = torch.tensor(with_normals(pcl_obj[i].cpu().numpy(), np), device=dev)
        nrm[i] = cache_n[key]
    lift_start = env.T_ref_start_lifting
    thr = args.contact_threshold

    def margin_step(acc, lift_mask):
        env.gym.refresh_net_contact_force_tensor(env.sim)
        F = ncf[:, tips]  # (B, 5, 3)
        rot = env.object_rot.unsqueeze(1)
        pw = quat_apply(rot.expand(-1, pcl_obj.shape[1], -1).reshape(-1, 4), pcl_obj.reshape(-1, 3)).view_as(pcl_obj) + env.object_pos.unsqueeze(1)
        nw = quat_apply(rot.expand(-1, nrm.shape[1], -1).reshape(-1, 4), nrm.reshape(-1, 3)).view_as(nrm)
        ft = env.fingertip_pos  # (B, 5, 3)
        idx = torch.cdist(ft, pw).argmin(-1)  # (B, 5)
        n = torch.gather(nw, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        fn_signed = (F * n).sum(-1)
        fn = fn_signed.clamp(min=0)
        ftan = (F - fn_signed.unsqueeze(-1) * n).norm(dim=-1)
        active = F.norm(dim=-1) > thr
        sfn = (fn * active).sum(-1)
        sft = (ftan * active).sum(-1)
        contact = sfn > thr
        mu_req = sft / sfn.clamp(min=1e-6)
        m = torch.where(contact, (1 - mu_req / (args.margin_scale * mu_t)).clamp(-1, 1), -torch.ones_like(sfn))
        w = lift_mask.float()
        acc["m"] += m * w
        acc["n"] += w
        acc["c"] += contact.float() * w
        acc["mu_req"] += torch.where(contact, mu_req, torch.zeros_like(mu_req)) * w
        acc["fn"] += sfn * w

    ids = torch.arange(num_envs, device=dev)
    env.reset_idx(ids)
    runner = build_runner(cfg, env)
    ac = runner.actor_critic
    ac.train()
    for q in ac.parameters():
        q.requires_grad_(False)
    params = [q for n_, q in ac.named_parameters() if n_.startswith(("backbone", "actor_mean"))]
    for q in params:
        q.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=args.lr)
    if args.init_optimizer:
        opt.load_state_dict(torch.load(str(require_e(args.init_optimizer)), map_location=dev))
        for group in opt.param_groups:
            group["lr"] = args.lr
    lo, hi = env.reset_position_range[:, 0], env.reset_position_range[:, 1]
    gen = torch.Generator(device="cpu").manual_seed(args.seed * 7919 + 1)

    def sample_poses(n):
        pos = lo.cpu() + (hi - lo).cpu() * torch.rand(n, 3, generator=gen)
        pos[:, 2] += float(env.table_heights[0])
        axis = torch.randn(n, 3, generator=gen)
        axis = axis / axis.norm(dim=1, keepdim=True)
        angle = (torch.rand(n, generator=gen) * 2 - 1) * math.pi
        return torch.cat([pos, quat_from_angle_axis(angle, axis)], 1).to(dev)

    def pre_mean(o):
        pc = o[:, ac.pc_start_idx:].reshape(-1, *ac.pc_shape)
        f = ac.backbone(pc).reshape(-1, ac.pc_emb_dim)
        return ac.actor_mean(torch.cat([o[:, :ac.pc_start_idx], f], 1))

    def logp(u, m, std):
        return (-0.5 * ((u - m) / std) ** 2 - torch.log(std)).sum(-1)

    def rollout(paired):
        poses = sample_poses(G).repeat_interleave(K, 0) if paired else sample_poses(num_envs)
        obs = env.reset_idx(ids, object_init_pose=poses)["obs"].clone()
        if args.pcl_noise > 0:
            npc = obs.shape[1] - ac.pc_start_idx
            z = torch.randn(G if paired else num_envs, npc, generator=gen)
            if paired:
                z = z.repeat_interleave(K, 0)
            obs[:, ac.pc_start_idx:] += args.pcl_noise * z.to(dev)
        acc = {k: torch.zeros(num_envs, device=dev) for k in ("m", "n", "c", "mu_req", "fn")}
        with torch.no_grad():
            m_old = pre_mean(obs)
            std = torch.exp(ac.log_std).detach().clamp(min=args.std_floor if train else 0.0).expand_as(m_old)
            if train and args.adverse_scale > 0:
                z = torch.randn(m_old.shape[0] // 2, m_old.shape[1], generator=gen).repeat_interleave(2, 0)
            else:
                z = torch.randn(m_old.shape, generator=gen)
            u = m_old + std * z.to(dev)
            if not train and args.deterministic:
                u = m_old.clone()
            env.generate_reaching_plan_idx(ids, actions=torch.tanh(u))
            for t in range(env.max_episode_length):
                env.step(env.compute_reference_actions())
                lift = env.progress_buf >= env.reaching_plan_timesteps + lift_start
                margin_step(acc, lift)
                if t == env.max_episode_length - 2:
                    r = env.successes.clone().float()
                    r = torch.where(env.has_hit_table.bool(), torch.zeros_like(r), r)
                    break
        nn_ = acc["n"].clamp(min=1)
        stats = {"margin": torch.where(acc["n"] > 0, acc["m"] / nn_, -torch.ones_like(nn_)),
                 "contact_frac": acc["c"] / nn_, "mu_req": acc["mu_req"] / acc["c"].clamp(min=1),
                 "normal_force": acc["fn"] / nn_, "lift_steps": acc["n"]}
        return obs, m_old, std, u, r, stats

    manifest = {"utc": datetime.now(timezone.utc).isoformat(), "args": vars(args), "overrides": overrides,
                "groups": G, "group_size": K, "init_checkpoint_sha256": digest(init), "script_sha256": digest(Path(__file__)),
                "policy_std": torch.exp(ac.log_std).tolist()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    t0 = time.time()
    if not train:
        episodes = []
        for e in range(args.episodes):
            obs, _, _, _, r, st = rollout(paired=False)
            d = output / ("episode_%03d" % e)
            d.mkdir()
            np.savez_compressed(d / "initial_state.npz", policy_observation=obs[:, :14].cpu().numpy())
            episodes.append({"episode": e, "official_ppo_print_successes": r.cpu().tolist(), "object_assets": list(rows),
                             "mu": mu_env, **{k: v.cpu().tolist() for k, v in st.items()}})
            print(json.dumps({"episode": e, "success": float(r.mean()), "margin_mean": float(st["margin"].mean()),
                              "contact_frac": float(st["contact_frac"].mean())}), flush=True)
        S = np.array([ep["official_ppo_print_successes"] for ep in episodes])
        M = np.array([ep["margin"] for ep in episodes])
        auc = None
        pos, neg = M[S > 0], M[S == 0]
        if len(pos) and len(neg):
            auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
        totals = {"success_rate": float(S.mean()), "episodes": int(S.size), "margin_mean": float(M.mean()), "margin_auc_vs_success": auc}
        (output / "result.json").write_text(json.dumps({"status": "completed", "totals": totals,
                                                       "evaluated_environment_episodes": int(S.size), "episodes": episodes}))
        (output / "timing.json").write_text(json.dumps({"wall_seconds": time.time() - t0}))
        print("FM_EVAL_DONE", json.dumps(totals), flush=True)
        env.gym.destroy_sim(env.sim)
        return
    curve = []
    for it in range(args.iterations):
        obs, m_old, std, u, r, st = rollout(paired=not args.unpaired)
        dense = st["margin"] if args.reward_mode == "margin" else 2 * st["contact_frac"] - 1
        reward = r + args.margin_weight * dense
        with torch.no_grad():
            if args.adverse_scale > 0:
                rp = reward.view(G, K // 2, 2)
                ra = (1 - args.adverse_weight) * rp[..., 0] + args.adverse_weight * rp[..., 1]
                rg = ra.repeat_interleave(2, 1)
            else:
                rg = reward.view(G, K)
            mean_g = rg.mean(1, keepdim=True)
            if args.adv == "grpo":
                adv = (rg - mean_g) / (rg.std(1, keepdim=True) + 1e-6)
            elif args.adv == "rloo":
                adv = rg - (rg.sum(1, keepdim=True) - rg) / (K - 1)
            else:
                adv = rg - mean_g
            adv = adv.reshape(-1)
            informative = float(((rg.max(1).values - rg.min(1).values) > 1e-6).float().mean())
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
        curve.append({"it": it, "success": float(r.mean()), "success_nominal": float(r.view(G, -1)[:, 0::2].mean()) if args.adverse_scale > 0 else float(r.mean()), "margin": float(st["margin"].mean()),
                      "contact_frac": float(st["contact_frac"].mean()), "informative_groups": informative,
                      "clipfrac": float(np.mean(clipfrac)), "approx_kl": float(np.mean(kl)), "seconds": time.time() - t0})
        print(json.dumps(curve[-1]), flush=True)
    torch.save(ac.state_dict(), output / "final.pt")
    torch.save(opt.state_dict(), output / "optimizer.pt")
    (output / "curve.json").write_text(json.dumps(curve, indent=1))
    (output / "train_log.json").write_text(json.dumps({"seconds": time.time() - t0, "final_sha256": digest(output / "final.pt"),
                                                       "episodes": args.iterations * num_envs}, indent=2))
    print("GRPO_FM_DONE", output, flush=True)
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
