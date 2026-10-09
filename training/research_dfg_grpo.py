"""DemoFunGrasp (A2, CVPR 2026) + state-paired GRPO (B): training and paired evaluation.

Same B as workspace/DemoGrasp/research_grpo_finetune.py, ported to DemoFunGrasp's functional-grasp
planner (25-d action = 6 wrist + 18 finger deltas + 1 style scale; the observation carries a sampled
affordance point and a one-hot grasp style). A group of K rollouts shares the object, the friction,
the initial object pose, the initial robot-joint noise, the style label and the affordance point, so
outcome differences inside a group come only from the sampled edit (and simulator sensitivity).

Modes
  train: --iterations N (clipped surrogate on pre-tanh Gaussian log-ratios, group-normalized
         binary success). Writes curve.json, final.pt, optimizer.pt, train_log.json.
  eval : --episodes E. Every condition (pose, joint noise, style, affordance point, action noise) is
         drawn from a dedicated generator seeded by --seed, so different checkpoints see identical
         conditions (paired). Success is read at step max-2 exactly like the official test loop.
         Writes result.json (+ episode_XXX/initial_state.npz) in the format read by
         tools/compare_policies.py, plus style accuracy and fingertip-affordance distance.
Objects and hand come from workspace/DemoGrasp/assets (shadow_hand_simple/right_with_base.urdf is
byte-identical; the union dataset layout is the same). bbox_dict.npy there is created by
tools/make_dfg_bbox.py (same trimesh OBB rule as dataset_processor/generate_object_dataset_pcl.py).
The task/train configuration is taken leaf-by-leaf from the checkpoint's config.json.
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
import types

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT.parent / "DemoGrasp" / "assets"
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


def flatten(x, prefix=""):
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(flatten(v, prefix + ("." if prefix else "") + k))
    else:
        out[prefix] = x
    return out


def stub_if_missing(name, attrs):
    """Upstream imports these at module level but never calls them on our code path."""
    try:
        __import__(name)
        return None
    except ImportError:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
        return name


def with_normals(pc, np, k=20):
    """xyz (N,3) -> (N,6): kNN (k=20, self excluded) PCA normal, flipped to point away from the origin.
    Same rule as upstream dataset_processor/generate_object_dataset_pcl.get_norm_vector_from_pc
    (exact kNN in NumPy instead of sklearn; eigenvectors are invariant to the covariance scale)."""
    pc = np.asarray(pc, dtype=np.float64)
    d = ((pc[:, None, :] - pc[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d, axis=1, kind="stable")[:, 1:k + 1]
    nb = pc[idx]
    c = nb - nb.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c)
    w, v = np.linalg.eigh(cov)
    nrm = v[np.arange(len(pc)), :, np.argmin(w, axis=1)]
    flip = (nrm * pc).sum(1) < 0
    nrm[flip] *= -1
    return np.hstack([pc, nrm]).astype(np.float32)


def ensure_bbox(rows, np):
    """DemoFunGrasp loads <dataset>/bbox_dict.npy (used only for its affordance-reward clip). Create or extend
    it for the listed objects with the upstream rule (trimesh OBB extents of the mesh), atomically."""
    import trimesh
    ds = ASSETS / "union_ycb_unidex"
    path = ds / "bbox_dict.npy"
    bbox = np.load(path, allow_pickle=True).item() if path.exists() else {}
    need = sorted({Path(r).stem for r in rows} - set(bbox))
    if not need:
        return
    meshes = {f.split(".")[0]: f for f in os.listdir(ds / "meshes")}
    for key in need:
        mesh = trimesh.load(str(ds / "meshes" / meshes[key]), force="mesh")
        bbox[key] = mesh.bounding_box_oriented.extents.astype(np.float32)
    if path.exists():  # merge with anything written meanwhile
        bbox = {**np.load(path, allow_pickle=True).item(), **bbox}
    tmp = ds / ("bbox_dict.%d.tmp.npy" % os.getpid())
    np.save(tmp, bbox)
    os.replace(str(tmp), str(path))
    print("bbox_dict: added %d objects (total %d)" % (len(need), len(bbox)), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("train", "eval"), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoint/shadow_example/model_8000.pt"))
    p.add_argument("--ckpt-config", default=str(ROOT / "checkpoint/shadow_example/config.json"))
    p.add_argument("--object-list", required=True, help="relative to DemoGrasp/assets; sorted, each object repeated K times")
    p.add_argument("--group-size", type=int, default=8, help="train: rollouts per group")
    p.add_argument("--mus", default="0.1,0.2,0.3", help="friction per group, cycled; 'none' keeps asset defaults")
    p.add_argument("--iterations", type=int, default=0)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--adv", choices=("grpo", "rloo", "drgrpo"), default="grpo")
    p.add_argument("--unpaired", action="store_true", help="ablation: independent conditions inside a group")
    p.add_argument("--learn-std", action="store_true")
    p.add_argument("--std-floor", type=float, default=0.0, help="train: per-dim lower bound on behavior/scoring std (pre-tanh)")
    p.add_argument("--init-optimizer", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--disable-gpu-jit-fusion", action="store_true")
    p.add_argument("--pcl-noise", type=float, default=0.0, help="iid Gaussian noise (m) on the xyz channels of the observed cloud; shared inside a paired group")
    p.add_argument("--reward", choices=("success", "functional", "official"), default="success",
                   help="train reward: official binary success, or functional = success AND style match AND affordance distance < --afford-thresh (the joint criterion of the DemoFunGrasp paper, Sec. 4.4), "
                        "or official = the full DemoFunGrasp training reward with the checkpoint's own weights (algo/ppo_onestep/ppo.py: success + qpos + affordance + close terms)")
    p.add_argument("--afford-thresh", type=float, default=0.04, help="functional reward: affordance-distance threshold in m")
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
    sys.path.insert(0, str(ROOT))
    stubbed = [s for s in (stub_if_missing("open3d", {}),
                           stub_if_missing("termcolor", {"cprint": lambda *a, **k: print(a[0] if a else "")})) if s]
    from isaacgym import gymapi  # noqa: F401
    from isaacgym.torch_utils import quat_from_angle_axis
    import isaacgymenvs
    import numpy as np
    import torch
    import yaml
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from isaacgymenvs.utils.utils import set_seed
    from run_demofungrasp import build_runner
    import tasks.grasp as tg
    # DemoGrasp's union_ycb_unidex point clouds are xyz only (512,3); DemoFunGrasp expects xyz+normal (512,6)
    # (its own object download is not reachable from here). Same xyz points, normals by the upstream rule.
    def load_pcl_with_normals(object_files, asset_root, if_aff=False):
        ret = []
        for fn in object_files:
            substrs = fn.split("/")
            assert len(substrs) == 3, fn
            pc = np.load(os.path.join(asset_root, substrs[0], "pointclouds", substrs[-1].replace(".urdf", ".npy")))
            if pc.shape[1] == 3:
                pc = with_normals(pc, np)
            if if_aff and pc.shape[1] == 6:
                pc = np.concatenate([pc, np.ones((pc.shape[0], 1), dtype=pc.dtype)], axis=-1)
            if sum(pc[:, -1] == 0) == pc.shape[0]:
                pc[:, -1] = 1
            ret.append(pc)
        return np.array(ret)

    tg.load_object_point_clouds = load_pcl_with_normals
    from isaacgymenvs.tasks import isaacgym_task_map

    rows = yaml.safe_load((ASSETS / args.object_list).read_text())
    ensure_bbox(rows, np)
    num_envs = len(rows)
    K = args.group_size if args.mode == "train" else 1
    if sorted(rows) != rows or num_envs % K or any(rows[i] != rows[i - i % K] for i in range(num_envs)):
        raise ValueError("object list must be sorted with each object repeated K times consecutively")
    G = num_envs // K
    ckpt = require_e(args.checkpoint)
    ck = json.loads(require_e(args.ckpt_config).read_text())
    overrides = ["task=grasp", "train=PPOOneStep", "hand=shadow_simple", "test=True", "if_visualize=False",
                 "num_envs=%d" % num_envs, "seed=%d" % args.seed, "headless=True", "force_render=False",
                 "capture_video=False", "graphics_device_id=-1"]
    with initialize_config_dir(config_dir=str(ROOT / "tasks"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    applied = {}
    keep = {"task.env.asset.assetRoot", "task.env.asset.multiObjectList", "task.env.numEnvs",
            "train.params.log_dir"}
    for section in ("task", "train"):
        for k, v in flatten(ck[section]).items():
            key = section + "." + k
            if key in keep or (isinstance(v, str) and "${" in v):
                continue
            try:
                cur = OmegaConf.select(cfg, key)
            except Exception:
                cur = None
            cur = OmegaConf.to_container(cur) if OmegaConf.is_config(cur) else cur
            if cur != v:
                OmegaConf.update(cfg, key, v, merge=False, force_add=True)
                applied[key] = {"was": cur, "now": v}
    cfg.checkpoint = str(ckpt)
    cfg.task.env.asset.assetRoot = str(require_e(ASSETS)) + "/"
    cfg.task.env.asset.multiObjectList = args.object_list
    cfg.task.env.numEnvs = num_envs
    cfg.task.env.trackingReferenceFile = str(require_e(ROOT / ck["task"]["env"]["trackingReferenceFile"]))
    cfg.task.func.style_dict_path = str(require_e(ROOT / ck["task"]["func"]["style_dict_path"]))
    if "render" in cfg.task.env:
        cfg.task.env.render.enable = False
    cfg.train.params.log_dir = str(output / "runner")
    cfg.run_name = "runner"
    cfg.seed = set_seed(args.seed, torch_deterministic=cfg.torch_deterministic, rank=0)
    if cfg.task.env.observationType != ck["task"]["env"]["observationType"]:
        raise RuntimeError("observation type does not match the checkpoint")
    if args.disable_gpu_jit_fusion:
        torch._C._jit_override_can_fuse_on_gpu(False)
        if hasattr(torch._C, "_jit_set_nvfuser_enabled"):
            torch._C._jit_set_nvfuser_enabled(False)

    readback = []
    base_cls = isaacgym_task_map[cfg.task_name]

    class GroupFrictionGrasp(base_cls):
        def create_sim(self):
            result = super().create_sim()
            if mus is None:
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
                readback.append({"env": i, "group": i // K, "mu": mu})
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
    gen = torch.Generator(device="cpu").manual_seed(args.seed * 7919 + 1)
    state = {"K": 1}
    style_list = [int(s) for s in cfg.task.func.style_list]

    # --- paired randomness hooks (all draws from `gen`, shared inside groups of size state["K"]) ---
    orig_trf = tg.torch_rand_float

    def paired_trf(lower, upper, shape, device):
        n, d = shape
        if d == env.num_robot_dofs and n == num_envs:
            k = state["K"]
            r = torch.rand((n // k, d), generator=gen) * (upper - lower) + lower
            return r.repeat_interleave(k, 0).to(device)
        return orig_trf(lower, upper, shape, device)

    tg.torch_rand_float = paired_trf
    fg = env.functional_generator
    if fg.afford_type != "score_based" or env.enable_dataset_aff:
        raise RuntimeError("expected score_based affordance without dataset labels (checkpoint config)")

    def paired_afford(point_cloud, object_pose, if_use_data_afford, afford=None):
        # identical scores to functional_generator.generate_affordance_points (score_based branch)
        pos, nrm = point_cloud[:, :, :3], point_cloud[:, :, 3:]
        height = (pos[:, :, 2] > object_pose[:, 2].unsqueeze(1)).float()
        z = torch.tensor([0, 0, 1], dtype=torch.float32, device=point_cloud.device)
        cos = torch.clamp(torch.nn.functional.cosine_similarity(nrm, z.expand_as(nrm), dim=-1), -1.0, 1.0)
        ang = torch.clamp(1 - torch.acos(cos) / (math.pi / 2), 0.0, 1.0)
        scores = height + ang
        probs = torch.where(scores.sum(1, keepdim=True) > 0, scores / scores.sum(1, keepdim=True),
                            torch.ones_like(scores) / scores.shape[1])
        k = state["K"]
        idx = torch.multinomial(probs[::k].double().cpu(), 1, generator=gen).squeeze(1)
        idx = idx.repeat_interleave(k).to(point_cloud.device)
        fg.num_envs, fg.num_points = point_cloud.shape[0], point_cloud.shape[1]
        fg.affordance_point = point_cloud[torch.arange(point_cloud.shape[0], device=point_cloud.device), idx].clone()
        return fg.affordance_point, idx

    fg.generate_affordance_points = paired_afford
    lo, hi = env.reset_position_range[:, 0].cpu(), env.reset_position_range[:, 1].cpu()

    def sample_conditions(n, k):
        m = n // k
        pos = lo + (hi - lo) * torch.rand(m, 3, generator=gen)
        pos[:, 2] += float(env.table_heights[0])
        axis = torch.randn(m, 3, generator=gen)
        axis = axis / axis.norm(dim=1, keepdim=True)
        angle = (torch.rand(m, generator=gen) * 2 - 1) * math.pi
        pose = torch.cat([pos, quat_from_angle_axis(angle, axis)], 1)
        style = torch.tensor(style_list)[torch.randint(0, len(style_list), (m,), generator=gen)]
        return pose.repeat_interleave(k, 0).to(dev), style.repeat_interleave(k, 0).to(dev)

    ids = torch.arange(num_envs, device=dev)

    def reset(k):
        state["K"] = k
        pose, style = sample_conditions(num_envs, k)
        env.manually_set_style_labels = style
        obs = env.reset_idx(ids, object_init_pose=pose)["obs"].clone()
        if not torch.equal(env.style_labels.long().view(-1), style.long()):
            raise RuntimeError("style labels were not applied")
        if env.afford_idx is None:
            raise RuntimeError("affordance point was not sampled at reset")
        if args.pcl_noise > 0:
            n, s0 = obs.shape[0], ac.pc_start_idx
            pc = obs[:, s0:].reshape(n, *ac.pc_shape).clone()
            z = torch.randn(n // k, pc.shape[1], 3, generator=gen).repeat_interleave(k, 0).to(obs.device)
            pc[..., :3] += args.pcl_noise * z
            obs[:, s0:] = pc.reshape(n, -1)
        return obs

    (output / "friction_readback.json").write_text(json.dumps(readback))
    manifest = {"utc": datetime.now(timezone.utc).isoformat(), "args": vars(args), "overrides": overrides,
                "applied_from_ckpt_config": applied, "stubbed_modules": stubbed, "num_envs": num_envs,
                "groups": G, "group_size": K, "checkpoint_sha256": digest(ckpt),
                "script_sha256": digest(Path(__file__)), "observation_type": cfg.task.env.observationType,
                "point_clouds": "DemoGrasp union_ycb_unidex xyz + kNN-PCA normals (k=20, upstream rule)"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    env.manually_set_style_labels = torch.tensor(style_list, device=dev)[torch.zeros(num_envs, dtype=torch.long, device=dev)]
    env.reset_idx(ids)
    runner = build_runner(cfg, env)  # test mode: loads checkpoint weights
    ac = runner.actor_critic
    act_dim = ac.act_dim
    for q in ac.parameters():
        q.requires_grad_(False)

    def pre_mean(o):
        pc = o[:, ac.pc_start_idx:].reshape(-1, *ac.pc_shape)
        f = ac.backbone(pc).reshape(-1, ac.pc_emb_dim)
        return ac.actor_mean(torch.cat([o[:, :ac.pc_start_idx], f], 1))

    official_parts = {}

    def rollout(obs, floor=0.0, official=False):
        with torch.no_grad():
            m = pre_mean(obs)
            std = torch.exp(ac.log_std).detach().clamp(min=floor).expand_as(m)
            u = m + std * torch.randn(m.shape, generator=gen).to(dev)
            env.generate_reaching_plan_idx(ids, actions=torch.tanh(u))
            dists = []
            for t in range(env.max_episode_length):
                env.step(env.compute_reference_actions())
                if official and env.cfg["func"]["if_use_close_reward"]:
                    dists.append(env.calcu_affordance_dist().clone().float())   # official: distance at every step
                if t == env.max_episode_length - 2:
                    r = env.successes.clone().float()
                    style_hit = (env.find_similar_style_label().long() == env.style_labels.long()).float()
                    adist = env.calcu_affordance_dist().clone().float()
                    if official:
                        # same terms, order and switches as the official PPOOneStep rollout (algo/ppo_onestep/ppo.py)
                        f = env.cfg["func"]
                        parts = {"success": r.clone()}
                        if f["if_use_qpos_reward"]:
                            parts["qpos"] = env.calcu_qpos_rewards(succ=r).clone().float().to(dev)
                        if f["use_affordance_reward"]:
                            parts["afford"] = env.calcu_affordance_rewards(mean_dists=adist, succ=r).clone().float().to(dev)
                        if f["if_use_close_reward"]:
                            mind = torch.stack(dists[:t], 1).min(1).values      # official: min over steps 0..t-1
                            parts["close"] = env.calcu_close_rewards(min_dists=mind, succ=r, type=f["close_type"]).clone().float().to(dev)
                        if f.get("if_use_contact_reward", False):
                            cs = env.calcu_contact_similarity(succ=r).clone().float().to(dev)
                            parts["contact"] = env.calcu_contact_rewards(contact_similarity=cs, succ=r).clone().float().to(dev)
                        official_parts.clear()
                        official_parts.update(parts)
                    break
        return m, std, u, r, style_hit, adist

    t0 = time.time()
    if args.mode == "eval":
        episodes = []
        for e in range(args.episodes):
            obs = reset(1)
            _, _, _, r, sh, ad = rollout(obs)
            d = output / ("episode_%03d" % e)
            d.mkdir()
            np.savez_compressed(d / "initial_state.npz", policy_observation=obs[:, :14].cpu().numpy(),
                                style_labels=env.style_labels.cpu().numpy(), afford_idx=env.afford_idx.cpu().numpy())
            episodes.append({"episode": e, "official_ppo_print_successes": r.cpu().tolist(),
                             "object_assets": list(rows), "style_match": sh.cpu().tolist(),
                             "afford_dist": ad.cpu().tolist(), "style_labels": env.style_labels.cpu().tolist()})
            succ = r.bool()
            print(json.dumps({"episode": e, "success": float(r.mean()),
                              "style_acc_on_success": float(sh[succ].mean()) if succ.any() else None,
                              "afford_dist_on_success": float(ad[succ].mean()) if succ.any() else None}), flush=True)
        S = np.array([ep["official_ppo_print_successes"] for ep in episodes])
        SH = np.array([ep["style_match"] for ep in episodes])
        AD = np.array([ep["afford_dist"] for ep in episodes])
        totals = {"success_rate": float(S.mean()), "episodes": int(S.size),
                  "style_acc_on_success": float(SH[S > 0].mean()) if (S > 0).any() else None,
                  "afford_dist_on_success": float(AD[S > 0].mean()) if (S > 0).any() else None}
        (output / "timing.json").write_text(json.dumps({"wall_seconds": time.time() - t0}))
        (output / "result.json").write_text(json.dumps({"status": "completed", "totals": totals,
                                                       "evaluated_environment_episodes": int(S.size),
                                                       "episodes": episodes}))
        print("DFG_EVAL_DONE", json.dumps(totals), flush=True)
        env.gym.destroy_sim(env.sim)
        return

    params = [q for n, q in ac.named_parameters() if n.startswith(("backbone", "actor_mean")) or (args.learn_std and n == "log_std")]
    for q in params:
        q.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=args.lr)
    if args.init_optimizer:
        opt.load_state_dict(torch.load(str(require_e(args.init_optimizer)), map_location=dev))
        for group in opt.param_groups:
            group["lr"] = args.lr

    def logp(u, m, std):
        return (-0.5 * ((u - m) / std) ** 2 - torch.log(std)).sum(-1)

    curve = []
    for it in range(args.iterations):
        obs = reset(1 if args.unpaired else K)
        m_old, std, u, r, sh, ad = rollout(obs, args.std_floor, official=(args.reward == "official"))
        if args.reward == "official":
            rew = sum(official_parts.values())
        else:
            rew = r if args.reward == "success" else r * sh * (ad < args.afford_thresh).float()
        rg = rew.view(G, K)
        mean_g = rg.mean(1, keepdim=True)
        if args.adv == "grpo":
            adv = (rg - mean_g) / (rg.std(1, keepdim=True) + 1e-6)
        elif args.adv == "rloo":
            adv = rg - (rg.sum(1, keepdim=True) - rg) / (K - 1)
        else:
            adv = rg - mean_g
        adv = adv.reshape(-1)
        informative = float(((rg.max(1).values - rg.min(1).values) > 0).float().mean())
        all_zero = float((rg.max(1).values == 0).float().mean())
        all_one = float((rg.min(1).values >= 1).float().mean())
        joint = r * sh * (ad < args.afford_thresh).float()
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
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                clipfrac.append(float(((ratio - 1).abs() > args.clip).float().mean()))
                kl.append(float((ratio - 1 - logr).mean()))
        succ = r.bool()
        curve.append({"it": it, "success": float(r.mean()), "reward": float(rew.mean()), **({"r_" + k_: float(v_.mean()) for k_, v_ in official_parts.items()} if args.reward == "official" else {}), "joint_4cm": float(joint.mean()),
                      "informative_groups": informative, "all_zero_groups": all_zero, "all_one_groups": all_one,
                      "style_acc_on_success": float(sh[succ].mean()) if succ.any() else None,
                      "clipfrac": float(np.mean(clipfrac)), "approx_kl": float(np.mean(kl)),
                      "seconds": time.time() - t0})
        print(json.dumps(curve[-1]), flush=True)
    torch.save(ac.state_dict(), output / "final.pt")
    torch.save(opt.state_dict(), output / "optimizer.pt")
    (output / "curve.json").write_text(json.dumps(curve, indent=1))
    (output / "train_log.json").write_text(json.dumps({"seconds": time.time() - t0, "final_sha256": digest(output / "final.pt"),
                                                       "episodes": args.iterations * num_envs, "act_dim": act_dim}, indent=2))
    print("GRPO_DONE", output, flush=True)
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
