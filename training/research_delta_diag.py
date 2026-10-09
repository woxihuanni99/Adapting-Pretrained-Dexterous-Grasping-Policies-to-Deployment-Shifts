"""Offline diagnostic (2026-10-02, no training, no rollouts): how far trained policies move the frozen planner's pre-tanh
mean on fixed nominal-state inputs, per action dimension.

For a set of training-object states (random initial object poses drawn exactly as in training, observation before any
noise injection) it evaluates
  residual heads  : delta(s) (noise-space shift), dm = sigma0 * delta, learned scale e^rho
  full-weight ckpt: dm = m_theta(s) - m0(s)
and reports per model
  kl_floor = mean_s 0.5 * sum_j (dm_j / s_j)^2  with s_j = max(sigma0_j, std_floor)   (the KL used by the KL-regularized
             full-weight GRPO baseline, research_grpo_finetune_v2_kl.py, for fixed-std Gaussians)
  kl_raw   = mean_s 0.5 * sum_j (dm_j / sigma0_j)^2  (= 0.5 * sum delta^2 for residual heads)
  per-dim mean dm, mean |dm|, mean |dm| / sigma0, and the share of dm variance that is state-dependent.
Writes OUTPUT/delta_diag.json. Nothing is simulated beyond env.reset_idx (no physics rollouts)."""
import argparse
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
    p.add_argument("--init-checkpoint", default=str(ROOT / "ckpt/shadow.pt"), help="frozen planner")
    p.add_argument("--object-list", required=True)
    p.add_argument("--rounds", type=int, default=4, help="pose draws per object")
    p.add_argument("--std-floor", type=float, default=0.05)
    p.add_argument("--models", required=True, help="comma list LABEL=PATH (head.pt -> residual, final.pt -> full weight)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--disable-gpu-jit-fusion", action="store_true")
    args = p.parse_args()
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
    import torch
    import torch.nn as nn
    import yaml
    import copy
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from isaacgymenvs.utils.utils import set_seed
    from run_rl_grasp import build_runner
    rows = yaml.safe_load((ROOT / "assets" / args.object_list).read_text())
    num_envs = len(rows)
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
    env = isaacgymenvs.make(cfg.seed, cfg.task_name, cfg.task.env.numEnvs, cfg.sim_device, cfg.rl_device,
                            cfg.graphics_device_id, cfg.headless, cfg.multi_gpu, cfg.capture_video, cfg.force_render, cfg)
    dev = env.device
    ids = torch.arange(num_envs, device=dev)
    env.reset_idx(ids)
    runner = build_runner(cfg, env)
    ac = runner.actor_critic
    ac.eval()
    for q in ac.parameters():
        q.requires_grad_(False)
    sigma0 = torch.exp(ac.log_std).detach()
    s_floor = sigma0.clamp(min=args.std_floor)
    feat_dim = int(ac.pc_start_idx + ac.pc_emb_dim)
    act_dim = int(sigma0.shape[0])
    lo, hi = env.reset_position_range[:, 0], env.reset_position_range[:, 1]
    gen = torch.Generator(device="cpu").manual_seed(args.seed * 7919 + 1)

    def sample_poses(n):
        pos = lo.cpu() + (hi - lo).cpu() * torch.rand(n, 3, generator=gen)
        pos[:, 2] += float(env.table_heights[0])
        axis = torch.randn(n, 3, generator=gen)
        axis = axis / axis.norm(dim=1, keepdim=True)
        angle = (torch.rand(n, generator=gen) * 2 - 1) * math.pi
        return torch.cat([pos, quat_from_angle_axis(angle, axis)], 1).to(dev)

    def feats(net, o):
        pc = o[:, net.pc_start_idx:].reshape(-1, *net.pc_shape)
        f = net.backbone(pc).reshape(-1, net.pc_emb_dim)
        return torch.cat([o[:, :net.pc_start_idx], f], 1)

    def mlp(i, o):
        return nn.Sequential(nn.LayerNorm(i), nn.Linear(i, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, o))

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.delta = mlp(feat_dim, act_dim)
            self.rho = nn.Parameter(torch.zeros(act_dim))

    with torch.no_grad():
        X, M0, OBS = [], [], []
        for _ in range(args.rounds):
            o = env.reset_idx(ids, object_init_pose=sample_poses(num_envs))["obs"].clone()
            x = feats(ac, o)
            X.append(x); M0.append(ac.actor_mean(x)); OBS.append(o)
        X, M0, OBS = torch.cat(X), torch.cat(M0), torch.cat(OBS)
        out = {"sigma0": sigma0.tolist(), "std_floor": args.std_floor, "n_states": int(X.shape[0]), "object_list": args.object_list,
               "init_sha256": digest(init), "script_sha256": digest(Path(__file__)), "models": {}}
        for item in args.models.split(","):
            label, path = item.split("=", 1)
            path = require_e(path)
            sd = torch.load(str(path), map_location=dev)
            rec = {"path": str(path), "sha256": digest(path)}
            if path.name == "head.pt":
                h = Head().to(dev)
                h.load_state_dict({k: v for k, v in sd.items() if k.startswith("delta.") or k == "rho"})
                d = h.delta(X)
                dm = sigma0 * d
                rec.update({"kind": "residual", "e_rho": torch.exp(h.rho).tolist(), "mean_abs_delta": d.abs().mean(0).tolist(),
                            "kl_raw_mean": float((0.5 * (d ** 2).sum(-1)).mean())})
            else:
                net = copy.deepcopy(ac)
                net.load_state_dict(sd)
                net.eval()
                dm = torch.cat([net.actor_mean(feats(net, OBS[i:i + num_envs])) for i in range(0, OBS.shape[0], num_envs)]) - M0
                rec.update({"kind": "full", "kl_raw_mean": float((0.5 * ((dm / sigma0) ** 2).sum(-1)).mean())})
            klf = 0.5 * ((dm / s_floor) ** 2).sum(-1)
            var_tot = dm.var(0)
            rec.update({"kl_floor_mean": float(klf.mean()), "kl_floor_sd": float(klf.std()), "kl_floor_median": float(klf.median()),
                        "mean_dm": dm.mean(0).tolist(), "mean_abs_dm": dm.abs().mean(0).tolist(),
                        "mean_abs_dm_over_sigma0": (dm.abs().mean(0) / sigma0).tolist(),
                        "state_dependent_share": (var_tot / (var_tot + dm.mean(0) ** 2 + 1e-12)).tolist()})
            out["models"][label] = rec
            print(json.dumps({"model": label, "kind": rec["kind"], "kl_floor_mean": rec["kl_floor_mean"], "kl_raw_mean": rec["kl_raw_mean"]}), flush=True)
    (output / "delta_diag.json").write_text(json.dumps(out, indent=1))
    print("DELTA_DIAG_DONE", output, flush=True)
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
