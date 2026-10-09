"""Output analysis on one saved set of common inputs (2026-10-08; plan: reports/补充实验锁定说明_偏置坐标对照_1008.md section 5).
No training and no physics rollouts (env.reset_idx only). Same state sampling as research_delta_diag.py: training-object list
(64 distinct objects repeated 8 times), --rounds pose draws, seed 0 -> 2048 states. Saves, for later analysis on E::
  inputs.npz   : observations, frozen pre-tanh mean m0(s), sigma0, object asset per state
  env_info.json: action layout (wrist translation 0-2, wrist rotation 3-5, hand 6-23 in simulator order), active hand joint names,
                 joint limits, demonstration grasp pose q_ref, edit ranges (as used by generate_reaching_plan_idx)
  models.npz   : per model the adapted pre-tanh mean m(s) on the same inputs
Model kinds: head.pt with delta.* -> residual (m = m0 + sigma0 * delta(s)); head.pt with c -> directly learned constant
(m = m0 + sigma0 * c); final.pt -> full planner state dict (full-weight fine-tuning, or a folded output shift)."""

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
    p.add_argument("--object-list", default="union_ycb_unidex/research_grpo_v1_phase00_64x8.yaml")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--models", required=True, help="comma list LABEL=PATH")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--disable-gpu-jit-fusion", action="store_true")
    args = p.parse_args()
    output = require_e(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cache = require_e(ROOT.parents[1] / "cache" / "shadow_probe")
    for key, sub in {"TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp", "XDG_CACHE_HOME": "xdg",
                     "TORCH_EXTENSIONS_DIR": "torch_extensions", "CUDA_CACHE_PATH": "cuda"}.items():
        (cache / sub).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(cache / sub))
    os.chdir(str(ROOT))
    from isaacgym import gymapi, gymtorch  # noqa: F401
    from isaacgym.torch_utils import quat_from_angle_axis
    import isaacgymenvs
    import numpy as np
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
    feat_dim = int(ac.pc_start_idx + ac.pc_emb_dim)
    act_dim = int(sigma0.shape[0])
    lo, hi = env.reset_position_range[:, 0], env.reset_position_range[:, 1]
    gen = torch.Generator(device="cpu").manual_seed(args.seed * 7919 + 1)

    def sample_poses(n):  # identical to research_delta_diag.py
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

    ref = env.tracking_reference["hand_qpos"]
    ref = ref.reshape(-1, ref.shape[-2], ref.shape[-1])
    q_ref = ref[:, env.T_ref_start_lifting - 1]
    assert torch.allclose(q_ref, q_ref[:1].expand_as(q_ref)), "demonstration grasp pose differs between envs"
    hidx = [int(i) for i in env.active_hand_dof_indices]
    info = {"action_layout": {"wrist_translation": [0, 1, 2], "wrist_rotation_euler_xyz": [3, 4, 5], "hand": list(range(6, act_dim))},
            "hand_joint_names": [env.robot_dof_names[i] for i in hidx],
            "hand_lower": env.robot_dof_lower_limits[hidx].tolist(), "hand_upper": env.robot_dof_upper_limits[hidx].tolist(),
            "q_ref": q_ref[0].tolist(), "grasp_pose_range": (env.randomize_grasp_pose_range.tolist() if hasattr(env.randomize_grasp_pose_range, "tolist") else (list(env.randomize_grasp_pose_range) if isinstance(env.randomize_grasp_pose_range, (list, tuple)) else float(env.randomize_grasp_pose_range))),
            "wrist_ranges": env.randomize_tracking_reference_range.tolist(), "T_ref_start_lifting": int(env.T_ref_start_lifting),
            "sigma0": sigma0.tolist(), "object_list": args.object_list, "rounds": args.rounds, "seed": args.seed,
            "init_sha256": digest(init), "script_sha256": digest(Path(__file__))}
    (output / "env_info.json").write_text(json.dumps(info, indent=1))
    with torch.no_grad():
        X, M0, OBS = [], [], []
        for _ in range(args.rounds):
            o = env.reset_idx(ids, object_init_pose=sample_poses(num_envs))["obs"].clone()
            x = feats(ac, o)
            X.append(x); M0.append(ac.actor_mean(x)); OBS.append(o)
        X, M0, OBS = torch.cat(X), torch.cat(M0), torch.cat(OBS)
        np.savez_compressed(output / "inputs.npz", obs=OBS.cpu().numpy().astype(np.float32), m0=M0.cpu().numpy(),
                            sigma0=sigma0.cpu().numpy(), object_asset=np.array([rows[i % num_envs] for i in range(OBS.shape[0])]))
        models, man = {}, {}
        for item in args.models.split(","):
            label, path = item.split("=", 1)
            path = require_e(path)
            sd = torch.load(str(path), map_location=dev)
            rec = {"path": str(path), "sha256": digest(path)}
            if path.name == "head.pt" and "c" in sd:
                m = M0 + sigma0 * sd["c"].to(dev)
                rec["kind"] = "constant"
            elif path.name == "head.pt":
                h = Head().to(dev)
                h.load_state_dict({k: v for k, v in sd.items() if k.startswith("delta.") or k == "rho"})
                m = M0 + sigma0 * h.delta(X)
                rec["kind"] = "residual"
            else:
                if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
                    sd = sd["model"]
                net = copy.deepcopy(ac)
                missing, unexpected = net.load_state_dict(sd, strict=False)
                bad = [k for k in missing if k.startswith(("backbone", "actor_mean"))]
                assert not bad, ("planner weights missing", label, bad[:5])
                net.eval()
                m = torch.cat([net.actor_mean(feats(net, OBS[i:i + num_envs])) for i in range(0, OBS.shape[0], num_envs)])
                rec.update({"kind": "full", "missing_nonplanner_keys": len(missing), "unexpected_keys": len(unexpected)})
            models[label] = m.cpu().numpy().astype(np.float32)
            man[label] = rec
            dm = m - M0
            print(json.dumps({"model": label, "kind": rec["kind"], "kl_raw_mean": float((0.5 * ((dm / sigma0) ** 2).sum(-1)).mean()),
                              "l1": float(dm.abs().mean(0).sum())}), flush=True)
        np.savez_compressed(output / "models.npz", **models)
        (output / "models_manifest.json").write_text(json.dumps(man, indent=1))
    print("OUTPUT_DIAG_DONE", output, flush=True)
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
