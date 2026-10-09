"""Baseline: continue official PPO training of the frozen DemoGrasp Shadow planner under
low friction, with an interaction budget matched to the offline dataset (fair-comparison baseline).

One process = one phase (fixed object list). Chain phases with --init-checkpoint.
Friction: every robot and object collision shape in env i gets mu = MUS[i % len(MUS)].
Everything else follows the official Shadow preset (play_policy.sh / train.sh); episodeLength=50
to match our evaluation. Writes OUTPUT/{manifest.json, friction_readback.json, final.pt, train_log.json}.
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
    p.add_argument("--object-list", required=True)
    p.add_argument("--mus", default="0.1,0.2,0.3")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--iterations", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=None, help="Override optim_stepsize (default: official)")
    p.add_argument("--minibatches", type=int, default=None, help="Override nminibatches (default: official 4)")
    p.add_argument("--epochs", type=int, default=None, help="Override noptepochs (default: official 5)")
    p.add_argument("--fixed-lr", action="store_true", help="Use a fixed learning-rate schedule instead of adaptive")
    p.add_argument("--disable-gpu-jit-fusion", action="store_true")
    args = p.parse_args()
    mus = [float(x) for x in args.mus.split(",")]
    if not all(math.isfinite(m) and 0 <= m <= 2 for m in mus):
        raise ValueError("bad friction list")
    output = require_e(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cache = require_e(ROOT.parents[1] / "cache" / "shadow_probe")
    for key, sub in {"TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp", "XDG_CACHE_HOME": "xdg",
                     "TORCH_EXTENSIONS_DIR": "torch_extensions", "CUDA_CACHE_PATH": "cuda"}.items():
        (cache / sub).mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(cache / sub)
    os.chdir(str(ROOT))
    from isaacgym import gymapi  # noqa: F401
    import isaacgymenvs
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from isaacgymenvs.utils.utils import set_seed
    from run_rl_grasp import build_runner
    init = require_e(args.init_checkpoint)
    overrides = ["task=grasp", "train=PPOOneStep", "hand=shadow_simple", "test=False",
                 "num_envs=%d" % args.num_envs, "seed=%d" % args.seed, "headless=True",
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
    cfg.train.params.log_dir = str(output)
    cfg.run_name = "ppo"
    cfg.train.params.max_iterations = args.iterations
    cfg.train.params.save_interval = 10 ** 9
    if args.lr is not None:
        cfg.train.params.optim_stepsize = args.lr
    if args.minibatches is not None:
        cfg.train.params.nminibatches = args.minibatches
    if args.epochs is not None:
        cfg.train.params.noptepochs = args.epochs
    if args.fixed_lr:
        cfg.train.params.schedule = "fixed"
    cfg.task.env.asset.assetRoot = str(require_e(ROOT / "assets"))
    cfg.task.env.trackingReferenceFile = str(require_e(ROOT / cfg.task.env.trackingReferenceFile))
    cfg.seed = set_seed(args.seed, torch_deterministic=cfg.torch_deterministic, rank=0)
    if args.disable_gpu_jit_fusion:
        torch._C._jit_override_can_fuse_on_gpu(False)
        if hasattr(torch._C, "_jit_set_nvfuser_enabled"):
            torch._C._jit_set_nvfuser_enabled(False)
    (output / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    readback = []
    from isaacgymenvs.tasks import isaacgym_task_map
    base_cls = isaacgym_task_map[cfg.task_name]

    class FrictionGrasp(base_cls):
        def create_sim(self):
            result = super().create_sim()
            for i, env_ptr in enumerate(self.envs):
                mu = mus[i % len(mus)]
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
                readback.append({"env": i, "mu": mu, "object": self.object_fns[i % len(self.object_fns)]})
            return result

    isaacgym_task_map[cfg.task_name] = FrictionGrasp
    try:
        env = isaacgymenvs.make(cfg.seed, cfg.task_name, cfg.task.env.numEnvs, cfg.sim_device, cfg.rl_device,
                                cfg.graphics_device_id, cfg.headless, cfg.multi_gpu, cfg.capture_video,
                                cfg.force_render, cfg)
    finally:
        isaacgym_task_map[cfg.task_name] = base_cls
    (output / "friction_readback.json").write_text(json.dumps(readback, indent=1))
    manifest = {"utc": datetime.now(timezone.utc).isoformat(), "args": vars(args), "overrides": overrides,
                "init_checkpoint_sha256": digest(init), "script_sha256": digest(Path(__file__)),
                "budget_episodes": args.iterations * args.num_envs,
                "note": "Official PPOOneStep update rule and hyperparameters; reward = official success at step max-2, zero if table hit."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    env.reset_idx(torch.arange(env.num_envs))
    runner = build_runner(cfg, env)
    t0 = time.time()
    runner.run()
    final = Path(runner.log_dir) / ("model_%d.pt" % args.iterations)
    target = output / "final.pt"
    target.write_bytes(final.read_bytes())
    (output / "train_log.json").write_text(json.dumps({"seconds": time.time() - t0, "final": str(target),
                                                      "final_sha256": digest(target)}, indent=2))
    print("PPO_FINETUNE_DONE", target, flush=True)
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
