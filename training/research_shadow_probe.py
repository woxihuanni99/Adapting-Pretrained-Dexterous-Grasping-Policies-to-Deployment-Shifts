"""Bounded, recorded evaluation of the unmodified official Shadow planner.

Run with the isolated Linux environment prepared on E:. This entry point is
separate from the upstream sources and never trains or calls PPO.run().
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


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
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, help="New E: output directory; never overwritten")
    p.add_argument("--num-envs", type=int, choices=(1, 2, 4, 8, 16), default=8)
    p.add_argument("--episodes", type=int, default=2, help="Complete rounds per environment, 1..20")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", default=str(ROOT / "ckpt/shadow.pt"))
    p.add_argument("--object-list", default="union_ycb_unidex/union_ycb_debugset.yaml")
    p.add_argument("--override", action="append", default=[], help="Extra Hydra config override, recorded verbatim")
    p.add_argument("--disable-gpu-jit-fusion", action="store_true",
                   help="Explicit, process-local compatibility switch for the existing Torch/CUDA stack")
    a = p.parse_args()
    if not 1 <= a.episodes <= 20:
        p.error("--episodes must be between 1 and 20")
    if a.seed < 0:
        p.error("An explicit nonnegative seed is required")
    return a


def configure_process_cache(output):
    # No persistent environment edits. Configure before importing simulator/Torch.
    # Share only disposable build/download caches between runs, all on E.
    # Results stay in separate immutable output directories.
    cache = require_e(ROOT.parents[1] / "cache" / "shadow_probe")
    locations = {"TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
                 "XDG_CACHE_HOME": "xdg", "TORCH_EXTENSIONS_DIR": "torch_extensions",
                 "CUDA_CACHE_PATH": "cuda", "MPLCONFIGDIR": "matplotlib",
                 "HF_HOME": "huggingface", "TORCH_HOME": "torch"}
    for key, directory in locations.items():
        target = cache / directory
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MAX_JOBS"):
        os.environ.setdefault(key, "4" if key != "MAX_JOBS" else "2")


def source_manifest():
    paths = [ROOT / "run_rl_grasp.py", ROOT / "research_shadow_probe.py"]
    for directory in (ROOT / "tasks", ROOT / "algo"):
        paths.extend(p for p in directory.rglob("*")
                     if p.is_file() and p.suffix in (".py", ".yaml", ".yml"))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))}


def physical_properties(env):
    records = []
    for i, env_ptr in enumerate(env.envs):
        actors = {}
        for name in ("object", "robot"):
            handle = env.gym.find_actor_handle(env_ptr, name)
            if handle < 0:
                raise RuntimeError("Actor not found: %s in environment %d" % (name, i))
            bodies = env.gym.get_actor_rigid_body_properties(env_ptr, handle)
            shapes = env.gym.get_actor_rigid_shape_properties(env_ptr, handle)
            actors[name] = {
                "handle": int(handle),
                "body_masses_kg": [float(prop.mass) for prop in bodies],
                "total_mass_kg": sum(float(prop.mass) for prop in bodies),
                "shape_frictions": [float(prop.friction) for prop in shapes],
                "shape_restitutions": [float(prop.restitution) for prop in shapes],
            }
        records.append({"env_index": i,
                        "object_asset": env.object_fns[i % len(env.object_fns)],
                        "actors": actors})
    return records


def run(args, output):
    # Isaac Gym must be imported before Torch. No dependency installation here.
    from isaacgym import gymapi, gymtorch  # noqa: F401
    import isaacgymenvs
    import torch
    import numpy as np
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from isaacgymenvs.utils.utils import set_seed
    from run_rl_grasp import build_runner

    checkpoint = require_e(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))
    overrides = [
        "task=grasp", "train=PPOOneStep", "hand=shadow_simple", "test=True",
        "num_envs=" + str(args.num_envs), "seed=" + str(args.seed),
        "headless=True", "force_render=False", "capture_video=False", "graphics_device_id=-1",
        "task.env.observationType=eefpose+objinitpose+objpcl", "task.env.armController=pose",
        "task.env.asset.multiObjectList=" + args.object_list,
        "task.env.randomizeTrackingReference=True", "task.env.randomizeGraspPose=True",
        "task.env.resetDofPosRandomInterval=0", "task.env.episodeLength=50",
        "task.env.enablePointCloud=True", "train.params.is_vision=True",
        "task.env.trackingReferenceFile=tasks/grasp_ref_shadow.pkl",
        "task.env.trackingReferenceLiftTimestep=11", "task.env.render.enable=False",
    ] + list(args.override)
    with initialize_config_dir(config_dir=str(ROOT / "tasks"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    # Force bounded inference/output choices even when additional overrides exist.
    cfg.test = True
    cfg.train.params.test = True
    cfg.checkpoint = str(checkpoint)
    cfg.num_envs = args.num_envs
    cfg.task.env.numEnvs = args.num_envs
    cfg.headless, cfg.force_render, cfg.capture_video = True, False, False
    cfg.task.env.render.enable = False
    cfg.multi_gpu = False
    cfg.graphics_device_id = -1
    cfg.train.params.log_dir = str(output / "unused_training_logs")
    cfg.task.env.asset.assetRoot = str(require_e(ROOT / "assets"))
    object_list = require_e(ROOT / "assets" / cfg.task.env.asset.multiObjectList)
    if not object_list.is_file():
        raise FileNotFoundError(str(object_list))
    if cfg.hand.name != "shadow_simple" or cfg.train.params.name != "ppo_onestep":
        raise ValueError("This probe only supports the official Shadow one-step planner")
    if cfg.task.env.randomEpisodeLength or cfg.task.task.randomize:
        raise ValueError("Random episode length / domain randomization require a separate protocol")
    if cfg.task.env.rewardType != "binary":
        raise ValueError("The recorded metric definitions require the official binary reward")
    if not 2 <= int(cfg.task.env.episodeLength) <= 200:
        raise ValueError("Episode length must be 2..200")
    reference = require_e(ROOT / cfg.task.env.trackingReferenceFile)
    cfg.task.env.trackingReferenceFile = str(reference)
    cfg.seed = set_seed(args.seed, torch_deterministic=cfg.torch_deterministic, rank=0)
    if args.disable_gpu_jit_fusion:
        torch._C._jit_override_can_fuse_on_gpu(False)
        if hasattr(torch._C, "_jit_set_nvfuser_enabled"):
            torch._C._jit_set_nvfuser_enabled(False)
    config_text = OmegaConf.to_yaml(cfg, resolve=True)
    (output / "resolved_config.yaml").write_text(config_text, encoding="utf-8")
    sources_before = source_manifest()
    references = {str(p): digest(p) for p in (checkpoint, reference, object_list)}
    write_json(output / "manifest.json", {
        "utc_started": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
        "hydra_overrides": overrides, "sources_sha256": sources_before,
        "references_sha256": references,
        "resolved_config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        "software": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                     "isaacgymenvs_module": str(Path(isaacgymenvs.__file__).resolve())},
        "scope": "Bounded official Shadow baseline probe; no training and no new method.",
        "cold_vs_warm": "Official main pre-run reset retained; all rounds use explicit in-process resets.",
    })
    env = None
    try:
        env = isaacgymenvs.make(cfg.seed, cfg.task_name, cfg.task.env.numEnvs,
                               cfg.sim_device, cfg.rl_device, cfg.graphics_device_id,
                               cfg.headless, cfg.multi_gpu, cfg.capture_video,
                               cfg.force_render, cfg)
        # Retain run_rl_grasp.main's reset before build_runner; PPO.run resets again.
        ids = torch.arange(env.num_envs)
        env.reset_idx(ids)
        runner = build_runner(cfg, env)
        runner.actor_critic.eval()
        runner.actor_critic.requires_grad_(False)
        contact = gymtorch.wrap_tensor(env.gym.acquire_net_contact_force_tensor(env.sim))
        contact = contact.view(env.num_envs, -1, 3)
        assets = {str(require_e(Path(cfg.task.env.asset.assetRoot) / p)):
                  digest(require_e(Path(cfg.task.env.asset.assetRoot) / p))
                  for p in sorted(set(env.object_fns))}
        write_json(output / "loaded_object_urdf_sha256.json", assets)

        def cpu(value):
            array = value.detach().cpu().numpy().copy()
            if array.dtype.kind in "fc" and not np.isfinite(array).all():
                raise FloatingPointError("Nonfinite recorded tensor")
            return array

        def snapshot():
            env.gym.refresh_net_contact_force_tensor(env.sim)
            return {
                "object_root_state": cpu(env.root_state_tensor[env.object_indices]),
                "reward_object_pos": cpu(env.object_pos),
                "object_init_state": cpu(env.object_init_states),
                "palm_state": cpu(env.palm_state), "palm_center_pos": cpu(env.palm_center_pos),
                "fingertip_state": cpu(env.fingertip_state),
                "reward_fingertip_pos": cpu(env.fingertip_pos),
                "eef_state": cpu(env.rigid_body_states.view(-1, 13)[env.eef_idx]),
                "robot_dof_pos": cpu(env.robot_dof_pos), "robot_dof_vel": cpu(env.robot_dof_vel),
                "net_contact_force": cpu(contact), "table_heights": cpu(env.table_heights),
                "progress_buf": cpu(env.progress_buf), "reset_buf": cpu(env.reset_buf),
                "official_successes": cpu(env.successes),
                "official_current_successes": cpu(env.current_successes),
                "has_hit_table": cpu(env.has_hit_table),
            }

        episodes = []
        for episode in range(args.episodes):
            episode_dir = output / ("episode_%03d" % episode)
            episode_dir.mkdir(exist_ok=False)
            obs = env.reset_idx(ids)["obs"].clone()
            states = env.get_state()
            initial = snapshot()
            initial["policy_observation"] = cpu(obs)
            initial["policy_state"] = cpu(states)
            initial["dof_target_initial"] = cpu(env.cur_targets)
            np.savez_compressed(str(episode_dir / "initial_state.npz"), **initial)
            write_json(episode_dir / "physical_properties_after_reset.json", physical_properties(env))
            with torch.no_grad():
                planner_action = runner.actor_critic(obs, states, inference=True)
                env.generate_reaching_plan_idx(ids, actions=planner_action)
            tracking_reference = getattr(env, "current_tracking_reference", None)
            if not isinstance(tracking_reference, dict) or any(
                    not torch.is_tensor(tracking_reference.get(key))
                    for key in ("wrist_initobj_pos", "wrist_quat", "hand_qpos")):
                raise RuntimeError("Official generated tracking reference is missing expected tensors")
            np.savez_compressed(str(episode_dir / "planner.npz"),
                                planner_action=cpu(planner_action),
                                reaching_plan_timesteps=cpu(env.reaching_plan_timesteps),
                                reaching_plan_ee=cpu(env.reaching_plan_ee),
                                **{"tracking_reference_" + k: cpu(v)
                                   for k, v in tracking_reference.items()
                                   if torch.is_tensor(v)})
            records = {}
            for step in range(env.max_episode_length):
                with torch.no_grad():
                    command = env.compute_reference_actions()
                    command_array = cpu(command)
                    previous_targets = cpu(env.prev_targets)
                    _, reward, reset, extras = env.step(command)
                sample = snapshot()
                sample.update({"reference_command_requested": command_array,
                               "reference_command_after_clip": cpu(env.actions),
                               "dof_target_start": previous_targets,
                               "dof_target_final": cpu(env.cur_targets),
                               "returned_reward": cpu(reward), "returned_reset": cpu(reset),
                               "extras_current_successes": cpu(extras["current_successes"]),
                               "extras_successes": cpu(extras["successes"])})
                for key in ("palm_object_dist", "fingertips_object_dist", "hand_approach_flag", "lift_object"):
                    if key in extras:
                        sample["reward_info_" + key] = cpu(extras[key])
                for key, value in sample.items():
                    records.setdefault(key, []).append(value)
            arrays = {k: np.stack(v, axis=0) for k, v in records.items()}
            np.savez_compressed(str(episode_dir / "trajectory.npz"), **arrays)
            reset_mask = arrays["returned_reset"].astype(bool)
            expected_mask = np.zeros_like(reset_mask)
            expected_mask[-1] = True
            reset_check = bool(np.array_equal(reset_mask, expected_mask))
            printed_success = arrays["official_successes"][-2] > .5
            terminal_success = arrays["official_current_successes"][-1] > .5
            last10 = np.all(arrays["official_successes"][-min(10, env.max_episode_length):] > .5, axis=0)
            metrics = {
                "episode": episode, "num_envs": env.num_envs,
                "reward_num_fingers": int(env.num_fingers),
                "control_steps": int(env.max_episode_length), "physics_dt": float(env.dt),
                "decimation": int(env.decimation), "exact_terminal_reset_pattern": reset_check,
                "diagnostic_last_window_control_steps": min(10, int(env.max_episode_length)),
                "official_ppo_print_step_0based": int(env.max_episode_length - 2),
                "official_ppo_print_successes": printed_success.tolist(),
                "official_ppo_print_success_count": int(printed_success.sum()),
                "official_terminal_current_successes": terminal_success.tolist(),
                "official_terminal_current_success_count": int(terminal_success.sum()),
                "diagnostic_last10_official_success_condition": last10.tolist(),
                "diagnostic_last10_official_success_count": int(last10.sum()),
                "final_has_hit_table": arrays["has_hit_table"][-1].astype(bool).tolist(),
                "initial_state_sha256": digest(episode_dir / "initial_state.npz"),
                "trajectory_sha256": digest(episode_dir / "trajectory.npz"),
                "planner_sha256": digest(episode_dir / "planner.npz"),
                "physical_properties_sha256": digest(episode_dir / "physical_properties_after_reset.json"),
            }
            write_json(episode_dir / "metrics.json", metrics)
            episodes.append(metrics)
            print("EPISODE", episode, "official_print", int(printed_success.sum()), "/", env.num_envs,
                  "official_terminal", int(terminal_success.sum()), "/", env.num_envs, flush=True)
            if not reset_check:
                raise RuntimeError("Unexpected reset pattern; preserve traces and do not treat as completed evaluation")
        sources_unchanged = source_manifest() == sources_before
        references_unchanged = all(digest(p) == h for p, h in references.items())
        denominator = args.episodes * env.num_envs
        totals = {key: sum(row[key] for row in episodes)
                  for key in ("official_ppo_print_success_count", "official_terminal_current_success_count",
                              "diagnostic_last10_official_success_count")}
        write_json(output / "result.json", {
            "status": "completed" if sources_unchanged and references_unchanged else "integrity_failed",
            "scope": "Local smoke/probe only; not the full published evaluation or independent final test.",
            "episodes_per_environment": args.episodes, "num_envs": env.num_envs,
            "evaluated_environment_episodes": denominator, "totals": totals,
            "rates": {key.replace("_count", "_rate"): value / denominator for key, value in totals.items()},
            "episodes": episodes, "source_files_unchanged": sources_unchanged,
            "references_unchanged": references_unchanged,
            "artifact_sha256": {name: digest(output / name) for name in
                                ("manifest.json", "resolved_config.yaml", "loaded_object_urdf_sha256.json")},
            "metric_note": "PPO-print metric is successes at the penultimate control step. Terminal metric is current_successes after terminal reset flag. Diagnostic last10 requires the official instantaneous success condition at each of the final 10 control steps; it is neither an official paper metric nor the earlier O6 hold10 metric. Intermediate reward/current_successes may retain the previous episode's value until the terminal update.",
            "command_note": "reference_command_requested is the policy-plan reference; reference_command_after_clip is env.actions; dof_target_final is the last position target actually sent in each control step. Intermediate simulator targets interpolate from dof_target_start to dof_target_final over decimation substeps, as in the unchanged official env.step.",
        })
        if not sources_unchanged or not references_unchanged:
            raise RuntimeError("Source/reference hash changed during probe")
    finally:
        if env is not None:
            if getattr(env, "viewer", None) is not None:
                env.gym.destroy_viewer(env.viewer)
            if getattr(env, "sim", None) is not None:
                env.gym.destroy_sim(env.sim)


def main():
    args = parse_args()
    require_e(ROOT)
    output = require_e(args.output)
    output.mkdir(parents=True, exist_ok=False)
    configure_process_cache(output)
    os.chdir(str(ROOT))
    began = time.perf_counter()
    try:
        run(args, output)
    except Exception:
        write_json(output / "failure.json", {"status": "failed", "traceback": traceback.format_exc(),
                                             "wall_seconds": time.perf_counter() - began})
        raise
    else:
        write_json(output / "timing.json", {"wall_seconds": time.perf_counter() - began})


if __name__ == "__main__":
    main()
