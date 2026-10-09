"""Stress-test variant of research_grasp_collect.py (2026-09-25): perturb what the frozen planner
SEES (not the simulator): --pcl-noise SIGMA (m, iid Gaussian on the 512x3 world-frame object cloud),
--pcl-partial (keep the half of the cloud facing a random view direction, resample to 512 points),
--pose-noise SIGMA (m, on the object initial position in the observation). Physics are unchanged;
config-level stressors (distractors, PD gains, workspace) go through --override.

Data-collection variant of research_grasp_edit_probe.py (2026-09-24).

Additionally records the frozen planner's own input features (142-d: 14 pose + 128
PointNet) and pre-tanh mean via a read-only forward hook, and supports
--edit policy:TEMP = tanh(mean + TEMP * exp(log_std) * eps), i.e. sampling from the
frozen PPO policy's own Gaussian as behavior policy. Weights are never modified.

Frozen-planner evaluation with an optional, predeclared planner-action edit.

Derived from research_shadow_extreme_probe.py (2026-09-24). Adds:
  --hand shadow|inspire   official DemoGrasp presets (inspire = FR3 + Inspire)
  --edit none | tighten:ETA | scale:BETA | noise:SIGMA[:hand]
     tighten: a_hand += ETA * d, d = sign(q_demo[lift-1] - q_demo[0]) (|diff|>0.05 rad), clip [-1,1]
     scale:   a_hand *= BETA (BETA=0 -> demo grasp pose)
     noise:   a += SIGMA * N(0,1) on all dims (or hand dims only), clip [-1,1]; behavior data
  --lean   save compact per-step success/height arrays instead of the full trajectory
The frozen policy output is saved as planner_action_frozen; the executed (edited)
action is saved as planner_action. No training happens here.

Original description: Bounded gravity/friction stress evaluation of the official Shadow planner.

Run with the isolated Linux environment prepared on E:. This entry point is
separate from the upstream sources and never trains or calls PPO.run().
Object mass scaling runs in a process-local task subclass before prepare_sim.
Gravity scaling changes the gravity vector, not object mass. Contact collection
remains disabled, so recorded net contact forces are not contact evidence.
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
    p.add_argument("--num-envs", type=int, choices=(1, 2, 4, 8, 16, 32, 64, 128), default=8)
    p.add_argument("--episodes", type=int, default=2, help="Complete rounds per environment, 1..20")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gravity-scale", type=float, default=1.0,
                   help="Multiply resolved base gravity vector by finite scale 0.1..5")
    p.add_argument("--robot-friction", type=float, default=None,
                   help="Set friction of every robot collision shape to finite value 0..2")
    p.add_argument("--object-mass-scale", type=float, default=1.0,
                   help="One-time per-body object mass multiplier, finite 0.5..4")
    p.add_argument("--object-friction", type=float, default=None,
                   help="Object shape friction, finite 0..2; overrides the final Hydra setting")
    p.add_argument("--checkpoint", default=str(ROOT / "ckpt/shadow.pt"))
    p.add_argument("--object-list", default="union_ycb_unidex/union_ycb_debugset.yaml")
    p.add_argument("--override", action="append", default=[], help="Extra Hydra config override, recorded verbatim")
    p.add_argument("--hand", choices=("shadow", "inspire"), default="shadow")
    p.add_argument("--pcl-noise", type=float, default=0.0)
    p.add_argument("--pcl-partial", action="store_true")
    p.add_argument("--pose-noise", type=float, default=0.0)
    p.add_argument("--edit", default="none", help="none | tighten:ETA | scale:BETA | noise:SIGMA[:hand]")
    p.add_argument("--edit-seed", type=int, default=0, help="Seed for noise edits (independent of env seed)")
    p.add_argument("--lean", action="store_true", help="Compact per-step records (for data collection)")
    p.add_argument("--disable-gpu-jit-fusion", action="store_true",
                   help="Explicit, process-local compatibility switch for the existing Torch/CUDA stack")
    a = p.parse_args()
    if not 1 <= a.episodes <= 20:
        p.error("--episodes must be between 1 and 20")
    if a.seed < 0:
        p.error("An explicit nonnegative seed is required")
    if not math.isfinite(a.gravity_scale) or not 0.1 <= a.gravity_scale <= 5.0:
        p.error("--gravity-scale must be finite and between 0.1 and 5")
    if a.robot_friction is not None and (
            not math.isfinite(a.robot_friction) or not 0 <= a.robot_friction <= 2):
        p.error("--robot-friction must be finite and between 0 and 2")
    if not math.isfinite(a.object_mass_scale) or not 0.5 <= a.object_mass_scale <= 4.0:
        p.error("--object-mass-scale must be finite and between 0.5 and 4")
    kind = a.edit.split(":")[0]
    if kind not in ("none", "tighten", "scale", "noise", "policy", "model"):
        p.error("--edit must be none | tighten:ETA | scale:BETA | noise:SIGMA[:hand] | policy:TEMP | model:E-PATH.pt")
    if kind == "model":
        model_path = require_e(a.edit.split(":", 1)[1] if not a.edit.split(":", 1)[1].startswith("/")
                               else a.edit.split(":", 1)[1])
        if model_path.suffix != ".pt" or not model_path.is_file():
            p.error("model edit needs an existing TorchScript .pt on E:")
    elif kind != "none":
        value = float(a.edit.split(":")[1])
        if not math.isfinite(value) or not -2.0 <= value <= 2.0:
            p.error("edit magnitude must be finite in [-2, 2]")
    if a.object_friction is not None and (
            not math.isfinite(a.object_friction) or not 0.0 <= a.object_friction <= 2.0):
        p.error("--object-friction must be finite and between 0 and 2")
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
    paths = [ROOT / "run_rl_grasp.py", ROOT / "research_shadow_probe.py",
             ROOT / "research_shadow_extreme_probe.py", ROOT / "research_grasp_edit_probe.py", ROOT / "research_grasp_collect.py",
             ROOT / "research_shadow_physics_probe.py", Path(__file__).resolve()]
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


PHYSICS_TOLERANCES = {
    "mass": {"rtol": 1e-5, "atol": 1e-9},
    "invMass": {"rtol": 1e-5, "atol": 1e-8},
    "inertia": {"rtol": 5e-4, "atol": 1e-12},
    "invInertia": {"rtol": 5e-4, "atol": 1e-6},
    "com": {"rtol": 0.0, "atol": 1e-7},
}


def external_mass_control_evidence():
    path = require_e(ROOT.parents[1] / "runs/gpu_mass_response_01/result.json")
    if not path.is_file():
        return {"available": False, "path": str(path),
                "note": "No external dynamical-response validation is claimed."}
    result = json.loads(path.read_text(encoding="utf-8"))
    checks = result.get("checks", {})
    return {"available": True, "path": str(path), "sha256": digest(path),
            "status": result.get("status"), "checks": checks,
            "passed": result.get("status") == "passed" and bool(checks)
                      and all(value is True for value in checks.values()),
            "physics_steps": result.get("physics_steps"), "bodies": result.get("bodies"),
            "gpu_pipeline": result.get("gpu_pipeline"), "source_sha256": result.get("source_sha256"),
            "scope": "Independent contact-free two-box force-response control; not grasping performance or per-object dynamical verification."}


def gym_value(value):
    """Serialize scalar, Vec3, Mat33 or Transform without importing simulator."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if hasattr(value, "p") and hasattr(value, "r"):
        return {"p": gym_value(value.p), "r": gym_value(value.r)}
    if all(hasattr(value, key) for key in ("x", "y", "z")):
        keys = ("x", "y", "z", "w") if hasattr(value, "w") else ("x", "y", "z")
        return [gym_value(getattr(value, key)) for key in keys]
    numeric = float(value)
    if not math.isfinite(numeric):
        raise FloatingPointError("Nonfinite simulator physical property")
    return numeric


def detailed_physics(env):
    """Record every body and exposed shape property for object/robot/table."""
    output = []
    for env_index, env_ptr in enumerate(env.envs):
        actors = {}
        for actor_name in ("object", "robot", "table"):
            handle = env.gym.find_actor_handle(env_ptr, actor_name)
            if handle < 0:
                raise RuntimeError("Required actor missing: " + actor_name)
            bodies = env.gym.get_actor_rigid_body_properties(env_ptr, handle)
            shapes = env.gym.get_actor_rigid_shape_properties(env_ptr, handle)
            actors[actor_name] = {
                "handle": int(handle),
                "bodies": [{key: gym_value(getattr(prop, key)) for key in
                            ("mass", "invMass", "inertia", "invInertia", "com", "flags")}
                           for prop in bodies],
                "shapes": [{key: gym_value(getattr(prop, key))
                            for key in dir(prop) if not key.startswith("_")
                            and not callable(getattr(prop, key))} for prop in shapes],
            }
        output.append({"env_index": env_index,
                       "object_asset": env.object_fns[env_index % len(env.object_fns)],
                       "actors": actors})
    return output


def scaled_tree(value, factor):
    if isinstance(value, dict):
        return {key: scaled_tree(item, factor) for key, item in value.items()}
    if isinstance(value, list):
        return [scaled_tree(item, factor) for item in value]
    return float(value) * factor


def close_tree(actual, expected, tolerance):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            close_tree(actual[key], value, tolerance) for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            close_tree(a, b, tolerance) for a, b in zip(actual, expected))
    return math.isclose(float(actual), float(expected),
                        rel_tol=tolerance["rtol"], abs_tol=tolerance["atol"])


def physics_errors(nominal, actual, mass_scale, friction, robot_friction=None):
    """Check fixed original values, never scale an already-scaled readback."""
    errors = []
    if len(actual) != len(nominal):
        return ["Environment count changed"]
    for original, current in zip(nominal, actual):
        prefix = "env %d" % original["env_index"]
        if (original["env_index"], original["object_asset"]) != (
                current["env_index"], current["object_asset"]):
            errors.append(prefix + ": object assignment changed")
        if current["actors"]["table"] != original["actors"]["table"]:
            errors.append(prefix + ": table body/shape properties changed")
        robot_before, robot_after = original["actors"]["robot"], current["actors"]["robot"]
        if robot_friction is None:
            if robot_after != robot_before:
                errors.append(prefix + ": robot body/shape properties changed without permission")
        else:
            if robot_before["handle"] != robot_after["handle"] or robot_before["bodies"] != robot_after["bodies"]:
                errors.append(prefix + ": robot handle/body properties changed")
            if len(robot_before["shapes"]) != len(robot_after["shapes"]):
                errors.append(prefix + ": robot collision shape count changed")
            for shape_index, (source_shape, target_shape) in enumerate(zip(robot_before["shapes"], robot_after["shapes"])):
                if ({k: v for k, v in source_shape.items() if k != "friction"}
                        != {k: v for k, v in target_shape.items() if k != "friction"}):
                    errors.append(prefix + ": robot shape %d non-friction property changed" % shape_index)
                if not math.isclose(float(target_shape["friction"]), robot_friction, rel_tol=0.0, abs_tol=1e-6):
                    errors.append(prefix + ": robot shape %d friction setting failed" % shape_index)
        source, target = original["actors"]["object"], current["actors"]["object"]
        if source["handle"] != target["handle"]:
            errors.append(prefix + ": object actor handle changed")
        if source["shapes"] != target["shapes"]:
            errors.append(prefix + ": object shape properties changed after construction")
        for shape_index, shape in enumerate(target["shapes"]):
            if not math.isclose(float(shape["friction"]), friction, rel_tol=0.0, abs_tol=1e-6):
                errors.append(prefix + ": incorrect object friction at shape %d" % shape_index)
        if len(source["bodies"]) != len(target["bodies"]) or not source["bodies"]:
            errors.append(prefix + ": object body count invalid or changed")
            continue
        for body_index, (before, after) in enumerate(zip(source["bodies"], target["bodies"])):
            body_prefix = prefix + ": object body %d " % body_index
            if before["mass"] <= 0 or before["invMass"] <= 0:
                errors.append(body_prefix + "must have positive nominal mass and invMass")
            if after["flags"] != before["flags"]:
                errors.append(body_prefix + "flags changed")
            for key, factor in (("mass", mass_scale), ("invMass", 1.0 / mass_scale),
                                ("inertia", mass_scale), ("invInertia", 1.0 / mass_scale),
                                ("com", 1.0)):
                if not close_tree(after[key], scaled_tree(before[key], factor), PHYSICS_TOLERANCES[key]):
                    errors.append(body_prefix + key + " scaling/invariance failed")
    return errors


def set_object_physics_once(env, args, cfg, output, gravity_spec):
    """Called exactly once after Grasp.create_sim(), before VecTask.prepare_sim."""
    before = detailed_physics(env)
    friction = float(cfg.task.env.objectFriction)
    write_json(output / "physics_before_mass_setting.json", {
        "note": "Raw simulator readback after official object-friction creation config, before object mass or robot-friction setters; never replaced by expected values.",
        "environments": before})
    setters = []
    for index, env_ptr in enumerate(env.envs):
        handle = env.gym.find_actor_handle(env_ptr, "object")
        original = before[index]["actors"]["object"]["bodies"]
        if args.object_mass_scale == 1.0:
            setters.append({"env_index": index, "called": False, "reason": "Identity mass scale; preserve official initialization"})
            continue
        props = env.gym.get_actor_rigid_body_properties(env_ptr, handle)
        for prop, reference in zip(props, original):
            if reference["mass"] <= 0 or reference["invMass"] <= 0:
                raise RuntimeError("Cannot scale non-dynamic/invalid object body")
            prop.mass = float(reference["mass"] * args.object_mass_scale)
        accepted = env.gym.set_actor_rigid_body_properties(env_ptr, handle, props, True)
        setters.append({"env_index": index, "called": True,
                        "recomputeInertia": True, "setter_returned_success": bool(accepted)})
    robot_friction_setters = []
    if args.robot_friction is not None:
        for index, env_ptr in enumerate(env.envs):
            handle = env.gym.find_actor_handle(env_ptr, "robot")
            shapes = env.gym.get_actor_rigid_shape_properties(env_ptr, handle)
            for shape in shapes:
                shape.friction = args.robot_friction
            accepted = env.gym.set_actor_rigid_shape_properties(env_ptr, handle, shapes)
            robot_friction_setters.append({"env_index": index, "shape_count": len(shapes),
                                           "friction": args.robot_friction,
                                           "setter_returned_success": bool(accepted)})
    after = detailed_physics(env)
    errors = physics_errors(before, after, args.object_mass_scale, friction, args.robot_friction)
    for setter in setters:
        if setter["called"] and not setter["setter_returned_success"]:
            errors.append("env %d: rigid-body setter returned failure" % setter["env_index"])
    for setter in robot_friction_setters:
        if not setter["setter_returned_success"]:
            errors.append("env %d: robot shape-friction setter returned failure" % setter["env_index"])
    setting = {
        "status": "verified_by_readback" if not errors else "verification_failed",
        "setting_stage": "Process-local Grasp subclass create_sim: after super().create_sim, before VecTask.prepare_sim; once only",
        "object_mass_scale": args.object_mass_scale, "object_shape_friction": friction,
        "robot_shape_friction_requested": args.robot_friction,
        "robot_friction_setter_calls": robot_friction_setters,
        "gravity_scale": args.gravity_scale,
        "gravity_specification": gravity_spec,
        "gravity_configured_m_s2": [float(value) for value in cfg.task.sim.gravity],
        "gravity_note": "Gravity multiplier is independent of mass multiplier; 5g is not mass times five.",
        "nominal_note": "Mass base is actor readback; object friction has already been set by official creation config.",
        "mass_policy": "All object bodies multiply mass by k, inertia by k, inverse quantities by 1/k; COM fixed.",
        "setter_policy": "recomputeInertia=True for nonidentity scale; no manual inertia correction or silent fallback.",
        "gpu_pipeline_note": "Set before prepare_sim, then validate readback after make and each reset. Attribute readback alone does not establish GPU dynamical response; a separate external-force control is required.",
        "gpu_dynamics_response_validated_here": False,
        "external_gpu_mass_response_control": external_mass_control_evidence(),
        "tolerances": PHYSICS_TOLERANCES, "setter_calls": setters, "errors": errors,
        "before": before, "after": after,
        "before_file_sha256": digest(output / "physics_before_mass_setting.json"),
    }
    write_json(output / "physics_setting.json", setting)
    if errors:
        raise RuntimeError("Physics setting failed readback: " + "; ".join(errors[:8]))
    return before


def verify_physics_after_reset(env, nominal, args, cfg, destination, stage="after_explicit_reset"):
    actual = detailed_physics(env)
    errors = physics_errors(nominal, actual, args.object_mass_scale, float(cfg.task.env.objectFriction), args.robot_friction)
    write_json(destination, {"status": "verified_by_readback" if not errors else "verification_failed",
                             "stage": stage, "check_only_no_setter_called": True, "errors": errors,
                             "tolerances": PHYSICS_TOLERANCES, "environments": actual})
    if errors:
        raise RuntimeError("Reset changed physical conditions: " + "; ".join(errors[:8]))


def verify_simulation_parameters(env, cfg, gravity_spec, destination, stage):
    """Read the simulator itself, not just the configuration or env cache."""
    params = env.gym.get_sim_params(env.sim)
    actual = {
        "gravity_m_s2": gym_value(params.gravity), "dt": float(params.dt),
        "substeps": int(params.substeps), "use_gpu_pipeline": bool(params.use_gpu_pipeline),
        "physx": {name: gym_value(getattr(params.physx, name)) for name in
                  ("solver_type", "use_gpu", "num_position_iterations", "num_velocity_iterations",
                   "contact_offset", "rest_offset", "max_gpu_contact_pairs")},
    }
    actual["physx"]["contact_collection"] = int(params.physx.contact_collection)
    errors = []
    if not close_tree(actual["gravity_m_s2"], gravity_spec["configured_m_s2"], {"rtol": 1e-6, "atol": 1e-6}):
        errors.append("Simulator gravity does not match scaled config")
    if int(actual["physx"]["contact_collection"]) != 0:
        errors.append("Contact collection must remain CC_NEVER (0)")
    if not math.isclose(actual["dt"], float(cfg.task.sim.dt), rel_tol=1e-6, abs_tol=1e-9):
        errors.append("Simulator dt changed")
    if actual["substeps"] != int(cfg.task.sim.substeps):
        errors.append("Simulator substeps changed")
    write_json(destination, {"status": "verified_by_readback" if not errors else "verification_failed",
                             "stage": stage, "gravity_specification": gravity_spec,
                             "actual_simulator_parameters": actual, "errors": errors,
                             "net_contact_force_valid_as_contact_evidence": False,
                             "contact_note": "contact_collection=0 disables contact-force collection; it does not disable physical collisions. Saved net_contact_force arrays are diagnostic storage only, not valid contact/force evidence."})
    if errors:
        raise RuntimeError("Simulator parameter check failed: " + "; ".join(errors))


CAPTURE = {}


def perturb_observation(args, obs, episode):
    import torch
    if args.pcl_noise <= 0 and not args.pcl_partial and args.pose_noise <= 0:
        return obs
    g = torch.Generator(device="cpu").manual_seed(int(args.seed) * 7907 + episode)
    out = obs.clone()
    n = obs.shape[0]
    pcl = out[:, 14:].reshape(n, -1, 3)
    if args.pcl_partial:
        view = torch.randn(n, 3, generator=g).to(obs.device)
        view = view / view.norm(dim=1, keepdim=True)
        center = pcl.mean(1, keepdim=True)
        keep = ((pcl - center) * view.unsqueeze(1)).sum(-1) > 0
        for i in range(n):
            idx = torch.nonzero(keep[i]).flatten()
            if len(idx) < 8:
                continue
            pick = idx[torch.randint(len(idx), (pcl.shape[1],), generator=g).to(obs.device)]
            pcl[i] = pcl[i, pick]
    if args.pcl_noise > 0:
        pcl += args.pcl_noise * torch.randn(pcl.shape, generator=g).to(obs.device)
    out[:, 14:] = pcl.reshape(n, -1)
    if args.pose_noise > 0:
        out[:, 7:10] += args.pose_noise * torch.randn(n, 3, generator=g).to(obs.device)
    return out


def install_capture(actor_critic):
    def hook(module, inputs, output):
        CAPTURE["features"] = inputs[0].detach().clone()
        CAPTURE["mean"] = output.detach().clone()
    actor_critic.actor_mean.register_forward_hook(hook)
    CAPTURE["log_std"] = actor_critic.log_std.detach().clone()


def apply_edit(args, env, frozen, episode):
    """Return (executed_action, info). Only the planner action is changed."""
    import torch
    kind = args.edit.split(":")[0]
    nh = int(env.num_active_hand_dofs)
    ref = env.tracking_reference["hand_qpos"]
    ref = ref[0] if ref.dim() == 3 else ref
    t_close = int(env.T_ref_start_lifting) - 1
    diff = ref[t_close] - ref[0]
    direction = torch.where(diff.abs() > 0.05, torch.sign(diff), torch.zeros_like(diff)).to(frozen.device)
    action = frozen.clone()
    info = {"kind": kind, "num_hand_dims": nh, "direction": direction,
            "active_direction_dims": int((direction != 0).sum().item())}
    if kind == "none":
        return action, info
    if kind == "model":
        path = args.edit.split(":", 1)[1]
        if "model" not in CAPTURE:
            CAPTURE["model"] = torch.jit.load(str(require_e(path)), map_location=frozen.device).eval()
            CAPTURE["model_sha256"] = digest(require_e(path))
        model = CAPTURE["model"]
        generator = torch.Generator(device="cpu").manual_seed(int(args.edit_seed) * 1000003 + episode)
        k = int(getattr(model, "num_candidates", 1))
        noise = torch.randn((frozen.shape[0], k, frozen.shape[1]), generator=generator).to(frozen.device)
        with torch.no_grad():
            action = model(CAPTURE["features"], frozen, noise).clamp(-1.0, 1.0)
        info.update(model=path, model_sha256=CAPTURE["model_sha256"], num_candidates=k,
                    mean_abs_change=float((action - frozen).abs().mean().item()))
        return action, info
    parts = args.edit.split(":")
    value = float(parts[1])
    info["value"] = value
    hand = slice(6, 6 + nh)
    if kind == "tighten":
        action[:, hand] = torch.clamp(action[:, hand] + value * direction.unsqueeze(0), -1.0, 1.0)
    elif kind == "scale":
        action[:, hand] = action[:, hand] * value
    elif kind == "policy":
        generator = torch.Generator(device="cpu").manual_seed(int(args.edit_seed) * 1000003 + episode)
        mean = CAPTURE["mean"]
        std = CAPTURE["log_std"].exp().unsqueeze(0)
        eps = torch.randn(mean.shape, generator=generator).to(mean.device)
        action = torch.tanh(mean + value * std * eps)
        info["policy_std_mean"] = float(std.mean().item())
    elif kind == "noise":
        generator = torch.Generator(device="cpu").manual_seed(int(args.edit_seed) * 1000003 + episode)
        noise = torch.randn(action.shape, generator=generator).to(action.device) * value
        if len(parts) > 2 and parts[2] == "hand":
            noise[:, :6] = 0
            info["noise_scope"] = "hand"
        else:
            info["noise_scope"] = "all"
        action = torch.clamp(action + noise, -1.0, 1.0)
    info["mean_abs_change"] = float((action - frozen).abs().mean().item())
    return action, info


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
    common = [
        "task=grasp", "train=PPOOneStep", "test=True",
        "num_envs=" + str(args.num_envs), "seed=" + str(args.seed),
        "headless=True", "force_render=False", "capture_video=False", "graphics_device_id=-1",
        "task.env.observationType=eefpose+objinitpose+objpcl", "task.env.armController=pose",
        "task.env.asset.multiObjectList=" + args.object_list,
        "task.env.randomizeTrackingReference=True", "task.env.randomizeGraspPose=True",
        "task.env.episodeLength=50",
        "task.env.enablePointCloud=True", "train.params.is_vision=True",
        "task.env.render.enable=False",
    ]
    if args.hand == "shadow":  # play_policy.sh "Shadow hand"
        overrides = ["hand=shadow_simple"] + common + [
            "task.env.resetDofPosRandomInterval=0",
            "task.env.trackingReferenceFile=tasks/grasp_ref_shadow.pkl",
            "task.env.trackingReferenceLiftTimestep=11"]
    else:  # play_policy.sh "FR3 + Inspire hand" (config default hand)
        overrides = ["hand=fr3_inspire_tac"] + common + [
            "task.env.trackingReferenceFile=tasks/grasp_ref_inspire.pkl",
            "task.env.trackingReferenceLiftTimestep=13"]
    overrides += list(args.override)
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
    base_gravity = [float(value) for value in cfg.task.sim.gravity]
    if len(base_gravity) != 3 or not all(math.isfinite(value) for value in base_gravity) or sum(value * value for value in base_gravity) == 0:
        raise ValueError("Resolved base gravity must be a finite nonzero three-vector")
    gravity_spec = {
        "base_m_s2": base_gravity,
        "scale": args.gravity_scale,
        "configured_m_s2": [value * args.gravity_scale for value in base_gravity],
        "definition": "Multiply the resolved config gravity vector; independent of object_mass_scale.",
    }
    cfg.task.sim.gravity = gravity_spec["configured_m_s2"]
    if int(cfg.task.sim.physx.contact_collection) != 0:
        raise ValueError("This protocol preserves the original contact_collection=0 setting")
    if args.object_friction is not None:
        cfg.task.env.objectFriction = args.object_friction
    if not math.isfinite(float(cfg.task.env.objectFriction)) or not 0 <= float(cfg.task.env.objectFriction) <= 2:
        raise ValueError("Effective object shape friction must be finite and between 0 and 2")
    object_list = require_e(ROOT / "assets" / cfg.task.env.asset.multiObjectList)
    if not object_list.is_file():
        raise FileNotFoundError(str(object_list))
    expected_hand = {"shadow": "shadow_simple", "inspire": "fr3_inspire_tac"}[args.hand]
    if args.checkpoint == str(ROOT / "ckpt/shadow.pt") and args.hand == "inspire":
        checkpoint = require_e(ROOT / "ckpt/inspire.pt")
        cfg.checkpoint = str(checkpoint)
    if cfg.hand.name != expected_hand or cfg.train.params.name != "ppo_onestep":
        raise ValueError("Hand preset/config mismatch or non one-step planner")
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
    external_mass_control = external_mass_control_evidence()
    if external_mass_control["available"]:
        references[external_mass_control["path"]] = external_mass_control["sha256"]
    write_json(output / "manifest.json", {
        "utc_started": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
        "hydra_overrides": overrides, "sources_sha256": sources_before,
        "references_sha256": references,
        "resolved_config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        "software": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                     "isaacgymenvs_module": str(Path(isaacgymenvs.__file__).resolve())},
        "scope": "Bounded physics-only Shadow probe; original policy/control/reward retained, no training or new method.",
        "baseline_probe_source_sha256": digest(ROOT / "research_shadow_probe.py"),
        "physics_probe_source_sha256": digest(ROOT / "research_shadow_physics_probe.py"),
        "external_gpu_mass_response_control": external_mass_control,
        "physics": {"object_mass_scale": args.object_mass_scale,
                    "object_shape_friction": float(cfg.task.env.objectFriction),
                    "robot_shape_friction_requested": args.robot_friction,
                    "gravity": gravity_spec,
                    "robot_friction_setter_stage": "Process-local create_sim after actor creation, before prepare_sim; only robot shape.friction changes",
                    "mass_setter_stage": "Process-local task create_sim after actor creation, before VecTask.prepare_sim",
                    "process_local_task_subclass_only": True, "tolerances": PHYSICS_TOLERANCES},
        "contact_collection": 0,
        "net_contact_force_valid_as_contact_evidence": False,
        "cold_vs_warm": "Official main pre-run reset retained; all rounds use explicit in-process resets.",
    })
    env = None
    try:
        from isaacgymenvs.tasks import isaacgym_task_map
        official_task_class = isaacgym_task_map[cfg.task_name]

        class ExtremeProbeGrasp(official_task_class):
            def create_sim(self):
                result = super().create_sim()
                try:
                    self._physics_probe_nominal = set_object_physics_once(self, args, cfg, output, gravity_spec)
                    verify_simulation_parameters(self, cfg, gravity_spec,
                                                 output / "sim_params_after_create_sim.json",
                                                 stage="after_create_sim_before_prepare_sim")
                except Exception:
                    # make() has not returned an env, so the outer finally cannot clean it.
                    self.gym.destroy_sim(self.sim)
                    self.sim = None
                    raise
                return result

        isaacgym_task_map[cfg.task_name] = ExtremeProbeGrasp
        try:
            env = isaacgymenvs.make(cfg.seed, cfg.task_name, cfg.task.env.numEnvs,
                                   cfg.sim_device, cfg.rl_device, cfg.graphics_device_id,
                                   cfg.headless, cfg.multi_gpu, cfg.capture_video,
                                   cfg.force_render, cfg)
        finally:
            isaacgym_task_map[cfg.task_name] = official_task_class
        physics_nominal = env._physics_probe_nominal
        verify_physics_after_reset(env, physics_nominal, args, cfg,
                                   output / "physics_after_make.json", stage="after_make_and_prepare_sim")
        verify_simulation_parameters(env, cfg, gravity_spec, output / "sim_params_after_make.json",
                                     stage="after_make_and_prepare_sim")
        # Retain run_rl_grasp.main's reset before build_runner; PPO.run resets again.
        ids = torch.arange(env.num_envs)
        env.reset_idx(ids)
        verify_physics_after_reset(env, physics_nominal, args, cfg,
                                   output / "physics_after_prerun_reset.json", stage="after_official_prerun_reset")
        verify_simulation_parameters(env, cfg, gravity_spec, output / "sim_params_after_prerun_reset.json",
                                     stage="after_official_prerun_reset")
        runner = build_runner(cfg, env)
        runner.actor_critic.eval()
        runner.actor_critic.requires_grad_(False)
        install_capture(runner.actor_critic)
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

        def lean_snapshot():
            return {"official_successes": cpu(env.successes),
                    "official_current_successes": cpu(env.current_successes),
                    "object_z": cpu(env.root_state_tensor[env.object_indices][:, 2]),
                    "object_init_z": cpu(env.object_init_states[:, 2]),
                    "has_hit_table": cpu(env.has_hit_table),
                    "progress_buf": cpu(env.progress_buf), "reset_buf": cpu(env.reset_buf)}

        def snapshot():
            if args.lean:
                return lean_snapshot()
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
            verify_physics_after_reset(env, physics_nominal, args, cfg,
                                       episode_dir / "physics_verification.json")
            verify_simulation_parameters(env, cfg, gravity_spec, episode_dir / "sim_params_after_reset.json",
                                         stage="after_episode_reset")
            states = env.get_state()
            initial = snapshot()
            initial["policy_observation"] = cpu(obs)
            initial["policy_state"] = cpu(states)
            initial["dof_target_initial"] = cpu(env.cur_targets)
            np.savez_compressed(str(episode_dir / "initial_state.npz"), **initial)
            write_json(episode_dir / "physical_properties_after_reset.json", physical_properties(env))
            obs_true = obs.clone()
            obs = perturb_observation(args, obs, episode)
            with torch.no_grad():
                frozen_action = runner.actor_critic(obs, states, inference=True)
                planner_action, edit_info = apply_edit(args, env, frozen_action, episode)
                env.generate_reaching_plan_idx(ids, actions=planner_action)
            tracking_reference = getattr(env, "current_tracking_reference", None)
            if not isinstance(tracking_reference, dict) or any(
                    not torch.is_tensor(tracking_reference.get(key))
                    for key in ("wrist_initobj_pos", "wrist_quat", "hand_qpos")):
                raise RuntimeError("Official generated tracking reference is missing expected tensors")
            np.savez_compressed(str(episode_dir / "planner.npz"),
                                policy_observation_seen=cpu(obs), policy_observation_true_delta=cpu(obs - obs_true),
                                planner_action=cpu(planner_action),
                                planner_action_frozen=cpu(frozen_action),
                                planner_features=cpu(CAPTURE["features"]),
                                planner_pre_tanh_mean=cpu(CAPTURE["mean"]),
                                planner_log_std=cpu(CAPTURE["log_std"]),
                                edit_direction=cpu(edit_info["direction"]),
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
                if args.lean:
                    sample.update({"returned_reset": cpu(reset)})
                    for key, value in sample.items():
                        records.setdefault(key, []).append(value)
                    continue
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
            np.savez_compressed(str(episode_dir / ("trajectory_lean.npz" if args.lean else "trajectory.npz")), **arrays)
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
                "trajectory_sha256": digest(episode_dir / ("trajectory_lean.npz" if args.lean else "trajectory.npz")),
                "edit": args.edit, "edit_info": {k: v for k, v in edit_info.items() if k != "direction"},
                "object_assets": [env.object_fns[i % len(env.object_fns)] for i in range(env.num_envs)],
                "planner_sha256": digest(episode_dir / "planner.npz"),
                "physical_properties_sha256": digest(episode_dir / "physical_properties_after_reset.json"),
                "physics_verification_sha256": digest(episode_dir / "physics_verification.json"),
                "sim_params_after_reset_sha256": digest(episode_dir / "sim_params_after_reset.json"),
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
            "physics_setting_readback_passed": True,
            "physics_after_make_and_reset_readback_passed": True,
            "gpu_dynamics_response_validated_here": False,
            "external_gpu_mass_response_control": external_mass_control,
            "object_mass_scale": args.object_mass_scale,
            "object_shape_friction": float(cfg.task.env.objectFriction),
            "robot_shape_friction_requested": args.robot_friction,
            "gravity": gravity_spec, "hand": args.hand, "edit": args.edit, "edit_seed": args.edit_seed,
            "lean": args.lean,
            "gravity_after_create_make_reset_readback_passed": True,
            "gravity_note": "Gravity times five is not object mass times five; inertial properties follow the separately recorded mass scale.",
            "contact_collection": 0,
            "net_contact_force_valid_as_contact_evidence": False,
            "contact_note": "Raw net_contact_force arrays are retained for schema compatibility only. CC_NEVER is active, so these arrays cannot establish contact presence, slip, grip force or force closure.",
            "artifact_sha256": {name: digest(output / name) for name in
                                ("manifest.json", "resolved_config.yaml", "loaded_object_urdf_sha256.json",
                                 "physics_before_mass_setting.json", "physics_setting.json",
                                 "physics_after_make.json", "physics_after_prerun_reset.json",
                                 "sim_params_after_create_sim.json", "sim_params_after_make.json",
                                 "sim_params_after_prerun_reset.json")},
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
