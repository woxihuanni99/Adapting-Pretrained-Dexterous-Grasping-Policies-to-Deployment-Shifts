"""DemoFunGrasp variant of research_shift_run.py: lives in the DemoGrasp folder (so the server daemon finds it) but
imports DemoFunGrasp's tasks.grasp and runs a DemoFunGrasp target (research_dfg_grpo.py). exbias is not supported here
(DemoFunGrasp's observation layout differs). Original docstring:
Run any DemoGrasp research script under a deployment shift, without touching the script itself.

Usage (daemon style): research_shift_run.py --output OUT --shift=SPEC TARGET.py [target args ...]
SPEC = items joined by '|':
  lock=FF,MF        hardware fault: the listed Shadow fingers (FF, MF, RF, LF, TH) are stuck at their start pose
                    (their joint targets are held at the tracking reference's first frame for the whole episode)
  lockgroup=G       with lock: only envs whose group index (env // G) is even are faulty (mixture training)
  mass=S            object mass x S (inertia recomputed), set once after Grasp.create_sim (before prepare_sim)
  exbias=DX,DY,DZ   calibration error: the robot believes the object is DX,DY,DZ metres away from where it is;
                    both the observation (cloud + initial object position) and the execution frame are shifted
Everything else (planner, reward, success check) is the target script's own. Patches only tasks.grasp.Grasp.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "DemoFunGrasp"
sys.dont_write_bytecode = True
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))


def parse(spec):
    out = {}
    for item in filter(None, spec.split("|")):
        k, v = item.split("=", 1)
        if k == "lock":
            out["lock"] = [f.strip().upper() for f in v.split(",") if f.strip()]
            if not all(f in ("FF", "MF", "RF", "LF", "TH") for f in out["lock"]):
                raise SystemExit("unknown finger in " + item)
        elif k == "lockgroup":
            out["lockgroup"] = int(v)
        elif k == "mass":
            out["mass"] = float(v)
        elif k == "exbias" and False:
            out["exbias"] = [float(x) for x in v.split(",")]
            if len(out["exbias"]) != 3:
                raise SystemExit("exbias needs 3 values")
        else:
            raise SystemExit("unknown shift " + k)
    return out


def install(sh):
    # DemoFunGrasp's tasks import open3d/termcolor; research_dfg_grpo.py stubs them before importing the task, so do the same
    import importlib.util, types
    for name, attrs in (("open3d", {}), ("termcolor", {"cprint": lambda *a, **k: print(a[0] if a else "")})):
        if importlib.util.find_spec(name) is None and name not in sys.modules:
            m = types.ModuleType(name)
            for k_, v_ in attrs.items():
                setattr(m, k_, v_)
            sys.modules[name] = m
    from isaacgym import gymapi  # noqa: F401
    import torch
    import tasks.grasp as tg
    G = tg.Grasp
    if "lock" in sh:
        orig_ref = G.compute_reference_actions

        def compute_reference_actions(self):
            action = orig_ref(self)
            if not hasattr(self, "_shift_lock_k"):
                idx = [int(i) for i in self.active_hand_dof_indices]
                names = [self.robot_dof_names[i] for i in idx]
                self._shift_lock_k = torch.tensor([k for k, n in enumerate(names) if any(("_%sJ" % f) in n for f in sh["lock"])],
                                                  dtype=torch.long, device=self.device)
                self._shift_lo = self.robot_dof_lower_limits[idx]
                self._shift_hi = self.robot_dof_upper_limits[idx]
                print("SHIFT lock dofs:", [names[k] for k in self._shift_lock_k.tolist()], flush=True)
            ks = self._shift_lock_k
            hq = self.current_tracking_reference["hand_qpos"]
            q0 = hq[:, 0] if hq.dim() == 3 else hq[0].unsqueeze(0).expand(self.num_envs, -1)
            un = 2.0 * (q0 - self._shift_lo) / (self._shift_hi - self._shift_lo) - 1.0
            if sh.get("lockgroup"):
                # only every other group of G consecutive envs is faulty (mixture training keeps nominal groups)
                rows = torch.nonzero((torch.arange(self.num_envs, device=self.device) // sh["lockgroup"]) % 2 == 0).squeeze(1)
                action[rows.unsqueeze(1), (self.hand_dof_start_idx + ks).unsqueeze(0)] = un[rows][:, ks]
            else:
                action[:, self.hand_dof_start_idx + ks] = un[:, ks]
            return action
        G.compute_reference_actions = compute_reference_actions
    if "mass" in sh:
        orig_cs = G.create_sim
        s = sh["mass"]

        def create_sim(self, *a, **k):
            res = orig_cs(self, *a, **k)
            ok = 0
            for env_ptr in self.envs:
                h = self.gym.find_actor_handle(env_ptr, "object")
                props = self.gym.get_actor_rigid_body_properties(env_ptr, h)
                for p in props:
                    p.mass = float(p.mass * s)
                ok += bool(self.gym.set_actor_rigid_body_properties(env_ptr, h, props, True))
            print("SHIFT mass x%s set on %d/%d envs" % (s, ok, len(self.envs)), flush=True)
            if ok != len(self.envs):
                raise RuntimeError("mass setter failed")
            return res
        G.create_sim = create_sim
    if "exbias" in sh:
        b = sh["exbias"]
        orig_obs = G.compute_observations
        orig_ref2 = G.compute_reference_actions

        def compute_observations(self, *a, **k):
            res = orig_obs(self, *a, **k)
            bt = torch.tensor(b, device=self.obs_buf.device, dtype=self.obs_buf.dtype)
            n = self.obs_buf.shape[0]
            self.obs_buf[:, 14:] = (self.obs_buf[:, 14:].view(n, -1, 3) + bt).view(n, -1)
            self.obs_buf[:, 7:10] += bt
            return res

        def compute_reference_actions(self):
            bt = torch.tensor(b, device=self.device, dtype=self.object_init_states.dtype)
            self.object_init_states[:, 0:3] += bt
            try:
                return orig_ref2(self)
            finally:
                self.object_init_states[:, 0:3] -= bt
        orig_plan = G.generate_reaching_plan_idx

        def generate_reaching_plan_idx(self, env_ids, actions=None):
            bt = torch.tensor(b, device=self.device, dtype=self.object_init_states.dtype)
            self.object_init_states[:, 0:3] += bt
            try:
                return orig_plan(self, env_ids, actions)
            finally:
                self.object_init_states[:, 0:3] -= bt
        G.compute_observations = compute_observations
        G.compute_reference_actions = compute_reference_actions
        G.generate_reaching_plan_idx = generate_reaching_plan_idx


if __name__ == "__main__":
    argv = sys.argv[1:]
    spec, out, rest = None, None, []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--shift="):
            spec = a.split("=", 1)[1]
        elif a == "--output":
            out = argv[i + 1]; i += 1
        else:
            rest.append(a)
        i += 1
    if spec is None or not rest or not rest[0].endswith(".py"):
        raise SystemExit("usage: research_shift_run.py --output OUT --shift=SPEC TARGET.py [args]")
    target = str(ROOT / rest[0])
    sh = parse(spec)
    import os
    os.chdir(str(ROOT))
    install(sh)
    print("SHIFT", spec, "->", rest[0], flush=True)
    sys.argv = [target] + (["--output", out] if out else []) + rest[1:]
    runpy.run_path(target, run_name="__main__")
