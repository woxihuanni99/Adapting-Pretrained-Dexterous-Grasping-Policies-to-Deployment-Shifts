"""Run any DemoGrasp research script under a deployment shift, without touching the script itself.

Usage (daemon style): research_shift_run.py --output OUT --shift=SPEC TARGET.py [target args ...]
SPEC = items joined by '|':
  lock=FF,MF        hardware fault: the listed Shadow fingers (FF, MF, RF, LF, TH) are stuck at their start pose
                    (their joint targets are held at the tracking reference's first frame for the whole episode)
  lockgroup=G       with lock: only envs whose group index (env // G) is even are faulty (mixture training)
  lockpos=R         with lock: hold the locked joints at q_min + R (q_max - q_min) instead of the start pose (R in [0,1])
  lockdebug=1       with lock: print the commanded and the actually reached positions of the locked joints (as fractions
                    of each joint's range) once per episode, for injection checks without looking at task success
  mass=S            object mass x S (inertia recomputed), set once after Grasp.create_sim (before prepare_sim)
  exbias=DX,DY,DZ   calibration error: the robot believes the object is DX,DY,DZ metres away from where it is;
                    both the observation (cloud + initial object position) and the execution frame are shifted
  exrand=SX,SY,SZ,G[,SEED]  unknown random calibration error: at every reset each group of G consecutive envs draws
                    one bias ~ U[-S, S]^3 (fixed within the episode, new every episode; dedicated CPU generator seeded
                    by SEED, default 12345, so all methods see the same sequence); added to exbias if both given
  excomp=CX,CY,CZ   fixed compensation of the execution frame only (the robot subtracts C from its believed object
                    position when planning/executing; observation unchanged) -- baseline for a known constant offset
  hand=inspire|allegro  run the target on another official DemoGrasp hand (FR3+Inspire / UR5+Allegro): rewrites the
                    target's hydra overrides (hand, tracking reference, lift step); pass that hand's checkpoint with
                    --init-checkpoint. Finger codes for lock: Inspire FF/MF/RF/LF/TH = index/middle/ring/little/thumb;
                    Allegro FF/MF/RF/TH = joints 0-3/4-7/8-11/12-15 (no LF).
Everything else (planner, reward, success check) is the target script's own. Patches only tasks.grasp.Grasp
(and, for hand=, hydra.compose).
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
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
        elif k == "lockpos":
            out["lockpos"] = float(v)
            if not 0.0 <= out["lockpos"] <= 1.0:
                raise SystemExit("lockpos must be in [0, 1]")
        elif k == "lockdebug":
            out["lockdebug"] = int(v)
        elif k == "mass":
            out["mass"] = float(v)
        elif k == "hand":
            if v not in ("inspire", "allegro"):
                raise SystemExit("unknown hand " + v)
            out["hand"] = v
        elif k == "exrand":
            vals = [float(x) for x in v.split(",")]
            if len(vals) not in (4, 5) or int(vals[3]) < 1:
                raise SystemExit("exrand needs SX,SY,SZ,G[,SEED]")
            out["exrand"] = {"s": vals[:3], "g": int(vals[3]), "seed": int(vals[4]) if len(vals) == 5 else 12345}
        elif k == "excomp":
            out["excomp"] = [float(x) for x in v.split(",")]
            if len(out["excomp"]) != 3:
                raise SystemExit("excomp needs 3 values")
        elif k == "exbias":
            out["exbias"] = [float(x) for x in v.split(",")]
            if len(out["exbias"]) != 3:
                raise SystemExit("exbias needs 3 values")
        else:
            raise SystemExit("unknown shift " + k)
    return out


HANDS = {"inspire": ("fr3_inspire_tac", "tasks/grasp_ref_inspire.pkl", 13),
         "allegro": ("ur5_allegro", "tasks/grasp_ref_allegro.pkl", 11)}
FNAME = {"FF": "index", "MF": "middle", "RF": "ring", "LF": "little", "TH": "thumb"}
ALLEGRO = {"FF": 0, "MF": 1, "RF": 2, "TH": 3}


def finger_of(name, f, hand):
    if hand == "inspire":
        return ("right_%s_" % FNAME[f]) in name
    if hand == "allegro":
        if not name.startswith("joint_") or f not in ALLEGRO:
            return False
        return int(float(name[len("joint_"):])) // 4 == ALLEGRO[f]
    return ("_%sJ" % f) in name


def install(sh):
    from isaacgym import gymapi  # noqa: F401
    import torch
    import tasks.grasp as tg
    G = tg.Grasp
    hand = sh.get("hand", "shadow")
    if hand != "shadow":
        import hydra
        hname, href, hlift = HANDS[hand]
        orig_compose = hydra.compose

        def compose(*a, **k):
            ov = []
            for o in k.get("overrides", []):
                if o == "hand=shadow_simple":
                    o = "hand=" + hname
                elif o.startswith("task.env.trackingReferenceFile="):
                    o = "task.env.trackingReferenceFile=" + href
                elif o.startswith("task.env.trackingReferenceLiftTimestep="):
                    o = "task.env.trackingReferenceLiftTimestep=%d" % hlift
                ov.append(o)
            if ("hand=" + hname) not in ov:
                raise RuntimeError("target script has no hand=shadow_simple override to rewrite")
            k["overrides"] = ov
            print("SHIFT hand", hname, href, hlift, flush=True)
            return orig_compose(*a, **k)
        hydra.compose = compose
    if "lock" in sh:
        orig_ref = G.compute_reference_actions

        def compute_reference_actions(self):
            action = orig_ref(self)
            if not hasattr(self, "_shift_lock_k"):
                idx = [int(i) for i in self.active_hand_dof_indices]
                names = [self.robot_dof_names[i] for i in idx]
                self._shift_lock_k = torch.tensor([k for k, n in enumerate(names) if any(finger_of(n, f, hand) for f in sh["lock"])],
                                                  dtype=torch.long, device=self.device)
                if self._shift_lock_k.numel() == 0:
                    raise RuntimeError("lock matched no hand dofs: %s" % names)
                self._shift_lo = self.robot_dof_lower_limits[idx]
                self._shift_hi = self.robot_dof_upper_limits[idx]
                print("SHIFT lock dofs:", [names[k] for k in self._shift_lock_k.tolist()], flush=True)
            ks = self._shift_lock_k
            hq = self.current_tracking_reference["hand_qpos"]
            q0 = hq[:, 0] if hq.dim() == 3 else hq[0].unsqueeze(0).expand(self.num_envs, -1)
            un = 2.0 * (q0 - self._shift_lo) / (self._shift_hi - self._shift_lo) - 1.0
            if sh.get("lockdebug"):
                self._shift_dbg_n = getattr(self, "_shift_dbg_n", 0) + 1
                if self._shift_dbg_n % 40 == 1:   # early in each episode: the start-pose (training) lock position
                    fr0 = ((un[:, ks] + 1.0) / 2.0)
                    print("LOCKDBG train_lock_fraction mean/min/max", [round(float(x), 4) for x in fr0.mean(0)],
                          [round(float(x), 4) for x in fr0.min(0).values], [round(float(x), 4) for x in fr0.max(0).values], flush=True)
                if self._shift_dbg_n % 40 == 30:  # late in the episode: where the locked joints actually are
                    idx_all = [int(i) for i in self.active_hand_dof_indices]
                    qa = self.robot_dof_pos[:, idx_all][:, ks]
                    fa = (qa - self._shift_lo[ks]) / (self._shift_hi[ks] - self._shift_lo[ks])
                    print("LOCKDBG reached_fraction mean/min/max", [round(float(x), 4) for x in fa.mean(0)],
                          [round(float(x), 4) for x in fa.min(0).values], [round(float(x), 4) for x in fa.max(0).values], flush=True)
            if sh.get("lockpos") is not None:
                un = torch.full_like(un, 2.0 * sh["lockpos"] - 1.0)
                if sh.get("lockdebug") and getattr(self, "_shift_dbg_n", 0) % 40 == 1:
                    print("LOCKDBG commanded_fraction", sh["lockpos"], flush=True)
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
    if "exbias" in sh or "exrand" in sh or "excomp" in sh:
        b0 = sh.get("exbias", [0.0, 0.0, 0.0])
        c0 = sh.get("excomp", [0.0, 0.0, 0.0])
        rnd = sh.get("exrand")
        rgen = torch.Generator(device="cpu")
        if rnd is not None:
            rgen.manual_seed(rnd["seed"])

        def _rb(self):
            if not hasattr(self, "_shift_rb"):
                self._shift_rb = torch.zeros(self.num_envs, 3)
            return self._shift_rb

        def obs_bias(self, dev, dt):
            bt = torch.tensor(b0, device=dev, dtype=dt).expand(self.num_envs, 3)
            return bt + _rb(self).to(device=dev, dtype=dt) if rnd is not None else bt

        def exec_bias(self, dev, dt):
            return obs_bias(self, dev, dt) - torch.tensor(c0, device=dev, dtype=dt)

        if rnd is not None:
            orig_reset = G.reset_idx

            def reset_idx(self, env_ids, *a, **k):
                rb = _rb(self)
                ids = torch.as_tensor(env_ids).detach().long().cpu().reshape(-1)
                if ids.numel():
                    ug, inv = torch.unique(ids // rnd["g"], return_inverse=True)
                    draw = (torch.rand(len(ug), 3, generator=rgen) * 2 - 1) * torch.tensor(rnd["s"])
                    rb[ids] = draw[inv]
                return orig_reset(self, env_ids, *a, **k)
            G.reset_idx = reset_idx
        orig_obs = G.compute_observations
        orig_ref2 = G.compute_reference_actions

        def compute_observations(self, *a, **k):
            res = orig_obs(self, *a, **k)
            n = self.obs_buf.shape[0]
            bt = obs_bias(self, self.obs_buf.device, self.obs_buf.dtype)
            self.obs_buf[:, 14:] = (self.obs_buf[:, 14:].view(n, -1, 3) + bt.unsqueeze(1)).view(n, -1)
            self.obs_buf[:, 7:10] += bt
            return res

        def compute_reference_actions(self):
            bt = exec_bias(self, self.device, self.object_init_states.dtype)
            self.object_init_states[:, 0:3] += bt
            try:
                return orig_ref2(self)
            finally:
                self.object_init_states[:, 0:3] -= bt
        orig_plan = G.generate_reaching_plan_idx

        def generate_reaching_plan_idx(self, env_ids, actions=None):
            bt = exec_bias(self, self.device, self.object_init_states.dtype)
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
