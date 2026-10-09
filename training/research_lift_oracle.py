"""Diagnostic wrapper around research_grasp_collect.py (official deterministic protocol): changes ONLY the lift
timing of the tracked demonstration, to test whether low-friction failures come from lifting too early/fast.
DEX_LIFT_DWELL=d  : hold the grasp pose d extra control steps before the lift phase starts
DEX_LIFT_SCALE=s  : advance through the lift part of the reference at s steps per control step (s<1 = slower)
Everything else (planner, edits, success check at step max-2) is unchanged; use --override task.env.episodeLength=N
to give a slower lift enough time. The patched method is Grasp.compute_reference_actions with one line replaced.
"""
import inspect
import os
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    dwell = int(os.environ.get("DEX_LIFT_DWELL", "0"))
    scale = float(os.environ.get("DEX_LIFT_SCALE", "1.0"))
    for a in list(sys.argv):
        if a.startswith("--lift-dwell="):
            dwell = int(a.split("=", 1)[1]); sys.argv.remove(a)
        elif a.startswith("--lift-scale="):
            scale = float(a.split("=", 1)[1]); sys.argv.remove(a)
    import research_grasp_collect as rgc
    if hasattr(rgc, "configure_process_cache"):
        rgc.configure_process_cache(None)
    from isaacgym import gymapi  # noqa: F401
    import tasks.grasp as tg
    src = textwrap.dedent(inspect.getsource(tg.Grasp.compute_reference_actions))
    old = "tracking_timestep_ids = (self.progress_buf - self.reaching_plan_timesteps).clamp(min=0, max=self.T_ref-1)"
    if src.count(old) != 1:
        raise SystemExit("upstream compute_reference_actions changed; refusing to patch")
    line = [l for l in src.splitlines() if old in l][0]
    ind = line[:len(line) - len(line.lstrip())]
    new = ("_k = (self.progress_buf - self.reaching_plan_timesteps).clamp(min=0)\n"
           + ind + "_L = self.T_ref_start_lifting - 1\n"
           + ind + "_after = (_k - _L - %d).clamp(min=0).float()\n"
           + ind + "tracking_timestep_ids = torch.where(_k < _L + %d, _k.clamp(max=_L), _L + torch.floor(_after * %r).long()).clamp(min=0, max=self.T_ref-1)"
           ) % (dwell, dwell, scale)
    src = src.replace(old, new)
    ns = {}
    exec(compile(src, tg.__file__ + " (compute_reference_actions, lift timing dwell=%d scale=%s)" % (dwell, scale), "exec"), tg.__dict__, ns)
    tg.Grasp.compute_reference_actions = ns["compute_reference_actions"]
    print("LIFT_ORACLE dwell=%d scale=%s" % (dwell, scale), flush=True)
    rgc.main()
