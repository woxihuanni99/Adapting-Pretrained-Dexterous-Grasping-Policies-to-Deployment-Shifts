"""Diagnostic wrapper around research_grasp_collect.py (official deterministic protocol): after the planner's action
has been turned into a tracking reference, the finger joint targets of the LIFT phase are pushed further along the
closing direction, q' = q + alpha * (q - q0) (q0 = pre-grasp finger pose at t=0, clamped to joint limits), so the
PD controller squeezes harder while the object is lifted. --squeeze-from=grasp applies it from the grasp frame
(T_lift-1) instead, i.e. a deeper grasp. Tests whether low-friction failures are fixed by more normal force at the
contacts (unlike task.env.pdParamScale, which stiffens every joint). Only Grasp.generate_reaching_plan_idx is wrapped.
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    alpha = float(os.environ.get("DEX_SQUEEZE", "0"))
    start = "lift"
    for a in list(sys.argv):
        if a.startswith("--squeeze="):
            alpha = float(a.split("=", 1)[1]); sys.argv.remove(a)
        elif a.startswith("--squeeze-from="):
            start = a.split("=", 1)[1]; sys.argv.remove(a)
    if start not in ("lift", "grasp"):
        raise SystemExit("--squeeze-from must be lift or grasp")
    import research_grasp_collect as rgc
    if hasattr(rgc, "configure_process_cache"):
        rgc.configure_process_cache(None)
    from isaacgym import gymapi  # noqa: F401
    import torch
    import tasks.grasp as tg
    orig = tg.Grasp.generate_reaching_plan_idx

    def generate_reaching_plan_idx(self, env_ids, actions=None):
        out = orig(self, env_ids, actions)
        if alpha != 0.0 and self.randomize_tracking_reference:
            hq = self.current_tracking_reference["hand_qpos"]
            L = self.T_ref_start_lifting
            s = L if start == "lift" else L - 1
            q0 = hq[:, 0:1]
            new = hq[:, s:] + alpha * (hq[:, s:] - q0)
            lo = self.robot_dof_lower_limits[self.active_hand_dof_indices]
            hi = self.robot_dof_upper_limits[self.active_hand_dof_indices]
            hq[:, s:] = torch.max(torch.min(new, hi), lo)
        return out

    tg.Grasp.generate_reaching_plan_idx = generate_reaching_plan_idx
    print("SQUEEZE_ORACLE alpha=%s from=%s" % (alpha, start), flush=True)
    rgc.main()
