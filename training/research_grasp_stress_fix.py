"""research_grasp_stress.py with a one-line fix of an upstream DemoGrasp bug in Grasp.reset_idx.

In the distractor branch, upstream indexes root_state_tensor with an int32 tensor
(`self.distractor_object_indices[env_ids].view(-1).to(torch.int32)`), which torch 1.12 rejects
("tensors used as indices must be long"). The patched method is the upstream source with only that
line changed to `.long()` (the later torch.cat(...).to(torch.int32) for the simulator call is kept).
Only needed for --override task.env.asset.useDistractorObjects=True; all arguments are passed through.
"""
import inspect
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    import research_grasp_stress  # stdlib-only at import time
    research_grasp_stress.configure_process_cache(None)  # same E-drive caches as the probe, before the simulator import
    from isaacgym import gymapi  # noqa: F401  (must precede torch)
    import tasks.grasp as tg
    src = textwrap.dedent(inspect.getsource(tg.Grasp.reset_idx))
    old = "distractor_object_indices = self.distractor_object_indices[env_ids].view(-1).to(torch.int32)"
    if src.count(old) != 1:
        raise SystemExit("upstream reset_idx changed; refusing to patch")
    src = src.replace(old, "distractor_object_indices = self.distractor_object_indices[env_ids].view(-1).long()")
    ns = {}
    exec(compile(src, tg.__file__ + " (reset_idx, distractor index dtype fix)", "exec"), tg.__dict__, ns)
    tg.Grasp.reset_idx = ns["reset_idx"]
    research_grasp_stress.main()
