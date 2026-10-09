"""PPO fine-tuning baseline under point-cloud noise (fair counterpart of research_grpo_finetune_v2.py --pcl-noise).
Wraps research_ppo_finetune.py unchanged and adds iid Gaussian noise (--pcl-noise=SIGMA, metres) to the object
point-cloud part of every observation the policy sees (obs[:, 14:], the same slice perturbed by
research_grasp_stress.py and research_grpo_finetune_v2.py). Physics are unchanged. Only Grasp.compute_observations
is wrapped; use --mus 1.0 for nominal friction (grasp.yaml objectFriction = 1.0).
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    sigma = 0.0
    for a in list(sys.argv):
        if a.startswith("--pcl-noise="):
            sigma = float(a.split("=", 1)[1]); sys.argv.remove(a)
    import os
    os.chdir(str(ROOT))
    from isaacgym import gymapi  # noqa: F401
    import torch
    import tasks.grasp as tg
    import research_ppo_finetune as rpf
    orig = tg.Grasp.compute_observations

    def compute_observations(self, *a, **k):
        out = orig(self, *a, **k)
        if sigma > 0:
            self.obs_buf[:, 14:] += sigma * torch.randn_like(self.obs_buf[:, 14:])
        return out

    tg.Grasp.compute_observations = compute_observations
    print("PPO_PCL_NOISE sigma=%s" % sigma, flush=True)
    rpf.main()
