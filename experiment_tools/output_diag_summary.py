#!/usr/bin/env python3
"""Summaries of the output analysis (plan: reports/补充实验锁定说明_偏置坐标对照_1008.md, section 5), computed only from the saved
common inputs (inputs.npz, env_info.json) and per-model adapted means (models.npz) written by research_output_diag.py.

Per model (one trained run):
  dm = m(s) - m0(s) (pre-tanh), over the same 2,048 states
  kl_raw   = mean_s 1/2 sum_j (dm_j / sigma0_j)^2           ("sigma0-normalised mean shift"; not a full policy KL)
  kl_floor = mean_s 1/2 sum_j (dm_j / max(sigma0_j, .05))^2  (the floor used by the KL baseline)
  l1       = sum_j mean_s |dm_j|
  high_share = sum_{j in H} mean|dm_j| / l1, H = the 12 dims with the largest frozen sigma0 (absolute changes)
  mean_dm  (signed, per dim); cos_ref = cosine(mean_dm, mean over the reference group's runs of mean_dm)
  state_share (primary)   = sum_j Var_s(dm_j) / sum_j E_s[dm_j^2]
  state_share_w (secondary) = same with every dim weighted by 1/sigma0_j^2
Action level (control targets, not reached joint angles): hand target q = clip(q_ref + tanh(u_hand) * range, lower, upper);
dq = q(m) - q(m0) per joint; mean |dq| per joint and summed; clip fraction = share of state x joint entries whose unclipped
target lies outside the joint limits; wrist translation (m) and rotation (rad) changes = (tanh(m) - tanh(m0)) * range.
Groups are summarised over training runs (mean and run-level SD); states are never treated as replicates."""
import argparse, json, re, statistics
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--dir', required=True)
ap.add_argument('--ref', default='rgLF')
ap.add_argument('--out', required=True)
a = ap.parse_args()
D = Path(a.dir)
info = json.loads((D / 'env_info.json').read_text())
inp = np.load(D / 'inputs.npz')
M = np.load(D / 'models.npz')
m0 = inp['m0'].astype(np.float64); s0 = np.array(info['sigma0']); sf = np.maximum(s0, 0.05)
H = np.argsort(-s0)[:12]
names = ['wrist_tx', 'wrist_ty', 'wrist_tz', 'wrist_rx', 'wrist_ry', 'wrist_rz'] + info['hand_joint_names']
rng = np.array(info['grasp_pose_range'], dtype=float); rng = rng if rng.ndim else np.full(len(info['hand_joint_names']), float(rng))
lo, hi, qref = np.array(info['hand_lower']), np.array(info['hand_upper']), np.array(info['q_ref'])
wr = np.array(info['wrist_ranges'], dtype=float)


def hand_targets(u):
    raw = qref + np.tanh(u[:, 6:]) * rng
    return np.clip(raw, lo, hi), (raw < lo) | (raw > hi)


q0, c0 = hand_targets(m0)
per = {}
for label in M.files:
    m = M[label].astype(np.float64); dm = m - m0
    e2 = (dm ** 2).mean(0); var = dm.var(0); mdm = dm.mean(0)
    q, cl = hand_targets(m); dq = q - q0
    w = (np.tanh(m[:, :6]) - np.tanh(m0[:, :6])) * wr
    per[label] = {
        'kl_raw': float((0.5 * (dm / s0) ** 2).sum(1).mean()), 'kl_floor': float((0.5 * (dm / sf) ** 2).sum(1).mean()),
        'l1': float(np.abs(dm).mean(0).sum()), 'high_share': float(np.abs(dm).mean(0)[H].sum() / np.abs(dm).mean(0).sum()),
        'mean_dm': mdm.tolist(), 'mean_abs_dm': np.abs(dm).mean(0).tolist(),
        'state_share': float(var.sum() / e2.sum()), 'state_share_w': float((var / s0 ** 2).sum() / (e2 / s0 ** 2).sum()),
        'hand_mean_abs_dq': np.abs(dq).mean(0).tolist(), 'hand_mean_dq': dq.mean(0).tolist(), 'hand_sum_abs_dq': float(np.abs(dq).mean(0).sum()),
        'clip_frac': float(cl.mean()), 'clip_frac_frozen': float(c0.mean()),
        'wrist_trans_mean_abs_m': np.abs(w[:, :3]).mean(0).tolist(), 'wrist_rot_mean_abs_rad': np.abs(w[:, 3:]).mean(0).tolist()}
group = lambda lab: re.sub(r'_s\d+$', '', lab)
groups = {}
for lab in per:
    groups.setdefault(group(lab), []).append(lab)
ref = np.mean([per[l]['mean_dm'] for l in groups[a.ref]], 0)
for lab, r in per.items():
    v = np.array(r['mean_dm']); r['cos_ref'] = float(v @ ref / (np.linalg.norm(v) * np.linalg.norm(ref) + 1e-12))
dom = int(np.argmax(np.abs(ref)))
summ = {}
for g, labs in groups.items():
    def agg(key, f=lambda x: x):
        xs = [f(per[l][key]) for l in labs]
        return {'mean': statistics.mean(xs), 'sd': statistics.stdev(xs) if len(xs) > 1 else None, 'runs': xs}
    summ[g] = {'n_runs': len(labs), 'kl_raw': agg('kl_raw'), 'kl_floor': agg('kl_floor'), 'l1': agg('l1'), 'high_share': agg('high_share'),
               'cos_ref': agg('cos_ref'), 'state_share': agg('state_share'), 'state_share_w': agg('state_share_w'),
               'dom_mean_dm': agg('mean_dm', lambda v: v[dom]), 'dom_mean_abs_dm': agg('mean_abs_dm', lambda v: v[dom]),
               'dom_mean_dq_rad': agg('hand_mean_dq', lambda v: v[dom - 6]) if dom >= 6 else None,
               'hand_sum_abs_dq': agg('hand_sum_abs_dq'), 'clip_frac': agg('clip_frac')}
out = {'n_states': int(m0.shape[0]), 'high_sigma0_dims': sorted(int(j) for j in H), 'dim_names': names,
       'reference_group': a.ref, 'dominant_dim': dom, 'dominant_dim_name': names[dom], 'sigma0_dominant': float(s0[dom]),
       'clip_frac_frozen': float(c0.mean()), 'groups': summ, 'per_model': per}
Path(a.out).write_text(json.dumps(out, indent=1))
print('states', out['n_states'], '| dominant dim', dom, names[dom], 'sigma0 %.3f' % s0[dom], '| frozen clip frac %.3f' % c0.mean())
print('%-12s %2s %8s %8s %6s %5s %6s %6s %6s %8s %8s %7s %6s' % ('group', 'n', 'KLraw', 'KLfloor', 'L1', 'hiS', 'cos', 'stS', 'stSw', 'dom_dm', 'dom_dq', 'sum|dq|', 'clip'))
for g, s in summ.items():
    f = lambda k: s[k]['mean'] if s[k] else float('nan')
    print('%-12s %2d %8.1f %8.1f %6.2f %5.2f %6.2f %6.2f %6.2f %8.2f %8.3f %7.3f %6.3f' % (g, s['n_runs'], f('kl_raw'), f('kl_floor'), f('l1'), f('high_share'),
          f('cos_ref'), f('state_share'), f('state_share_w'), f('dom_mean_dm'), f('dom_mean_dq_rad'), f('hand_sum_abs_dq'), f('clip_frac')))
