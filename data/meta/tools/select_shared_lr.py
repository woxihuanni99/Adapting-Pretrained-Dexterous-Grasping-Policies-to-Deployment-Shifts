#!/usr/bin/env python3
"""Shared learning-rate selection (protocol agreed with the advisor on 2026-09-29).
Four methods (full/residual x PPO/GRPO); selection conditions LF (little-finger lock) and PCL (1 cm point-cloud noise).
  N  : nominal dev success, 2048 episodes (fmdet2k_*); N0 = frozen planner on the same 2048-episode set
  S  : fault dev success, 512 episodes (fmdet_munom_<cond>_*)
  feasible : for EACH selection condition, mean_seeds N_c(a) - N0 >= -delta (unrounded); delta 0.05 main, 0.10 supplementary
  objective: maximise (S_LF + S_PCL)/2 ; ties -> smaller max nominal loss, then smaller learning rate
Two stages (one-shot seed top-up rule):
  --stage trigger : with seeds s1-s3, every candidate (method, lr) with |dN_c + 0.05| <= 0.01 on ANY selection condition
                    is put on the trigger list (written once to temp/shared_lr_trigger.json; refuses to overwrite).
                    Triggered candidates get seeds s4,s5 on BOTH selection conditions; no further top-ups.
  --stage select  : triggered candidates use s1-s5, others s1-s3; writes temp/shared_lr_selection.json.
Sensitivity (reported, never replaces the main choice): recovery-ratio objective R_c = (S_c - F_c)/(N0 - F_c), same feasible set.
Reference: per-condition selection (constraint on that condition only) and the ideal-switching table for LF."""
import argparse, json, os
ROOT = '/root/autodl-tmp/DexGraspResearch/'
R = ROOT + 'runs/'
C = {
 "full_grpo": {
  "LF": {
   "1e-5": [
    "shlockLFgrpos1",
    "shlockLFgrpos2",
    "shlockLFgrpos3"
   ],
   "3e-5": [
    "shlockLFgrpolr3s1",
    "shlockLFgrpolr3s2",
    "shlockLFgrpolr3s3"
   ],
   "1e-4": [
    "shlockLFgrpolr10s1",
    "shlockLFgrpolr10s2",
    "shlockLFgrpolr10s3"
   ]
  },
  "PCL": {
   "1e-5": [
    "pcl1cms1",
    "pcl1cms2",
    "pcl1cms3"
   ],
   "3e-5": [
    "pcl1cmgrpolr3s1",
    "pcl1cmgrpolr3s2",
    "pcl1cmgrpolr3s3"
   ],
   "1e-4": [
    "pcl1cmgrpolr10s1",
    "pcl1cmgrpolr10s2",
    "pcl1cmgrpolr10s3"
   ]
  }
 },
 "full_ppo": {
  "LF": {
   "1e-5": [
    "shlockLFppolr1s1",
    "shlockLFppolr1s2",
    "shlockLFppolr1s3"
   ],
   "3e-5": [
    "shlockLFppos1",
    "shlockLFppos2",
    "shlockLFppos3"
   ],
   "1e-4": [
    "shlockLFppolr10s1",
    "shlockLFppolr10s2",
    "shlockLFppolr10s3"
   ]
  },
  "PCL": {
   "1e-5": [
    "pcl1cmppolr1s1",
    "pcl1cmppolr1s2",
    "pcl1cmppolr1s3"
   ],
   "3e-5": [
    "ppopcls1",
    "ppopcls2",
    "ppopcls3"
   ],
   "1e-4": [
    "pcl1cmppolr10s1",
    "pcl1cmppolr10s2",
    "pcl1cmppolr10s3"
   ]
  }
 },
 "res_ppo": {
  "LF": {
   "1e-4": [
    "shlockLFplugppolr1e4s1",
    "shlockLFplugppolr1e4s2",
    "shlockLFplugppolr1e4s3"
   ],
   "3e-4": [
    "shlockLFplugppos1",
    "shlockLFplugppos2",
    "shlockLFplugppos3"
   ],
   "1e-3": [
    "shlockLFplugppolr1e3s1",
    "shlockLFplugppolr1e3s2",
    "shlockLFplugppolr1e3s3"
   ],
   "3e-3": [
    "shlockLFplugppolr3e3s1",
    "shlockLFplugppolr3e3s2",
    "shlockLFplugppolr3e3s3"
   ],
   "1e-2": [
    "shlockLFplugppolr1e2s1",
    "shlockLFplugppolr1e2s2",
    "shlockLFplugppolr1e2s3"
   ]
  },
  "PCL": {
   "1e-4": [
    "pcl1cmplugppolr1e4s1",
    "pcl1cmplugppolr1e4s2",
    "pcl1cmplugppolr1e4s3"
   ],
   "3e-4": [
    "pcl1cmplugppos1",
    "pcl1cmplugppos2",
    "pcl1cmplugppos3"
   ],
   "1e-3": [
    "pcl1cmplugppolr1e3s1",
    "pcl1cmplugppolr1e3s2",
    "pcl1cmplugppolr1e3s3"
   ],
   "3e-3": [
    "pcl1cmplugppolr3e3s1",
    "pcl1cmplugppolr3e3s2",
    "pcl1cmplugppolr3e3s3"
   ],
   "1e-2": [
    "pcl1cmplugppolr1e2s1",
    "pcl1cmplugppolr1e2s2",
    "pcl1cmplugppolr1e2s3"
   ]
  }
 },
 "res_grpo": {
  "LF": {
   "1e-4": [
    "shlockLFpluggrpolr1e4s1",
    "shlockLFpluggrpolr1e4s2",
    "shlockLFpluggrpolr1e4s3"
   ],
   "3e-4": [
    "shlockLFpluggrpos1",
    "shlockLFpluggrpos2",
    "shlockLFpluggrpos3"
   ],
   "1e-3": [
    "shlockLFpluggrpolr1e3s1",
    "shlockLFpluggrpolr1e3s2",
    "shlockLFpluggrpolr1e3s3"
   ],
   "3e-3": [
    "shlockLFpluggrpolr3e3s1",
    "shlockLFpluggrpolr3e3s2",
    "shlockLFpluggrpolr3e3s3"
   ],
   "1e-2": [
    "shlockLFpluggrpolr1e2s1",
    "shlockLFpluggrpolr1e2s2",
    "shlockLFpluggrpolr1e2s3"
   ]
  },
  "PCL": {
   "1e-4": [
    "pcl1cmpluggrpolr1e4s1",
    "pcl1cmpluggrpolr1e4s2",
    "pcl1cmpluggrpolr1e4s3"
   ],
   "3e-4": [
    "pcl1cmpluggrpos1",
    "pcl1cmpluggrpos2",
    "pcl1cmpluggrpos3"
   ],
   "1e-3": [
    "pcl1cmpluggrpolr1e3s1",
    "pcl1cmpluggrpolr1e3s2",
    "pcl1cmpluggrpolr1e3s3"
   ],
   "3e-3": [
    "pcl1cmpluggrpolr3e3s1",
    "pcl1cmpluggrpolr3e3s2",
    "pcl1cmpluggrpolr3e3s3"
   ],
   "1e-2": [
    "pcl1cmpluggrpolr1e2s1",
    "pcl1cmpluggrpolr1e2s2",
    "pcl1cmpluggrpolr1e2s3"
   ]
  }
 }
}
FAULT = {'LF': 'fmdet_munom_shlockLF_%s_64x8', 'PCL': 'fmdet_munom_pn10_%s_64x8'}
FROZEN_FAULT = {'LF': 'fmdet_munom_shlockLF_frozen_64x8', 'PCL': 'fmdet_munom_pn10_frozen_64x8'}
TRIG = ROOT + 'temp/shared_lr_trigger.json'
ap = argparse.ArgumentParser(); ap.add_argument('--stage', choices=('trigger', 'select'), required=True); a = ap.parse_args()
def sr(name):
    f = R + name + '/result.json'
    return json.load(open(f))['totals']['success_rate'] if os.path.exists(f) else None
def seeds(shorts, n):
    base = shorts[-1]
    assert base.endswith('s3'), base
    return shorts + [base[:-1] + str(k) for k in range(4, n + 1)] if n > 3 else shorts
N0 = sr('fmdet2k_munom_frozen')
F = {c: sr(v) for c, v in FROZEN_FAULT.items()}
def load_trig():  # 2026-09-29 fix: the first trigger run wrote its full report to the same path; the list is its 'trigger_list' field
    t = json.load(open(TRIG))
    return t['trigger_list'] if isinstance(t, dict) else t
trig = load_trig() if (a.stage == 'select' and os.path.exists(TRIG)) else []
out = {'stage': a.stage, 'N0_frozen_2048': N0, 'F_frozen_fault': F, 'trigger_list': trig, 'methods': {}, 'missing': []}
new_trig = []
for m, cc in C.items():
    table = {}
    for c, g in cc.items():
        for lr, shorts in g.items():
            n = 5 if [m, lr] in trig else 3
            ss = seeds(shorts, n)
            N = [sr('fmdet2k_munom_%s' % s) for s in ss]
            S = [sr(FAULT[c] % s) for s in ss]
            out['missing'] += [nm for nm, v in zip(['fmdet2k_munom_%s' % s for s in ss] + [FAULT[c] % s for s in ss], N + S) if v is None]
            ok = None not in N and None not in S
            row = table.setdefault(lr, {})
            row[c] = {'seeds': ss, 'N_seeds': N, 'S_seeds': S, 'N': sum(N) / len(N) if ok else None, 'S': sum(S) / len(S) if ok else None}
            if ok and N0 is not None:
                row[c]['dN'] = row[c]['N'] - N0
                if a.stage == 'trigger' and abs(row[c]['dN'] + 0.05) <= 0.01 and [m, lr] not in new_trig:
                    new_trig.append([m, lr])
    res = {'table': table}
    for delta in (0.05, 0.10):
        key = 'd%02d' % round(delta * 100)
        feas, feas_ratio, per = [], [], {'LF': [], 'PCL': []}
        for lr, row in table.items():
            if any(row.get(c, {}).get('dN') is None for c in ('LF', 'PCL')):
                continue
            dN = {c: row[c]['dN'] for c in ('LF', 'PCL')}
            for c in ('LF', 'PCL'):
                row[c]['critical'] = abs(dN[c] + 0.05) <= 0.01
                if dN[c] >= -delta:
                    per[c].append((row[c]['S'], -float(lr), lr))
            if all(dN[c] >= -delta for c in ('LF', 'PCL')):
                feas.append(((row['LF']['S'] + row['PCL']['S']) / 2, min(dN.values()), -float(lr), lr))
                if None not in F.values():
                    rr = [(row[c]['S'] - F[c]) / (N0 - F[c]) for c in ('LF', 'PCL')]
                    feas_ratio.append((sum(rr) / 2, min(dN.values()), -float(lr), lr))
        feas.sort(reverse=True); feas_ratio.sort(reverse=True)
        res[key] = {'shared_lr': feas[0][3] if feas else 'no feasible shared learning rate',
                    'objective': feas[0][0] if feas else None,
                    'sensitivity_recovery_ratio_lr': feas_ratio[0][3] if feas_ratio else None,
                    'per_condition_reference': {c: (max(per[c])[2] if per[c] else 'no feasible') for c in per}}
    # ideal switching reference on LF: nominal = frozen by routing; fault = main config / unconstrained dev-best
    lf = [(row['LF']['S'], -float(lr), lr) for lr, row in table.items() if row.get('LF', {}).get('S') is not None]
    res['switch_LF'] = {'keep_main_config': res['d05']['shared_lr'], 'unconstrained_dev_best': max(lf)[2] if lf else None}
    out['methods'][m] = res
if a.stage == 'trigger':
    if os.path.exists(TRIG):
        raise SystemExit('trigger list already fixed: ' + TRIG)
    json.dump(new_trig, open(TRIG, 'w'))
    out['trigger_list'] = new_trig
json.dump(out, open(ROOT + ('temp/shared_lr_select.json' if a.stage == 'select' else 'temp/shared_lr_trigger_report.json'), 'w'), indent=1)
print('stage', a.stage, '| N0 =', N0, '| frozen fault', F, '| missing', len(out['missing']))
if a.stage == 'trigger':
    print('TRIGGERED (method, lr):', new_trig)
for m, res in out['methods'].items():
    print('==', m, '| main d05:', res['d05'], '| supp d10:', res['d10'], '| switch LF:', res['switch_LF'])
    for lr, row in sorted(res['table'].items(), key=lambda kv: float(kv[0])):
        print('   lr %-5s' % lr, '  '.join('%s n=%d S=%s dN=%s%s' % (c, len(row[c]['seeds']), None if row[c]['S'] is None else '%.3f' % row[c]['S'],
              '%+.4f' % row[c]['dN'] if 'dN' in row[c] else '-', ' CRIT' if row[c].get('critical') else '') for c in ('LF', 'PCL') if c in row))
