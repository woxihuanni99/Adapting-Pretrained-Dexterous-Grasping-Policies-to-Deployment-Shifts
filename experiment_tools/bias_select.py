#!/usr/bin/env python3
"""Development-set rate selection for the output-shift coordinate control (plan section 3; 2026-10-08). Reads ONLY development
results. Separately for each arm (raw, sigma): shared rate over LF and PCL; feasible if mean nominal change N_c - N0 >= -0.05 in both
conditions (unrounded; N0 = this machine's frozen 2048-episode dev nominal, bx_fmdet2k_munom_frozen); among feasible rates maximise the
mean of the two conditions' mean fault success; ties -> smaller worst-case nominal loss, then smaller rate. None feasible -> 'no feasible
configuration' (no grid extension, no threshold change, no seed top-up). --selftest uses synthetic scores."""
import argparse, json, statistics, sys
sys.path.insert(0, '/root/autodl-tmp/DexGraspResearch/tools'); import fs_common as F
ap = argparse.ArgumentParser(); ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
LABEL = {'LF': 'shlockLF', 'PCL': 'pn10'}
LRS = {'raw': ('3e-3', '1e-2', '3e-2'), 'sigma': ('1e-2', '3e-2', '1e-1')}
PRE = {'raw': 'bxr', 'sigma': 'bxs'}; LS = {'3e-3': '3em3', '1e-2': '1em2', '3e-2': '3em2', '1e-1': '1em1'}
if a.selftest:
    F.FAKE = lambda name, h: 0.86 + 0.03 * h if '2k' in name else 0.3 + 0.5 * h
    man = {'arms': {arm: {'%s|%s' % (c, lr): [{'tag': '%s_%s_%s_s%d' % (PRE[arm], c, LS[lr], k), 'short': '%s%s%ss%d' % (PRE[arm], c, LS[lr], k)}
                                              for k in (1, 2, 3)] for c in ('LF', 'PCL') for lr in LRS[arm]} for arm in LRS}}
else:
    man = json.load(open(F.ROOT + 'temp/bx_manifest.json'))
N0 = F.score('bx_fmdet2k_munom_frozen', 2048)
def stats(entries, c):
    S = [F.score('fmdet_munom_%s_%s_64x8' % (LABEL[c], e['short']), 512) for e in entries]
    N = [F.score('fmdet2k_munom_%s' % e['short'], 2048) for e in entries]
    return {'S_runs': S, 'N_runs': N, 'S': statistics.mean(S), 'N': statistics.mean(N), 'dN': statistics.mean(N) - N0, 'tags': [e['tag'] for e in entries]}
out = {'N0_this_machine': N0, 'rule': 'per arm; shared over LF and PCL; dN>=-0.05 both; max mean S; ties worst dN then smaller lr', 'arms': {}}
for arm, lrs in LRS.items():
    rec = {'rates': {}}
    for lr in lrs:
        st = {c: stats(man['arms'][arm]['%s|%s' % (c, lr)], c) for c in ('LF', 'PCL')}
        rec['rates'][lr] = {'per_condition': st, 'feasible': all(st[c]['dN'] >= -0.05 for c in st),
                            'mean_S': statistics.mean(st[c]['S'] for c in st), 'worst_dN': min(st[c]['dN'] for c in st)}
    feas = [lr for lr in lrs if rec['rates'][lr]['feasible']]
    feas.sort(key=lambda lr: (-rec['rates'][lr]['mean_S'], -rec['rates'][lr]['worst_dN'], float(lr)))
    rec['selected'] = feas[0] if feas else None
    rec['selected_at_grid_edge'] = (feas[0] in (lrs[0], lrs[-1])) if feas else None
    if not feas: rec['note'] = 'no feasible configuration'
    out['arms'][arm] = rec
    for lr, v in rec['rates'].items():
        print('%-5s lr %-5s feasible %-5s meanS %.4f worst dN %+.4f | %s' % (arm, lr, v['feasible'], v['mean_S'], v['worst_dN'],
              ' '.join('%s S %.3f dN %+.4f' % (c, v['per_condition'][c]['S'], v['per_condition'][c]['dN']) for c in ('LF', 'PCL'))))
    print('%-5s selected: %s' % (arm, rec['selected']))
print('N0 this machine %.5f' % N0)
if not a.selftest: print('sha256', F.dump(F.ROOT + 'temp/bx_select.json', out))
