#!/usr/bin/env python3
"""Development-set rate selection for the matched residual GRPO (plan section C). Reads ONLY development results.
Shared rate over LF and PCL: feasible if mean nominal change N_c - N0 >= -0.05 in both conditions (unrounded; N0 = this machine's
frozen 2048-episode dev nominal, fs_fmdet2k_munom_frozen); among feasible rates maximise the mean of the two conditions' mean fault
success; ties -> smaller worst-case nominal loss, then smaller rate. None feasible -> 'no feasible configuration' (no grid
extension, no threshold change, no seed top-up). Also reports the new full-weight GRPO chains. --selftest uses synthetic scores."""
import argparse, json, statistics, sys
sys.path.insert(0, '/root/autodl-tmp/DexGraspResearch/tools'); import fs_common as F
ap = argparse.ArgumentParser(); ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
LABEL = {'LF': 'shlockLF', 'PCL': 'pn10'}
if a.selftest:
    F.FAKE = lambda name, h: 0.86 + 0.03 * h if '2k' in name else 0.3 + 0.5 * h
    man = {'fsr': {'%s|%s' % (c, lr): [{'tag': 'fsr_%s_%s_s%d' % (c, s, k), 'short': 'fsr%s%ss%d' % (c, s, k)} for k in (1, 2, 3)]
                   for c in ('LF', 'PCL') for lr, s in (('1e-3', '1em3'), ('3e-3', '3em3'))},
           'fsf': {c: [{'tag': 'fsf_%s_s%d' % (c, k), 'short': 'fsf%ss%d' % (c, k)} for k in (1, 2, 3)] for c in ('LF', 'PCL')}}
else:
    man = json.load(open(F.ROOT + 'temp/fs_manifest.json'))
N0 = F.score('fs_fmdet2k_munom_frozen', 2048)
def stats(entries, c):
    S = [F.score('fmdet_munom_%s_%s_64x8' % (LABEL[c], e['short']), 512) for e in entries]
    N = [F.score('fmdet2k_munom_%s' % e['short'], 2048) for e in entries]
    return {'S_runs': S, 'N_runs': N, 'S': statistics.mean(S), 'N': statistics.mean(N), 'dN': statistics.mean(N) - N0, 'tags': [e['tag'] for e in entries]}
out = {'N0_this_machine': N0, 'N0_original': 0.88623046875, 'rule': 'shared; dN>=-0.05 both; max mean S; ties worst dN then smaller lr', 'rates': {}, 'fullweight_new': {}}
for lr in ('1e-3', '3e-3'):
    st = {c: stats(man['fsr']['%s|%s' % (c, lr)], c) for c in ('LF', 'PCL')}
    out['rates'][lr] = {'per_condition': st, 'feasible': all(st[c]['dN'] >= -0.05 for c in st),
                        'mean_S': statistics.mean(st[c]['S'] for c in st), 'worst_dN': min(st[c]['dN'] for c in st)}
for c in ('LF', 'PCL'): out['fullweight_new'][c] = stats(man['fsf'][c], c)
feas = [lr for lr in ('1e-3', '3e-3') if out['rates'][lr]['feasible']]
if feas:
    feas.sort(key=lambda lr: (-out['rates'][lr]['mean_S'], -out['rates'][lr]['worst_dN'], float(lr)))
    out['selected'] = feas[0]
else:
    out['selected'] = None; out['note'] = 'no feasible configuration'
for lr, v in out['rates'].items():
    print('lr %s feasible %s meanS %.4f worst dN %+.4f | %s' % (lr, v['feasible'], v['mean_S'], v['worst_dN'],
          ' '.join('%s S %.3f dN %+.4f' % (c, v['per_condition'][c]['S'], v['per_condition'][c]['dN']) for c in ('LF', 'PCL'))))
for c, v in out['fullweight_new'].items(): print('full-weight new %s S %.3f dN %+.4f' % (c, v['S'], v['dN']))
print('N0 this machine %.5f (original %.5f) -> selected: %s' % (N0, out['N0_original'], out['selected']))
if not a.selftest: print('sha256', F.dump(F.ROOT + 'temp/fs_select.json', out))
