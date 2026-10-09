#!/usr/bin/env python3
"""Reproduction check and full robot-table collision re-evaluation analysis (plan sections A and F). Descriptive only.
--repro : compare repro_<name> (and reproorig_<name>) with the archived runs/<name>: per-episode per-env success flags and
          episode_XXX/initial_state.npz; verdict 'exact' only if every flag and every initial state is identical.
default : collision analysis. Y_coll from coll_<name>; Y_orig from the archived runs (or collref_<name> with --ref-arm).
          Parameterization contrasts (residual minus full-weight, weights +/-1/2 over the four schemes) for LF and PCL fault
          and nominal under the collision setting; change of the contrast from per-run differences Y_coll - Y_orig;
          frozen planner values; optimizer contrast (GRPO minus PPO) descriptive."""
import argparse, glob, json, os, sys
sys.path.insert(0, '/root/autodl-tmp/DexGraspResearch/tools'); import fs_common as F
ap = argparse.ArgumentParser(); ap.add_argument('--repro', action='store_true'); ap.add_argument('--ref-arm', action='store_true')
ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
fm = json.load(open(F.ROOT + 'temp/final_manifest.json'))
if a.repro:
    import numpy as np
    rep = {}
    for d in sorted(glob.glob(F.ROOT + 'runs/repro_final_*')) + sorted(glob.glob(F.ROOT + 'runs/reproorig_final_*')):
        new = os.path.basename(d); old = new.split('_', 1)[1]
        rn, ro = json.load(open(d + '/result.json')), json.load(open(F.ROOT + 'runs/' + old + '/result.json'))
        fl_same = [e1['official_ppo_print_successes'] == e2['official_ppo_print_successes'] for e1, e2 in zip(rn['episodes'], ro['episodes'])]
        diffs = []
        for i in range(len(rn['episodes'])):
            p1, p2 = d + '/episode_%03d/initial_state.npz' % i, F.ROOT + 'runs/%s/episode_%03d/initial_state.npz' % (old, i)
            if os.path.exists(p1) and os.path.exists(p2):
                diffs.append(float(np.abs(np.load(p1)['policy_observation'] - np.load(p2)['policy_observation']).max()))
        rep[new] = {'flags_identical_all': all(fl_same) and len(fl_same) == len(ro['episodes']), 'episodes_identical': sum(fl_same),
                    'success_new': rn['totals']['success_rate'], 'success_archived': ro['totals']['success_rate'],
                    'initial_state_max_abs_diff': max(diffs) if diffs else None, 'initial_states_compared': len(diffs)}
        print(new, rep[new])
    verdict = 'exact' if rep and all(v['flags_identical_all'] and (v['initial_state_max_abs_diff'] in (0.0, None)) for v in rep.values()) else 'not exact'
    print('VERDICT', verdict); print('sha256', F.dump(F.ROOT + 'temp/coll_repro.json', {'verdict': verdict, 'runs': rep})); raise SystemExit(0)
if a.selftest:
    F.FAKE = lambda name, h: (0.6 if ('plug' in name and '_LF_' in name) else 0.3) + 0.1 * h - (0.02 if name.startswith('coll_') else 0)
orig = (lambda n: 'collref_' + n) if a.ref_arm else (lambda n: n)
def Yp(src, pre): return (F.score(pre(src['seen']), 1128) + F.score(pre(src['unseen']), 800)) / 2
coll = lambda n: 'coll_' + n
out = {'arm_original': 'collref' if a.ref_arm else 'archived', 'frozen': {}, 'conditions': {}}
for cond in ('LF', 'pn10', 'nom'): out['frozen'][cond] = {'orig': Yp(fm['frozen'][cond], orig), 'coll': Yp(fm['frozen'][cond], coll)}
M = ('res_ppo', 'res_grpo', 'full_ppo', 'full_grpo'); PW = (0.5, 0.5, -0.5, -0.5); OW = (-0.5, 0.5, -0.5, 0.5)
for cset, fc in (('LF', 'LF'), ('PCL', 'pn10')):
    for cond in (fc, 'nom'):
        g = {}
        for m in M:
            tags = [e['tag'] for e in fm['methods'][m]['sets'][cset]]
            yo = [Yp(fm['evals'][t][cond], orig) for t in tags]; yc = [Yp(fm['evals'][t][cond], coll) for t in tags]
            g[m] = {'orig': yo, 'coll': yc, 'diff': [c - o for c, o in zip(yc, yo)]}
        out['conditions']['%s|%s' % (cset, cond)] = {
            'per_method': g,
            'param_orig': F.welch([(w, g[m]['orig']) for m, w in zip(M, PW)]),
            'param_coll': F.welch([(w, g[m]['coll']) for m, w in zip(M, PW)]),
            'param_change': F.welch([(w, g[m]['diff']) for m, w in zip(M, PW)]),
            'opt_coll': F.welch([(w, g[m]['coll']) for m, w in zip(M, OW)])}
# initial-state agreement between the collision runs and the original-setting runs (per run: envs whose 14-dim
# initial observation differs in any episode; the collision setting adds a ground plane, which can stop a falling object)
import numpy as np
init = {}
for d in sorted(glob.glob(F.ROOT + 'runs/coll_final_*')):
    new = os.path.basename(d); old = new.split('_', 1)[1]; ref = (F.ROOT + 'runs/collref_' + old) if a.ref_arm else (F.ROOT + 'runs/' + old)
    envs = set(); eps = 0
    for p1 in sorted(glob.glob(d + '/episode_*/initial_state.npz')):
        p2 = ref + '/' + p1.split('/')[-2] + '/initial_state.npz'
        if os.path.exists(p2):
            x, y = np.load(p1)['policy_observation'], np.load(p2)['policy_observation']; eps += 1
            envs |= set(np.nonzero(np.abs(x - y).max(1) > 1e-6)[0].tolist())
    init[new] = {'episodes_compared': eps, 'envs_with_any_difference': sorted(envs)}
out['initial_state_check'] = init
nd = [k for k, v in init.items() if v['envs_with_any_difference']]
print('initial-state check: runs compared', len(init), '| runs with differing envs', len(nd), '| max envs differing', max([len(v['envs_with_any_difference']) for v in init.values()] or [0]))
def pp(w): return '%+.1f [%.1f, %.1f]' % (100 * w['est'], 100 * w['ci'][0], 100 * w['ci'][1])
print('frozen', {k: '%.3f -> %.3f' % (v['orig'], v['coll']) for k, v in out['frozen'].items()})
for k, v in out['conditions'].items():
    print(k, '| param orig', pp(v['param_orig']), '| param coll', pp(v['param_coll']), '| change', pp(v['param_change']), '| opt coll', pp(v['opt_coll']))
if not a.selftest: print('sha256', F.dump(F.ROOT + 'temp/coll_stats.json', out))
