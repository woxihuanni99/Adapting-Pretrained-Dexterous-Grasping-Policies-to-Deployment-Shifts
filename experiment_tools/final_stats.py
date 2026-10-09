#!/usr/bin/env python3
"""Locked final analysis, hypothesis -> run mapping (lock step 2, protocol of 2026-09-29; see claude/DexGrasp_锁定说明_草稿.md 6b).
Inputs: temp/final_manifest.json (written by tools/final_plan_gen.py) and runs/final_*/result.json (eval seed in manifest).
Unit = independent training run. Y(run, cond) = (S_seen + S_unseen) / 2, computed inside the run.
Structure contrast over four independent groups, order (res_ppo, res_grpo, full_ppo, full_grpo), weights (+1/2, +1/2, -1/2, -1/2):
  H1  LF models, cond LF                         D_LF > 0
  H2  LF models on LF and PCL models on pn10      D_LF - D_PCL1 > 0   (8 groups, weights +-1/2)
  H3a LF models, cond LF25; H3b cond LF75         D > 0
  H4  per run Z = [Y(pi,A) - B(A)] - [Y(pi,A+B) - B(A+B)], B = frozen planner on the same evaluation; methods weighted 1/4 each; Z > 0
      (LF+1cm noise: source LF models A=LF, source PCL models A=pn10; TH+U(-3,3)cm z: source TH models A=TH, source ZR models A=zr)
  H5  LF models at the per-method unconstrained LF dev-best lr (LF_H5; = main models if that lr equals the shared lr), cond LF, D > 0
Holm over the 9 one-sided p values (H1, H2, H3a, H3b, H4 x 4, H5). Outcome categories: supported (Holm p < 0.05, predicted sign) /
consistent but insufficient / opposite but insufficient / evidence for the opposite (estimate opposite and two-sided 95% CI excludes 0).
Descriptive (uncorrected): H2b estimates D(1), D(1.5), D(2), D(1.5)-D(1), D(2)-D(1.5) with two-sided 95% CIs (differences inside runs);
per-algorithm structure differences; seen / unseen separately; per method x set x condition means and frozen references.
A method without a feasible shared lr has no main-protocol models: contrasts needing it are reported as not computable."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_stats_core import contrast, holm, classify
ROOT = '/root/autodl-tmp/DexGraspResearch/'
M = json.load(open(sys.argv[1] if len(sys.argv) > 1 else ROOT + 'temp/final_manifest.json'))
ORDER = ['res_ppo', 'res_grpo', 'full_ppo', 'full_grpo']; W = [0.5, 0.5, -0.5, -0.5]
def sr(name): return json.load(open(ROOT + 'runs/' + name + '/result.json'))['totals']['success_rate']
def Ys(tag, cond, split): return sr(M['evals'][tag][cond][split])
def Y(tag, cond): return (Ys(tag, cond, 'seen') + Ys(tag, cond, 'unseen')) / 2
def B(cond, split=None):
    e = M['frozen'][cond]
    return sr(e[split]) if split else (sr(e['seen']) + sr(e['unseen'])) / 2
def tags(m, s): return [x['tag'] for x in M['methods'][m]['sets'].get(s, [])]
def h5set(m): return 'LF_H5' if tags(m, 'LF_H5') else 'LF'
def per_run(set_of, f):
    return [[f(t) for t in tags(m, set_of(m))] for m in ORDER]
def safe(fn):
    try: return fn()
    except (ValueError, KeyError, FileNotFoundError, ZeroDivisionError) as ex: return {'not_computable': repr(ex)}
def struct(set_of, fn):  # set_of: method -> model set name; fn: tag -> per-run value
    return contrast(per_run(set_of, fn), W, +1)
out = {'eval_seed': M['eval_seed'], 'lr': {m: (M['methods'][m]['lr_main'], M['methods'][m]['lr_H5']) for m in ORDER}, 'primary': {}, 'descriptive': {}}
P = out['primary']
P['H1'] = safe(lambda: struct(lambda m: 'LF', lambda t: Y(t, 'LF')))
P['H2'] = safe(lambda: contrast(per_run(lambda m: 'LF', lambda t: Y(t, 'LF')) + per_run(lambda m: 'PCL', lambda t: Y(t, 'pn10')), W + [-w for w in W], +1))
P['H3a'] = safe(lambda: struct(lambda m: 'LF', lambda t: Y(t, 'LF25')))
P['H3b'] = safe(lambda: struct(lambda m: 'LF', lambda t: Y(t, 'LF75')))
H4 = {'H4_LFpn_srcLF': ('LF', 'LF', 'LFpn'), 'H4_LFpn_srcPCL': ('PCL', 'pn10', 'LFpn'),
      'H4_THzr_srcTH': ('TH', 'TH', 'THzr'), 'H4_THzr_srcZR': ('ZR', 'zr', 'THzr')}
for k, (s, A, AB) in H4.items():
    P[k] = safe(lambda s=s, A=A, AB=AB: contrast(per_run(lambda m: s, lambda t: (Y(t, A) - B(A)) - (Y(t, AB) - B(AB))), [0.25] * 4, +1))
P['H5'] = safe(lambda: struct(h5set, lambda t: Y(t, 'LF')))
pv = {k: v['p_one_sided'] for k, v in P.items() if 'p_one_sided' in v}
adj = holm(pv) if pv else {}
for k, v in P.items():
    if k not in adj: v['outcome'] = 'not computable'; continue
    v['p_holm'] = adj[k]; lo, hi = v['ci95_two_sided']
    v['outcome'] = ('evidence for the opposite direction' if (v['estimate'] < 0 and hi < 0) else classify(v['estimate'], +1, adj[k]))
D = out['descriptive']
D['H2b'] = {
 'D(1)': safe(lambda: struct(lambda m: 'PCL', lambda t: Y(t, 'pn10'))), 'D(1.5)': safe(lambda: struct(lambda m: 'PCL', lambda t: Y(t, 'pn15'))),
 'D(2)': safe(lambda: struct(lambda m: 'PCL', lambda t: Y(t, 'pn20'))),
 'D(1.5)-D(1)': safe(lambda: struct(lambda m: 'PCL', lambda t: Y(t, 'pn15') - Y(t, 'pn10'))),
 'D(2)-D(1.5)': safe(lambda: struct(lambda m: 'PCL', lambda t: Y(t, 'pn20') - Y(t, 'pn15')))}
def per_alg(set_name, cond, split=None):
    f = (lambda t: Y(t, cond)) if split is None else (lambda t: Ys(t, cond, split))
    return {alg: safe(lambda alg=alg: contrast([[f(t) for t in tags('res_' + alg, set_name)], [f(t) for t in tags('full_' + alg, set_name)]], [1, -1], +1))
            for alg in ('ppo', 'grpo')}
D['per_algorithm'] = {'H1_LF': per_alg('LF', 'LF'), 'H3a_LF25': per_alg('LF', 'LF25'), 'H3b_LF75': per_alg('LF', 'LF75'), 'PCL_pn10': per_alg('PCL', 'pn10')}
D['split'] = {sp: {'H1_LF': safe(lambda sp=sp: struct(lambda m: 'LF', lambda t: Ys(t, 'LF', sp))),
                   'PCL_pn10': safe(lambda sp=sp: struct(lambda m: 'PCL', lambda t: Ys(t, 'pn10', sp)))} for sp in ('seen', 'unseen')}
tab = {}
for m in ORDER:
    for s, lst in M['methods'][m]['sets'].items():
        for t in [x['tag'] for x in lst]:
            for cond in M['evals'].get(t, {}):
                try: tab.setdefault(m, {}).setdefault(s, {}).setdefault(cond, {'runs': {}})['runs'][t] = {'Y': Y(t, cond), 'seen': Ys(t, cond, 'seen'), 'unseen': Ys(t, cond, 'unseen')}
                except FileNotFoundError: tab.setdefault(m, {}).setdefault(s, {}).setdefault(cond, {'runs': {}})['runs'][t] = 'missing'
D['table'] = tab
D['frozen'] = {c: safe(lambda c=c: {'Y': B(c), 'seen': B(c, 'seen'), 'unseen': B(c, 'unseen')}) for c in M['frozen']}
json.dump(out, open(ROOT + 'temp/final_stats.json', 'w'), indent=1, default=str)
for k, v in P.items():
    if 'estimate' in v:
        print('%-16s est %+.4f  CI [%+.4f, %+.4f]  df %.1f  p1 %.4f  Holm %.4f  -> %s  (n=%s)' % (k, v['estimate'], *v['ci95_two_sided'], v['df'], v['p_one_sided'], v['p_holm'], v['outcome'], [g['n'] for g in v['groups']]))
    else: print('%-16s %s' % (k, v))
for k, v in D['H2b'].items():
    print('H2b %-12s' % k, ('est %+.4f CI [%+.4f, %+.4f]' % (v['estimate'], *v['ci95_two_sided'])) if 'estimate' in v else v)
