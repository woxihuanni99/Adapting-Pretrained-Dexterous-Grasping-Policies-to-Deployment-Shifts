#!/usr/bin/env python3
"""Test-set analysis of the matched-GRPO control (plan sections D-E). Run only after fs_select.py has written temp/fs_select.json.
Primary: LF fault, within GRPO. G_new = Ybar(matched residual, selected rate) - Ybar(new full-weight GRPO) (Welch, weights +1/-1);
G_old = Ybar(original residual GRPO) - Ybar(original full-weight GRPO) from the locked runs (32.1 pp); Delta = G_new - G_old
(four independent groups, weights +1/-1/-1/+1). Category via fs_common.classify (pre-specified order). Everything else is
descriptive: nominal, PCL, LF25/LF75, the non-selected rate, replication of full-weight GRPO (new minus original seeds) and
the residual change (matched minus original). If no rate was feasible, both rates are reported descriptively, no category."""
import argparse, json, statistics, sys
sys.path.insert(0, '/root/autodl-tmp/DexGraspResearch/tools'); import fs_common as F
ap = argparse.ArgumentParser(); ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
fm = json.load(open(F.ROOT + 'temp/final_manifest.json'))
if a.selftest:
    F.FAKE = lambda name, h: (0.6 if ('fsr' in name or 'pluggrpo' in name) and '_LF_' in name else 0.3) + 0.1 * h
    man = {'fsr': {'%s|%s' % (c, lr): [{'tag': 'fsr_%s_%s_s%d' % (c, s, k)} for k in (1, 2, 3)]
                   for c in ('LF', 'PCL') for lr, s in (('1e-3', '1em3'), ('3e-3', '3em3'))},
           'fsf': {c: [{'tag': 'fsf_%s_s%d' % (c, k)} for k in (1, 2, 3)] for c in ('LF', 'PCL')}}
    sel = {'selected': '1e-3'}
else:
    man = json.load(open(F.ROOT + 'temp/fs_manifest.json')); sel = json.load(open(F.ROOT + 'temp/fs_select.json'))
CONDS = {'LF': ['LF', 'nom', 'LF25', 'LF75'], 'PCL': ['pn10', 'nom']}
def vals(tags, cond): return [F.Y(F.yname(cond, t)) for t in tags]
def old(m, c): return [e['tag'] for e in fm['methods'][m]['sets'][c]]
frozen = {cond: F.Y(fm['frozen'][cond]) for cond in ('LF', 'LF25', 'LF75', 'pn10', 'nom')}
out = {'selected_rate': sel['selected'], 'frozen_archived': frozen, 'conditions': {}}
rates = [sel['selected']] if sel['selected'] else ['1e-3', '3e-3']
for c, conds in CONDS.items():
    F_new = [e['tag'] for e in man['fsf'][c]]; R_old = old('res_grpo', c); F_old = old('full_grpo', c)
    for cond in conds:
        fn, ro, fo = vals(F_new, cond), vals(R_old, cond), vals(F_old, cond)
        rec = {'fullweight_new': fn, 'residual_old': ro, 'fullweight_old': fo,
               'G_old': F.welch([(1, ro), (-1, fo)]), 'fullweight_new_minus_old': F.welch([(1, fn), (-1, fo)]), 'rates': {}}
        for lr in ('1e-3', '3e-3'):
            rn = vals([e['tag'] for e in man['fsr']['%s|%s' % (c, lr)]], cond)
            g = F.welch([(1, rn), (-1, fn)]); d = F.welch([(1, rn), (-1, fn), (-1, ro), (1, fo)])
            r = {'residual_matched': rn, 'G_new': g, 'Delta': d, 'residual_matched_minus_old': F.welch([(1, rn), (-1, ro)]),
                 'role': 'primary' if lr == sel['selected'] else 'descriptive'}
            if lr == sel['selected'] and c == 'LF' and cond == 'LF': r['category'] = F.classify(g, d)
            rec['rates'][lr] = r
        out['conditions']['%s|%s' % (c, cond)] = rec
def pp(w): return '%+.1f [%.1f, %.1f]' % (100 * w['est'], 100 * w['ci'][0], 100 * w['ci'][1])
print('selected rate:', sel['selected'])
for k, rec in out['conditions'].items():
    print(k, '| G_old', pp(rec['G_old']), '| FW new-old', pp(rec['fullweight_new_minus_old']))
    for lr, r in rec['rates'].items():
        print('    lr %s (%s) G_new %s  Delta %s  R matched-old %s %s' % (lr, r['role'], pp(r['G_new']), pp(r['Delta']), pp(r['residual_matched_minus_old']), r.get('category', '')))
if not a.selftest: print('sha256', F.dump(F.ROOT + 'temp/fs_stats.json', out))
