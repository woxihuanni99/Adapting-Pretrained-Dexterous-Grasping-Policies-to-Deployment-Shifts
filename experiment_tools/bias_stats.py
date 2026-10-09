#!/usr/bin/env python3
"""Test-set analysis of the output-shift coordinate control (plan section 5; 2026-10-08). Run only after bias_select.py.
Descriptive, outside the nine-test family. Units: independent training runs; Welch-Satterthwaite 95% intervals (unadjusted).
Y = (seen + unseen)/2 success per run. Conditions: LF models -> LF, nom, LF25, LF75; PCL models -> pn10, nom.
Pre-specified contrasts:
  D_sel[cond]  = Ybar(sigma arm, its selected rate) - Ybar(raw arm, its selected rate)    (only if both arms have a selected rate)
  D_idx[i]     = Ybar(sigma, i-th rate) - Ybar(raw, i-th rate), i = 1..3 (grids matched by mean per-dimension step: raw ~ 0.34 x sigma)
  context      = each arm (selected, else every rate) minus the locked residual GRPO and full-weight GRPO groups (other machines;
                 computed only where temp/final_manifest.json and the archived runs exist, otherwise on E: by the paper audit).
Wording for D_sel under LF fault and LF nominal: interval excludes 0 -> 'the coordinate changed this outcome at the selected operating
points'; otherwise 'no difference detected at the selected operating points (not evidence of equivalence)'."""
import argparse, json, sys
sys.path.insert(0, '/root/autodl-tmp/DexGraspResearch/tools'); import fs_common as F
ap = argparse.ArgumentParser(); ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
LRS = {'raw': ('3e-3', '1e-2', '3e-2'), 'sigma': ('1e-2', '3e-2', '1e-1')}
PRE = {'raw': 'bxr', 'sigma': 'bxs'}; LS = {'3e-3': '3em3', '1e-2': '1em2', '3e-2': '3em2', '1e-1': '1em1'}
CONDS = {'LF': ['LF', 'nom', 'LF25', 'LF75'], 'PCL': ['pn10', 'nom']}
import os
fm = json.load(open(F.ROOT + 'temp/final_manifest.json')) if os.path.exists(F.ROOT + 'temp/final_manifest.json') else None  # absent on the 10/8 server (9/29 data-disk snapshot): context is then computed on E:
if a.selftest:
    F.FAKE = lambda name, h: (0.6 if 'bxs' in name else 0.5) + 0.1 * h
    man = {'arms': {arm: {'%s|%s' % (c, lr): [{'tag': '%s_%s_%s_s%d' % (PRE[arm], c, LS[lr], k)} for k in (1, 2, 3)]
                          for c in ('LF', 'PCL') for lr in LRS[arm]} for arm in LRS}}
    sel = {'arms': {'raw': {'selected': '1e-2'}, 'sigma': {'selected': '3e-2'}}}
else:
    man = json.load(open(F.ROOT + 'temp/bx_manifest.json')); sel = json.load(open(F.ROOT + 'temp/bx_select.json'))
def vals(tags, cond): return [F.Y(F.yname(cond, t)) for t in tags]
def old(m, c): return [e['tag'] for e in fm['methods'][m]['sets'][c]]
def word(w): return 'the coordinate changed this outcome at the selected operating points' if (w['ci'][0] > 0 or w['ci'][1] < 0) else 'no difference detected at the selected operating points (not evidence of equivalence)'
out = {'selected': {arm: sel['arms'][arm]['selected'] for arm in LRS}, 'conditions': {}}
for c, conds in CONDS.items():
    for cond in conds:
        rec = {'runs': {}, 'D_idx': {}, 'context': {}}
        for arm in LRS:
            for lr in LRS[arm]:
                rec['runs']['%s|%s' % (arm, lr)] = vals([e['tag'] for e in man['arms'][arm]['%s|%s' % (c, lr)]], cond)
        R_old = F_old = None
        if fm is not None:
            R_old, F_old = vals(old('res_grpo', c), cond), vals(old('full_grpo', c), cond)
            rec['residual_grpo_locked'], rec['fullweight_grpo_locked'] = R_old, F_old
        for i in range(3):
            lr_r, lr_s = LRS['raw'][i], LRS['sigma'][i]
            rec['D_idx']['%s_vs_%s' % (lr_s, lr_r)] = F.welch([(1, rec['runs']['sigma|' + lr_s]), (-1, rec['runs']['raw|' + lr_r])])
        if all(out['selected'][arm] for arm in LRS):
            d = F.welch([(1, rec['runs']['sigma|' + out['selected']['sigma']]), (-1, rec['runs']['raw|' + out['selected']['raw']])])
            rec['D_sel'] = d
            if c == 'LF' and cond in ('LF', 'nom'): rec['D_sel_wording'] = word(d)
        for arm in (LRS if fm is not None else ()):
            for lr in ([out['selected'][arm]] if out['selected'][arm] else LRS[arm]):
                v = rec['runs']['%s|%s' % (arm, lr)]
                rec['context']['%s|%s' % (arm, lr)] = {'minus_residual_grpo_locked': F.welch([(1, v), (-1, R_old)]),
                                                       'minus_fullweight_grpo_locked': F.welch([(1, v), (-1, F_old)])}
        out['conditions']['%s|%s' % (c, cond)] = rec
def pp(w): return '%+.1f [%.1f, %.1f]' % (100 * w['est'], 100 * w['ci'][0], 100 * w['ci'][1])
print('selected', out['selected'])
for k, rec in out['conditions'].items():
    means = ' '.join('%s %.1f' % (kk, 100 * sum(v) / len(v)) for kk, v in rec['runs'].items())
    print(k, '|', means)
    for kk, w in rec['D_idx'].items(): print('    D_idx', kk, pp(w))
    if 'D_sel' in rec: print('    D_sel', pp(rec['D_sel']), rec.get('D_sel_wording', ''))
if not a.selftest: print('sha256', F.dump(F.ROOT + 'temp/bx_stats.json', out))
