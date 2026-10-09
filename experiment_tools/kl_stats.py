#!/usr/bin/env python3
"""Lock step 3 analysis (descriptive; supplementary KL-retention baseline, section 10 of the lock document).
Y = (seen + unseen)/2 per independent training run, final-protocol test evaluation (seed 4242).
Per condition (LF, PCL=pn10, TH): selected KL full-weight GRPO (3e-5, beta_A) vs main full-weight GRPO (1e-5) and vs main
residual GRPO (1e-3): fault and nominal differences with Welch 95% CIs (final_stats_core.contrast; no correction, no
supported/not-supported verdict). Pareto table: every KL grid config (mean fault Y, nominal loss vs frozen) plus the
beta = 0 full-weight GRPO and residual GRPO learning-rate points of the S3 curve. Writes temp/kl_stats.json."""
import json, os, sys
ROOT = '/root/autodl-tmp/DexGraspResearch/'
sys.path.insert(0, ROOT + 'tools')
from final_stats_core import contrast
F = json.load(open(ROOT + 'temp/final_manifest.json')); K = json.load(open(ROOT + 'temp/kl_manifest.json'))
SEL = json.load(open(ROOT + 'temp/kl_select.json')); S3 = json.load(open(ROOT + 'temp/supp_manifest.json'))['S3_lrcurve']
def sr(n):
    f = ROOT + 'runs/' + n + '/result.json'
    return json.load(open(f))['totals']['success_rate'] if os.path.exists(f) else None
def Yn(cond, tag):
    a, b = sr('final_seen_%s_%s' % (cond, tag)), sr('final_unseen_%s_%s' % (cond, tag))
    return None if a is None or b is None else (a + b) / 2
def Ym(tag, cond):  # main-protocol models: names from the final manifest
    e = F['evals'][tag][cond]; return (sr(e['seen']) + sr(e['unseen'])) / 2
def B(cond):
    e = F['frozen'][cond]; return (sr(e['seen']) + sr(e['unseen'])) / 2
CN = {'LF': 'LF', 'PCL': 'pn10', 'TH': 'TH'}; SETS = {'LF': 'LF', 'PCL': 'PCL', 'TH': 'TH'}
lr, bi = SEL['selected']['lr'], SEL['selected']['beta_id']
out = {'selected': SEL['selected'], 'frozen': {c: B(c) for c in ('nom', 'LF', 'pn10', 'TH')}, 'compare': {}, 'pareto': {}}
print('frozen', {k: round(v, 4) for k, v in out['frozen'].items()})
for c in ('LF', 'PCL', 'TH'):
    kl = [t['tag'] for t in K['configs']['%s|%s|%s' % (c, lr, bi)]] if c != 'TH' else SEL['thumb_tags']
    fg = [x['tag'] for x in F['methods']['full_grpo']['sets'][SETS[c]]]; rg = [x['tag'] for x in F['methods']['res_grpo']['sets'][SETS[c]]]
    cond = CN[c]
    g = {'kl': {'fault': [Yn(cond, t) for t in kl], 'nom': [Yn('nom', t) for t in kl]},
         'full': {'fault': [Ym(t, cond) for t in fg], 'nom': [Ym(t, 'nom') for t in fg]},
         'res': {'fault': [Ym(t, cond) for t in rg], 'nom': [Ym(t, 'nom') for t in rg]}}
    rec = {'means': {k: {m: sum(v[m]) / len(v[m]) for m in v} for k, v in g.items()}, 'n': {k: len(v['fault']) for k, v in g.items()}}
    for ref in ('full', 'res'):
        for m in ('fault', 'nom'):
            r = contrast([g['kl'][m], g[ref][m]], [1, -1], +1)
            rec['kl_minus_%s_%s' % (ref, m)] = {'est': r['estimate'], 'ci': r['ci95_two_sided'], 'df': r['df']}
    out['compare'][c] = rec
    mm = rec['means']
    print('== %s (frozen fault %.3f)' % (c, out['frozen'][cond]))
    for k in ('kl', 'full', 'res'): print('   %-4s fault %.3f  nom %.3f (loss %.1f)  n=%d' % (k, mm[k]['fault'], mm[k]['nom'], 100 * (out['frozen']['nom'] - mm[k]['nom']), rec['n'][k]))
    for ref in ('full', 'res'):
        for m in ('fault', 'nom'):
            x = rec['kl_minus_%s_%s' % (ref, m)]; print('   KL - %-4s %-5s %+.1f [%+.1f, %+.1f] df %.1f' % (ref, m, 100 * x['est'], 100 * x['ci'][0], 100 * x['ci'][1], x['df']))
N0 = out['frozen']['nom']
for c in ('LF', 'PCL'):
    rows = []
    for key, tags in K['configs'].items():
        cc, l, b = key.split('|')
        if cc != c: continue
        f = [Yn(CN[c], t['tag']) for t in tags]; nm = [Yn('nom', t['tag']) for t in tags]
        rows.append({'method': 'KL full GRPO', 'lr': l, 'beta': K['beta'][b], 'fault': sum(f) / 3, 'nom_loss': N0 - sum(nm) / 3, 'per_seed_fault': f})
    for m, lab in (('full_grpo', 'full GRPO (beta 0)'), ('res_grpo', 'residual GRPO')):
        for l, d in S3[m].items():
            tags = d[c]; f = [Yn(CN[c], t) for t in tags]; nm = [Yn('nom', t) for t in tags]
            if None in f or None in nm: continue
            rows.append({'method': lab, 'lr': l, 'beta': 0.0, 'fault': sum(f) / len(f), 'nom_loss': N0 - sum(nm) / len(nm), 'n': len(f)})
    out['pareto'][c] = rows
    print('== Pareto %s (fault, nominal loss in points)' % c)
    for r in sorted(rows, key=lambda r: (r['method'], float(r['lr']), r['beta'])):
        print('   %-20s lr %-5s beta %-8s fault %.3f  loss %5.1f' % (r['method'], r['lr'], r['beta'], r['fault'], 100 * r['nom_loss']))
json.dump(out, open(ROOT + 'temp/kl_stats.json', 'w'), indent=1); print('written')
