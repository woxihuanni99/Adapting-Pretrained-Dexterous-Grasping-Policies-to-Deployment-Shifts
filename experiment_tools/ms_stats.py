#!/usr/bin/env python3
"""Lock step 4 analysis (descriptive; supplementary, lock document section 11; written 2026-10-02 before any lock-step-4
result exists). Y = (seen + unseen)/2 per independent training run, final-protocol test evaluation (seed 4242).
A) Mean-scale intervention, residual GRPO, S = D_c (block RMS) vs S = diag(sigma0) (= D0, existing models, s1-s3 only):
   - restricted development selection for each scaling separately (same rule as the main shared-lr protocol, delta = 0.05,
     lr in {1e-3, 3e-3}, models of the condition they were trained on): N_c = mean nominal dev success (2048 episodes),
     S_c = mean fault dev success (512 episodes); feasible: N_LF - N0 >= -0.05 and N_PCL - N0 >= -0.05 (unrounded);
     choose max (S_LF + S_PCL)/2; ties: smaller worst nominal loss, then smaller lr. No feasible lr -> reported as such.
   - same-lr test comparisons D_c - D0 (fault Y, nominal Y) per condition and lr: Welch 95% (final_stats_core.contrast).
   - each scaling's selected point (fault Y, nominal loss vs frozen).
B) Constant latent-offset diagnostic: per training run, Y_const - Y_full (own fault condition and nominal), paired within
   run, mean with t interval (n - 1 df). Original gain G = Y_full - Y0 (Y0 = frozen, same condition); only if its
   one-sample 95% lower bound > 0 is R_gain = (mean Y_const - Y0)/(mean Y_full - Y0) reported.
No multiple-comparison correction and no supported / not-supported verdict. Writes temp/ms_stats.json."""
import json, math, os, sys
from scipy import stats
ROOT = '/root/autodl-tmp/DexGraspResearch/'
sys.path.insert(0, ROOT + 'tools')
from final_stats_core import contrast
M = json.load(open(ROOT + 'temp/ms_manifest.json'))
def sr(n):
    f = ROOT + 'runs/' + n + '/result.json'
    return json.load(open(f))['totals']['success_rate'] if os.path.exists(f) else None
def Y(cond, tag):
    a, b = sr('final_seen_%s_%s' % (cond, tag)), sr('final_unseen_%s_%s' % (cond, tag))
    return None if a is None or b is None else (a + b) / 2
def mean(v): return sum(v) / len(v)
def tci(v):
    n = len(v); m = mean(v); s = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)); h = stats.t.ppf(0.975, n - 1) * s / math.sqrt(n)
    return m, [m - h, m + h]
CN = {'LF': 'LF', 'PCL': 'pn10'}; LABEL = {'LF': 'shlockLF', 'PCL': 'pn10'}; LRS = ('1e-3', '3e-3')
Y0 = {c: Y(c, 'frozen') for c in ('nom', 'LF', 'pn10', 'LF75', 'RFLF')}
N0 = sr('fmdet2k_munom_frozen')
out = {'frozen_Y': Y0, 'N0_dev': N0, 'select': {}, 'same_lr': {}, 'selected_points': {}, 'const': {}, 'missing': []}
print('frozen Y', {k: round(v, 4) for k, v in Y0.items()}, '| N0 dev', round(N0, 4))
# ---- A) selection (development data only)
for sc in ('d0', 'ms'):
    rows = {}
    for lr in LRS:
        row = {}
        for c in ('LF', 'PCL'):
            ent = M[sc]['%s|%s' % (c, lr)]
            N = [sr('fmdet2k_munom_%s' % e['short']) for e in ent]; S = [sr('fmdet_munom_%s_%s_64x8' % (LABEL[c], e['short'])) for e in ent]
            out['missing'] += [e['short'] for e, x, y in zip(ent, N, S) if x is None or y is None]
            row[c] = {'N': N, 'S': S, 'dN': mean(N) - N0 if None not in N else None, 'S_mean': mean(S) if None not in S else None}
        ok = None not in (row['LF']['dN'], row['PCL']['dN'], row['LF']['S_mean'], row['PCL']['S_mean'])
        row['feasible'] = ok and all(row[c]['dN'] >= -0.05 for c in ('LF', 'PCL'))
        row['objective'] = (row['LF']['S_mean'] + row['PCL']['S_mean']) / 2 if ok else None
        row['worst_loss'] = -min(row[c]['dN'] for c in ('LF', 'PCL')) if ok else None
        rows[lr] = row
        print('dev %-2s lr %s feasible=%s obj=%s dN_LF=%s dN_PCL=%s' % (sc, lr, row['feasible'], ok and round(row['objective'], 4),
              ok and round(100 * row['LF']['dN'], 2), ok and round(100 * row['PCL']['dN'], 2)))
    feas = sorted([lr for lr in LRS if rows[lr]['feasible']], key=lambda lr: (-rows[lr]['objective'], rows[lr]['worst_loss'], float(lr)))
    out['select'][sc] = {'rows': rows, 'selected_lr': feas[0] if feas else None}
    print('   selected', sc, feas[0] if feas else 'no feasible lr')
# ---- A) test: same-lr comparisons and selected points
for c in ('LF', 'PCL'):
    cond = CN[c]
    for lr in LRS:
        g = {sc: {'fault': [Y(cond, e['tag']) for e in M[sc]['%s|%s' % (c, lr)]], 'nom': [Y('nom', e['tag']) for e in M[sc]['%s|%s' % (c, lr)]]} for sc in ('d0', 'ms')}
        if any(None in g[sc][m] for sc in g for m in ('fault', 'nom')):
            out['missing'].append('test %s %s' % (c, lr)); print('MISSING test', c, lr); continue
        rec = {'means': {sc: {m: mean(v) for m, v in g[sc].items()} for sc in g}, 'per_run': g}
        for m in ('fault', 'nom'):
            r = contrast([g['ms'][m], g['d0'][m]], [1, -1], +1); rec['ms_minus_d0_' + m] = {'est': r['estimate'], 'ci': r['ci95_two_sided'], 'df': r['df']}
        out['same_lr']['%s|%s' % (c, lr)] = rec; mm = rec['means']
        print('== %s lr %s: D0 fault %.3f nom %.3f (loss %.1f) | Dc fault %.3f nom %.3f (loss %.1f)' % (c, lr, mm['d0']['fault'], mm['d0']['nom'],
              100 * (Y0['nom'] - mm['d0']['nom']), mm['ms']['fault'], mm['ms']['nom'], 100 * (Y0['nom'] - mm['ms']['nom'])))
        for m in ('fault', 'nom'):
            x = rec['ms_minus_d0_' + m]; print('   Dc - D0 %-5s %+.1f [%+.1f, %+.1f] df %.1f' % (m, 100 * x['est'], 100 * x['ci'][0], 100 * x['ci'][1], x['df']))
    for sc in ('d0', 'ms'):
        lr = out['select'][sc]['selected_lr']; k = '%s|%s' % (c, lr)
        if lr and k in out['same_lr']:
            mm = out['same_lr'][k]['means'][sc]
            out['selected_points']['%s|%s' % (c, sc)] = {'lr': lr, 'fault': mm['fault'], 'nom_loss': Y0['nom'] - mm['nom']}
            print('   selected point %-2s %s: lr %s fault %.3f nominal loss %.1f' % (sc, c, lr, mm['fault'], 100 * (Y0['nom'] - mm['nom'])))
# ---- B) constant latent-offset diagnostic
groups = {}
for tag, e in M['const'].items():
    groups.setdefault(e['cond'], []).append((tag, e['const_tag']))
for cond, lst in sorted(groups.items()):
    lst.sort()
    full = {m: [Y(cond if m == 'fault' else 'nom', t) for t, _ in lst] for m in ('fault', 'nom')}
    const = {m: [Y(cond if m == 'fault' else 'nom', ct) for _, ct in lst] for m in ('fault', 'nom')}
    if any(None in v for d in (full, const) for v in d.values()):
        out['missing'].append('const ' + cond); print('MISSING const', cond); continue
    rec = {'models': [t for t, _ in lst], 'full': full, 'const': const, 'Y0_fault': Y0[cond], 'Y0_nom': Y0['nom']}
    for m in ('fault', 'nom'):
        d = [a - b for a, b in zip(const[m], full[m])]; est, ci = tci(d); rec['const_minus_full_' + m] = {'est': est, 'ci': ci, 'per_run': d}
    gain = [y - Y0[cond] for y in full['fault']]; gm, gci = tci(gain); rec['orig_gain'] = {'est': gm, 'ci': gci}
    rec['R_gain'] = (mean(const['fault']) - Y0[cond]) / (mean(full['fault']) - Y0[cond]) if gci[0] > 0 else None
    out['const'][cond] = rec
    print('== const %s (n=%d, frozen fault %.3f, nom %.3f)' % (cond, len(lst), Y0[cond], Y0['nom']))
    print('   full  fault %.3f nom %.3f | const fault %.3f nom %.3f' % (mean(full['fault']), mean(full['nom']), mean(const['fault']), mean(const['nom'])))
    for m in ('fault', 'nom'):
        x = rec['const_minus_full_' + m]; print('   const - full %-5s %+.1f [%+.1f, %+.1f]  per run %s' % (m, 100 * x['est'], 100 * x['ci'][0], 100 * x['ci'][1], [round(100 * v, 1) for v in x['per_run']]))
    print('   original gain %+.1f [%+.1f, %+.1f] -> R_gain %s' % (100 * gm, 100 * gci[0], 100 * gci[1], 'n/a (gain not clearly positive)' if rec['R_gain'] is None else '%.2f' % rec['R_gain']))
if out['missing']: print('MISSING', out['missing'])
json.dump(out, open(ROOT + 'temp/ms_stats.json', 'w'), indent=1); print('written')
