#!/usr/bin/env python3
"""Lock step 5 analysis (written 2026-10-02 before any lock-step-5 result exists; lock document section 12).
Directly optimised state-independent constant mean c (residual GRPO head with delta(s) replaced by c), exploratory baseline.
Selection, per condition separately (development data only): N = mean over s1-s3 of nominal dev success (2048 episodes),
N0 = frozen planner on the same set; S = mean fault dev success (512 episodes). Feasible: N - N0 >= -0.05 (unrounded);
choose max S; ties: smaller nominal loss, then smaller lr. No feasible lr -> reported, threshold not relaxed.
Test (final protocol, seed 4242; Y = (seen + unseen)/2 per independent training run): full grid table; selected point vs four
references - frozen planner Y0, state-dependent residual GRPO (LF s1-s3, RFLF s1-s5), post-hoc constant delta-bar of those
residual models, directly optimised c. Contrasts between INDEPENDENT training groups (Welch 95%, final_stats_core.contrast);
no within-model pairing. R_direct = (mean Y_c - Y0)/(mean Y_res - Y0), point-estimate ratio. Writes temp/cst_stats.json."""
import json, os, sys
ROOT = '/root/autodl-tmp/DexGraspResearch/'
sys.path.insert(0, ROOT + 'tools')
from final_stats_core import contrast
M = json.load(open(ROOT + 'temp/cst_manifest.json'))
def sr(n):
    f = ROOT + 'runs/' + n + '/result.json'
    return json.load(open(f))['totals']['success_rate'] if os.path.exists(f) else None
def Y(cond, tag):
    a, b = sr('final_seen_%s_%s' % (cond, tag)), sr('final_unseen_%s_%s' % (cond, tag))
    return None if a is None or b is None else (a + b) / 2
def mean(v): return sum(v) / len(v)
N0 = sr('fmdet2k_munom_frozen'); LRS = ('3e-3', '1e-2', '3e-2')
out = {'N0_dev': N0, 'select': {}, 'grid_test': {}, 'compare': {}, 'missing': []}
for c, ref in M['refs'].items():
    cond, label = ref['cond'], ref['label']
    rows = {}
    for lr in LRS:
        ent = M['cst']['%s|%s' % (c, lr)]
        N = [sr('fmdet2k_munom_%s' % e['short']) for e in ent]; S = [sr('fmdet_munom_%s_%s_64x8' % (label, e['short'])) for e in ent]
        out['missing'] += [e['short'] for e, x, y in zip(ent, N, S) if x is None or y is None]
        ok = None not in N and None not in S
        rows[lr] = {'N': N, 'S': S, 'dN': mean(N) - N0 if ok else None, 'S_mean': mean(S) if ok else None}
        rows[lr]['feasible'] = ok and rows[lr]['dN'] >= -0.05
        print('dev %-4s c-lr %-5s S %s dN %s feasible %s' % (c, lr, ok and round(rows[lr]['S_mean'], 4), ok and round(100 * rows[lr]['dN'], 2), rows[lr]['feasible']))
    feas = sorted([lr for lr in LRS if rows[lr]['feasible']], key=lambda lr: (-rows[lr]['S_mean'], -rows[lr]['dN'], float(lr)))
    sel = feas[0] if feas else None
    out['select'][c] = {'rows': rows, 'selected_lr': sel}; print('   selected', c, sel or 'no feasible learning rate')
    Y0f, Y0n = Y(cond, 'frozen'), Y('nom', 'frozen')
    grid = {}
    for lr in LRS:
        tags = [e['tag'] for e in M['cst']['%s|%s' % (c, lr)]]
        f = [Y(cond, t) for t in tags]; nm = [Y('nom', t) for t in tags]
        if None in f or None in nm: out['missing'].append('test %s %s' % (c, lr)); continue
        grid[lr] = {'fault': f, 'nom': nm, 'fault_mean': mean(f), 'nom_loss': Y0n - mean(nm)}
        print('test %-4s c-lr %-5s fault %.3f  nominal loss %.1f  per run %s' % (c, lr, mean(f), 100 * (Y0n - mean(nm)), [round(x, 3) for x in f]))
    out['grid_test'][c] = grid
    res = {'fault': [Y(cond, t) for t in ref['residual']], 'nom': [Y('nom', t) for t in ref['residual']]}
    pc = {'fault': [Y(cond, t) for t in ref['posthoc_const']], 'nom': [Y('nom', t) for t in ref['posthoc_const']]}
    rec = {'Y0_fault': Y0f, 'Y0_nom': Y0n, 'residual': res, 'posthoc_const': pc, 'selected_lr': sel}
    print('== %s refs: frozen fault %.3f nom %.3f | residual fault %.3f nom %.3f (n=%d) | post-hoc const fault %.3f nom %.3f' % (
        c, Y0f, Y0n, mean(res['fault']), mean(res['nom']), len(res['fault']), mean(pc['fault']), mean(pc['nom'])))
    if sel and sel in grid:
        g = grid[sel]; rec['direct'] = {'fault': g['fault'], 'nom': g['nom']}
        for refn, refd in (('residual', res), ('posthoc_const', pc)):
            for m in ('fault', 'nom'):
                r = contrast([g[m], refd[m]], [1, -1], +1)
                rec['direct_minus_%s_%s' % (refn, m)] = {'est': r['estimate'], 'ci': r['ci95_two_sided'], 'df': r['df']}
        rec['R_direct'] = (mean(g['fault']) - Y0f) / (mean(res['fault']) - Y0f)
        print('   direct c (lr %s) fault %.3f nom %.3f (loss %.1f), R_direct %.2f' % (sel, mean(g['fault']), mean(g['nom']), 100 * (Y0n - mean(g['nom'])), rec['R_direct']))
        for refn in ('residual', 'posthoc_const'):
            for m in ('fault', 'nom'):
                x = rec['direct_minus_%s_%s' % (refn, m)]
                print('   direct - %-13s %-5s %+.1f [%+.1f, %+.1f] df %.1f' % (refn, m, 100 * x['est'], 100 * x['ci'][0], 100 * x['ci'][1], x['df']))
    out['compare'][c] = rec
if out['missing']: print('MISSING', out['missing'])
json.dump(out, open(ROOT + 'temp/cst_stats.json', 'w'), indent=1); print('written')
