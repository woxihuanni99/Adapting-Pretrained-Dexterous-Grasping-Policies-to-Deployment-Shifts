#!/usr/bin/env python3
"""Build the manuscript numerical record from archived results and locked summaries, and draw Figures 2-4.
Inputs: data/runs/<run>/result.json (final-protocol evaluations, seed 4242), data/meta/temp/*.json (manifests and
locked statistics). Output: numbers.json (single source for manuscript numbers) and figures/*.pdf|png.
Statistics follow the locked analysis: Y = (seen + unseen)/2 per independent training run; equal-weight linear
contrasts with Welch-Satterthwaite df (final_stats_core.contrast). No values are edited by hand."""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
D = os.environ.get('DEXGRASP_DATA_ROOT', os.path.join(ROOT, '..', 'data')); FIG = os.path.join(ROOT, 'figures'); os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.join(D, 'meta', 'tools'))
from final_stats_core import contrast  # locked statistics core (sha256 0e6a9835...)
from scipy import stats

def J(p): return json.load(open(os.path.join(D, 'meta', 'temp', p)))
def sr(run):
    f = os.path.join(D, 'runs', run, 'result.json')
    return json.load(open(f))['totals']['success_rate'] if os.path.exists(f) else None
def Y(cond, tag):
    a, b = sr('final_seen_%s_%s' % (cond, tag)), sr('final_unseen_%s_%s' % (cond, tag))
    return None if a is None or b is None else (a + b) / 2
def mean(v): return float(np.mean(v))
def tci(v):
    v = np.asarray(v, float); n = len(v); m = v.mean(); h = stats.t.ppf(.975, n - 1) * v.std(ddof=1) / math.sqrt(n); return [m - h, m + h]
def pt(x): return round(100 * x, 1)

FS = J('final_stats.json'); FM = J('final_manifest.json'); SM = J('supp_manifest.json')
TAB = FS['descriptive']['table']; FROZ = {k: v['Y'] for k, v in FS['descriptive']['frozen'].items()}
N0 = FROZ['nom']
M4 = ['full_grpo', 'full_ppo', 'res_ppo', 'res_grpo']
LAB = {'full_grpo': 'Full-weight GRPO', 'full_ppo': 'Full-weight PPO', 'res_ppo': 'Residual PPO', 'res_grpo': 'Residual GRPO'}
CONDS = [('LF', 'LF', 'Little-finger\nlock'), ('PCL', 'pn10', 'Point-cloud\nnoise 1 cm'), ('TH', 'TH', 'Thumb\nlock'),
         ('ZR', 'zr', 'Random $z$ offset\n$\\pm$3 cm'), ('BZ2PN', 'bz2pn', '$z$ offset 2 cm\n+ noise'), ('THPN', 'THpn', 'Thumb lock\n+ noise')]
out = {'frozen': FROZ, 'main': {}, 'tests': {}, 'lrcurve': {}, 'boundary': {}, 'kl': {}, 'ms': {}, 'const': {}, 'cst': {}}

# ---------------- main table (Shadow) ----------------
for ts, ec, name in CONDS:
    row = {'name': name, 'frozen_fault': FROZ[ec]}
    for m in M4:
        r = TAB[m][ts]
        f = [x['Y'] for x in r[ec]['runs'].values()]; nm = [x['Y'] for x in r['nom']['runs'].values()]
        row[m] = {'fault': f, 'nom': nm, 'fault_mean': mean(f), 'nom_mean': mean(nm), 'n': len(f)}
    # structure difference: equal-weight (res_ppo, res_grpo, full_ppo, full_grpo) = (+.5,+.5,-.5,-.5)
    for key in ('fault', 'nom'):
        c = contrast([row['res_ppo'][key], row['res_grpo'][key], row['full_ppo'][key], row['full_grpo'][key]], [.5, .5, -.5, -.5], +1)
        row['struct_' + key] = {'est': c['estimate'], 'ci': c['ci95_two_sided'], 'df': c['df']}
        a = contrast([row['full_grpo'][key], row['res_grpo'][key], row['full_ppo'][key], row['res_ppo'][key]], [.5, .5, -.5, -.5], +1)
        row['alg_' + key] = {'est': a['estimate'], 'ci': a['ci95_two_sided']}  # GRPO - PPO
    out['main'][ts] = row
# nine pre-specified tests
for k, v in FS['primary'].items():
    out['tests'][k] = {'est': v['estimate'], 'ci': v['ci95_two_sided'], 'p': v['p_one_sided'], 'p_holm': v['p_holm'], 'df': v['df'], 'outcome': v['outcome']}
out['tests_H2b'] = {k: {'est': v['estimate'], 'ci': v['ci95_two_sided']} for k, v in FS['descriptive']['H2b'].items()}

# ---------------- learning-rate curves (S3 + main configs) ----------------
S3 = SM['S3_lrcurve']
for c, ec in (('LF', 'LF'), ('PCL', 'pn10')):
    cur = {}
    for m in M4:
        for lr, d in S3[m].items():
            tags = d[c]; f = [Y(ec, t) for t in tags]; nm = [Y('nom', t) for t in tags]
            if None in f or None in nm: continue
            cur.setdefault(m, []).append({'lr': lr, 'fault': mean(f), 'fault_sd': float(np.std(f, ddof=1)), 'nom_loss': N0 - mean(nm),
                                          'nom_sd': float(np.std(nm, ddof=1)), 'n': len(f), 'fault_runs': f, 'nom_runs': nm})
        cur[m].sort(key=lambda r: float(r['lr']))
    out['lrcurve'][c] = cur

# ---------------- boundary configurations ----------------
def struct(groups_res_ppo, groups_res_grpo, groups_full_ppo, groups_full_grpo):
    c = contrast([groups_res_ppo, groups_res_grpo, groups_full_ppo, groups_full_grpo], [.5, .5, -.5, -.5], +1)
    return {'est': c['estimate'], 'ci': c['ci95_two_sided'], 'df': c['df']}
B = {}
for key, h in (('LF_ref', 'H1'), ('LF25', 'H3a'), ('LF75_eval', 'H3b')):
    B[key] = {'fault': {'est': FS['primary'][h]['estimate'], 'ci': FS['primary'][h]['ci95_two_sided']}}
B['LF_ref']['nom'] = out['main']['LF']['struct_nom']; B['LF_ref']['frozen'] = FROZ['LF']
B['LF25']['frozen'] = FROZ['LF25']; B['LF75_eval']['frozen'] = FROZ['LF75']
S2 = SM['S2_LF75train']
g = {m: ([Y('LF75', t) for t in S2[m]['models']], [Y('nom', t) for t in S2[m]['models']]) for m in M4}
B['LF75_retrain'] = {'fault': struct(g['res_ppo'][0], g['res_grpo'][0], g['full_ppo'][0], g['full_grpo'][0]),
                     'nom': struct(g['res_ppo'][1], g['res_grpo'][1], g['full_ppo'][1], g['full_grpo'][1]), 'frozen': FROZ['LF75'],
                     'means': {m: {'fault': mean(g[m][0]), 'nom': mean(g[m][1])} for m in M4}}
R5 = J('supp_rflf5_manifest.json')['models']
g = {m: ([Y('RFLF', t) for t in R5[m]['tags']], [Y('nom', t) for t in R5[m]['tags']]) for m in M4}
B['RFLF'] = {'fault': struct(g['res_ppo'][0], g['res_grpo'][0], g['full_ppo'][0], g['full_grpo'][0]),
             'nom': struct(g['res_ppo'][1], g['res_grpo'][1], g['full_ppo'][1], g['full_grpo'][1]), 'frozen': Y('RFLF', 'frozen'),
             'means': {m: {'fault': mean(g[m][0]), 'nom': mean(g[m][1]), 'n': len(g[m][0])} for m in M4}}
out['boundary'] = B


# ---------------- Allegro (learning-rate and training-procedure transfer) ----------------
AM = J('allegro_manifest.json'); AL = {}
for cond, ts, ec in (('TH', 'shlockTH', 'alTH'), ('Z2', 'shbz2', 'albz2')):
    g = {}
    for key, m in (('grpo', 'full_grpo'), ('ppo', 'full_ppo'), ('plugppo', 'res_ppo'), ('pluggrpo', 'res_grpo'), ('pluggrpoU', 'res_grpo_unpaired')):
        tags = AM['models'].get('%s|%s' % (key, ts))
        if not tags: continue
        g[m] = {'fault': [Y(ec, t) for t in tags], 'nom': [Y('alnom', t) for t in tags]}
    rec = {'frozen_fault': Y(ec, 'alfrozen'), 'frozen_nom': Y('alnom', 'alfrozen'), 'groups': g}
    for key in ('fault', 'nom'):
        c = contrast([g['res_ppo'][key], g['res_grpo'][key], g['full_ppo'][key], g['full_grpo'][key]], [.5, .5, -.5, -.5], +1)
        rec['struct_' + key] = {'est': c['estimate'], 'ci': c['ci95_two_sided']}
        a = contrast([g['full_grpo'][key], g['res_grpo'][key], g['full_ppo'][key], g['res_ppo'][key]], [.5, .5, -.5, -.5], +1)
        rec['alg_' + key] = {'est': a['estimate'], 'ci': a['ci95_two_sided']}
    AL[cond] = rec
out['allegro'] = AL

# ---------------- KL baseline ----------------
KS = J('kl_stats.json'); KM = J('kl_manifest.json')
out['kl'] = {'selected': KS['selected'], 'compare': KS['compare'], 'grid': {}}
for c, ec in (('LF', 'LF'), ('PCL', 'pn10')):
    rows = []
    for key, tags in KM['configs'].items():
        cc, lr, bi = key.split('|')
        if cc != c: continue
        f = [Y(ec, t['tag']) for t in tags]; nm = [Y('nom', t['tag']) for t in tags]
        rows.append({'lr': lr, 'beta': KM['beta'][bi], 'beta_id': bi, 'fault': mean(f), 'nom_loss': N0 - mean(nm), 'fault_runs': f})
    out['kl']['grid'][c] = rows

# ---------------- mean-scale intervention ----------------
MS = J('ms_stats.json'); out['ms'] = {'select': {k: v['selected_lr'] for k, v in MS['select'].items()}, 'same_lr': MS['same_lr']}
MM = J('ms_manifest.json'); stab = {}
for k, ents in MM['ms'].items():
    rows = []
    for e in ents:
        dn, s = [], []
        for p in range(12):
            f = os.path.join(D, 'runs', '%s_p%02d' % (e['tag'], p), 'curve.json')
            if not os.path.exists(f): break
            cv = json.load(open(f)); dn.append(max(x['delta_norm'] for x in cv)); s.append(mean([x['success'] for x in cv]))
        rows.append({'tag': e['tag'], 'max_delta_norm': max(dn) if dn else None, 'train_success': s})
    stab[k] = rows
out['ms']['stability'] = stab
# ---------------- constant diagnostics ----------------
out['const'] = J('ms_stats.json')['const']
CS = J('cst_stats.json'); out['cst'] = CS

# ---------------- additional quantities used in the text ----------------
X = {}
def ctr(groups, w): c = contrast(groups, w, +1); return {'est': c['estimate'], 'ci': c['ci95_two_sided'], 'df': c['df'], 'p1': c['p_one_sided']}
def lrrow(c, m, lr):
    for r in out['lrcurve'][c][m]:
        if r['lr'] == lr: return r
    raise KeyError((c, m, lr))
# H5 (ideal switching, unconstrained per-condition selection on LF): group means and nominal loss of those models
X['H5'] = {'groups': {g: {'mean': x['mean'], 'n': x['n']} for g, x in zip(['res_ppo', 'res_grpo', 'full_ppo', 'full_grpo'], FS['primary']['H5']['groups'])},
           'lr': {m: FS['lr'][m][1] for m in M4},
           'nom_loss': {m: lrrow('LF', m, FS['lr'][m][1])['nom_loss'] for m in M4}}
# S1: delta = 0.10 shared selection (dev) -> lrs; and LF-only per-condition selection at delta = 0.05
S1lr = {'full_grpo': '3e-5', 'full_ppo': '3e-5', 'res_ppo': '3e-3', 'res_grpo': '3e-3'}
LFonly = {'full_grpo': '1e-5', 'full_ppo': '3e-5', 'res_ppo': '3e-3', 'res_grpo': '1e-3'}
for key, sel in (('S1_d10', S1lr), ('LFonly_d05', LFonly)):
    rec = {'lr': sel}
    for c in ('LF', 'PCL'):
        rr = {m: lrrow(c, m, sel[m]) for m in M4}
        rec[c] = {'fault': ctr([rr['res_ppo']['fault_runs'], rr['res_grpo']['fault_runs'], rr['full_ppo']['fault_runs'], rr['full_grpo']['fault_runs']], [.5, .5, -.5, -.5]),
                  'nom': ctr([rr['res_ppo']['nom_runs'], rr['res_grpo']['nom_runs'], rr['full_ppo']['nom_runs'], rr['full_grpo']['nom_runs']], [.5, .5, -.5, -.5]),
                  'means': {m: {'fault': rr[m]['fault'], 'nom_loss': rr[m]['nom_loss'], 'n': rr[m]['n']} for m in M4}}
    X[key] = rec
# S4: fixed z offset 2 cm, four methods at the shared learning rates, and constant execution compensation
S4 = SM['S4_bz2']; g = {}
for m in M4:
    tags = S4[m]['models']; g[m] = {'fault': [Y('bz2', t) for t in tags], 'nom': [Y('nom', t) for t in tags]}
X['S4'] = {'frozen': Y('bz2', 'frozen'), 'comp_fault': Y('bz2comp', 'frozen'), 'comp_nom_loss': N0 - Y('nomcomp', 'frozen'),
           'means': {m: {'fault': mean(g[m]['fault']), 'nom_loss': N0 - mean(g[m]['nom'])} for m in M4},
           'struct_fault': ctr([g['res_ppo']['fault'], g['res_grpo']['fault'], g['full_ppo']['fault'], g['full_grpo']['fault']], [.5, .5, -.5, -.5]),
           'struct_nom': ctr([g['res_ppo']['nom'], g['res_grpo']['nom'], g['full_ppo']['nom'], g['full_grpo']['nom']], [.5, .5, -.5, -.5]),
           'zr_frozen': FROZ['zr'], 'zr_comp_fault': Y('zrcomp', 'frozen'), 'zr_comp_nom_loss': N0 - Y('nomcompzr', 'frozen'), 'zr_comp_c_m': SM['frozen']['zr_comp_c']}
# change relative to the frozen planner per method (one-sample t over runs), all main conditions
gain = {}
for ts, ec, _ in CONDS:
    gain[ts] = {m: {'est': mean(out['main'][ts][m]['fault']) - FROZ[ec], 'ci': [v - FROZ[ec] for v in tci(out['main'][ts][m]['fault'])]} for m in M4}
X['gain_vs_frozen'] = gain
# H4 components: G_A = Y(pi_A, A) - Y0(A); G_AB = Y(pi_A, A+B) - Y0(A+B)
H4 = {}
for key, ts, a, ab in (('LFpn_srcLF', 'LF', 'LF', 'LFpn'), ('LFpn_srcPCL', 'PCL', 'pn10', 'LFpn'), ('THzr_srcTH', 'TH', 'TH', 'THzr'), ('THzr_srcZR', 'ZR', 'zr', 'THzr')):
    rec = {}
    for m in M4:
        r = TAB[m][ts]
        ga = mean([x['Y'] for x in r[a]['runs'].values()]) - FROZ[a]; gab = mean([x['Y'] for x in r[ab]['runs'].values()]) - FROZ[ab]
        rec[m] = {'G_A': ga, 'G_AB': gab}
    rec['avg'] = {k: mean([rec[m][k] for m in M4]) for k in ('G_A', 'G_AB')}
    H4[key] = rec
X['H4'] = H4
# PCL strength curve per structure (raw success), for H2b description
pc = {}
for ec in ('pn10', 'pn15', 'pn20', 'pn05'):
    pc[ec] = {'frozen': FROZ[ec]}
    for m in M4: pc[ec][m] = mean([x['Y'] for x in TAB[m]['PCL'][ec]['runs'].values()])
X['pcl_strength'] = pc
# Allegro nominal loss relative to the Allegro frozen planner
X['allegro_nom_loss'] = {c: {m: AL[c]['frozen_nom'] - mean(v['nom']) for m, v in AL[c]['groups'].items()} for c in AL}
X['allegro_unpaired'] = {k: mean(AL['TH']['groups']['res_grpo_unpaired'][k]) - mean(AL['TH']['groups']['res_grpo'][k]) for k in ('fault', 'nom')}
# per-algorithm and seen/unseen descriptive (H1)
pa = FS['descriptive']['per_algorithm']; sp = FS['descriptive']['split']
X['per_alg'] = {k: {a: {'est': v[a]['estimate'], 'ci': v[a]['ci95_two_sided']} for a in ('ppo', 'grpo')} for k, v in pa.items()}
X['split'] = {s: {k: {'est': v['estimate'], 'ci': v['ci95_two_sided']} for k, v in sp[s].items()} for s in sp}
# model counts: distinct trained models with a final-protocol evaluation (constant heads are derived, not trained)
import re as _re
tags = set()
for dname in os.listdir(os.path.join(D, 'runs')):
    mm = _re.match(r'final_(seen|unseen)_([A-Za-z0-9]+)_(.*)$', dname)
    if mm and 'frozen' not in mm.group(3): tags.add(mm.group(3))
X['n_models'] = {'evaluated_tags': len(tags), 'derived_const_heads': sum(t.startswith('const_') for t in tags),
                 'trained': len([t for t in tags if not t.startswith('const_')]),
                 'allegro': len([t for t in tags if t.startswith('al_')])}
# ---------------- DemoFunGrasp case study (development stage; full-weight GRPO, 12 phases) ----------------
DJ = J('dfg_joint_metric.json')
def dj(key, f): return DJ[key][f]
DFG = {}
for cond, short, lr, so_seeds in (('LF', 'shlockLF', 'lr3', [1, 2, 3, 4]), ('TH', 'shlockTH', 'lr3', [1, 2, 3, 4, 5]), ('PCL', 'pn10', None, [1, 2, 3])):
    if cond == 'PCL':
        arms = {'SO': ['dfgpcl1cms%d' % s for s in so_seeds], 'FN': ['dfgpcl1cmfuncs%d' % s for s in (1, 2, 3)]}
        fault_dev = 'dfgsel_v2_munom_pn10_%s_64x8'; fault_test = 'dfgtest_%s_munom_pn10_%s_x8'
    else:
        arms = {'SO': ['dfg%s%ss%d' % (short, lr, s) for s in so_seeds], 'FN': ['dfg%s%sfuncs%d' % (short, lr, s) for s in (1, 2, 3)],
                'OF': ['dfg%s%soffrws%d' % (short, lr, s) for s in (1, 2, 3)]}
        fault_dev = 'dfgsel_v2_munom_' + short + '_%s_64x8'; fault_test = 'dfgtest_%s_munom_' + short + '_%s_x8'
    rec = {'frozen': {'fault_dev_succ': dj(fault_dev % 'base', 'success'), 'fault_dev_J': dj(fault_dev % 'base', 'J_4cm'),
                      'fault_test_J': (dj(fault_test % ('seen', 'base'), 'J_4cm') + dj(fault_test % ('unseen', 'base'), 'J_4cm')) / 2,
                      'nom_dev_J': dj('dfgsel_v2_munom_base_64x8', 'J_4cm'), 'nom_dev_succ': dj('dfgsel_v2_munom_base_64x8', 'success')}}
    for arm, tags in arms.items():
        r = {'n': len(tags), 'fault_dev_succ': [dj(fault_dev % t, 'success') for t in tags], 'fault_dev_J': [dj(fault_dev % t, 'J_4cm') for t in tags],
             'fault_test_J': [(dj(fault_test % ('seen', t), 'J_4cm') + dj(fault_test % ('unseen', t), 'J_4cm')) / 2 for t in tags],
             'nom_dev_J': [dj('dfgsel_v2_munom_%s_64x8' % t, 'J_4cm') for t in tags], 'nom_dev_succ': [dj('dfgsel_v2_munom_%s_64x8' % t, 'success') for t in tags]}
        r['mean'] = {k: mean(v) for k, v in r.items() if k != 'n'}
        rec[arm] = r
    for arm in [a for a in arms if a != 'SO']:
        rec[arm + '_minus_SO'] = {k: ctr([rec[arm][k], rec['SO'][k]], [1, -1]) for k in ('fault_dev_succ', 'fault_dev_J', 'fault_test_J', 'nom_dev_J')}
    DFG[cond] = rec
X['dfg'] = DFG
# ring + little finger: estimate from the first three runs per scheme (before the one-time extension to five)
g3 = {}
for m in M4:
    tags = [t for t in R5[m]['tags'] if t.endswith(('_s1', '_s2', '_s3'))]
    g3[m] = ([Y('RFLF', t) for t in tags], [Y('nom', t) for t in tags])
X['RFLF3'] = {'fault': ctr([g3['res_ppo'][0], g3['res_grpo'][0], g3['full_ppo'][0], g3['full_grpo'][0]], [.5, .5, -.5, -.5]),
              'nom': ctr([g3['res_ppo'][1], g3['res_grpo'][1], g3['full_ppo'][1], g3['full_grpo'][1]], [.5, .5, -.5, -.5]),
              'n': {m: len(g3[m][0]) for m in M4}}
# evaluation-only conditions of the locked final evaluation (all reported)
EO = {}
for ts, ecs in (('LF', ['LF25', 'LF75', 'LFpn']), ('PCL', ['pn05', 'pn15', 'pn20', 'LFpn']), ('TH', ['THzr']),
                ('ZR', ['zp1', 'zp3', 'zm2', 'x2', 'y2', 'zp4', 'zm4', 'THzr'])):
    for ec in ecs:
        gg = {m: [x['Y'] for x in TAB[m][ts][ec]['runs'].values()] for m in M4}
        EO['%s|%s' % (ts, ec)] = {'frozen': FROZ[ec], 'means': {m: mean(gg[m]) for m in M4}, 'n': {m: len(gg[m]) for m in M4},
                                  'struct': ctr([gg['res_ppo'], gg['res_grpo'], gg['full_ppo'], gg['full_grpo']], [.5, .5, -.5, -.5])}
X['evalonly'] = EO
out['extra'] = X
from make_feedback_analyses import augment
augment(out, D, contrast)
from audit_collision_split import build_collision_split
out['collision_split'] = build_collision_split(D)
from build_supp1007_record import build_record
SUPP_ROOT = os.environ.get('DEXGRASP_SUPP1007_ROOT', os.path.abspath(os.path.join(D, 'supp_20261007')))  # public-release layout
out['supp_1007'] = build_record(SUPP_ROOT, D, os.path.join(ROOT, 'audit', 'supp1007_recomputed.json'))
sel_dev=J('shared_lr_select.json')['methods']
out['development_boundary']={
 'full_ppo_3e5_pcl_loss':-sel_dev['full_ppo']['table']['3e-5']['PCL']['dN'],
 'full_grpo_3e5_lf_loss':-sel_dev['full_grpo']['table']['3e-5']['LF']['dN'],
 'full_ppo_3e5_pcl_excess':-sel_dev['full_ppo']['table']['3e-5']['PCL']['dN']-0.05,
 'source':'data/meta/temp/shared_lr_select.json; unrounded development values'}
json.dump(out, open(os.path.join(ROOT, 'numbers.json'), 'w'), indent=1)

# ======================= figures =======================
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
STYLE = os.path.join(ROOT, 'scripts', 'publication.mplstyle')
plt.style.use(STYLE)
mpl.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7})
COL = {'full_grpo': '#0072B2', 'full_ppo': '#009E73', 'res_ppo': '#CC79A7', 'res_grpo': '#D55E00', 'kl': '#000000', 'cst': '#E69F00'}
MK = {'full_grpo': 'o', 'full_ppo': '^', 'res_ppo': 'D', 'res_grpo': 's'}
MMW = 1 / 25.4

# Figure 2: recovery (Y - frozen) and nominal loss per condition, per-run points + means
fig, axes = plt.subplots(2, 1, figsize=(190 * MMW, 105 * MMW), sharex=True, layout='constrained')
rng = np.random.default_rng(20261004)
for i, (ts, ec, name) in enumerate(CONDS):
    row = out['main'][ts]
    for j, m in enumerate(M4):
        x0 = i + (j - 1.5) * 0.18
        rec = 100 * (np.array(row[m]['fault']) - row['frozen_fault']); los = 100 * (N0 - np.array(row[m]['nom']))
        for ax, v in ((axes[0], rec), (axes[1], los)):
            ax.scatter(x0 + rng.uniform(-.03, .03, len(v)), v, s=7, marker=MK[m], facecolor='none', edgecolor=COL[m], linewidth=.6, zorder=3)
            ax.plot([x0 - .07, x0 + .07], [v.mean()] * 2, color=COL[m], lw=1.6, zorder=4)
for ax in axes:
    ax.axhline(0, color='0.6', lw=.6, zorder=1)
    for i in range(1, len(CONDS)): ax.axvline(i - .5, color='0.88', lw=.5, zorder=0)
axes[1].axhline(5, color='0.3', lw=.6, ls='--', zorder=1); axes[1].text(5.48, 5.3, 'retention threshold (5 points)', ha='right', va='bottom', fontsize=6.5, color='0.3')
axes[0].set_ylabel('Fault recovery\n(points above frozen planner)'); axes[1].set_ylabel('Nominal success loss\n(points)')
axes[1].set_xticks(range(len(CONDS))); axes[1].set_xticklabels([f"{c[2]}\n(frozen {100*out['main'][c[0]]['frozen_fault']:.1f}%)" for c in CONDS])
handles = [plt.Line2D([], [], color=COL[m], marker=MK[m], markerfacecolor='none', lw=1.6, ms=4, label=LAB[m]) for m in M4]
axes[0].legend(handles=handles, ncol=4, loc='upper right', frameon=False)
fig.savefig(os.path.join(FIG, 'fig_main.pdf')); fig.savefig(os.path.join(FIG, 'fig_main.png'), dpi=300); plt.close(fig)

# Figure 3: recovery-retention plane, LF and PCL
# These main-text figures must remain legible in the narrower elsarticle layout.
mpl.rcParams.update({'font.size': 10.5, 'axes.labelsize': 10.5,
                     'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
                     'legend.fontsize': 10})
fig, axes = plt.subplots(1, 2, figsize=(190 * MMW, 88 * MMW), layout='constrained')
MAIN_LR = {'full_grpo': '1e-5', 'full_ppo': '1e-5', 'res_ppo': '3e-3', 'res_grpo': '1e-3'}
for ax, (c, title, fz) in zip(axes, (('LF', 'Little-finger lock', FROZ['LF']), ('PCL', 'Point-cloud noise 1 cm', FROZ['pn10']))):
    for m in M4:
        pts = out['lrcurve'][c][m]
        xs = [100 * p['nom_loss'] for p in pts]; ys = [100 * p['fault'] for p in pts]
        ax.plot(xs, ys, color=COL[m], lw=.9, marker=MK[m], ms=3.5, markerfacecolor='white', label=LAB[m])
        for p in pts:
            if p['lr'] != MAIN_LR[m]:
                assert len(p['nom_runs']) == len(p['fault_runs']) == p['n']
                ax.scatter(100*(N0-np.asarray(p['nom_runs'])),100*np.asarray(p['fault_runs']),
                           s=10,marker=MK[m],color=COL[m],alpha=.26,linewidths=0,zorder=2)
            if p['lr'] == MAIN_LR[m]:
                # Each point uses paired nominal/fault scores from one model,
                # in the common manifest tag order; no jitter or run linkage.
                assert len(p['nom_runs']) == len(p['fault_runs']) == p['n']
                ax.scatter(100 * (N0 - np.asarray(p['nom_runs'])),
                           100 * np.asarray(p['fault_runs']), s=12,
                           marker=MK[m], color=COL[m], alpha=.35,
                           linewidths=0, zorder=3)
                ax.plot(100 * p['nom_loss'], 100 * p['fault'], marker=MK[m], ms=6, color=COL[m], zorder=5)
    kl = out['kl']['grid'][c]
    ax.scatter([100 * r['nom_loss'] for r in kl], [100 * r['fault'] for r in kl], marker='x', s=16, color=COL['kl'], linewidth=.8, label='Full-weight GRPO + KL (grid)', zorder=6)
    if c == 'LF':
        g = out['cst']['grid_test']['LF']
        ax.plot([100 * g[l]['nom_loss'] for l in ('3e-3', '1e-2', '3e-2')], [100 * g[l]['fault_mean'] for l in ('3e-3', '1e-2', '3e-2')], color=COL['cst'], lw=.9, ls=':', marker='v', ms=3.5, label='Direct constant $(c,\\rho)$')
    ax.axhline(100 * fz, color='0.6', lw=.6, ls='--'); ax.text(0.2, 100 * fz + .6, 'frozen planner', fontsize=9.5, color='0.3', bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .2})
    ax.axvline(5, color='0.3', lw=.6, ls='--')
    ax.set_xlabel('Nominal success loss (points)'); ax.set_ylabel('Fault success (%)'); ax.set_title(title, fontsize=10.5)
for ax, condition in zip(axes, ('LF', 'PCL')):
    all_losses = [100*(N0-value) for method in out['lrcurve'][condition].values()
                  for point in method for value in point['nom_runs']]
    ax.set_xlim(min(-1, math.floor(min(all_losses))-1), 5*math.ceil(max(all_losses)/5))
axes[0].legend(loc='lower right', bbox_to_anchor=(1, .08), frameon=False, fontsize=10)
fig.savefig(os.path.join(FIG, 'fig_tradeoff.pdf')); fig.savefig(os.path.join(FIG, 'fig_tradeoff.png'), dpi=300); plt.close(fig)

# Figure 4: boundary forest plot
rowsB = [('LF_ref', 'Little finger, reference lock\n(main analysis, H1)'), ('LF25', 'Little finger, 25% lock\n(LF-trained models, H3a)'),
         ('LF75_eval', 'Little finger, 75% lock\n(LF-trained models, H3b)'), ('LF75_retrain', 'Little finger, 75% lock\n(retrained at 75%)'),
         ('RFLF', 'Ring + little finger lock\n(5 runs per method)')]
fig, axes = plt.subplots(1, 2, figsize=(190 * MMW, 76 * MMW), sharey=True, layout='constrained', gridspec_kw={'width_ratios': [1.6, 1]})
for k, (key, lab) in enumerate(rowsB):
    yk = len(rowsB) - 1 - k
    f = B[key]['fault']; axes[0].errorbar(100 * f['est'], yk, xerr=[[100 * (f['est'] - f['ci'][0])], [100 * (f['ci'][1] - f['est'])]], fmt='o', color='#D55E00', ms=4, capsize=2, lw=1)
    if 'nom' in B[key]:
        nn = B[key]['nom']; axes[1].errorbar(100 * nn['est'], yk, xerr=[[100 * (nn['est'] - nn['ci'][0])], [100 * (nn['ci'][1] - nn['est'])]], fmt='s', color='#0072B2', ms=4, capsize=2, lw=1)
for ax in axes: ax.axvline(0, color='0.5', lw=.6)
axes[0].set_yticks(range(len(rowsB))); axes[0].set_yticklabels([l for _, l in rowsB][::-1])
axes[0].set_xlabel('Fault-success difference\n(residual $-$ full-weight, points)'); axes[1].set_xlabel('Nominal-success difference\n(points)')
axes[1].text(-2.6, 2.5, 'Same models as reference:\nnominal success unchanged\nby test-time lock posture', ha='center', va='center', fontsize=9.5, color='0.3')
fig.savefig(os.path.join(FIG, 'fig_boundary.pdf')); fig.savefig(os.path.join(FIG, 'fig_boundary.png'), dpi=300); plt.close(fig)
print('numbers.json and figures written')

# Rebuild the 8 October analyses from the archived attempts and shared input bank.
from pathlib import Path
import subprocess
base=Path(D)  # public-release layout: all archives unpack under data/
subprocess.run([sys.executable,'-B',str(Path(HERE)/'audit_bias1008.py'),
 '--new-root',str(base/'supp_20261008'),'--data-root',str(D),
 '--out',str(Path(ROOT)/'audit/bias1008_recomputed.json')],check=True)
subprocess.run([sys.executable,'-B',str(Path(HERE)/'make_supp1008.py'),
 '--diag',str(base/'output_analysis_20261008'),'--tags',str(base/'output_analysis_20261008/outdiag_tags.json'),
 '--perf',str(D),str(base/'supp_20261007'),str(base/'supp_20261008'),
 '--summary',str(base/'output_analysis_20261008/summary.json'),
 '--bias-audit',str(Path(ROOT)/'audit/bias1008_recomputed.json')],check=True)
