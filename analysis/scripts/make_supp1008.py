#!/usr/bin/env python3
"""Record, tables, macros and figure for the analyses added on 8 October 2026 (manuscript v13).

(1) Checkpoint output analysis (no training, no simulation). For every trained LF model, the change of the pre-tanh action mean
    dm(s) = m(s) - m0(s) on one saved set of 2,048 nominal training-object states (64 objects x 8 repeats x 4 pose draws, seed 0),
    read from inputs.npz / models.npz written by research_output_diag.py. Metrics (per training run):
      shift      = mean_s 1/2 sum_j (dm_j / sigma0_j)^2        sigma0-normalised mean shift (not a policy KL)
      l1         = sum_j mean_s |dm_j|
      high_share = share of l1 on the 12 dimensions with the largest frozen sigma0
      state_share= sum_j Var_s(dm_j) / sum_j E_s[dm_j^2]       (unweighted; 0 for a constant shift)
      cos_ref    = cosine between the run's mean shift and the mean over selected residual GRPO runs
      dq_rfj3    = mean change of the RFJ3 joint control target (rad), q = clip(q_ref + tanh(u) * range, limits)
    Recomputes tools/output_diag_summary.py and requires agreement with its summary.json (1e-9).
(2) Output-shift coordinate control: selection, test contrasts and context, from audit_bias1008.py (attempt flags).
(3) Per-run test success of the analysed models, rebuilt from attempt flags.

Writes only numbers.json['supp_1008'], tables/tab_output.tex, tables/tab_supp1008_*.tex, tables/supp1008_macros.tex and
figures/fig_output.{pdf,png}. Everything here is descriptive and outside the nine-test family."""
import argparse, json, math, os, statistics
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--diag')
ap.add_argument('--from-record', action='store_true', help='Render tables and figure from numbers.json without reading raw inputs')
ap.add_argument('--tags')
ap.add_argument('--perf', nargs='+', help='roots containing runs/<name>/result.json')
ap.add_argument('--bias-audit', default=None)
ap.add_argument('--summary', default=None, help='summary.json of tools/output_diag_summary.py, for cross-checking')
ap.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
a = ap.parse_args()
ROOT = Path(a.root)
N = json.loads((ROOT / 'numbers.json').read_text())

if a.from_record:
    out=N['supp_1008']; oa=out['output_analysis']
    G=oa['groups']; X=oa['extra_groups']; bias=out.get('bias_control')
    m0=np.empty((oa['states'],0)); c0=np.array([oa['clip_frac_frozen']])
    rho=oa['spearman_dq_rfj3_vs_LF']; rho_shift=oa['spearman_shift_vs_LF']
    allx=[None]*oa['n_runs_rank']
    LS={'3e-3':'3em3','1e-2':'1em2','3e-2':'3em2','1e-1':'1em1'}
else:
    # ---------------------------------------------------------------- per-run test success from attempt flags
    SLOTS = {'seen': 141, 'unseen': 100}
    ATT = {'seen': 1128, 'unseen': 800}
    _cache = {}


    def score(name, split):
        if name not in _cache:
            for r in a.perf:
                p = Path(r) / 'runs' / name / 'result.json'
                if p.exists():
                    rec = json.loads(p.read_text())
                    break
            else:
                raise FileNotFoundError(name)
            assert rec.get('status') == 'completed', name
            eps = [ep['official_ppo_print_successes'] for ep in rec['episodes']]
            assert len(eps) == 8 and all(len(f) == SLOTS[split] for f in eps), name
            assert all(x in (0, 1) for f in eps for x in f), name
            n, s = sum(map(len, eps)), sum(map(sum, eps))
            assert n == ATT[split] and abs(s / n - rec['totals']['success_rate']) < 1e-12, name
            _cache[name] = s / n
        return _cache[name]


    def Y(cond, tag):
        return (score('final_seen_%s_%s' % (cond, tag), 'seen') + score('final_unseen_%s_%s' % (cond, tag), 'unseen')) / 2


    # ---------------------------------------------------------------- output metrics
    D = Path(a.diag)
    info = json.loads((D / 'env_info.json').read_text())
    inp = np.load(D / 'inputs.npz')
    M = np.load(D / 'models.npz')
    man = json.loads((D / 'models_manifest.json').read_text())
    tags = json.loads(Path(a.tags).read_text())
    m0 = inp['m0'].astype(np.float64)
    s0 = np.array(info['sigma0'], dtype=float)
    H = np.argsort(-s0)[:12]
    names = ['wrist_tx', 'wrist_ty', 'wrist_tz', 'wrist_rx', 'wrist_ry', 'wrist_rz'] + info['hand_joint_names']
    rng_ = float(info['grasp_pose_range'])
    lo, hi, qref = np.array(info['hand_lower']), np.array(info['hand_upper']), np.array(info['q_ref'])
    J = names.index('rh_RFJ3')
    assert J == 17


    def targets(u):
        raw = qref + np.tanh(u[:, 6:]) * rng_
        return np.clip(raw, lo, hi), (raw < lo) | (raw > hi)


    q0, c0 = targets(m0)
    met = {}
    for lab in M.files:
        dm = M[lab].astype(np.float64) - m0
        e2, var, mabs = (dm ** 2).mean(0), dm.var(0), np.abs(dm).mean(0)
        q, cl = targets(M[lab].astype(np.float64))
        met[lab] = {'shift': float((0.5 * (dm / s0) ** 2).sum(1).mean()), 'l1': float(mabs.sum()),
                    'high_share': float(mabs[H].sum() / mabs.sum()), 'state_share': float(var.sum() / e2.sum()),
                    'mean_dm': dm.mean(0), 'dq_rfj3': float((q - q0)[:, J - 6].mean()), 'dm_rfj3': float(dm[:, J].mean()),
                    'sum_abs_dq': float(np.abs(q - q0).mean(0).sum()), 'clip': float(cl.mean()), 'kind': man[lab]['kind']}
    REF = [l for l in met if l.startswith('rgLF_s')]
    ref = np.mean([met[l]['mean_dm'] for l in REF], 0)
    assert int(np.argmax(np.abs(ref))) == J  # RFJ3 carries the largest mean shift of the reference group
    for lab, r in met.items():
        v = r.pop('mean_dm')
        r['cos_ref'] = float(v @ ref / (np.linalg.norm(v) * np.linalg.norm(ref)))
        r['mean_dm'] = v.tolist()
    if a.summary:
        S = json.loads(Path(a.summary).read_text())['per_model']
        for lab, r in met.items():
            s = S[lab]
            for k1, k2 in (('shift', 'kl_raw'), ('l1', 'l1'), ('high_share', 'high_share'), ('state_share', 'state_share'),
                           ('cos_ref', 'cos_ref'), ('clip', 'clip_frac'), ('sum_abs_dq', 'hand_sum_abs_dq')):
                assert abs(r[k1] - s[k2]) <= 1e-9 * max(1, abs(s[k2])), (lab, k1, r[k1], s[k2])
            assert abs(r['dq_rfj3'] - s['hand_mean_dq'][J - 6]) <= 1e-9, lab

    # ---------------------------------------------------------------- groups
    bias = json.loads(Path(a.bias_audit).read_text()) if a.bias_audit else None
    selB = {arm: bias['selection']['arms'][arm]['selected'] for arm in ('raw', 'sigma')} if bias else {}
    LS = {'3e-3': '3em3', '1e-2': '1em2', '3e-2': '3em2', '1e-1': '1em1'}
    GROUPS = [  # key, scheme, rate, diag label stem, selected, test condition
        ('rgLF', 'res_grpo', '1e-3', True), ('rg3LF', 'res_grpo', '3e-3', False), ('rpLF', 'res_ppo', '3e-3', True),
        ('fgLF1em5', 'full_grpo', '1e-5', True), ('fgLF3em5', 'full_grpo', '3e-5', False), ('fgLF1em4', 'full_grpo', '1e-4', False),
        ('fpLF1em5', 'full_ppo', '1e-5', True), ('fpLF3em5', 'full_ppo', '3e-5', False), ('cstLF', 'cst', '3e-2', True)]
    if bias:
        GROUPS += [('bxrLF' + LS[lr], 'bias_raw', lr, lr == selB['raw']) for lr in ('3e-3', '1e-2', '3e-2')]
        GROUPS += [('bxsLF' + LS[lr], 'bias_sigma', lr, lr == selB['sigma']) for lr in ('1e-2', '3e-2', '1e-1')]
    EXTRA = [('fsrLF1em3', 'matched_res', '1e-3', 'LF'), ('fsrLF3em3', 'matched_res', '3e-3', 'LF'), ('fsfLF', 'matched_full', '1e-5', 'LF'),
             ('rgLF75', 'res_grpo', '1e-3', 'LF75'), ('rpLF75', 'res_ppo', '3e-3', 'LF75'), ('fgLF75', 'full_grpo', '1e-5', 'LF75'),
             ('fpLF75', 'full_ppo', '1e-5', 'LF75')]
    KEYS = ('shift', 'l1', 'high_share', 'state_share', 'cos_ref', 'dq_rfj3', 'dm_rfj3', 'sum_abs_dq', 'clip')


    def runs_of(stem):
        labs = sorted([l for l in met if l.rsplit('_s', 1)[0] == stem], key=lambda l: int(l.rsplit('_s', 1)[1]))
        assert labs, stem
        return labs


    def summarise(stem, scheme, lr, sel, cond):
        labs = runs_of(stem)
        rec = {'scheme': scheme, 'lr': lr, 'selected': sel, 'test_condition': cond, 'runs': labs,
               'tags': [tags[l] for l in labs], 'per_run': {k: [met[l][k] for l in labs] for k in KEYS}}
        rec['per_run']['fault'] = [Y(cond, tags[l]) for l in labs]
        rec['per_run']['nom'] = [Y('nom', tags[l]) for l in labs]
        rec['mean'] = {k: statistics.mean(v) for k, v in rec['per_run'].items()}
        rec['sd'] = {k: (statistics.stdev(v) if len(v) > 1 else None) for k, v in rec['per_run'].items()}
        return rec


    G = {g[0]: summarise(g[0], g[1], g[2], g[3], 'LF') for g in GROUPS}
    X = {g[0]: summarise(g[0], g[1], g[2], False, g[3]) for g in EXTRA}

    # test success of the locked groups must match the archived record exactly (same runs, any order)
    LR = N['lrcurve']['LF']
    for key, rec in G.items():
        sch = rec['scheme']
        if sch in LR:
            p = [q for q in LR[sch] if q['lr'] == rec['lr']][0]
            assert sorted(p['fault_runs']) == sorted(rec['per_run']['fault']), key
            assert sorted(p['nom_runs']) == sorted(rec['per_run']['nom']), key
    assert sorted(N['cst']['grid_test']['LF']['3e-2']['fault']) == sorted(G['cstLF']['per_run']['fault'])

    # rank association between the RFJ3 target change and LF success across all analysed LF runs (descriptive)
    def ranks(x):
        o = np.argsort(x, kind='mergesort'); r = np.empty(len(x)); r[o] = np.arange(len(x))
        xs = np.asarray(x)[o]; i = 0
        while i < len(x):  # average ties
            j = i
            while j + 1 < len(x) and xs[j + 1] == xs[i]: j += 1
            r[o[i:j + 1]] = (i + j) / 2; i = j + 1
        return r


    allx = [v for rec in G.values() for v in rec['per_run']['dq_rfj3']]
    ally = [v for rec in G.values() for v in rec['per_run']['fault']]
    rho = float(np.corrcoef(ranks(allx), ranks(ally))[0, 1])
    allsh = [v for rec in G.values() for v in rec['per_run']['shift']]
    rho_shift = float(np.corrcoef(ranks(allsh), ranks(ally))[0, 1])

    import hashlib
    H256 = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    inputs_rec = {f: H256(D / f) for f in ('inputs.npz', 'models.npz', 'env_info.json', 'models_manifest.json')}
    inputs_rec['outdiag_tags.json'] = H256(a.tags)
    inputs_rec['result_files_read'] = len(_cache)
    inputs_rec['research_output_diag_sha256'] = info.get('script_sha256')
    out = {'role': 'descriptive; post-lock analyses added on 8 October 2026; outside the nine-test family',
           'output_analysis': {
               'states': int(m0.shape[0]), 'state_set': 'research_grpo_v1_phase00_64x8: 64 training objects x 8 repeats x 4 pose draws, seed 0',
               'sigma0': s0.tolist(), 'dim_names': names, 'high_sigma0_dims': sorted(int(j) for j in H), 'rfj3_dim': J,
               'clip_frac_frozen': float(c0.mean()), 'reference_runs': REF, 'groups': G, 'extra_groups': X,
               'inputs': inputs_rec, 'spearman_dq_rfj3_vs_LF': rho, 'spearman_shift_vs_LF': rho_shift, 'n_runs_rank': len(allx)}}
    if bias:
        out['bias_control'] = {k: bias[k] for k in ('selection', 'stats', 'context', 'inputs', 'checks')}
    N['supp_1008'] = out
    (ROOT / 'numbers.json').write_text(json.dumps(N, indent=1))

    # ---------------------------------------------------------------- console summary
    print('states %d | frozen clip %.3f | RFJ3 sigma0 %.3f | Spearman(dq_RFJ3, LF) %.3f over %d runs | Spearman(shift, LF) %.3f'
          % (m0.shape[0], c0.mean(), s0[J], rho, len(allx), rho_shift))
    print('%-11s %-10s %-5s %3s %2s %6s %6s %6s %6s %5s %5s %5s %6s' % ('key', 'scheme', 'lr', 'sel', 'n', 'LF', 'nom', 'shift', 'L1', 'hi', 'state', 'cos', 'dqRF'))
    for key, rec in list(G.items()) + list(X.items()):
        m = rec['mean']
        print('%-11s %-10s %-5s %3s %2d %6.1f %6.1f %6.1f %6.2f %5.2f %5.2f %5.2f %6.3f' % (
            key, rec['scheme'], rec['lr'], 'Y' if rec['selected'] else '', len(rec['runs']), 100 * m['fault'], 100 * m['nom'], m['shift'],
            m['l1'], m['high_share'], m['state_share'], m['cos_ref'], m['dq_rfj3']))


# ---------------------------------------------------------------- tables and macros (from the record only)
T = ROOT / 'tables'
RATE = {'1e-5': '$10^{-5}$', '3e-5': r'$3\cdot10^{-5}$', '1e-4': '$10^{-4}$', '1e-3': '$10^{-3}$', '3e-3': r'$3\cdot10^{-3}$',
        '1e-2': '$10^{-2}$', '3e-2': r'$3\cdot10^{-2}$', '1e-1': '$10^{-1}$'}
NAME = {'res_grpo': 'Residual GRPO', 'res_ppo': 'Residual PPO', 'full_grpo': 'Full-weight GRPO', 'full_ppo': 'Full-weight PPO',
        'cst': 'Constant + scale (48)', 'bias_raw': 'Raw shift (24)', 'bias_sigma': r'$\sigma_0$-scaled shift (24)',
        'matched_res': 'Residual GRPO, setting-matched', 'matched_full': 'Full-weight GRPO, new runs'}


def pc(x): return '%.1f' % (100 * x)


def sg(x, d=1):
    v = round(x, d)
    if v == 0: v = 0.0
    return ('$+%.*f$' % (d, v)) if v > 0 else ('$%.*f$' % (d, v) if v < 0 else '$%.*f$' % (d, abs(v)))


def num(x, d=1):
    v = round(x, d)
    return ('$-%.*f$' % (d, abs(v))) if v < 0 else '%.*f' % (d, abs(v))


def ci(w, scale=100):
    return '[%s, %s]' % (num(scale * w['ci'][0]), num(scale * w['ci'][1]))


def row_output(rec, label):
    m = rec['mean']
    return '%s & %d & %s & %s & %.1f & %.2f & %.2f & %.2f & %.2f\\\\' % (
        label, len(rec['runs']), pc(m['fault']), pc(m['nom']), m['shift'], m['high_share'], m['state_share'], m['cos_ref'], m['dq_rfj3'])


fzLF, fzN = N['frozen']['LF'], N['frozen']['nom']
order_main = ['rgLF', 'rpLF', 'fgLF1em5', 'fpLF1em5', 'cstLF'] + ([k for k, r in G.items() if r['scheme'] == 'bias_raw' and r['selected']] +
                                                               [k for k, r in G.items() if r['scheme'] == 'bias_sigma' and r['selected']])
L = [r'\begin{table}[t]', r'\centering',
     r'\caption{Selected reference-lock configurations. LF and nominal success are test percentages; output statistics use common nominal training-object states. High-$\sigma_0$: fraction of absolute mean change on the 12 largest-scale dimensions. State: fraction of squared mean change that varies across states. RFJ3: signed control-target change after squashing and clipping. Parenthesized numbers give parameter counts for constant shifts. All entries are means over training runs; rates and complete output statistics are in Supplementary Table~\ref{tab:supp1008_output}.}',
     r'\label{tab:output}', r'\small', r'\setlength{\tabcolsep}{5pt}', r'\begin{adjustbox}{max width=\textwidth}', r'\begin{tabular}{@{}lrrrrrr@{}}', r'\toprule',
     r'Scheme & Runs & LF (\%) & Nominal (\%) & High-$\sigma_0$ & State & RFJ3 (rad)\\',r'\midrule',
     r'Frozen planner & -- & %s & %s & -- & -- & 0\\' % (pc(fzLF),pc(fzN))]
for k in order_main:
    r=G[k];m=r['mean']
    L.append('%s & %d & %s & %s & %.2f & %.2f & %.2f\\\\' %
             (NAME[r['scheme']],len(r['runs']),pc(m['fault']),pc(m['nom']),m['high_share'],m['state_share'],m['dq_rfj3']))
L += [r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}',r'\end{table}']
(T/'tab_output.tex').write_text('\n'.join(L)+'\n')

L = [r'\begin{table}[!htbp]', r'\centering',
     r'\caption{Output changes of every analysed LF model (means over training runs; definitions in Section~\ref{supp-sec:outputs}). Upper block: models trained and evaluated under the reference LF lock, including learning rates that were not selected; a dagger marks the selected rate. Middle block: the setting-matched residual and new full-weight GRPO runs of Section~\ref{supp-sec:matched}. Lower block: models retrained at the 75\% lock posture, with success under that posture; their cosine is taken against the reference-LF direction. L1 is $\sum_j\mathbb E_s|\Delta m_j|$; clip is the share of hand control targets outside the joint limits before clipping (frozen planner \FrozenClip).}',
     r'\label{tab:supp1008_output}', r'\small', r'\setlength{\tabcolsep}{3pt}', r'\begin{adjustbox}{max width=\textwidth}', r'\begin{tabular}{@{}llrrrrrrrrrr@{}}', r'\toprule',
     r'Scheme & Rate & Runs & Fault & Nominal & Shift & L1 & High-$\sigma_0$ & State & Cosine & RFJ3 & Clip\\', r'\midrule']


def row_full(rec):
    m = rec['mean']
    return '%s & %s%s & %d & %s & %s & %.1f & %.2f & %.2f & %.2f & %.2f & %.2f & %.3f\\\\' % (
        NAME[rec['scheme']], RATE[rec['lr']], r'\,$^{\dagger}$' if rec['selected'] else '', len(rec['runs']), pc(m['fault']), pc(m['nom']),
        m['shift'], m['l1'], m['high_share'], m['state_share'], m['cos_ref'], m['dq_rfj3'], m['clip'])


for k, r in G.items(): L.append(row_full(r))
L.append(r'\midrule')
for k in ('fsrLF1em3', 'fsrLF3em3', 'fsfLF'): L.append(row_full(X[k]))
L.append(r'\midrule')
for k in ('rgLF75', 'rpLF75', 'fgLF75', 'fpLF75'): L.append(row_full(X[k]))
L += [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{table}']
(T / 'tab_supp1008_output.tex').write_text('\n'.join(L) + '\n')

MAC = ['% Generated from numbers.json supp_1008 by make_supp1008.py; do not edit.']
def mac(n, t): MAC.append('\\newcommand{\\%s}{%s}' % (n, t))
mac('FrozenClip', '%.2f' % c0.mean()); mac('OutStates', '{,}'.join([str(m0.shape[0])[:-3], str(m0.shape[0])[-3:]]))
mac('OutRhoRF', '%.2f' % rho); mac('OutRhoShift', '%.2f' % rho_shift); mac('OutRankRuns', str(len(allx)))
for k, nm in (('rgLF', 'RG'), ('rpLF', 'RP'), ('fgLF1em5', 'FG'), ('fpLF1em5', 'FP'), ('cstLF', 'CST'), ('fgLF3em5', 'FGthree'), ('fgLF1em4', 'FGten'), ('fpLF3em5', 'FPthree'), ('rg3LF', 'RGthree')):
    m = G[k]['mean']
    mac('Out%sShift' % nm, '%.1f' % m['shift']); mac('Out%sHigh' % nm, '%.2f' % m['high_share']); mac('Out%sState' % nm, '%.2f' % m['state_share'])
    mac('Out%sCos' % nm, '%.2f' % m['cos_ref']); mac('Out%sRF' % nm, '%.2f' % m['dq_rfj3']); mac('Out%sLone' % nm, '%.2f' % m['l1'])
hs = [G[k]['mean']['high_share'] for k in G if G[k]['scheme'] in ('res_grpo', 'res_ppo')]
mac('OutResHighMin', '%.2f' % min(hs)); mac('OutResHighMax', '%.2f' % max(hs))
hs = [G[k]['mean']['high_share'] for k in G if G[k]['scheme'] in ('full_grpo', 'full_ppo')]
mac('OutFullHighMin', '%.2f' % min(hs)); mac('OutFullHighMax', '%.2f' % max(hs))
c75 = [X[k]['mean']['cos_ref'] for k in ('rgLF75', 'rpLF75', 'fgLF75', 'fpLF75')]
r75 = [X[k]['mean']['dq_rfj3'] for k in ('rgLF75', 'rpLF75', 'fgLF75', 'fpLF75')]
s75 = [X[k]['mean']['state_share'] for k in ('rgLF75', 'rpLF75', 'fgLF75', 'fpLF75')]
f75 = [X[k]['mean']['fault'] for k in ('rgLF75', 'rpLF75', 'fgLF75', 'fpLF75')]
mac('OutSFCosMin', '%.2f' % min(c75)); mac('OutSFCosMax', '%.2f' % max(c75)); mac('OutSFRFMax', '%.2f' % max(r75))
mac('OutSFStateMin', '%.2f' % min(s75)); mac('OutSFStateMax', '%.2f' % max(s75)); mac('OutSFSuccMin', pc(min(f75))); mac('OutSFSuccMax', pc(max(f75)))

if bias:
    SEL, ST, CX = bias['selection'], bias['stats'], bias['context']
    ARMS = (('raw', 'Raw, $m_0+\Delta b$'), ('sigma', r'$\sigma_0$-scaled, $m_0+\sigma_0\odot c$'))
    LRS = {'raw': ('3e-3', '1e-2', '3e-2'), 'sigma': ('1e-2', '3e-2', '1e-1')}
    # S25 development selection
    L = [r'\begin{table}[!htbp]', r'\centering',
         r'\caption{Development selection in the output-shift coordinate control. $S$: development fault success (\%%), 512 attempts per run; $\Delta N$: change of nominal development success from the frozen planner re-measured on the same machine (%s\%%), in points, 2,048 attempts per run. Means over three runs. A rate is feasible if $\Delta N\ge-5$ in both conditions; each arm selects its feasible rate with the highest mean $S$.}' % pc(SEL['N0_this_machine']),
         r'\label{tab:supp1008_selection}', r'\small', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{@{}llrrrrll@{}}', r'\toprule',
         r'Coordinate & Rate & \multicolumn{2}{c}{LF} & \multicolumn{2}{c}{PCL} & Feasible & Selected\\', r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',
         r' & & $S$ & $\Delta N$ & $S$ & $\Delta N$ & & \\', r'\midrule']
    for arm, an in ARMS:
        for lr in LRS[arm]:
            r = SEL['arms'][arm]['rates'][lr]; pcd = r['per_condition']
            L.append(r'%s & %s & %s & %s & %s & %s & %s & %s\\' % (an if lr == LRS[arm][0] else '', RATE[lr], pc(pcd['LF']['S']), sg(100 * pcd['LF']['dN']),
                     pc(pcd['PCL']['S']), sg(100 * pcd['PCL']['dN']), 'yes' if r['feasible'] else 'no', 'yes' if SEL['arms'][arm]['selected'] == lr else ''))
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (T / 'tab_supp1008_selection.tex').write_text('\n'.join(L) + '\n')
    # S26 test means
    CK = [('LF|LF', 'LF'), ('LF|nom', 'Nom.'), ('LF|LF25', '25\\%'), ('LF|LF75', '75\\%'), ('PCL|pn10', 'PCL'), ('PCL|nom', 'Nom.')]
    L = [r'\begin{table}[!htbp]', r'\centering',
         r'\caption{Test success (\%) in the output-shift coordinate control: means over three training runs of the average of seen and unseen success (1,928 attempts per run and condition). A dagger marks each arm\textquoteright s selected rate. Reference rows use one frozen checkpoint and three trained runs per locked GRPO group, evaluated on another machine.}',
         r'\label{tab:supp1008_means}', r'\small', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{@{}llrrrrrr@{}}', r'\toprule',
         r' & & \multicolumn{4}{c}{LF-trained models} & \multicolumn{2}{c}{PCL-trained models}\\', r'\cmidrule(lr){3-6}\cmidrule(l){7-8}',
         r'Coordinate & Rate & ' + ' & '.join(c[1] for c in CK) + r'\\', r'\midrule']
    for arm, an in ARMS:
        for lr in LRS[arm]:
            vals = [pc(statistics.mean(ST['conditions'][k]['runs']['%s|%s' % (arm, lr)])) for k, _ in CK]
            L.append(r'%s & %s%s & %s\\' % (an if lr == LRS[arm][0] else '', RATE[lr], r'\,$^{\dagger}$' if SEL['arms'][arm]['selected'] == lr else '', ' & '.join(vals)))
    L.append(r'\midrule')
    fzk = {'LF|LF': 'LF', 'LF|nom': 'nom', 'LF|LF25': 'LF25', 'LF|LF75': 'LF75', 'PCL|pn10': 'pn10', 'PCL|nom': 'nom'}
    L.append(r'Frozen planner & -- & ' + ' & '.join(pc(CX['frozen_locked'][fzk[k]]) for k, _ in CK) + r'\\')
    L.append(r'Residual GRPO & $10^{-3}$ & ' + ' & '.join(pc(statistics.mean(CX['conditions'][k]['residual_grpo_locked'])) for k, _ in CK) + r'\\')
    L.append(r'Full-weight GRPO & $10^{-5}$ & ' + ' & '.join(pc(statistics.mean(CX['conditions'][k]['fullweight_grpo_locked'])) for k, _ in CK) + r'\\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (T / 'tab_supp1008_means.tex').write_text('\n'.join(L) + '\n')
    # S27 contrasts
    CN = {'LF|LF': 'LF models, LF', 'LF|nom': 'LF models, nominal', 'LF|LF25': 'LF models, 25\\% posture', 'LF|LF75': 'LF models, 75\\% posture',
          'PCL|pn10': 'PCL models, PCL', 'PCL|nom': 'PCL models, nominal'}
    sr, ss = SEL['arms']['raw']['selected'], SEL['arms']['sigma']['selected']
    L = [r'\begin{table}[!htbp]', r'\centering',
         r'\caption{Contrasts in the output-shift coordinate control, in points with unadjusted 95\%% Welch intervals (three runs per group). $D_{\mathrm{sel}}$: $\sigma_0$-scaled arm at its selected rate (%s) minus raw arm at its selected rate (%s). $D_{\mathrm{idx}}$: rates paired by grid position, $\sigma_0$-scaled minus raw. Context: each selected arm minus the locked residual GRPO and full-weight GRPO groups, evaluated on another machine.}' % (RATE[ss], RATE[sr]),
         r'\label{tab:supp1008_contrasts}', r'\small', r'\setlength{\tabcolsep}{3pt}', r'\begin{adjustbox}{max width=\textwidth}', r'\begin{tabular}{@{}lllll@{}}', r'\toprule',
         r'Models, condition & $D_{\mathrm{sel}}$ & $D_{\mathrm{idx}}$: $10^{-2}$ vs $3\cdot10^{-3}$ & $3\cdot10^{-2}$ vs $10^{-2}$ & $10^{-1}$ vs $3\cdot10^{-2}$\\', r'\midrule']
    for k, nm in CN.items():
        c = ST['conditions'][k]; di = c['D_idx']
        L.append(r'%s & %s %s & %s %s & %s %s & %s %s\\' % (nm, sg(100 * c['D_sel']['est']), ci(c['D_sel']),
                 sg(100 * di['1e-2_vs_3e-3']['est']), ci(di['1e-2_vs_3e-3']), sg(100 * di['3e-2_vs_1e-2']['est']), ci(di['3e-2_vs_1e-2']),
                 sg(100 * di['1e-1_vs_3e-2']['est']), ci(di['1e-1_vs_3e-2'])))
    L += [r'\midrule', r' & \multicolumn{2}{l}{Raw (%s) minus} & \multicolumn{2}{l}{$\sigma_0$-scaled (%s) minus}\\' % (RATE[sr], RATE[ss]),
          r' & residual GRPO & full-weight GRPO & residual GRPO & full-weight GRPO\\', r'\midrule']
    for k, nm in CN.items():
        a1 = CX['conditions'][k]['arms']['raw|' + sr]; a2 = CX['conditions'][k]['arms']['sigma|' + ss]
        L.append(r'%s & %s %s & %s %s & %s %s & %s %s\\' % (nm, sg(100 * a1['minus_residual_grpo_locked']['est']), ci(a1['minus_residual_grpo_locked']),
                 sg(100 * a1['minus_fullweight_grpo_locked']['est']), ci(a1['minus_fullweight_grpo_locked']),
                 sg(100 * a2['minus_residual_grpo_locked']['est']), ci(a2['minus_residual_grpo_locked']),
                 sg(100 * a2['minus_fullweight_grpo_locked']['est']), ci(a2['minus_fullweight_grpo_locked'])))
    L += [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{table}']
    (T / 'tab_supp1008_contrasts.tex').write_text('\n'.join(L) + '\n')
    # macros
    mac('BxRawSel', RATE[sr]); mac('BxSigSel', RATE[ss]); mac('BxNzero', pc(SEL['N0_this_machine']))
    for arm, nm in (('raw', 'Raw'), ('sigma', 'Sig')):
        for lr in LRS[arm]:
            r = SEL['arms'][arm]['rates'][lr]['per_condition']; t = nm + LS[lr].replace('3em3', 'A').replace('1em2', 'B').replace('3em2', 'C').replace('1em1', 'D')
            mac('BxDev%sLF' % t, pc(r['LF']['S'])); mac('BxDev%sLFloss' % t, '%.1f' % (-100 * r['LF']['dN']))
            mac('BxDev%sPCLloss' % t, '%.1f' % (-100 * r['PCL']['dN']))
            for k, cn in (('LF|LF', 'LF'), ('LF|nom', 'Nom'), ('LF|LF75', 'SF'), ('PCL|pn10', 'PCL')):
                mac('Bx%s%s' % (t, cn), pc(statistics.mean(ST['conditions'][k]['runs']['%s|%s' % (arm, lr)])))
        sel = SEL['arms'][arm]['selected']
        for k, cn in (('LF|LF', 'LF'), ('LF|nom', 'Nom'), ('LF|LF25', 'TF'), ('LF|LF75', 'SF'), ('PCL|pn10', 'PCL'), ('PCL|nom', 'PNom')):
            mac('Bx%sSel%s' % (nm, cn), pc(statistics.mean(ST['conditions'][k]['runs']['%s|%s' % (arm, sel)])))
            w = CX['conditions'][k]['arms']['%s|%s' % (arm, sel)]
            mac('Bx%sMinusRes%s' % (nm, cn), sg(100 * w['minus_residual_grpo_locked']['est'])); mac('Bx%sMinusRes%sCI' % (nm, cn), ci(w['minus_residual_grpo_locked']))
            mac('Bx%sMinusFull%s' % (nm, cn), sg(100 * w['minus_fullweight_grpo_locked']['est'])); mac('Bx%sMinusFull%sCI' % (nm, cn), ci(w['minus_fullweight_grpo_locked']))
    for k, cn in (('LF|LF', 'LF'), ('LF|nom', 'Nom'), ('LF|LF25', 'TF'), ('LF|LF75', 'SF'), ('PCL|pn10', 'PCL'), ('PCL|nom', 'PNom')):
        c = ST['conditions'][k]
        mac('BxDsel%s' % cn, sg(100 * c['D_sel']['est'])); mac('BxDsel%sCI' % cn, ci(c['D_sel'])); mac('BxDsel%sAbs' % cn, '%.1f' % abs(100 * c['D_sel']['est']))
        for key, nm2 in (('1e-2_vs_3e-3', 'A'), ('3e-2_vs_1e-2', 'B'), ('1e-1_vs_3e-2', 'C')):
            mac('BxDidx%s%s' % (nm2, cn), sg(100 * c['D_idx'][key]['est'])); mac('BxDidx%s%sCI' % (nm2, cn), ci(c['D_idx'][key]))
    for k, nm in ((next(kk for kk, r in G.items() if r['scheme'] == 'bias_raw' and r['selected']), 'BR'),
                  (next(kk for kk, r in G.items() if r['scheme'] == 'bias_sigma' and r['selected']), 'BS')):
        m = G[k]['mean']
        mac('Out%sShift' % nm, '%.1f' % m['shift']); mac('Out%sHigh' % nm, '%.2f' % m['high_share']); mac('Out%sCos' % nm, '%.2f' % m['cos_ref'])
        mac('Out%sRF' % nm, '%.2f' % m['dq_rfj3']); mac('Out%sLone' % nm, '%.2f' % m['l1'])
(T / 'supp1008_macros.tex').write_text('\n'.join(MAC) + '\n')

# ---------------------------------------------------------------- figure (real per-run values only)
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use(str(ROOT / 'scripts' / 'publication.mplstyle'))
mpl.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 10, 'axes.labelsize': 10, 'xtick.labelsize': 9,
                     'ytick.labelsize': 9, 'legend.fontsize': 9})
MMW = 1 / 25.4
COL = {'full_grpo': '#0072B2', 'full_ppo': '#009E73', 'res_ppo': '#CC79A7', 'res_grpo': '#D55E00', 'cst': '#E69F00',
       'bias_raw': '#56B4E9', 'bias_sigma': '#000000'}
MK = {'full_grpo': 'o', 'full_ppo': '^', 'res_ppo': 'D', 'res_grpo': 's', 'cst': 'v', 'bias_raw': 'X', 'bias_sigma': 'P'}
LAB = {'res_grpo': 'Residual GRPO', 'res_ppo': 'Residual PPO', 'full_grpo': 'Full-weight GRPO', 'full_ppo': 'Full-weight PPO',
       'cst': 'Constant + scale (48)', 'bias_raw': 'Raw shift (24)', 'bias_sigma': r'$\sigma_0$-scaled shift (24)'}
fig, axes = plt.subplots(1, 2, figsize=(190 * MMW, 90 * MMW), layout='constrained')
for key, rec in G.items():
    sch, pr = rec['scheme'], rec['per_run']
    kw = dict(marker=MK[sch], s=30 if rec['selected'] else 22, linewidths=.8, zorder=4 if rec['selected'] else 3,
              facecolor=COL[sch] if rec['selected'] else 'white', edgecolor=COL[sch], alpha=.9)
    axes[0].scatter(pr['shift'], pr['high_share'], **kw)
    axes[1].scatter(pr['dq_rfj3'], [100 * v for v in pr['fault']], **kw)
fz = 100 * N['frozen']['LF']
axes[1].scatter([0], [fz], marker='*', s=70, color='0.35', zorder=5)
axes[1].annotate('frozen planner', (0, fz), xytext=(0.07, fz + 1.5), fontsize=9, color='0.3')
axes[0].set_xscale('log')
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
axes[0].xaxis.set_major_locator(FixedLocator([5, 10, 20, 50, 100, 200, 500])); axes[0].xaxis.set_major_formatter(FixedFormatter(['5', '10', '20', '50', '100', '200', '500'])); axes[0].xaxis.set_minor_locator(NullLocator())
axes[0].set_xlabel(r'$\sigma_0$-normalized mean shift (log scale)')
axes[0].set_ylabel(r'Share of change on high-$\sigma_0$ dimensions')
axes[1].set_xlabel('RFJ3 control-target change (rad)')
axes[1].set_ylabel('LF test success (%)')
axes[0].set_title('(a) Allocation of mean changes', fontsize=10, loc='left')
axes[1].set_title('(b) RFJ3 target change and recovery', fontsize=10, loc='left')
axes[1].text(.04,.94,'Descriptive rank correlation: %.2f\n%d training runs' % (rho,len(allx)), transform=axes[1].transAxes, va='top',fontsize=8.5,color='0.25')
present = []
for k in ('res_grpo', 'res_ppo', 'full_grpo', 'full_ppo', 'cst', 'bias_raw', 'bias_sigma'):
    if any(r['scheme'] == k for r in G.values()): present.append(k)
h = [plt.Line2D([], [], ls='', marker=MK[k], color=COL[k], markerfacecolor=COL[k], ms=6, label=LAB[k]) for k in present]
fig.legend(handles=h, loc='outside lower center', ncol=4, frameon=False, fontsize=9, columnspacing=1.5, handletextpad=.3)
fig.savefig(ROOT / 'figures' / 'fig_output.pdf'); fig.savefig(ROOT / 'figures' / 'fig_output.png', dpi=300); plt.close(fig)
print('figure written')
