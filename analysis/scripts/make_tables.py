#!/usr/bin/env python3
"""Write every numeric table of the manuscript from numbers.json (and the locked selection file) into tables/*.tex.
No value is typed by hand; rerun after make_numbers_and_figs.py."""
import json, os, math
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
N = json.load(open(os.path.join(ROOT, 'numbers.json')))
DATA = os.environ.get('DEXGRASP_DATA_ROOT', os.path.join(ROOT, '..', 'data'))
SEL = json.load(open(os.path.join(DATA, 'meta', 'temp', 'shared_lr_select.json')))
TRIGGER = json.load(open(os.path.join(DATA, 'meta', 'temp', 'shared_lr_trigger.json')))
T = os.path.join(ROOT, 'tables'); os.makedirs(T, exist_ok=True)
X = N['extra']; FZ = N['frozen']; N0 = FZ['nom']
M4 = ['full_ppo', 'full_grpo', 'res_ppo', 'res_grpo']
SHORT = {'full_ppo': 'FW-PPO', 'full_grpo': 'FW-GRPO', 'res_ppo': 'Res-PPO', 'res_grpo': 'Res-GRPO'}
LONG = {'full_ppo': 'Full-weight PPO', 'full_grpo': 'Full-weight GRPO', 'res_ppo': 'Residual PPO', 'res_grpo': 'Residual GRPO'}

def p1(x): return '%.1f' % (100 * x)
def sg(x):
    s = '%+.1f' % (100 * x)
    if s in ('+0.0', '-0.0'): s = '0.0'
    return '$%s$' % s
def ci(c): return '[%s, %s]' % (sg(c[0]).replace('$+', '$'), sg(c[1]).replace('$+', '$'))
def eci(e): return '%s %s' % (sg(e['est']), ci(e['ci']))
def mean(v): return sum(v) / len(v)
def lrtex(s):
    a, b = s.split('e-'); return ('$10^{-%s}$' % b) if a == '1' else ('$%s{\\cdot}10^{-%s}$' % (a, b))
def pfmt(p):
    if p < 1e-4: return '$<10^{-4}$'
    if p < 0.001: return '$<0.001$'
    if p >= 0.9995: return '1.000'
    return '%.3f' % p
def W(name, s):
    # shrink a table to the text width only when it is wider (adjustbox max width); content is unchanged
    s = s.replace('\\begin{tabular}', '\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}', 1).replace('\\end{tabular}', '\\end{tabular}\n\\end{adjustbox}', 1)
    open(os.path.join(T, name + '.tex'), 'w').write(s)

# ---------------- Table: main results ----------------
order = [('LF', 'LF'), ('PCL', 'PCL'), ('TH', 'TH'), ('ZR', 'ZR'), ('BZ2PN', 'Z2+PCL'), ('THPN', 'TH+PCL')]
rows = []
for k, lab in order:
    r = N['main'][k]
    cells = []
    for m in M4:
        x = r[m]; cells.append('%s (%s)%s' % (p1(x['fault_mean']), p1(N0 - x['nom_mean']), '$^{\\dagger}$' if x['n'] == 5 else ''))
    rows.append('%s & %s & %s & %s & %s & %s \\\\' % (lab, p1(r['frozen_fault']), ' & '.join(cells), eci(r['struct_fault']), eci(r['struct_nom']), eci(r['alg_fault'])))
W('tab_main', r'''\begin{table}[t]
\centering
\caption{Final-test results on the Shadow Hand. Entries for the four schemes are fault success $Y$ in \% with the nominal success loss relative to the frozen planner (91.0\%) in parentheses, averaged over independent training runs (three per cell; $^{\dagger}$five, from the one-time seed top-up of the selection protocol). Structure: residual minus full-weight, equal-weight average of the PPO and GRPO contrasts; optimizer: GRPO minus PPO, equal-weight average over the two parameterizations. Differences are in percentage points with two-sided 95\% Welch intervals. Only the little-finger structure contrast is a pre-specified test (H1, \cref{tab:tests}); the other intervals are descriptive and not corrected for multiplicity.}
\label{tab:main}
\footnotesize
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{@{}l r r r r r l l l@{}}
\toprule
 & & \multicolumn{4}{c}{Fault $Y$ (nominal loss)} & \multicolumn{2}{c}{Structure difference} & Optimizer difference\\
\cmidrule(lr){3-6}\cmidrule(lr){7-8}\cmidrule(l){9-9}
Condition & Frozen & FW-PPO & FW-GRPO & Res-PPO & Res-GRPO & Fault & Nominal & Fault\\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table}
''')

# ---------------- Table: nine pre-specified tests ----------------
desc = {'H1': ('LF, reference posture', 'residual $>$ full-weight'),
        'H2': ('LF vs PCL', 'parameterization contrast larger under LF'),
        'H3a': ('LF locked at 25\\%', 'residual $>$ full-weight'),
        'H3b': ('LF locked at 75\\%', 'residual $>$ full-weight'),
        'H4_LFpn_srcLF': ('LF + noise, LF models', 'gain shrinks'),
        'H4_LFpn_srcPCL': ('LF + noise, PCL models', 'gain shrinks'),
        'H4_THzr_srcTH': ('TH + random $z$, TH models', 'gain shrinks'),
        'H4_THzr_srcZR': ('TH + random $z$, ZR models', 'gain shrinks'),
        'H5': ('LF, ideal switching', 'residual $>$ full-weight')}
lab = {'H1': 'H1', 'H2': 'H2', 'H3a': 'H3a', 'H3b': 'H3b', 'H4_LFpn_srcLF': 'H4a', 'H4_LFpn_srcPCL': 'H4b', 'H4_THzr_srcTH': 'H4c', 'H4_THzr_srcZR': 'H4d', 'H5': 'H5'}
dec = {'direction supported (Holm)': 'supported', 'mean direction opposite, evidence insufficient': 'opposite mean, insufficient',
       'mean direction consistent, evidence insufficient': 'consistent mean, insufficient', 'evidence for opposite direction': 'evidence for opposite'}
rows = []
for k in ['H1', 'H2', 'H3a', 'H3b', 'H4_LFpn_srcLF', 'H4_LFpn_srcPCL', 'H4_THzr_srcTH', 'H4_THzr_srcZR', 'H5']:
    t = N['tests'][k]
    rows.append('%s & %s & %s & %s & %s & %.1f & %s & %s & %s \\\\' % (lab[k], desc[k][0], desc[k][1], sg(t['est']), ci(t['ci']), t['df'], pfmt(t['p']), pfmt(t['p_holm']), dec[t['outcome']]))
W('tab_tests', r'''\begin{table}[t]
\centering
\caption{The nine pre-specified directional tests. Estimates and two-sided 95\% intervals are in percentage points; $p$ is the one-sided $p$-value in the predicted direction and $p_{\mathrm{Holm}}$ the Holm-adjusted value over the nine tests. H4a--H4d estimate $G_A - G_{A+B}$, the reduction of the gain over the frozen planner when a second shift $B$ is added. H5 uses the configurations selected per condition on the little-finger development set without the nominal-loss limit.}
\label{tab:tests}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}l l l r l r l l l@{}}
\toprule
Test & Condition & Prediction & Est. & 95\% CI & df & $p$ & $p_{\mathrm{Holm}}$ & Decision\\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table}
''')

# ---------------- Table: development-set selection (appendix) ----------------
rows = []
sel = {m: SEL['methods'][m]['d05']['shared_lr'] for m in M4}
for m in M4:
    tab = SEL['methods'][m]['table']
    lrs = sorted(tab.keys(), key=float)
    for i, lr in enumerate(lrs):
        e = tab[lr]; n = len(e['LF']['seeds'])
        feas = e['LF']['dN'] >= -0.05 and e['PCL']['dN'] >= -0.05
        crit = [c for c in ('LF', 'PCL') if TRIGGER['methods'][m]['table'][lr][c]['critical']]
        def dn(c):
            s = sg(e[c]['dN']); return s + ('$^{*}$' if c in crit else '')
        rows.append('%s & %s & %d & %s & %s & %s & %s & %s & %s \\\\' % (LONG[m] if i == 0 else '', lrtex(lr), n, p1(e['LF']['S']), dn('LF'), p1(e['PCL']['S']), dn('PCL'),
                                                                      p1((e['LF']['S'] + e['PCL']['S']) / 2), ('\\textbf{selected}' if lr == sel[m] else ('yes' if feas else 'no'))))
    rows.append('\\addlinespace')
W('tab_selection', r'''\begin{table}[!h]
\centering
\caption{Shared learning-rate selection on the development objects (512 fault episodes and 2048 nominal episodes per model; frozen planner: nominal 88.6\%, little-finger lock 15.2\%, point-cloud noise 67.4\%). $S$ is the mean fault success over training runs and $\Delta N$ the change of nominal success relative to the frozen planner, both in \%. A rate is feasible if $\Delta N \ge -5$ points under both conditions; the feasible rate with the largest mean $S$ is selected. $^{*}$Borderline value ($|\Delta N + 5| \le 1$ point) that triggered the one-time top-up to five training runs.}
\label{tab:selection}
\footnotesize
\begin{tabular}{@{}l l r r r r r r l@{}}
\toprule
 & & & \multicolumn{2}{c}{Little-finger lock} & \multicolumn{2}{c}{Point-cloud noise} & & \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Scheme & Rate & Runs & $S$ & $\Delta N$ & $S$ & $\Delta N$ & Mean $S$ & Feasible\\
\midrule
''' + '\n'.join(rows[:-1]) + r'''
\bottomrule
\end{tabular}
\end{table}
''')

# ---------------- Table: sensitivity of the LF parameterization contrast to the selection rule ----------------
H5 = X['H5']
def cfg_row(name, lrs, fault, nom):
    return '%s & %s & %s & %s & %s & %s & %s \\\\' % (name, *[lrtex(lrs[m]) for m in M4], fault, nom)
main_lr = {m: SEL['methods'][m]['d05']['shared_lr'] for m in M4}
rows = [cfg_row('Shared, 5-point limit (main protocol)', main_lr, eci(N['main']['LF']['struct_fault']), eci(N['main']['LF']['struct_nom'])),
        cfg_row('Little-finger condition only, 5-point limit', X['LFonly_d05']['lr'], eci(X['LFonly_d05']['LF']['fault']), eci(X['LFonly_d05']['LF']['nom'])),
        cfg_row('Shared, 10-point limit', X['S1_d10']['lr'], eci(X['S1_d10']['LF']['fault']), eci(X['S1_d10']['LF']['nom'])),
        cfg_row('Little-finger condition only, no limit (H5)', H5['lr'], eci(N['tests']['H5']), '--')]
W('tab_selrule', r'''\begin{table}[t]
\centering
\caption{Little-finger parameterization contrast (residual minus full-weight, percentage points, 95\% Welch interval) under alternative learning-rate selection rules, all applied to the development data. The first and last rows are the main protocol (H1) and H5; the two middle rows are post-lock sensitivity analyses. The nominal difference is not reported for H5, where ideal switching assigns the frozen planner to nominal conditions.}
\label{tab:selrule}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}l l l l l l l@{}}
\toprule
Selection rule & FW-PPO & FW-GRPO & Res-PPO & Res-GRPO & Fault difference & Nominal difference\\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table}
''')

# ---------------- Table: Allegro ----------------
AL = N['allegro']; rows = []
for c, lab in (('TH', 'Thumb lock'), ('Z2', '$z$ offset 2\\,cm')):
    a = AL[c]; cells = []
    for m in M4:
        g = a['groups'][m]; cells.append('%s (%s)' % (p1(mean(g['fault'])), p1(a['frozen_nom'] - mean(g['nom']))))
    rows.append('%s & %s & %s & %s & %s \\\\' % (lab, p1(a['frozen_fault']), ' & '.join(cells), eci(a['struct_fault']), eci(a['alg_fault'])))
W('tab_allegro', r'''\begin{table}[t]
\centering
\caption{Allegro Hand: transfer of the selected learning rates and training procedure (three runs per cell). Entries are fault success $Y$ in \% with nominal loss relative to the frozen Allegro planner (86.8\%) in parentheses. Differences in percentage points with 95\% Welch intervals (descriptive).}
\label{tab:allegro}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}l r r r r r l l@{}}
\toprule
Condition & Frozen & FW-PPO & FW-GRPO & Res-PPO & Res-GRPO & Structure & Optimizer\\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table}
''')

# ---------------- Table: fixed z offset and constant compensation ----------------
S4 = X['S4']
rows = ['Frozen planner & %s & 0.0 \\\\' % p1(S4['frozen']),
        'Frozen planner + constant compensation (1.5\\,cm) & %s & %s \\\\' % (p1(S4['comp_fault']), p1(S4['comp_nom_loss']))]
for m in M4:
    rows.append('%s & %s & %s \\\\' % (LONG[m], p1(S4['means'][m]['fault']), p1(S4['means'][m]['nom_loss'])))
W('tab_zoffset', r'''\begin{table}[t]
\centering
\caption{Fixed 2\,cm $z$ offset (post-lock robustness analysis, three runs per scheme). The compensation shifts the commanded object position by a constant chosen before the lock on post-training objects and is applied in every episode, including nominal ones. Values in \%; nominal loss relative to the frozen planner.}
\label{tab:zoffset}
\footnotesize
\begin{tabular}{@{}l r r@{}}
\toprule
Policy & Fault $Y$ & Nominal loss\\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table}
''')

# ---------------- Table: boundary configurations (S2, S5 with 3 and 5 runs) ----------------
B = N['boundary']
def struct_first3(cond):
    return None
rows = []
for key, lab, cond in (('LF75_retrain', 'LF at 75\\%, retrained at 75\\%', 'LF75'), ('RFLF', 'Ring + little finger', 'RFLF')):
    b = B[key]; cells = []
    for m in M4:
        mm = b['means'][m]; cells.append('%s (%s)' % (p1(mm['fault']), p1(N0 - mm['nom'])))
    rows.append('%s & %s & %s & %s & %s \\\\' % (lab, p1(b['frozen']), ' & '.join(cells), eci(b['fault']), eci(b['nom'])))
r3 = X.get('RFLF3')
if r3:
    rows.append('Ring + little finger, first 3 runs & %s & \\multicolumn{4}{c}{--} & %s & %s \\\\' % (p1(B['RFLF']['frozen']), eci(r3['fault']), eci(r3['nom'])))
W('tab_boundary', r'''\begin{table}[t]
\centering
\caption{Post-lock exploratory configurations. Method columns show fault success $Y$ in \% with nominal loss in parentheses. The last two columns are residual-minus-full-weight contrasts in fault and nominal success, respectively (percentage points; descriptive 95\% Welch intervals); the final column is not a method's nominal loss. The 75\% models were retrained after H3b was known (three runs per scheme). The two-finger condition was first run with three runs per scheme and then extended once to five; the five-run estimate contains the first three runs.}
\label{tab:boundary}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}l r r r r r l l@{}}
\toprule
Condition & Frozen & FW-PPO & FW-GRPO & Res-PPO & Res-GRPO & Fault diff. & Nominal diff.\\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table}
''')

# ---------------- Table: KL baseline ----------------
K = N['kl']['compare']; rows = []
for c, lab in (('LF', 'LF'), ('PCL', 'PCL'), ('TH', 'TH (transfer)')):
    k = K[c]; mm = k['means']
    def cell(x): return '%s (%s)' % (p1(x['fault']), p1(N0 - x['nom']))
    rows.append('%s & %s & %s & %s & %s & %s \\\\' % (lab, cell(mm['full']), cell(mm['kl']), cell(mm['res']), eci(k['kl_minus_full_fault']), eci(k['kl_minus_res_fault'])))
W('tab_kl', r'''\begin{table}[t]
\centering
\caption{Full-weight GRPO with a KL penalty to the pretrained planner (selected configuration: learning rate $3\cdot10^{-5}$, $\beta = 0.00528$), compared with the main full-weight GRPO and residual GRPO configurations. Entries are fault success $Y$ in \% with nominal loss in parentheses (three runs each); differences in fault success are in percentage points with 95\% Welch intervals (descriptive).}
\label{tab:kl}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}l r r r l l@{}}
\toprule
Condition & FW-GRPO & FW-GRPO + KL & Res-GRPO & KL $-$ FW-GRPO & KL $-$ Res-GRPO\\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table}
''')

# ---------------- Table: mean-scale intervention ----------------
MS = N['ms']; rows = []
for key, lab in (('LF|1e-3', 'LF'), ('PCL|1e-3', 'PCL'), ('LF|3e-3', 'LF'), ('PCL|3e-3', 'PCL')):
    e = MS['same_lr'][key]; d0 = e['means']['d0']; dc = e['means']['ms']
    lr = key.split('|')[1]
    unstable = sum(1 for r in MS['stability'][key] if r['max_delta_norm'] is not None and r['max_delta_norm'] > 100)
    rows.append('%s & %s & %s (%s) & %s (%s) & %s & %s & %d of 3 \\\\' % (lab, lrtex(lr), p1(d0['fault']), p1(N0 - d0['nom']), p1(dc['fault']), p1(N0 - dc['nom']),
                                                                           eci(e['ms_minus_d0_fault']), eci(e['ms_minus_d0_nom']), unstable))
W('tab_ms', r'''\begin{table}[t]
\centering
\caption{Mean-scale intervention for residual GRPO. $D_0 = \operatorname{diag}(\sigz)$ is the original scaling of the residual mean and $D_c$ the block-wise root-mean-square scaling. Entries are fault success $Y$ in \% with nominal loss in parentheses (three runs each). Differences ($D_c - D_0$) in percentage points with 95\% Welch intervals (descriptive). The last column counts $D_c$ runs whose residual-mean norm $\lVert\delta_\phi\rVert$ exceeded 100 during training; the other $D_c$ runs stayed between 3.6 and 13.3. The development rule selected $10^{-3}$ for both scalings.}
\label{tab:ms}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}l l r r l l l@{}}
\toprule
Condition & Rate & $D_0$ & $D_c$ & Fault diff. & Nominal diff. & Unstable $D_c$ runs\\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table}
''')

# ---------------- Table: state-independent constants ----------------
C = N['const']; CS = N['cst']['compare']; rows = []
def yc(f, n): return '%s (%s)' % (p1(mean(f)), p1(N0 - mean(n)))
for key, lab in (('LF', 'LF, reference posture'), ('LF75', 'LF at 75\\% (retrained)'), ('RFLF', 'Ring + little finger')):
    c = C[key]
    res = yc(c['full']['fault'], c['full']['nom']) + ' [%d]' % len(c['full']['fault'])
    post = yc(c['const']['fault'], c['const']['nom'])
    if key in CS:
        d = CS[key]['direct']; direct = yc(d['fault'], d['nom']) + ' [%d]' % len(d['fault'])
        dres = eci(CS[key]['direct_minus_residual_fault']); R = '%.2f' % CS[key]['R_direct']
    else:
        direct, dres, R = 'not tested', '--', '--'
    rows.append('%s & %s & %s & %s & %s & %s & %.2f & %s \\\\' % (lab, p1(c['Y0_fault']), res, post, direct, eci(c['const_minus_full_fault']), c['R_gain'], dres))
    rows[-1] = rows[-1]  # R_direct reported in text
W('tab_const', r'''\begin{table}[t]
\centering
\caption{State dependence of the residual correction. Entries are fault success $Y$ in \% with nominal loss in parentheses and the number of training runs in brackets. \emph{Residual}: residual GRPO at learning rate $10^{-3}$. \emph{Post-hoc constant}: each residual model with its correction replaced by its own state average $\dbar$ (no additional training; paired with its source model). \emph{Direct constant}: a state-independent latent offset $c$ and scale $\rho$ (48 parameters) trained from the pretrained planner with the same budget, with the learning rate of $c$ selected per condition on development data (both conditions selected the largest grid value, $3\cdot10^{-2}$). Post-hoc minus residual: paired $t$ interval; $R_{\text{post}} = (\bar Y_{\text{const}} - Y_0)/(\bar Y_{\text{res}} - Y_0)$; direct minus residual: independent groups, Welch interval. Differences in percentage points; all comparisons descriptive.}
\label{tab:const}
\footnotesize
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}l r l l l l r l@{}}
\toprule
Condition & Frozen & Residual & Post-hoc constant & Direct constant & Post-hoc $-$ res. & $R_{\text{post}}$ & Direct $-$ res.\\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table}
''')

# ---------------- Table: DemoFunGrasp case study ----------------
D = X['dfg']; rows = []
for c, lab in (('LF', 'Little-finger lock'), ('TH', 'Thumb lock'), ('PCL', 'Point-cloud noise 1\\,cm')):
    r = D[c]
    for arm, alab in (('frozen', 'Frozen planner'), ('SO', 'Success-only reward'), ('FN', 'Joint endpoint reward'), ('OF', 'Official full reward')):
        if arm not in r: continue
        v = r[arm] if arm == 'frozen' else r[arm]['mean']; n = '--' if arm == 'frozen' else str(r[arm]['n'])
        rows.append('%s & %s & %s & %s & %s & %s & %s \\\\' % (lab if arm == 'frozen' else '', alab, n, p1(v['fault_dev_succ']), p1(v['fault_dev_J']), p1(v['fault_test_J']), p1(v['nom_dev_J'])))
    rows.append('\\addlinespace')
W('tab_dfg', r'''\begin{table}[t]
\centering
\caption{DemoFunGrasp case study (development stage, full-weight GRPO, same budget as the main study; not part of the locked protocol). Success and $J_{4}$ (the fraction of attempts that succeed, match the requested grasp style and place the functional contact within 4\,cm of the target) are in \%. Development values use 512 episodes per model; the test value of $J_4$ is the mean of the seen and unseen object sets. Nominal $J_4$ of the frozen planner on the development objects is 53.1\%.}
\label{tab:dfg}
\footnotesize
\begin{tabular}{@{}l l r r r r r@{}}
\toprule
 & & & \multicolumn{3}{c}{Under the shift} & Nominal\\
\cmidrule(lr){4-6}\cmidrule(l){7-7}
Condition & Policy & Runs & Success (dev) & $J_4$ (dev) & $J_4$ (test) & $J_4$ (dev)\\
\midrule
''' + '\n'.join(rows[:-1]) + r'''
\bottomrule
\end{tabular}
\end{table}
''')
# ---------------- Table: evaluation-only conditions (appendix) ----------------
EO = X['evalonly']; rows = []
names = {'LF25': 'LF locked at 25\\%', 'LF75': 'LF locked at 75\\%', 'LFpn': 'LF + noise 1\\,cm', 'pn05': 'Noise 0.5\\,cm', 'pn15': 'Noise 1.5\\,cm',
         'pn20': 'Noise 2\\,cm', 'THzr': 'TH + random $z$', 'zp1': '$z$ +1\\,cm', 'zp3': '$z$ +3\\,cm', 'zm2': '$z$ $-$2\\,cm', 'x2': '$x$ +2\\,cm',
         'y2': '$y$ +2\\,cm', 'zp4': '$z$ +4\\,cm', 'zm4': '$z$ $-$4\\,cm'}
src = {'LF': 'LF', 'PCL': 'PCL', 'TH': 'TH', 'ZR': 'ZR'}
last = None
for key, e in EO.items():
    ts, ec = key.split('|')
    if last is not None and ts != last: rows.append('\\addlinespace')
    rows.append('%s & %s & %s & %s & %s \\\\' % (src[ts] if ts != last else '', names[ec], p1(e['frozen']), ' & '.join(p1(e['means'][m]) for m in M4), eci(e['struct'])))
    last = ts
W('tab_evalonly', r'''\begin{table}[!h]
\centering
\caption{All evaluation-only conditions of the locked final evaluation, evaluated with the models trained on the listed training condition (fault success $Y$ in \%, mean over runs). Structure: residual minus full-weight in percentage points (95\% Welch interval). Rows used by the pre-specified tests H3 and H4 are repeated here for completeness; all other rows are descriptive. The $z$ offsets of $\pm$4\,cm lie outside the training range of the random-offset models.}
\label{tab:evalonly}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}l l r r r r r l@{}}
\toprule
Trained on & Evaluated under & Frozen & FW-PPO & FW-GRPO & Res-PPO & Res-GRPO & Structure\\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table}
''')
print('tables written:', sorted(os.listdir(T)))
from make_feedback_analyses import write_tables
write_tables(N, T)
import runpy
runpy.run_path(os.path.join(HERE, 'make_layout_tables.py'), run_name='__main__')
runpy.run_path(os.path.join(HERE, 'make_collision_tables.py'), run_name='__main__')

runpy.run_path(os.path.join(HERE, 'make_supp1007_tables.py'), run_name='__main__')

# Additional prose values and S9 tables also read the single numerical record.
dev=N['development_boundary']
with open(os.path.join(T,'v14_macros.tex'),'w',encoding='utf-8') as f:
    for key,name in [('full_ppo_3e5_pcl_excess','DevPPOMiss'),('full_ppo_3e5_pcl_loss','DevPPOLoss'),('full_grpo_3e5_lf_loss','DevGRPOLoss')]:
        f.write('\\newcommand{\\%s}{%.2f}\n' % (name,100*dev[key]))
import subprocess,sys
subprocess.run([sys.executable,'-B',os.path.join(HERE,'make_supp1008.py'),'--from-record'],check=True)
