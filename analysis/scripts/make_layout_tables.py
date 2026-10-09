"""Generate readable v2 table layouts from numbers.json, without new estimates."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N = json.loads((ROOT / 'numbers.json').read_text(encoding='utf-8'))
T = ROOT / 'tables'
METHODS = ['full_ppo', 'full_grpo', 'res_ppo', 'res_grpo']

def mean(a):
    return sum(a) / len(a)

def p(x):
    value=f'{100*x:.1f}'
    return '0.0' if value=='-0.0' else value

def signed(x):
    s = f'{100*x:+.1f}'
    return '0.0' if s in ['+0.0', '-0.0'] else s

def interval(e):
    # v5: interval bounds are typeset in math mode so negative bounds use a true minus sign.
    return '$' + signed(e['est']) + '$ [$' + p(e['ci'][0]) + '$, $' + p(e['ci'][1]) + '$]'

def pval(x):
    return '$<10^{-4}$' if x < 1e-4 else ('$<0.001$' if x < .001 else f'{x:.3f}')

def write(name, caption, label, body, placement='t'):
    text = '\\begin{table}['+placement+']\n\\centering\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.13}\n'+body+'\n\\end{table}\n'
    (T / (name+'.tex')).write_text(text, encoding='utf-8')

order = [('LF','Shadow LF'), ('PCL','Shadow PCL'), ('TH','Shadow TH'), ('ZR','Shadow ZR'), ('BZ2PN','Shadow Z2+PCL'), ('THPN','Shadow TH+PCL')]
entries = []
for k, label in order:
    r = N['main'][k]
    values = [p(r[m]['fault_mean'])+' ('+p(N['frozen']['nom']-r[m]['nom_mean'])+')'+('$^{\\dagger}$' if r[m]['n']==5 else '') for m in METHODS]
    entries.append((label, r['frozen_fault'], values, r['struct_fault'], r['struct_nom'], r['alg_fault']))
for k, label in [('TH','Allegro TH'),('Z2','Allegro Z2')]:
    r=N['allegro'][k]
    values = [p(mean(r['groups'][m]['fault']))+' ('+p(r['frozen_nom']-mean(r['groups'][m]['nom']))+')' for m in METHODS]
    entries.append((label, r['frozen_fault'], values, r['struct_fault'], r['struct_nom'], r['alg_fault']))
body = r'''\begin{tabularx}{\textwidth}{@{}l r *{4}{>{\centering\arraybackslash}X}@{}}
\toprule
Condition & Frozen & FW-PPO & FW-GRPO & Res-PPO & Res-GRPO\\
\midrule
'''
for label, frozen, cells, *_ in entries:
    body += ' & '.join([label,p(frozen)]+cells)+r' \\'+'\n'
body += r'''\bottomrule
\end{tabularx}
\par\medskip
\begin{tabularx}{\textwidth}{@{}l *{3}{>{\centering\arraybackslash}X}@{}}
\toprule
Condition & Parameterization: fault & Parameterization: nominal & Optimizer: fault\\
\midrule
'''
for label, _, _, sf, sn, af in entries:
    body += ' & '.join([label,interval(sf),interval(sn),interval(af)])+r' \\'+'\n'
body += '\\bottomrule\n\\end{tabularx}'
write('tab_main_v2', 'Post-lock results across eight conditions. Upper panel: mean shifted-condition success $Y$ (\\%), with nominal loss in parentheses. Three independent training runs per cell, except $^{\\dagger}$five after the selection top-up. Frozen nominal success is '+p(N['frozen']['nom'])+'\\% for Shadow and '+p(N['allegro']['TH']['frozen_nom'])+'\\% for Allegro. Lower panel: equal-weight contrasts in percentage points with unadjusted 95\\% Welch intervals. Parameterization is residual minus full-weight; optimizer is GRPO minus PPO. Negative nominal contrasts indicate worse residual retention. Except LF (H1), these contrasts are descriptive.', 'tab:main', body)

tests = [('H1','H1','Reference LF: residual advantage'),('H2','H2','Larger advantage under LF than PCL'),('H3a','H3a','LF at 25\\%: residual advantage'),('H3b','H3b','LF at 75\\%: residual advantage'),('H4_LFpn_srcLF','H4a','LF models: gain shrinks under LF+PCL'),('H4_LFpn_srcPCL','H4b','PCL models: gain shrinks under LF+PCL'),('H4_THzr_srcTH','H4c','TH models: gain shrinks under TH+ZR'),('H4_THzr_srcZR','H4d','ZR models: gain shrinks under TH+ZR'),('H5','H5','LF with ideal switching: residual advantage')]
body=r'''\begin{tabularx}{\textwidth}{@{}l X l r l@{}}
\toprule
Test & Directional prediction & Estimate [95\% CI] & $p_{\mathrm{Holm}}$ & Decision\\
\midrule
'''
for key, label, desc in tests:
    r=N['tests'][key]
    body += ' & '.join([label,desc,interval(r),pval(r['p_holm']), 'Supported' if r['p_holm']<.05 else 'Not supported'])+r' \\'+'\n'
body += '\\bottomrule\n\\end{tabularx}'
write('tab_tests_v2', 'All nine pre-specified directional tests. Estimates and two-sided, unadjusted 95\\% Welch confidence intervals are in percentage points. Decisions use one-sided $p$-values with Holm correction across the nine tests; the intervals are not Holm-adjusted decision intervals. H3 evaluates reference-trained policies without retraining. H4 compares gains over the frozen planner within runs. H5 selects rates on LF development data without a retention limit and assumes ideal fault routing. The two unsupported predictions have opposite-sign means with intervals containing zero. Degrees of freedom and raw $p$-values are in Supplementary Table~\\ref{tab:tests_full}.', 'tab:tests', body)

body=r'''\begin{tabularx}{\textwidth}{@{}X r c c c@{}}
\toprule
Condition & Frozen & Residual GRPO & Post-hoc constant & Direct constant\\
\midrule
'''
for k,label in [('LF','Reference LF'),('LF75','LF at 75\\%, retrained'),('RFLF','Ring + little finger')]:
    r=N['const'][k]
    def cell(d,n=True):
        return p(mean(d['fault']))+' ('+p(r['Y0_nom']-mean(d['nom']))+')'+(' ['+str(len(d['fault']))+']' if n else '')
    direct=cell(N['cst']['compare'][k]['direct']) if k in N['cst']['compare'] else 'Not tested'
    body += ' & '.join([label,p(r['Y0_fault']),cell(r['full']),cell(r['const']),direct])+r' \\'+'\n'
body+=r'''\bottomrule
\end{tabularx}
\par\medskip
\begin{tabularx}{\textwidth}{@{}X c c r@{}}
\toprule
Condition & Post-hoc $-$ residual & Direct $-$ residual & Gain retained\\
\midrule
'''
for k,label in [('LF','Reference LF'),('LF75','LF at 75\\%, retrained'),('RFLF','Ring + little finger')]:
    r=N['const'][k]
    direct=interval(N['cst']['compare'][k]['direct_minus_residual_fault']) if k in N['cst']['compare'] else '--'
    body+=' & '.join([label,interval(r['const_minus_full_fault']),direct,p(r['R_gain'])+r'\%'])+r' \\'+'\n'
body+='\\bottomrule\n\\end{tabularx}'
write('tab_const_v2','State-independent corrections: extraction after residual training and direct optimization. Upper panel: shifted-condition $Y$ (\\%), nominal loss in parentheses and number of runs in brackets. The reference and retrained 75\\% rows use three source runs; the two-finger row uses five, including the initial three. Post-hoc constants are paired with their source models; each direct-constant condition uses three independently trained runs. Lower panel: success differences in percentage points with descriptive 95\\% intervals (paired $t$ for post-hoc, Welch for direct); gain retained is the post-hoc gain over the frozen planner divided by the source model gain. Both direct-constant searches selected the highest tested learning rate.','tab:const',body)

print('Generated v2 main, test and constant tables from numbers.json')

body=r'''\begin{tabularx}{\textwidth}{@{}l l r r r X@{}}
\toprule
Test & Estimate [95\% CI] & df & Raw $p$ & $p_{\mathrm{Holm}}$ & Decision\\
\midrule
'''
for key, label, _ in tests:
    r=N['tests'][key]
    outcome='Supported' if r['p_holm']<.05 else 'Opposite mean, insufficient evidence'
    body+=' & '.join([label,interval(r),f"{r['df']:.1f}",pval(r['p']),pval(r['p_holm']),outcome])+r' \\'+'\n'
body+='\\bottomrule\n\\end{tabularx}'
write('tab_tests_full_v2','Complete statistics for the nine directional tests. Estimates and two-sided, unadjusted 95\\% Welch confidence intervals are in percentage points. Raw $p$ is one-sided in the predicted direction; decisions use Holm correction across all nine tests. The intervals are not Holm-adjusted decision intervals. The test definitions are given in the main manuscript.','tab:tests_full',body)
