#!/usr/bin/env python3
"""Generate the 7 October experiment tables and prose macros solely from numbers.json."""
import json
from pathlib import Path
V = Path(__file__).resolve().parents[1]
T = V / 'tables'
S = json.loads((V / 'numbers.json').read_text(encoding='utf-8'))['supp_1007']
def mean(v):
    return sum(v) / len(v)

# ---------------- step 2: tables and macros ------------------------------------------------------------------
def p1(x):
    # same convention as the v10 generators: negative numbers are whole math expressions, so the Word export keeps
    # the sign inside one native equation (a lone math minus is dropped by some renderers)
    v = f'{100 * x:.1f}'
    if v == '-0.0':
        v = '0.0'
    return f'${v}$' if v.startswith('-') else v


def sg(x):
    v = f'{100 * x:+.1f}'
    return '$0.0$' if v in ('+0.0', '-0.0') else f'${v}$'


def eci(r):
    return f"{sg(r['est'])} [{p1(r['ci'][0])}, {p1(r['ci'][1])}]"


def table(label, caption, cols, head, rows, size='\\small'):
    return ('\\begin{table}[!htbp]\n\\centering\n\\caption{' + caption + '}\n\\label{' + label + '}\n' + size + '\n'
            '\\setlength{\\tabcolsep}{3pt}\n\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}{@{}' + cols + '@{}}\n'
            '\\toprule\n' + head + '\\\\\n\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n'
            '\\end{adjustbox}\n\\end{table}\n')


sl = S['selection']
rows = []
for lr, lab in (('1e-3', 'Setting-matched residual, $10^{-3}$'), ('3e-3', 'Setting-matched residual, $3\\cdot10^{-3}$')):
    r = sl['rates'][lr]
    rows.append(' & '.join([lab, p1(r['LF']['S']), sg(r['LF']['dN']), p1(r['PCL']['S']), sg(r['PCL']['dN']),
                            'eligible' if r['feasible'] else 'infeasible']) + r'\\')
f = sl['fullweight_new']
rows.append(' & '.join(['Full-weight GRPO, new runs, $10^{-5}$', p1(f['LF']['S']), sg(f['LF']['dN']), p1(f['PCL']['S']),
                        sg(f['PCL']['dN']), 'fixed rate']) + r'\\')
cap = ('Development record of the setting-matched GRPO control. $S$ is mean development fault success (\\%; 512 attempts per run) '
       'and $\\Delta N$ the mean change in nominal development success (points; 2,048 attempts per run) relative to the frozen '
       f"planner re-measured on the new machine ({p1(sl['N0_this_machine'])}\\%; {p1(sl['N0_original'])}\\% originally). "
       'A rate was feasible only if $\\Delta N\\ge -5$ points in both conditions. Three runs per entry. Neither rate was feasible, '
       'so no rate was selected. The new full-weight runs used the originally selected rate and did not take part in selection.')
(T / 'tab_supp1007_selection.tex').write_text(table(
    'tab:supp1007_selection', cap, 'lrrrrl',
    'Configuration & LF $S$ & LF $\\Delta N$ & PCL $S$ & PCL $\\Delta N$ & Selection status', rows), encoding='utf-8')

CL = [('LF|LF', 'LF models, LF'), ('LF|nom', 'LF models, nominal'), ('LF|LF25', 'LF models, 25\\% posture'),
      ('LF|LF75', 'LF models, 75\\% posture'), ('PCL|pn10', 'PCL models, PCL'), ('PCL|nom', 'PCL models, nominal')]
sm = S['setting_matched']
rows = []
for k, lab in CL:
    e = sm[k]
    rows.append(' & '.join([lab, p1(e['frozen']), p1(mean(e['fullweight_old'])), p1(mean(e['fullweight_new'])),
                            p1(mean(e['residual_old'])), p1(mean(e['matched_1e-3']['runs'])),
                            p1(mean(e['matched_3e-3']['runs']))]) + r'\\')
cap = ('Test-set success (\\%) in the setting-matched GRPO control: means over training runs of the average of seen and unseen '
       'success (1,928 attempts per run and condition). Original runs are the locked GRPO runs; all other columns are new runs '
       'on the new machine. The frozen values are single-checkpoint references from the locked evaluations. Three training runs per adapted configuration and condition.')
(T / 'tab_supp1007_means.tex').write_text(table(
    'tab:supp1007_means', cap, 'lrrrrrr',
    'Models, condition & Frozen & \\multicolumn{2}{c}{Full-weight GRPO} & \\multicolumn{3}{c}{Residual GRPO}\\\\\n'
    '\\cmidrule(lr){3-4}\\cmidrule(l){5-7}\n & & original & new & original & matched $10^{-3}$ & matched $3\\cdot10^{-3}$',
    rows), encoding='utf-8')

rows = []
for k, lab in CL:
    e = sm[k]
    rows.append(' & '.join([lab, eci(e['G_old']), eci(e['fullweight_new_minus_old']), eci(e['matched_1e-3']['G_new']),
                            eci(e['matched_1e-3']['Delta']), eci(e['matched_3e-3']['G_new']),
                            eci(e['matched_3e-3']['Delta'])]) + r'\\')
cap = ('Descriptive contrasts of the setting-matched GRPO control, in percentage points with unadjusted 95\\% Welch intervals; '
       'training runs are the units. $G$ is residual minus full-weight GRPO success: $G_{\\mathrm{orig}}$ for the original pair '
       'and $G_{\\mathrm{new}}$ for the setting-matched residual minus the new full-weight runs. $\\Delta=G_{\\mathrm{new}}-'
       'G_{\\mathrm{orig}}$ uses four independent groups. Because no rate met the development retention limit, no rate is primary '
       'and the pre-specified classification was not applied.')
(T / 'tab_supp1007_contrasts.tex').write_text(table(
    'tab:supp1007_contrasts', cap, 'lllllll',
    'Models, condition & $G_{\\mathrm{orig}}$ & Full-weight new$-$original & $G_{\\mathrm{new}}$, $10^{-3}$ & $\\Delta$, $10^{-3}$ & '
    '$G_{\\mathrm{new}}$, $3\\cdot10^{-3}$ & $\\Delta$, $3\\cdot10^{-3}$', rows, size='\\footnotesize'), encoding='utf-8')

M = ['% Generated from numbers.json supp_1007 by make_supp1007_tables.py; do not edit.']


def mac(name, text):
    M.append('\\newcommand{\\' + name + '}{' + text + '}')


def macw(name, r):
    mac(name, sg(r['est']))
    mac(name + 'CI', f"[{p1(r['ci'][0])}, {p1(r['ci'][1])}]")


mac('SuppNzero', p1(sl['N0_this_machine']))
for lr, t in (('1e-3', 'A'), ('3e-3', 'B')):
    for c in ('LF', 'PCL'):
        mac('SuppDevLoss' + c + t, p1(-sl['rates'][lr][c]['dN']))
for c in ('LF', 'PCL'):
    mac('SuppDevLossFW' + c, p1(-sl['fullweight_new'][c]['dN']))
e = sm['LF|LF']
mac('SuppFWoldLF', p1(mean(e['fullweight_old'])))
mac('SuppFWnewLF', p1(mean(e['fullweight_new'])))
mac('SuppResOldLF', p1(mean(e['residual_old'])))
mac('SuppMatchedALF', p1(mean(e['matched_1e-3']['runs'])))
mac('SuppMatchedBLF', p1(mean(e['matched_3e-3']['runs'])))
mac('SuppMatchedBLFmin', p1(min(e['matched_3e-3']['runs'])))
mac('SuppMatchedBLFmax', p1(max(e['matched_3e-3']['runs'])))
macw('SuppFWdiffLF', e['fullweight_new_minus_old'])
macw('SuppGoldLF', e['G_old'])
macw('SuppGALF', e['matched_1e-3']['G_new'])
macw('SuppGBLF', e['matched_3e-3']['G_new'])
macw('SuppDeltaALF', e['matched_1e-3']['Delta'])
macw('SuppDeltaBLF', e['matched_3e-3']['Delta'])
n = sm['LF|nom']
macw('SuppGoldNom', n['G_old'])
macw('SuppGANom', n['matched_1e-3']['G_new'])
macw('SuppGBNom', n['matched_3e-3']['G_new'])
mac('SuppMatchedANom', p1(mean(n['matched_1e-3']['runs'])))
mac('SuppMatchedBNom', p1(mean(n['matched_3e-3']['runs'])))
mac('SuppResOldNom', p1(mean(n['residual_old'])))
mac('SuppFWnewNom', p1(mean(n['fullweight_new'])))
macw('SuppGALFsf', sm['LF|LF75']['matched_1e-3']['G_new'])
macw('SuppGBLFsf', sm['LF|LF75']['matched_3e-3']['G_new'])
macw('SuppGAPCL', sm['PCL|pn10']['matched_1e-3']['G_new'])
macw('SuppGBPCL', sm['PCL|pn10']['matched_3e-3']['G_new'])
macw('SuppGoldPCL', sm['PCL|pn10']['G_old'])

# reproduction check: successes out of attempts
rp = S['reproduction']['runs']
seen_exact = [k for k, v in rp.items() if '_seen_' in k and v['flags_identical_all']]
unseen = [k for k in rp if '_unseen_' in k]
unseen_exact = [k for k in unseen if rp[k]['flags_identical_all']]
mac('SuppReproSeenExact', str(len(seen_exact)))
mac('SuppReproSeenTotal', str(sum(1 for k in rp if '_seen_' in k)))
mac('SuppReproUnseenExact', str(len(unseen_exact)))
mac('SuppReproUnseenTotal', str(len(unseen)))
for k, t in (('repro_final_unseen_LF_frozen', 'FrozenLF'), ('repro_final_unseen_nom_grpo_shlockLF_s1', 'FWNom'),
             ('repro_final_unseen_nom_pluggrpo_shlockLF_lr1e3_s1', 'ResNom')):
    v = rp[k]
    mac('SuppRepro' + t + 'New', str(round(v['success_new'] * 800)))
    mac('SuppRepro' + t + 'Old', str(round(v['success_archived'] * 800)))
    d = v['initial_state_max_abs_diff_server']
    mac('SuppRepro' + t + 'Init', '0' if not d else f'{d:.3f}')

if 'collision_full' in S:
    C = S['collision_full']
    rows = []
    for k, lab in (('LF|LF', 'LF models, LF'), ('LF|nom', 'LF models, nominal'), ('PCL|pn10', 'PCL models, PCL'),
                   ('PCL|nom', 'PCL models, nominal')):
        v = C['conditions'][k]
        rows.append(' & '.join([lab, eci(v['param_orig']), eci(v['param_coll']), eci(v['param_change']),
                                eci(v['opt_orig']), eci(v['opt_coll'])]) + r'\\')
    fz = C['frozen']
    rows.append('\\midrule')
    rows.append('Frozen planner (success, \\%) & \\multicolumn{5}{l}{LF ' + p1(fz['LF']['orig']) + ' $\\to$ ' + p1(fz['LF']['coll']) +
                '; PCL ' + p1(fz['pn10']['orig']) + ' $\\to$ ' + p1(fz['pn10']['coll']) + '; nominal ' + p1(fz['nom']['orig']) +
                ' $\\to$ ' + p1(fz['nom']['coll']) + '}\\\\')
    cap = ('Re-evaluation of the main LF and PCL models with robot--table collisions enabled in all environments and a ground '
           'plane added, compared with a same-session re-run of the original setting. Parameterization contrasts are residual '
           'minus full-weight and optimizer contrasts GRPO minus PPO, each averaged equally over the other factor, in percentage '
           'points with unadjusted 95\\% Welch intervals; training runs are the units. The change uses per-run differences '
           'between the two settings. The last row gives frozen success in the original and the collision setting. Training '
           'used the original setting.')
    (T / 'tab_supp1007_collision.tex').write_text(table(
        'tab:supp1007_collision', cap, 'llllll',
        'Models, condition & \\multicolumn{3}{c}{Parameterization contrast} & \\multicolumn{2}{c}{Optimizer contrast}\\\\\n'
        '\\cmidrule(lr){2-4}\\cmidrule(l){5-6}\n & original & all collisions & change & original & all collisions',
        rows, size='\\footnotesize'), encoding='utf-8')
    for k, t in (('LF|LF', 'LF'), ('LF|nom', 'LFNom'), ('PCL|pn10', 'PCL'), ('PCL|nom', 'PCLNom')):
        v = C['conditions'][k]
        macw('CollFull' + t, v['param_coll'])
        macw('CollFullOrig' + t, v['param_orig'])
        macw('CollFullChange' + t, v['param_change'])
        macw('CollFullOpt' + t, v['opt_coll'])
    mac('CollFullChangeLFAbs', p1(C['conditions']['LF|LF']['param_change']['est']))
    for c, t in (('LF', 'LF'), ('pn10', 'PCL'), ('nom', 'Nom')):
        mac('CollFullFrozen' + t, p1(fz[c]['coll']))
        mac('CollFullFrozenOrig' + t, p1(fz[c]['orig']))
    opts = [C['conditions'][k]['opt_coll'] for k in ('LF|LF', 'LF|nom', 'PCL|pn10', 'PCL|nom')]
    # the prose states that every optimizer interval includes zero; fail loudly if the data say otherwise
    assert all(o['ci'][0] <= 0 <= o['ci'][1] for o in opts), 'an optimizer interval excludes zero: rewrite the prose'
    mac('CollFullOptRange', sg(min(o['est'] for o in opts)) + ' to ' + sg(max(o['est'] for o in opts)))
    ini = C['initial_state']
    n_eval, n_diff = ini['evaluations_compared'], ini['evaluations_with_differences']
    mac('CollFullInitDiffPhrase', ('all %d' % n_eval) if n_diff == n_eval else ('%d of the %d' % (n_diff, n_eval)))
    mac('CollFullInitEvals', str(ini['evaluations_compared']))
    mac('CollFullInitDiffEvals', str(ini['evaluations_with_differences']))
    mac('CollFullInitMaxEnvs', str(ini['max_envs_differing']))
    mac('CollFullChangedEvals', str(C['evaluations_with_any_changed_outcome']))
    mac('CollFullInitSlots', str(C['initial_state_slots']['n_slots']))
    drops = [m['orig'] - m['coll'] for m in C['conditions']['LF|LF']['means'].values()]
    assert all(d > 0 for d in drops), 'prose says every scheme was lower under LF with all collisions'
    mac('CollFullLFDropMin', p1(min(drops)))
    mac('CollFullLFDropMax', p1(max(drops)))
    rr = C['repeat_original_setting']
    mac('SuppRepeatTotal', str(rr['n']))
    mac('SuppRepeatDiffer', str(rr['n'] - rr['n_identical']))
    mac('SuppRepeatMaxDiff', p1(rr['max_abs_diff']))
(T / 'supp1007_macros.tex').write_text('\n'.join(M) + '\n', encoding='utf-8')
print('supp_1007 tables written from numbers.json;', len(M) - 1, 'macros; collision part:', 'collision_full' in S)
