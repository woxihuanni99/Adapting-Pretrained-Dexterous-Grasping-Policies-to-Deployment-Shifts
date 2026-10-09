#!/usr/bin/env python3
"""Write the collision-setting table and macros solely from numbers.json."""
import json
from pathlib import Path

VERSION = Path(__file__).resolve().parents[1]
S = json.loads((VERSION / 'numbers.json').read_text())['collision_split']
T = VERSION / 'tables'


def pp(x):
    v = f'{100 * x:.1f}'
    return '0.0' if v == '-0.0' else v


def signed(x):
    v = f'{100 * x:+.1f}'
    return '0.0' if v in ('+0.0', '-0.0') else v


def eci(r):
    return f"${signed(r['est'])}$ [${pp(r['ci'][0])}$, ${pp(r['ci'][1])}$]"


labels = {'Shadow LF': 'Shadow LF', 'Shadow PCL': 'Shadow PCL', 'Shadow TH': 'Shadow TH',
          'Shadow random z': 'Shadow random $z$', 'Shadow z + noise': 'Shadow $z$ + noise', 'Shadow TH + noise': 'Shadow TH + noise'}
rows = []
for key, label in labels.items():
    for outcome, field in (('Fault', 'param_fault'), ('Nominal', 'param_nom')):
        rows.append(' & '.join([label, outcome] + [eci(S[h][key][field]) for h in ('all', 'on', 'off')]) + r'\\')
rows.append(r'\midrule')
rows.append(' & '.join(['H5 (ideal switching)', 'Fault'] + [eci(S[h]['locked_estimands']['H5']) for h in ('all', 'on', 'off')]) + r'\\')
z = S['subset_sizes']
caption = ('Post hoc split of the locked Shadow evaluation by configured robot--table collision masks. Entries are '
           'residual-minus-full-weight contrasts in percentage points, with unadjusted 95\\% Welch intervals; '
           'training runs are the units. Each run averages seen and unseen success within its subset. '
           f"Enabled/disabled groups contain {z['seen']['on']}/{z['seen']['off']} seen slots and "
           f"{z['unseen']['on']}/{z['unseen']['off']} unseen slots. "
           'The groups contain different objects, so the descriptive split does not isolate the collision setting. '
           'Enabled denotes a configuration, not observed contact on every attempt.')
tex = ('\\begin{table}[!htbp]\n\\centering\n\\caption{' + caption + '}\n\\label{tab:collision_split}\n\\small\n'
       '\\setlength{\\tabcolsep}{3pt}\n\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}{@{}lllll@{}}\n\\toprule\n'
       'Condition & Outcome & All environments & Collisions enabled & Collisions disabled\\\\\n\\midrule\n' + '\n'.join(rows) +
       '\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n')
(T / 'tab_collision_split.tex').write_text(tex, encoding='utf-8')

M = []
def macro(name, r):
    M.append('\\newcommand{\\' + name + '}{$' + pp(r['est']) + '$}')
    M.append('\\newcommand{\\' + name + 'CI}{[$' + pp(r['ci'][0]) + '$, $' + pp(r['ci'][1]) + '$]}')
macro('CollisionOnLF', S['on']['Shadow LF']['param_fault'])
macro('CollisionOffLF', S['off']['Shadow LF']['param_fault'])
macro('CollisionOnLFNominal', S['on']['Shadow LF']['param_nom'])
macro('CollisionOffLFNominal', S['off']['Shadow LF']['param_nom'])
macro('CollisionOnZR', S['on']['Shadow random z']['param_fault'])
macro('CollisionOnHFive', S['on']['locked_estimands']['H5'])
on_other = [S['on'][k]['param_fault']['est'] for k in labels if k not in ('Shadow LF', 'Shadow random z')]
M.append('\\newcommand{\\CollisionOnOtherRange}{$' + pp(max(on_other)) + '$ to $' + pp(min(on_other)) + '$}')
M.append('\\newcommand{\\CollisionSeenOn}{' + str(z['seen']['on']) + '}')
M.append('\\newcommand{\\CollisionSeenOff}{' + str(z['seen']['off']) + '}')
M.append('\\newcommand{\\CollisionUnseenOn}{' + str(z['unseen']['on']) + '}')
M.append('\\newcommand{\\CollisionUnseenOff}{' + str(z['unseen']['off']) + '}')
(T / 'collision_macros.tex').write_text('% Generated from numbers.json collision_split by make_collision_tables.py; do not edit.\n' + '\n'.join(M) + '\n', encoding='utf-8')
print('wrote tab_collision_split.tex and collision_macros.tex')
