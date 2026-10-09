"""Recompute and compact the 7 October supplementary records before table generation."""
import json, subprocess, sys
from pathlib import Path

def w(r):
    return {k: r[k] for k in ('est', 'ci', 'df', 'n')}

def mean(v):
    return sum(v) / len(v)

def compact_record(REC):
    sel = REC['selection']
    fs = REC['setting_matched']
    field = {
        'role': 'descriptive; post-lock supplementary experiments planned on 7 October 2026; outside the nine-test family',
        'inputs': REC['inputs'],
        'reproduction': {'verdict': REC['reproduction']['verdict'], 'runs': {}},
        'selection': {'N0_this_machine': sel['N0_this_machine'], 'N0_original': sel['N0_original'], 'selected': sel['selected'],
                      'rates': {lr: {'feasible': v['feasible'], 'mean_S': v['mean_S'], 'worst_dN': v['worst_dN'],
                                     **{c: {'S': v['per_condition'][c]['S'], 'dN': v['per_condition'][c]['dN'],
                                            'S_runs': v['per_condition'][c]['S_runs'], 'N_runs': v['per_condition'][c]['N_runs']}
                                        for c in ('LF', 'PCL')}}
                                for lr, v in sel['rates'].items()},
                      'fullweight_new': {c: {'S': v['S'], 'dN': v['dN'], 'S_runs': v['S_runs'], 'N_runs': v['N_runs']}
                                         for c, v in sel['fullweight_new'].items()}},
        'setting_matched': {},
    }
    for nm, v in REC['reproduction']['runs'].items():
        field['reproduction']['runs'][nm] = v
    for key, c in fs['conditions'].items():
        cond = key.split('|')[1]
        e = {'frozen': fs['frozen_archived'][cond], 'fullweight_old': c['fullweight_old'], 'fullweight_new': c['fullweight_new'],
             'residual_old': c['residual_old'], 'G_old': w(c['G_old']), 'fullweight_new_minus_old': w(c['fullweight_new_minus_old'])}
        for lr, r in c['rates'].items():
            e['matched_' + lr] = {'runs': r['residual_matched'], 'G_new': w(r['G_new']), 'Delta': w(r['Delta']),
                                  'matched_minus_original_residual': w(r['residual_matched_minus_old'])}
        field['setting_matched'][key] = e
    if 'collision_full' in REC:
        cs = REC['collision_full']
        init = cs['initial_state_check']
        field['collision_full'] = {
            'original_arm': 'same-session re-run of the original setting (collref)',
            'frozen': cs['frozen'],
            'conditions': {k: {'param_orig': w(v['param_orig']), 'param_coll': w(v['param_coll']),
                               'param_change': w(v['param_change']), 'opt_coll': w(v['opt_coll']), 'opt_orig': w(v['opt_orig']),
                               'means': {m: {'orig': mean(g['orig']), 'coll': mean(g['coll']), 'n': len(g['orig'])}
                                         for m, g in v['per_method'].items()}}
                           for k, v in cs['conditions'].items()},
            'initial_state': {'evaluations_compared': len(init),
                              'evaluations_with_differences': sum(1 for x in init.values() if x['envs_with_any_difference']),
                              'max_envs_differing': max(len(x['envs_with_any_difference']) for x in init.values()),
                              'per_evaluation': {k: x['envs_with_any_difference'] for k, x in init.items() if x['envs_with_any_difference']}},
            'evaluations_with_any_changed_outcome': cs['evaluations_with_any_changed_outcome'],
            'initial_state_slots': cs['initial_state_slots'],
            'repeat_original_setting': cs['repeat_original_setting'],
        }
    return field

def build_record(new_root, data_root, out_path):
    script = Path(__file__).with_name('audit_supp1007.py')
    subprocess.run([sys.executable, '-B', str(script), '--new-root', str(new_root),
                    '--data-root', str(data_root), '--out', str(out_path)], check=True)
    return compact_record(json.loads(Path(out_path).read_text(encoding='utf-8')))
