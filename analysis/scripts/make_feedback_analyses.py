#!/usr/bin/env python3
"""Descriptive analyses added after manuscript feedback on 2026-10-05.

No training, evaluation, reselection, or changes to the nine locked tests.
Called by make_numbers_and_figs.py and make_tables.py. All success outcomes
below are read from the archived per-model result.json files.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = ['full_ppo', 'full_grpo', 'res_ppo', 'res_grpo']
SHORT = dict(zip(METHODS, ['FW-PPO', 'FW-GRPO', 'Res-PPO', 'Res-GRPO']))
SHADOW = [('LF', 'LF'), ('PCL', 'pn10'), ('TH', 'TH'), ('ZR', 'zr'),
          ('BZ2PN', 'bz2pn'), ('THPN', 'THpn')]
LABELS = {'LF': 'Shadow LF', 'PCL': 'Shadow PCL', 'TH': 'Shadow TH',
          'ZR': 'Shadow random $z$', 'BZ2PN': 'Shadow $z$ + noise',
          'THPN': 'Shadow TH + noise', 'Allegro_TH': 'Allegro TH',
          'Allegro_Z2': 'Allegro $z$'}


def augment(out, data_root, contrast):
    """Append one new key, leaving all previously generated fields untouched."""
    data_root = Path(data_root)
    def meta(name):
        return json.loads((data_root / 'meta' / 'temp' / name).read_text())
    fs = meta('final_stats.json')
    am = meta('allegro_manifest.json')
    sm = meta('supp_manifest.json')
    km = meta('kl_manifest.json')
    cm = meta('cst_manifest.json')
    selection = meta('shared_lr_select.json')
    cache = {}
    slot_sequences = {}

    def raw(split, condition, tag):
        name = 'final_%s_%s_%s' % (split, condition, tag)
        if name in cache:
            return cache[name]
        payload = json.loads((data_root / 'runs' / name / 'result.json').read_text())
        episodes = payload['episodes']
        assert len(episodes) == 8, name
        by_asset = defaultdict(list)
        all_successes = []
        for ep in episodes:
            successes = ep['official_ppo_print_successes']
            assets = ep['object_assets']
            assert len(successes) == len(assets), name
            assert len(assets) == (141 if split == 'seen' else 100), name
            assert all(x in (0, 1) for x in successes), name
            # This also verifies identity/order is consistent across episodes,
            # hands, models, shifted conditions, and nominal evaluations.
            if split not in slot_sequences:
                slot_sequences[split] = assets
            assert slot_sequences[split] == assets, name
            all_successes.extend(successes)
            for asset, success in zip(assets, successes):
                by_asset[asset].append(success)
        official = float(np.mean(all_successes))
        assert abs(official - payload['totals']['success_rate']) < 1e-12, name
        cache[name] = {'official': official,
                       'asset_balanced': float(np.mean([np.mean(v) for v in by_asset.values()])),
                       'slots': len(episodes[0]['object_assets']), 'assets': len(by_asset)}
        return cache[name]

    def values(tags, condition, split=None, balance='official'):
        if split is None:
            return [(raw('seen', condition, t)[balance] + raw('unseen', condition, t)[balance]) / 2
                    for t in tags]
        return [raw(split, condition, t)[balance] for t in tags]

    def ctr(groups, weights):
        c = contrast(groups, weights, +1)
        return {'est': c['estimate'], 'ci': list(c['ci95_two_sided']),
                'se': c['se'], 'df': c['df'], 't': c['t'],
                'n': [len(x) for x in groups]}

    def structure(groups):
        return ctr([groups[m] for m in METHODS], [-.5, -.5, .5, .5])

    configs = {}
    for ts, ec in SHADOW:
        tags = {m: list(fs['descriptive']['table'][m][ts][ec]['runs']) for m in METHODS}
        configs[ts] = {'fault_condition': ec, 'nominal_condition': 'nom', 'frozen_tag': 'frozen', 'tags': tags}
    for c, code, ec in [('TH', 'shlockTH', 'alTH'), ('Z2', 'shbz2', 'albz2')]:
        keys = dict(zip(METHODS, ['ppo', 'grpo', 'plugppo', 'pluggrpo']))
        tags = {m: am['models'][keys[m] + '|' + code] for m in METHODS}
        configs['Allegro_' + c] = {'fault_condition': ec, 'nominal_condition': 'alnom',
                                 'frozen_tag': 'alfrozen', 'tags': tags}

    result = {'provenance': {'analysis_date': '2026-10-05',
              'status': 'Descriptive analyses added after manuscript feedback; outside the nine locked tests.',
              'unit': 'Independent training run. No new training, evaluation, or selection.',
              'confidence_intervals': 'Unadjusted two-sided 95% Welch-Satterthwaite intervals.',
              'source': 'Archived result.json episode success flags and object asset identifiers.'},
              'optimizer_within_parameterization': {}, 'components': {}, 'object_balance': {},
              'curve_base': {}, 'curve_diagnostics': []}

    for c, config in configs.items():
        ec, nc, tags = config['fault_condition'], config['nominal_condition'], config['tags']
        grouped = {outcome: {m: values(tags[m], cond) for m in METHODS}
                   for outcome, cond in [('fault', ec), ('nominal', nc)]}
        # Independent check against the previously reported aggregation.
        old = out['main'][c] if c in out['main'] else out['allegro'][c.split('_', 1)[1]]['groups']
        for m in METHODS:
            for key, oldkey in [('fault', 'fault'), ('nominal', 'nom')]:
                assert np.allclose(grouped[key][m], old[m][oldkey], rtol=0, atol=1e-12), (c, m, key)
        result['optimizer_within_parameterization'][c] = {
            p: {outcome: ctr([grouped[outcome][p + '_grpo'], grouped[outcome][p + '_ppo']], [1, -1])
                for outcome in grouped} for p in ('full', 'res')}
        comp = {}
        for split in ('seen', 'unseen'):
            one = {'frozen_fault': raw(split, ec, config['frozen_tag'])['official'],
                   'frozen_nominal': raw(split, nc, config['frozen_tag'])['official'], 'methods': {}}
            g = {outcome: {m: values(tags[m], cond, split) for m in METHODS}
                 for outcome, cond in [('fault', ec), ('nominal', nc)]}
            for m in METHODS:
                one['methods'][m] = {'tags': tags[m], 'n': len(tags[m]),
                     'fault_runs': g['fault'][m], 'nominal_runs': g['nominal'][m],
                     'fault': float(np.mean(g['fault'][m])), 'nominal': float(np.mean(g['nominal'][m]))}
                one['methods'][m]['nominal_loss'] = one['frozen_nominal'] - one['methods'][m]['nominal']
            one['structure_fault'] = structure(g['fault'])
            one['structure_nominal'] = structure(g['nominal'])
            comp[split] = one
        result['components'][c] = comp
        balanced = {outcome: {m: values(tags[m], cond, balance='asset_balanced') for m in METHODS}
                    for outcome, cond in [('fault', ec), ('nominal', nc)]}
        result['object_balance'][c] = {
            outcome: {'original': structure(grouped[outcome]), 'asset_balanced': structure(balanced[outcome]),
                      'methods': {m: {'tags': tags[m], 'original_runs': grouped[outcome][m],
                                      'asset_balanced_runs': balanced[outcome][m]} for m in METHODS}}
            for outcome in grouped}

    # Same trained models, evaluated under two lock postures; subtract INSIDE
    # each run before estimating between-parameterization difference of changes.
    paired = {}
    for m, tags in configs['LF']['tags'].items():
        ref, alt = values(tags, 'LF'), values(tags, 'LF75')
        diffs = [a - b for a, b in zip(ref, alt)]
        paired[m] = {'tags': tags, 'ref_runs': ref, 'at75_runs': alt, 'difference_runs': diffs,
                     'ref_mean': float(np.mean(ref)), 'at75_mean': float(np.mean(alt)),
                     'ref_minus_75': ctr([diffs], [1])}
    result['paired_posture'] = {'methods': paired,
        'difference_of_structure_contrasts': structure({m: paired[m]['difference_runs'] for m in METHODS})}

    # Complete Figure 2 coordinates: keep development-only failures distinct.
    for c, ec in [('LF', 'LF'), ('PCL', 'pn10')]:
        rows = []
        for m in METHODS:
            for lr, r in selection['methods'][m]['table'].items():
                found = [p for p in out['lrcurve'][c][m] if p['lr'] == lr]
                row = {'method': m, 'lr': lr, 'selected': lr == selection['methods'][m]['d05']['shared_lr']}
                if found:
                    p = found[0]
                    tags = sm['S3_lrcurve'][m][lr][c]
                    fault, nominal = values(tags, ec), values(tags, 'nom')
                    assert abs(np.mean(fault) - p['fault']) < 1e-12
                    assert abs(out['frozen']['nom'] - np.mean(nominal) - p['nom_loss']) < 1e-12
                    row.update({'status': 'evaluated after lock', 'tags': tags, 'n': len(tags),
                                'fault': float(np.mean(fault)), 'nominal_loss': out['frozen']['nom'] - float(np.mean(nominal))})
                else:
                    assert m.startswith('res_') and lr == '1e-2', (c, m, lr)
                    row.update({'status': 'not evaluated after lock; excluded after development failure',
                                'n': None, 'fault': None, 'nominal_loss': None,
                                'development': r[c]})
                rows.append(row)
        result['curve_base'][c] = rows
        for row in out['kl']['grid'][c]:
            key = '%s|%s|%s' % (c, row['lr'], row['beta_id'])
            tags = [x['tag'] for x in km['configs'][key]]
            fault, nominal = values(tags, ec), values(tags, 'nom')
            assert abs(np.mean(fault) - row['fault']) < 1e-12
            assert abs(out['frozen']['nom'] - np.mean(nominal) - row['nom_loss']) < 1e-12
            result['curve_diagnostics'].append({'condition': c, 'method': 'Full-weight GRPO + KL',
                'lr': row['lr'], 'beta': row['beta'], 'tags': tags, 'n': len(tags),
                'fault': float(np.mean(fault)), 'nominal_loss': out['frozen']['nom'] - float(np.mean(nominal))})
        if c == 'LF':
            for lr in ('3e-3', '1e-2', '3e-2'):
                tags = [x['tag'] for x in cm['cst'][c + '|' + lr]]
                fault, nominal = values(tags, ec), values(tags, 'nom')
                plotted = out['cst']['grid_test'][c][lr]
                assert abs(np.mean(fault) - plotted['fault_mean']) < 1e-12
                assert abs(out['frozen']['nom'] - np.mean(nominal) - plotted['nom_loss']) < 1e-12
                result['curve_diagnostics'].append({'condition': c, 'method': 'Direct constant',
                    'lr': lr, 'rho_lr': cm['lr_rho'], 'beta': None, 'tags': tags, 'n': len(tags),
                    'fault': float(np.mean(fault)), 'nominal_loss': out['frozen']['nom'] - float(np.mean(nominal))})
    result['object_balance_metadata'] = {
        'definition': 'Mean success per exact object_assets identifier (including scale suffix) within a split; equal mean across identifiers; then equal mean of seen and unseen within a training run.',
        'interpretation': 'Asset-instance weighting sensitivity; not category balancing, mesh-identity balancing, or a new evaluation population.',
        'slots': {s: len(v) for s, v in slot_sequences.items()},
        'unique_asset_instances': {s: len(set(v)) for s, v in slot_sequences.items()},
        'records_checked': len(cache),
        'checks': 'All records have eight episodes, binary successes, matching slot/asset arrays, identical slot identity/order within split, and reconstructed official score equal to recorded total.'}
    result['curve_axis_check'] = {c: {'xlim': [-1, 34 if c == 'LF' else 28],
        'outside_count': sum(not -1 <= 100 * r['nominal_loss'] <= (34 if c == 'LF' else 28)
          for r in result['curve_base'][c] + [r for r in result['curve_diagnostics'] if r['condition'] == c]
          if r['nominal_loss'] is not None)} for c in ('LF', 'PCL')}
    out['feedback_1005'] = result
    # v8 addition: pair seen/unseen outcomes WITHIN each trained model before
    # comparing independent method cells. Preserve every v7 field above.
    lf_components = result['components']['LF']
    nominal_split_differences = {}
    for m in METHODS:
        seen = lf_components['seen']['methods'][m]
        unseen = lf_components['unseen']['methods'][m]
        assert seen['tags'] == unseen['tags']
        differences = [u - s for s, u in zip(seen['nominal_runs'], unseen['nominal_runs'])]
        nominal_split_differences[m] = {
            'tags': seen['tags'], 'seen_nominal_runs': seen['nominal_runs'],
            'unseen_nominal_runs': unseen['nominal_runs'],
            'unseen_minus_seen_runs': differences,
        }
    nominal_interaction = structure({m: nominal_split_differences[m]['unseen_minus_seen_runs'] for m in METHODS})
    assert abs(nominal_interaction['est'] - (
        lf_components['unseen']['structure_nominal']['est']
        - lf_components['seen']['structure_nominal']['est'])) < 1e-12
    operating_points = {}
    for name, m, lr in [('full_ppo_3em5', 'full_ppo', '3e-5'), ('res_ppo_3em3', 'res_ppo', '3e-3')]:
        matches = [r for r in result['curve_base']['LF'] if r['method'] == m and r['lr'] == lr]
        assert len(matches) == 1
        operating_points[name] = dict(matches[0])
    out['feedback_v8'] = {
        'provenance': {
            'revision': 'v8',
            'status': 'Additional descriptive analysis after manuscript feedback; outside the nine locked tests.',
            'unit': 'Independent training run, with unseen-minus-seen nominal success paired within each run.',
            'definition': '(Residual-minus-full nominal success contrast on unseen) minus (the same contrast on seen).',
            'interpretation': 'A negative estimate means the residual-minus-full nominal-success difference is more negative on unseen. It is not either scheme\'s loss relative to the frozen planner.',
            'interval': 'Unadjusted two-sided 95% Welch-Satterthwaite interval across independent method cells.',
            'source': 'Existing final_seen_nom_<tag>/result.json and final_unseen_nom_<tag>/result.json, same tags as the main LF configurations.',
        },
        'lf_frozen_nominal': {
            split: {'success': lf_components[split]['frozen_nominal'],
                    'source': 'final_%s_nom_frozen/result.json' % split}
            for split in ('seen', 'unseen')
        },
        'lf_nominal_split_interaction': {'methods': nominal_split_differences, 'contrast': nominal_interaction},
        'lf_operating_points': operating_points,
    }
    return result


def write_tables(numbers, table_dir):
    """Tables exclusively from numbers.json; no hard-coded empirical values."""
    n = numbers['feedback_1005']
    table_dir = Path(table_dir)
    def pp(x):
        s = '%.1f' % (100 * x)
        return '0.0' if s == '-0.0' else s
    def signed(x):
        s = '%+.1f' % (100 * x)
        return '0.0' if s in ('+0.0', '-0.0') else s
    def eci(x):
        return '$%s$ [$%s$, $%s$]' % (signed(x['est']), pp(x['ci'][0]), pp(x['ci'][1]))
    def lr(s):
        a, b = s.split('e-')
        return '$%s10^{-%s}$' % ('' if a == '1' else a + r'\times ', b)
    def table(name, caption, columns, header, rows):
        text = '\n'.join([r'\begin{table}[!htbp]', r'\centering', r'\caption{' + caption + '}',
            r'\label{tab:' + name + '}', r'\small', r'\setlength{\tabcolsep}{3pt}',
            r'\begin{adjustbox}{max width=\textwidth}', r'\begin{tabular}{@{}' + columns + '@{}}',
            r'\toprule', header + r'\\', r'\midrule'] + [r + r'\\' for r in rows] +
            [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{table}', ''])
        (table_dir / ('tab_' + name + '.tex')).write_text(text, encoding='utf-8')
    rows = []
    for c, v in n['optimizer_within_parameterization'].items():
        for param, label in [('full', 'Full-weight'), ('res', 'Residual')]:
            z = v[param]
            rows.append('%s & %s & %s/%s & %s & %s' %
                (LABELS[c], label, z['fault']['n'][1], z['fault']['n'][0], eci(z['fault']), eci(z['nominal'])))
    table('feedback_optimizer', r'Descriptive optimizer contrasts within each update parameterization (GRPO minus PPO, percentage points). The run counts are PPO/GRPO. Intervals are unadjusted two-sided 95\% Welch intervals. Positive nominal contrasts indicate higher nominal success for GRPO. These post-feedback analyses do not isolate optimizer choice from implementation differences and are outside the nine locked tests.',
          'llrll', r'Condition & Update & Runs & Fault success & Nominal success', rows)
    rows = []
    for m, p in n['paired_posture']['methods'].items():
        rows.append('%s & %d & %s & %s & %s' % (SHORT[m], len(p['tags']), pp(p['ref_mean']), pp(p['at75_mean']), eci(p['ref_minus_75'])))
    d = n['paired_posture']['difference_of_structure_contrasts']
    rows.append(r'\midrule \multicolumn{4}{l}{Residual-minus-full-weight difference in paired changes} & ' + eci(d))
    table('feedback_posture', r'Descriptive reference-minus-75\% lock-posture changes for the same LF-trained models. Success means are percentages and changes are percentage points with unadjusted 95\% intervals. Each change is computed within a training run before averaging; the last row contrasts these changes with equal weight for the two optimizers. Thus its uncertainty retains within-run covariance across postures. No model was retrained for this comparison.',
          'lrrrl', r'Scheme & Runs & Reference & 75\% & Reference minus 75\%', rows)
    for outcome in ('fault', 'nominal'):
        rows = []
        for c, v in n['components'].items():
            for split, r in v.items():
                vals = [pp(r['methods'][m][outcome]) for m in METHODS]
                if outcome == 'nominal':
                    vals.insert(0, pp(r['frozen_nominal']))
                rows.append('%s & %s & %s & %s' % (LABELS[c], split, ' & '.join(vals), eci(r['structure_' + outcome])))
        extra_caption = ''
        if outcome == 'nominal':
            direct = numbers['feedback_v8']['lf_nominal_split_interaction']['contrast']
            rows.append(r'\midrule \multicolumn{6}{l}{Shadow LF: unseen minus seen in the residual-minus-full contrast} & \multicolumn{2}{l}{' + eci(direct) + '}')
            extra_caption = r' Frozen columns give the corresponding baseline success, not a loss. The final row directly contrasts unseen-minus-seen nominal success computed within each LF-trained run; a negative value denotes a more negative residual-minus-full contrast on unseen, not the loss of either scheme relative to the frozen planner.'
        table('feedback_components_' + outcome, r'Descriptive seen/unseen decomposition of ' + outcome + r' success (\%). Method columns average independent training runs within the specified split; the last column is residual minus full-weight (percentage points; unadjusted 95\% Welch interval). Seen and unseen refer to the existing object splits, which were used during development. The main score gives these two components equal weight within each run.' + extra_caption,
              'llrrrrrl' if outcome == 'nominal' else 'llrrrrl',
              (r'Condition & Split & Frozen & FW-PPO & FW-GRPO & Res-PPO & Res-GRPO & Contrast' if outcome == 'nominal' else r'Condition & Split & FW-PPO & FW-GRPO & Res-PPO & Res-GRPO & Contrast'), rows)
    rows = []
    for c, v in n['object_balance'].items():
        for outcome in ('fault', 'nominal'):
            rows.append('%s & %s & %s & %s' % (LABELS[c], outcome.capitalize(),
                eci(v[outcome]['original']), eci(v[outcome]['asset_balanced'])))
    counts = n['object_balance_metadata']['unique_asset_instances']
    table('feedback_object_balance', r'Descriptive sensitivity to equal asset-instance weighting. Entries are residual-minus-full-weight contrasts (percentage points; unadjusted 95\% Welch intervals). Original scores weight environment slots equally within each split; balanced scores first average all recorded episodes for each exact asset identifier (including its scale suffix), then average the ' + str(counts['seen']) + ' seen and ' + str(counts['unseen']) + r' unseen identifiers separately and give the splits equal weight. Training runs remain the units for intervals. This reweights existing observations; it does not create independent objects or a new test set.',
          'llll', r'Condition & Outcome & Original & Asset-balanced', rows)
    for c, records in n['curve_base'].items():
        rows = []
        for r in records:
            method = SHORT[r['method']] + (r'$^{*}$' if r['selected'] else '')
            if r['fault'] is None:
                rows.append(r'%s & %s & -- & \multicolumn{2}{l}{Not evaluated after lock}' % (method, lr(r['lr'])))
            else:
                rows.append('%s & %s & %d & $%s$ & $%s$' % (method, lr(r['lr']), r['n'], pp(r['fault']), pp(r['nominal_loss'])))
        table('feedback_curve_' + c, r'Complete base-scheme configuration list for the ' + ('little-finger lock' if c == 'LF' else 'point-cloud noise') + r' panel of Figure~\ref{main-fig:tradeoff}. Fault success is in \%; nominal loss is in percentage points. A star identifies the shared learning rate highlighted by a filled symbol; the symbol repeats the same configuration, rather than adding runs. Residual $10^{-2}$ configurations were excluded from post-lock evaluation after development failure (Table~\ref{tab:selection}); their development scores are not substituted here. All evaluated rows fall inside the plotted horizontal range.',
              'llrrr', r'Scheme & Learning rate & Runs & Fault success & Nominal loss', rows)
    rows = []
    for r in n['curve_diagnostics']:
        beta = '%.5g' % r['beta'] if r['beta'] is not None else '--'
        rows.append('%s & %s & %s & %s & %d & $%s$ & $%s$' % (r['condition'], r['method'], lr(r['lr']), beta, r['n'], pp(r['fault']), pp(r['nominal_loss'])))
    table('feedback_curve_diagnostics', r'All diagnostic configurations plotted in Figure~\ref{main-fig:tradeoff}: full-weight GRPO with a KL penalty and directly trained constants. Fault success is in \%; nominal loss is in percentage points. The constant rows use the listed learning rate for $c$ and a fixed learning rate of $10^{-3}$ for $\rho$. All rows have post-lock evaluations and lie inside the plotted horizontal range. They are descriptive additions, outside the nine locked tests.',
          'lllrrrr', r'Condition & Diagnostic & Learning rate & $\beta$ & Runs & Fault & Nominal loss', rows)

    # Prose uses these macros so every newly reported empirical quantity is
    # regenerated from the same numeric source as the supplementary tables.
    macro_sources = {
        'PairedPostureContrast': n['paired_posture']['difference_of_structure_contrasts'],
        'BalancedLF': n['object_balance']['LF']['fault']['asset_balanced'],
        'AllegroFWOptimizer': n['optimizer_within_parameterization']['Allegro_TH']['full']['fault'],
        'LFSeen': n['components']['LF']['seen']['structure_fault'],
        'LFUnseen': n['components']['LF']['unseen']['structure_fault'],
        'LFNominalSeen': n['components']['LF']['seen']['structure_nominal'],
        'LFNominalUnseen': n['components']['LF']['unseen']['structure_nominal'],
        'LFNominalSplitContrast': numbers['feedback_v8']['lf_nominal_split_interaction']['contrast'],
    }
    macros = ['% Generated from numbers.json by make_feedback_analyses.py; do not edit.']
    for name, value in macro_sources.items():
        ci_text = '[$%s$, $%s$]' % (pp(value['ci'][0]), pp(value['ci'][1]))
        for suffix, text in [('', '$' + pp(value['est']) + '$'), ('CI', ci_text), ('EstCI', eci(value))]:
            macros.append('\\newcommand{\\' + name + suffix + '}{' + text + '}')
    scalar_sources = {
        'LFFrozenNominalSeen': numbers['feedback_v8']['lf_frozen_nominal']['seen']['success'],
        'LFFrozenNominalUnseen': numbers['feedback_v8']['lf_frozen_nominal']['unseen']['success'],
        'LFPPONearFullFault': numbers['feedback_v8']['lf_operating_points']['full_ppo_3em5']['fault'],
        'LFPPONearFullLoss': numbers['feedback_v8']['lf_operating_points']['full_ppo_3em5']['nominal_loss'],
        'LFPPONearResidualFault': numbers['feedback_v8']['lf_operating_points']['res_ppo_3em3']['fault'],
        'LFPPONearResidualLoss': numbers['feedback_v8']['lf_operating_points']['res_ppo_3em3']['nominal_loss'],
    }
    for name, value in scalar_sources.items():
        macros.append('\\newcommand{\\' + name + '}{$' + pp(value) + '$}')
    (table_dir / 'feedback_macros.tex').write_text('\n'.join(macros) + '\n', encoding='utf-8')

