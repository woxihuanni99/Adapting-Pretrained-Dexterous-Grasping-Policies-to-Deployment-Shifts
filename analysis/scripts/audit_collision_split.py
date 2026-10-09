#!/usr/bin/env python3
"""Fresh descriptive collision-mask audit; no training, simulation or new evaluation.

build_collision_split(data_root) counts and summarizes the same raw inputs in
one invocation. The manuscript calls it before writing numbers.json. The CLI
writes an audit copy only and never reads a cached collision_halves.json.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

VERSION = Path(__file__).resolve().parents[1]


def count(data_root):
    data_root = Path(data_root)
    manifest_bytes = (data_root/'meta/temp/final_manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    names = {n for conds in manifest['evals'].values() for d in conds.values() for n in d.values()}
    names |= {n for d in manifest['frozen'].values() for n in d.values()}
    counts, sequences = {}, {}
    digest = hashlib.sha256()
    for name in sorted(names):
        raw_bytes = (data_root/'runs'/name/'result.json').read_bytes()
        digest.update(name.encode('utf-8') + b'\0' + hashlib.sha256(raw_bytes).digest())
        record = json.loads(raw_bytes)
        assert record['status'] == 'completed', name
        episodes = record['episodes']
        assert len(episodes) == 8, name
        split = name.split('_')[1]
        nslots = {'seen': 141, 'unseen': 100}[split]
        boundary = nslots//2
        sequence = episodes[0]['object_assets']
        assert len(sequence) == nslots and sequence == sorted(sequence), name
        sequences.setdefault(split, sequence)
        assert sequence == sequences[split], name
        off = on = 0
        for ep in episodes:
            flags = ep['official_ppo_print_successes']
            assert len(flags) == nslots and ep['object_assets'] == sequence, name
            assert all(value in (0, 1) for value in flags), name
            off += sum(flags[:boundary]); on += sum(flags[boundary:])
        attempts = nslots*len(episodes)
        assert abs((off+on)/attempts-record['totals']['success_rate']) < 1e-12, name
        assert record['evaluated_environment_episodes'] == attempts, name
        counts[name] = {'n_envs': nslots, 'episodes': len(episodes),
            'off': [int(off), boundary*len(episodes)], 'on': [int(on), (nslots-boundary)*len(episodes)],
            'assets_off_first': sequence[0], 'assets_on_first': sequence[boundary]}
    def source(asset):
        if asset.startswith('unidex_core_'): return 'unidex_core'
        if asset.startswith('unidex_sem_'): return 'unidex_sem'
        return asset.split('_')[0]
    composition = {}
    for split, sequence in sequences.items():
        boundary = len(sequence)//2
        composition[split] = {}
        for key, subset, start in [('off', sequence[:boundary], 0), ('on', sequence[boundary:], boundary)]:
            composition[split][key] = {'index_range': [start, start+len(subset)-1],
                'slots': len(subset), 'unique_asset_ids': len(set(subset)),
                'source_counts': dict(collections.Counter(source(a) for a in subset))}
        composition[split]['asset_ids_shared_across_halves'] = sorted(set(sequence[:boundary]) & set(sequence[boundary:]))
    validation = {'records': len(counts), 'episodes_per_slot': 8, 'completed_status': True,
        'binary_outcomes': True, 'slot_counts_checked': {'seen':141, 'unseen':100},
        'fixed_sorted_object_order_across_records': True, 'reconstructed_totals_match': True,
        'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'ordered_result_digest_sha256': digest.hexdigest(), 'object_composition': composition}
    return manifest, counts, validation


def build_collision_split(data_root):
    """Fresh payload for numbers.json; estimates are descriptive, without decisions."""
    import audit_recompute_primary as A
    man, halves, validation = count(data_root)
    class Subset:
        def __init__(self, half): self.half = half
        def score(self, name, expected_attempts=None, final_split=None):
            h = halves[name]
            if final_split is not None: assert name.split('_')[1] == final_split, name
            if self.half == 'all':
                # Count weighting retains 70/71 seen slots; equal half means would be wrong.
                s, n = h['off'][0]+h['on'][0], h['off'][1]+h['on'][1]
                if expected_attempts is not None: assert n == expected_attempts, name
            else:
                s, n = h[self.half]
            return s/n
    def y(arch, src):
        return (arch.score(src['seen'], final_split='seen') + arch.score(src['unseen'], final_split='unseen'))/2
    def contrast(arch, cset, cond, weights):
        groups = [{'method':m, 'weight':w, 'values':[y(arch,man['evals'][e['tag']][cond])
                   for e in man['methods'][m]['sets'][cset]]} for m,w in zip(A.METHODS,weights)]
        r = A.summarize(groups)
        return {'est':r['estimate'], 'ci':r['ci95_two_sided'], 'df':r['df'],
                'means':{g['method']:g['mean'] for g in r['groups']}, 'n':{g['method']:g['n'] for g in r['groups']}}
    conds = [('LF','LF','Shadow LF'), ('PCL','pn10','Shadow PCL'), ('TH','TH','Shadow TH'),
             ('ZR','zr','Shadow random z'), ('BZ2PN','bz2pn','Shadow z + noise'), ('THPN','THpn','Shadow TH + noise')]
    out = {'provenance': {
        'status':'Post hoc descriptive grouping of archived outcomes; no new training, simulation, evaluation or selection.',
        'interpretation':'Configured collision-mask groups inferred from the inspected implementation; enabled does not imply actual contact on every attempt.',
        'runtime_limit':'Result files contain no collision-filter/configuration snapshot or executed source hash. Current code evidence does not independently attest the historical runtime state.',
        'replication_unit':'Independent training run; object slots and episodes are not additional training replications.',
        'intervals':'Unadjusted two-sided 95% Welch-Satterthwaite intervals.',
        'locked_estimands':'Existing labels identify reused estimators. Subset contrasts are post hoc; no subset p-values, Holm decisions or support outcomes are reported.',
        'input_freshness':'Counts and statistics rebuilt together from the same raw bytes in this invocation; no cached summary is read.'},
        'definition':'off: indices i < num_envs//2 have robot-table collisions disabled under the inspected implementation; on: remaining indices have them enabled',
        'recombination':'Combine successes and attempts across halves within each seen/unseen split, then average the two split rates equally; do not equally average the aggregate half scores.',
        'subset_sizes':{s:{h:validation['object_composition'][s][h]['slots'] for h in ('off','on')} for s in ('seen','unseen')},
        'episodes_per_slot':validation['episodes_per_slot'], 'input_validation':validation, 'input_counts':halves}
    for half in ('all','on','off'):
        arch = Subset(half); res = {}
        for cset, cond, label in conds:
            res[label] = {'frozen_fault':y(arch,man['frozen'][cond]), 'frozen_nom':y(arch,man['frozen']['nom']),
                'param_fault':contrast(arch,cset,cond,(.5,.5,-.5,-.5)),
                'param_nom':contrast(arch,cset,'nom',(.5,.5,-.5,-.5)),
                'opt_fault':contrast(arch,cset,cond,(-.5,.5,-.5,.5))}
        tests, _ = A.reconstruct_primary(arch,man)
        # The shared audit helper also computes primary-test decisions. Drop them
        # explicitly: these subset analyses report only descriptive uncertainty.
        res['locked_estimands'] = {k:{'est':v['estimate'], 'ci':v['ci95_two_sided'], 'df':v['df']} for k,v in tests.items()}
        out[half] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--count', action='store_true', help='Write a fresh count audit only; never used as manuscript input.')
    ap.add_argument('--data-root', type=Path, default=VERSION.parent/'data')
    ap.add_argument('--output-dir', type=Path, default=VERSION/'audit')
    a = ap.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.count:
        _, payload, validation = count(a.data_root.resolve())
        output = a.output_dir/'collision_halves.json'
        (a.output_dir/'collision_count_validation.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8')
    else:
        payload = build_collision_split(a.data_root.resolve())
        output = a.output_dir/'collision_split_stats.json'
    output.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print('Wrote independent audit copy:', output)


if __name__ == '__main__':
    main()
