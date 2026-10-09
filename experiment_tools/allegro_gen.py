#!/usr/bin/env python3
"""Allegro transfer of the selected shared learning rates (protocol 2026-09-29, section 7: learning-rate and training-procedure
transfer inside the DemoGrasp framework; not an independent-system validation). Clones the held srv_al_* training plans
(TH lock and fixed z bias 2 cm, seeds s1-s3; plus the unpaired residual GRPO control on TH) with the selected lr per method,
new unused training seeds and new tags; replaces their dev evals by final-style evaluations (eval seed 4242, sealed seen/unseen,
fault condition + nominal); Allegro frozen planner re-evaluated on the same conditions. Writes temp/allegro_manifest.json.
Run on the server. --dry-run prints only."""
import argparse, copy, glob, json, os, re, sys
ROOT = '/root/autodl-tmp/DexGraspResearch/'; Q = ROOT + 'server_queue/'; PL = ROOT + 'reports/plans/'
sys.path.insert(0, ROOT + 'tools'); os.chdir(PL)
import plangen as P
ap = argparse.ArgumentParser(); ap.add_argument('--dry-run', action='store_true'); a = ap.parse_args()
ES = 4242
LR = {'grpo': ('fg', '1e-5'), 'ppo': ('fp', '1e-5'), 'plugppo': ('rp', '3e-3'), 'pluggrpo': ('rg', '1e-3'), 'pluggrpoU': ('rgU', '1e-3')}
CONDS = {'shlockTH': 'alTH', 'shbz2': 'albz2'}
used = P.used_seeds(); seedbase = max(used) // 1000 * 1000 + 2000
plans, man, n = [], {'eval_seed': ES, 'lr': {k: v[1] for k, v in LR.items()}, 'models': {}, 'frozen': {}}, 0
def final_eval(tmpl, shift, split, cond, mid):
    r = copy.deepcopy(tmpl); args = r['args']
    args[0] = '--shift=' + shift
    args[args.index('--object-list') + 1] = 'union_ycb_unidex/test_set_%s_cat.yaml' % split
    args[args.index('--episodes') + 1] = '8'; args[args.index('--seed') + 1] = str(ES)
    r['name'] = 'final_%s_%s_%s' % (split, cond, mid); r.pop('done_file', None); return r
for cond, ccode in CONDS.items():
    for alg, (mc, lr) in LR.items():
        for k in (1, 2, 3):
            fn = 'srv_al_%s_%s_s%d.json' % (alg, cond, k)
            if not os.path.exists(fn): continue
            q = json.load(open(fn)); tr = [r for r in q['runs'] if r.get('done_file') == 'train_log.json']
            old = re.sub(r'_p\d+$', '', tr[0]['name']); new = 'al_%s_%s_%s_s%d' % (cond, mc, lr.replace('e-', 'em'), k)
            runs = []
            for i, r in enumerate(tr):
                r = copy.deepcopy(r); r['name'] = r['name'].replace(old, new)
                r['args'] = [x.replace('/runs/' + old + '_', '/runs/' + new + '_') for x in r['args']]
                r['args'][r['args'].index('--lr') + 1] = lr
                v = seedbase + 1000 * n + i; assert v not in used, v; used.add(v); r['args'][r['args'].index('--seed') + 1] = str(v)
                runs.append(r)
            assert not any(old in x for r in runs for x in r['args']), fn
            ev = [r for r in q['runs'] if r['name'].startswith('testdet_seen_')][0]
            ev = copy.deepcopy(ev); ev['args'] = [x.replace('/runs/' + old + '_', '/runs/' + new + '_') for x in ev['args']]
            spec = ev['args'][0].split('=', 1)[1]
            for c2, sh in ((ccode, spec), ('alnom', 'hand=allegro')):
                for sp in ('seen', 'unseen'):
                    runs.append(final_eval(ev, sh, sp, c2, new)); man['models'].setdefault(alg + '|' + cond, []).append(new) if (c2 == ccode and sp == 'seen') else None
            plans.append(('srv_%s' % new, runs)); n += 1
    fz = json.load(open('srv_al_frozen_%s.json' % cond))
    tm = [r for r in fz['runs'] if r['name'].startswith('testdet_seen_munom_al_%s' % cond)][0]
    spec = tm['args'][0].split('=', 1)[1]
    runs = [final_eval(tm, sh, sp, c2, 'alfrozen') for c2, sh in ((ccode, spec), ('alnom', 'hand=allegro')) for sp in ('seen', 'unseen')]
    if cond != 'shlockTH': runs = [r for r in runs if '_alnom_' not in r['name']]
    for r in runs: man['frozen'].setdefault(r['name'].split('_')[2], {})[r['name'].split('_')[1]] = r['name']
    plans.insert(0, ('srv_final_alfrozen_%s' % cond, runs))
for name, runs in plans:
    tr = [r for r in runs if r.get('done_file') == 'train_log.json']
    print(name, len(tr), 'train', len(runs) - len(tr), 'eval |', tr[0]['args'][tr[0]['args'].index('--lr') + 1] if tr else '-', '|', ' '.join(runs[-1]['args'])[:200])
print('plans', len(plans), 'chains', sum(1 for p in plans if not p[0].startswith('srv_final')), 'evals', sum(sum(1 for r in p[1] if r['name'].startswith('final_')) for p in plans))
if a.dry_run: print('dry-run'); raise SystemExit(0)
for j, (name, runs) in enumerate(plans):
    pf = name + '.json'; assert not os.path.exists(pf), pf
    json.dump({'name': name, 'purpose': 'Allegro transfer of selected shared lr + final-style eval (seed %d)' % ES, 'runs': runs}, open(pf, 'w'), indent=1)
    json.dump({'plan': 'reports/plans/' + pf}, open(Q + 'pending/05375%02d_%s.json' % (j, name), 'w'))
json.dump(man, open(ROOT + 'temp/allegro_manifest.json', 'w'), indent=1); print('written')
