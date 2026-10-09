#!/usr/bin/env python3
"""Lock step 3 selection for the supplementary KL-retention baseline (written 2026-10-02 before any KL-baseline result).
Uses development data only (same rule as the main shared-lr selection, delta = 0.05):
  N_c(cfg) = mean over s1-s3 of nominal dev success, 2048 episodes (fmdet2k_munom_<short>) of the models trained on c
  N0       = frozen planner on the same 2048-episode set (fmdet2k_munom_frozen)
  S_c(cfg) = mean over s1-s3 of fault dev success, 512 episodes (fmdet_munom_<label>_<short>_64x8)
  feasible : N_LF - N0 >= -0.05 and N_PCL - N0 >= -0.05 (unrounded)
  choose   : max (S_LF + S_PCL) / 2 over the four beta > 0 configs; ties: smaller worst nominal loss, smaller lr, smaller beta.
No feasible config -> 'no feasible configuration', no thumb transfer, nothing else is tried. Otherwise the selected
config is transferred unchanged to the thumb lock (lock=TH, seeds s1-s3, same budget) with final-protocol test evals
(seed 4242, seen/unseen, TH + nominal). Writes temp/kl_select.json. --dry-run prints only."""
import argparse, json, os, sys
ROOT = '/root/autodl-tmp/DexGraspResearch/'; E = '/mnt/e/DexGraspResearch/runs/'; Q = ROOT + 'server_queue/'
sys.path.insert(0, ROOT + 'tools'); os.chdir(ROOT + 'reports/plans')
import plangen as P
ap = argparse.ArgumentParser(); ap.add_argument('--dry-run', action='store_true'); a = ap.parse_args()
M = json.load(open(ROOT + 'temp/kl_manifest.json')); ES = M['eval_seed']
LABEL = {'LF': 'shlockLF', 'PCL': 'pn10'}
def sr(name):
    f = ROOT + 'runs/' + name + '/result.json'
    return json.load(open(f))['totals']['success_rate'] if os.path.exists(f) else None
N0 = sr('fmdet2k_munom_frozen'); assert N0 is not None
rows, missing = {}, []
for lr in ('3e-5', '1e-4'):
    for bi in ('bA', 'bB'):
        row = {}
        for c in ('LF', 'PCL'):
            tags = M['configs']['%s|%s|%s' % (c, lr, bi)]
            N = [sr('fmdet2k_munom_%s' % t['short']) for t in tags]; S = [sr('fmdet_munom_%s_%s_64x8' % (LABEL[c], t['short'])) for t in tags]
            missing += [t['short'] for t, x, y in zip(tags, N, S) if x is None or y is None]
            row[c] = {'N': N, 'S': S, 'dN': (sum(N) / len(N) - N0) if None not in N else None, 'S_mean': (sum(S) / len(S)) if None not in S else None}
        row['feasible'] = all(row[c]['dN'] is not None and row[c]['dN'] >= -0.05 for c in ('LF', 'PCL'))
        row['objective'] = (row['LF']['S_mean'] + row['PCL']['S_mean']) / 2 if None not in (row['LF']['S_mean'], row['PCL']['S_mean']) else None
        row['worst_loss'] = -min(row[c]['dN'] for c in ('LF', 'PCL')) if None not in (row['LF']['dN'], row['PCL']['dN']) else None
        rows['%s|%s' % (lr, bi)] = row
out = {'N0': N0, 'beta': M['beta'], 'rows': rows, 'missing': missing}
for k, r in rows.items():
    print('%-9s feasible=%s obj=%s dN_LF=%s dN_PCL=%s' % (k, r['feasible'], r['objective'] and round(r['objective'], 4),
          r['LF']['dN'] and round(r['LF']['dN'], 4), r['PCL']['dN'] and round(r['PCL']['dN'], 4)))
if missing: print('MISSING', missing); raise SystemExit(1)
feas = [(k, r) for k, r in rows.items() if r['feasible']]
if not feas:
    out['selected'] = None; out['note'] = 'no feasible configuration'
    print('no feasible configuration'); json.dump(out, open(ROOT + 'temp/kl_select.json', 'w'), indent=1) if not a.dry_run else None; raise SystemExit(0)
order = {'3e-5': 0, '1e-4': 1, 'bA': 0, 'bB': 1}
feas.sort(key=lambda kr: (-kr[1]['objective'], kr[1]['worst_loss'], order[kr[0].split('|')[0]], order[kr[0].split('|')[1]]))
sel = feas[0][0]; lr, bi = sel.split('|'); beta = M['beta'][bi]; out['selected'] = {'lr': lr, 'beta_id': bi, 'beta': beta}
print('selected', out['selected'])
OLD, NEW = 'research_grpo_finetune_v2.py', 'research_grpo_finetune_v2_kl.py'
used = P.used_seeds(); seedbase = max(used) // 1000 * 1000 + 2000
chains, finals = [], []
for k in (1, 2, 3):
    tag = 'kl_TH_%s_%s_s%d' % (lr.replace('e-', 'em'), bi, k); short = tag.replace('_', '')
    tr = [r for r in P.grpo(tag, short, 'lock=TH', None, 'shlockTH', lr, seedbase + 1000 * (k - 1)) if r.get('done_file') == 'train_log.json']
    for r in tr:
        r['args'] = [NEW if x == OLD else x for x in r['args']] + ['--kl-coef', str(beta)]
        assert NEW in r['args']
    ck = E + '%s_p11/final.pt' % tag
    ev = []
    for spec, cname in (('lock=TH', 'TH'), (None, 'nom')):
        for sp in ('seen', 'unseen'):
            args = ['--mode', 'eval', '--object-list', 'union_ycb_unidex/test_set_%s_cat.yaml' % sp, '--mus', 'none', '--episodes', '8', '--seed', str(ES),
                    '--disable-gpu-jit-fusion', '--deterministic', '--init-checkpoint', ck]
            nm = 'final_%s_%s_%s' % (sp, cname, tag)
            ev.append({'name': nm, 'probe': 'research_shift_run.py', 'args': ['--shift=' + spec, 'research_grpo_fm.py'] + args} if spec else {'name': nm, 'probe': 'research_grpo_fm.py', 'args': args})
    chains.append(('srv_%s' % tag, tr)); finals.append(('srv_klfinal_%s' % tag, ev, ['runs/%s_p11/final.pt' % tag]))
    out.setdefault('thumb_tags', []).append(tag)
print('thumb chains', len(chains), '|', ' '.join(chains[0][1][0]['args'])[:300])
if a.dry_run: print('dry-run'); raise SystemExit(0)
for j, (name, runs) in enumerate(chains):
    pf = name + '.json'; assert not os.path.exists(pf), pf
    json.dump({'name': name, 'purpose': 'KL-retention baseline thumb-lock transfer of the selected config (lock step 3)', 'runs': runs}, open(pf, 'w'), indent=1)
    json.dump({'plan': 'reports/plans/' + pf}, open(Q + 'pending/0539c%02d_%s.json' % (j, name), 'w'))
for j, (name, runs, req) in enumerate(finals):
    pf = name + '.json'; assert not os.path.exists(pf), pf
    json.dump({'name': name, 'purpose': 'KL-retention baseline final-protocol test evaluation, eval seed %d' % ES, 'runs': runs}, open(pf, 'w'), indent=1)
    json.dump({'plan': 'reports/plans/' + pf, 'requires': req}, open(Q + 'waiting/0539d%03d_%s.json' % (j, name), 'w'))
json.dump(out, open(ROOT + 'temp/kl_select.json', 'w'), indent=1); print('written')
