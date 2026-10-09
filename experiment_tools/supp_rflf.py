#!/usr/bin/env python3
"""Targeted supplementary S5 (post-lock, exploratory; decided 2026-10-01 after the Shadow final and Allegro results).
Question: is the large residual-over-full-weight advantage specific to the little-finger lock (where the shared lr was
selected), or does it hold for another severe finger lock? Allegro thumb lock (frozen 0.42) and all moderate Shadow shifts
showed full-weight >= residual. Test: Shadow ring+little double lock (lock=RF,LF; frozen dev 0.16), 4 methods at the main
shared lr, seeds s1-s3, no retuning; final protocol evaluation (seed 4242, sealed seen/unseen) on RFLF and nominal, plus frozen.
Queued after the earlier supplementary items (prefix 05383). Writes temp/supp_rflf_manifest.json. --dry-run prints only."""
import argparse, json, os, sys
ROOT = '/root/autodl-tmp/DexGraspResearch/'; E = '/mnt/e/DexGraspResearch/runs/'; Q = ROOT + 'server_queue/'
sys.path.insert(0, ROOT + 'tools'); os.chdir(ROOT + 'reports/plans')
import plangen as P
ap = argparse.ArgumentParser(); ap.add_argument('--dry-run', action='store_true'); a = ap.parse_args()
ES = 4242
SEL = json.load(open(ROOT + 'temp/shared_lr_select.json'))
MC = {'full_grpo': 'fg', 'full_ppo': 'fp', 'res_ppo': 'rp', 'res_grpo': 'rg'}
COND = {'nom': (None, None), 'RFLF': ('lock=RF,LF', None)}
used = P.used_seeds(); seedbase = max(used) // 1000 * 1000 + 2000
def ev(model, cond, split):
    spec, pn = COND[cond]; mid = model['tag'] if model else 'frozen'
    args = ['--mode', 'eval', '--object-list', 'union_ycb_unidex/test_set_%s_cat.yaml' % split, '--mus', 'none', '--episodes', '8', '--seed', str(ES),
            '--disable-gpu-jit-fusion', '--deterministic'] + (['--pcl-noise', pn] if pn else []) + ([model['flag'], model['ck']] if model else [])
    probe = model['probe'] if model else 'research_grpo_fm.py'; name = 'final_%s_%s_%s' % (split, cond, mid)
    return {'name': name, 'probe': 'research_shift_run.py', 'args': ['--shift=' + spec, probe] + args} if spec else {'name': name, 'probe': probe, 'args': args}
chains, evals, man, n = [], [], {'eval_seed': ES, 'models': {}}, 0
for m in MC:
    lr = SEL['methods'][m]['d05']['shared_lr']; man['models'][m] = {'lr': lr, 'tags': []}
    for k in (1, 2, 3):
        tag = 'sx_RFLF_%s_%s_s%d' % (MC[m], lr.replace('e-', 'em'), k); short = tag.replace('_', ''); sb = seedbase + 1000 * n; n += 1
        if m == 'full_grpo': runs = P.grpo(tag, short, 'lock=RF,LF', None, 'shlockRFLF', lr, sb)
        elif m == 'full_ppo': runs = P.ppo(tag, short, 'lock=RF,LF', None, 'shlockRFLF', lr, sb)
        else:
            runs = P.plug(tag, short, m.split('_')[1], 'lock=RF,LF', None, 'shlockRFLF', sb)
            for r in runs:
                if r.get('done_file') == 'train_log.json': r['args'][r['args'].index('--lr') + 1] = lr
        tr = [r for r in runs if r.get('done_file') == 'train_log.json']; fin = tr[-1]['name']; res = m.startswith('res')
        md = {'tag': tag, 'ck': E + '%s/%s' % (fin, 'head.pt' if res else 'final.pt'), 'flag': '--init-head' if res else '--init-checkpoint',
              'probe': 'research_plugin_rl.py' if res else 'research_grpo_fm.py'}
        chains.append(('srv_%s' % tag, tr)); man['models'][m]['tags'].append(tag)
        evals.append(('srv_supp_%s' % tag, [ev(md, c, sp) for c in ('RFLF', 'nom') for sp in ('seen', 'unseen')], ['runs/' + md['ck'].split('/runs/')[1]]))
evals.append(('srv_supp_frozen_RFLF', [ev(None, 'RFLF', sp) for sp in ('seen', 'unseen')], None))
for c in chains: print('chain', c[0], len(c[1]), 'train runs | lr', c[1][0]['args'][c[1][0]['args'].index('--lr') + 1], '|', c[1][0]['args'][0])
print('eval plans', len(evals), 'evals', sum(len(e[1]) for e in evals))
if a.dry_run: print('dry-run'); raise SystemExit(0)
for j, (name, runs) in enumerate(chains):
    pf = name + '.json'; assert not os.path.exists(pf), pf
    json.dump({'name': name, 'purpose': 'supplementary S5 (RFLF severe-lock generality)', 'runs': runs}, open(pf, 'w'), indent=1)
    json.dump({'plan': 'reports/plans/' + pf}, open(Q + 'pending/05383a%02d_%s.json' % (j, name), 'w'))
for j, (name, runs, req) in enumerate(evals):
    pf = name + '.json'; assert not os.path.exists(pf), pf
    json.dump({'name': name, 'purpose': 'supplementary S5 final-protocol evaluation, eval seed %d' % ES, 'runs': runs}, open(pf, 'w'), indent=1)
    item = {'plan': 'reports/plans/' + pf}
    if req: item['requires'] = req
    json.dump(item, open(Q + ('waiting' if req else 'pending') + '/05383b%03d_%s.json' % (j, name), 'w'))
json.dump(man, open(ROOT + 'temp/supp_rflf_manifest.json', 'w'), indent=1); print('written')
