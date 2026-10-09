#!/usr/bin/env python3
"""Output-shift coordinate control (2026-10-08; plan: reports/补充实验锁定说明_偏置坐标对照_1008.md). Run on the server in the repo root.
36 training chains, fixed budget. Both arms freeze the whole planner and train only a 24-dim output shift with the full-weight
GRPO procedure (research_grpo_finetune_v2_bias.py; same 12 phases, objects, iterations, group size 8, 2 epochs, 4 minibatches,
clip 0.2, grad-norm clip 1.0, Adam state across phases, sampling/scoring scale max(sigma0, 0.05)):
  * bxr_<C>_<lr>_s<k>: --bias-coord raw   (m = m0 + b),          lr in {3e-3, 1e-2, 3e-2}
  * bxs_<C>_<lr>_s<k>: --bias-coord sigma (m = m0 + sigma0 * c), lr in {1e-2, 3e-2, 1e-1}
C in {LF, PCL}, k = 1..3, fresh unused seeds. Phase k>0 starts from the base planner plus the previous cumulative shift (shift.pt)
and the previous Adam state; the last phase also writes final.pt (shift folded into actor_mean[-1].bias), which the unchanged
eval script research_grpo_fm.py evaluates. Per chain: development evals (fault 512 fmdet_munom_<label>_<short>_64x8; nominal 2048
fmdet2k_munom_<short>; seed 777). Final-protocol test evals (seed 4242, seen/unseen) for every chain: LF -> LF, nom, LF25, LF75;
PCL -> pn10, nom. Also re-measures the frozen planner's 2048-episode development nominal success (bx_fmdet2k_munom_frozen).
Writes temp/bx_manifest.json. --dry-run prints only."""
import argparse, copy, json, os, sys
ROOT = '/root/autodl-tmp/DexGraspResearch/'; E = '/mnt/e/DexGraspResearch/runs/'; Q = ROOT + 'server_queue/'
sys.path.insert(0, ROOT + 'tools'); os.chdir(ROOT + 'reports/plans')
import plangen as P
ap = argparse.ArgumentParser(); ap.add_argument('--dry-run', action='store_true'); a = ap.parse_args()
ES = 4242; OLD, NEW = 'research_grpo_finetune_v2.py', 'research_grpo_finetune_v2_bias.py'
ARMS = {'raw': ('bxr', (('3e-3', '3em3'), ('1e-2', '1em2'), ('3e-2', '3em2'))),
        'sigma': ('bxs', (('1e-2', '1em2'), ('3e-2', '3em2'), ('1e-1', '1em1')))}
COND = {'LF': ('lock=LF', None, 'shlockLF', 'LF', [('lock=LF|lockpos=0.25', None, 'LF25'), ('lock=LF|lockpos=0.75', None, 'LF75')]),
        'PCL': (None, '0.01', 'pn10', 'pn10', [])}
used = P.used_seeds(); seedbase = max(max(used) // 1000 * 1000 + 2000, 400000)  # >= 400000: above the 10/7 seeds (277000-295011), whose plans are not in this plans/ copy
def ev(script, ol, eps, seed, pn, ck):
    return [script, '--mode', 'eval', '--object-list', ol, '--mus', 'none', '--episodes', str(eps), '--seed', str(seed),
            '--disable-gpu-jit-fusion', '--deterministic'] + (['--pcl-noise', pn] if pn else []) + ['--init-checkpoint', ck]
def run(name, spec, args):
    return {'name': name, 'probe': 'research_shift_run.py', 'args': ['--shift=' + spec] + args} if spec else {'name': name, 'probe': args[0], 'args': args[1:]}
DEV = 'union_ycb_unidex/research_dfg_dev64_x2.yaml'
def dev_evals(spec, pn, label, short, ck):
    return [run('fmdet_munom_%s_%s_64x8' % (label, short), spec, ev('research_grpo_fm.py', DEV, 4, 777, pn, ck)),
            run('fmdet2k_munom_%s' % short, None, ev('research_grpo_fm.py', DEV, 16, 777, None, ck))]
def final_evals(c, tag, ck):
    spec, pn, _, cname, more = COND[c]
    out = []
    for (sp_, pn_, cn) in [(spec, pn, cname), (None, None, 'nom')] + more:
        for split in ('seen', 'unseen'):
            out.append(run('final_%s_%s_%s' % (split, cn, tag), sp_, ev('research_grpo_fm.py', 'union_ycb_unidex/test_set_%s_cat.yaml' % split, 8, ES, pn_, ck)))
    return out
chains, finals, man, n = [], [], {'eval_seed': ES, 'std_floor': 0.05, 'epochs': 2, 'arms': {}}, 0
for coord, (pre, lrs) in ARMS.items():
    for c, (spec, pn, label, cname, more) in COND.items():
        for lr, lrs_ in lrs:
            for k in (1, 2, 3):
                tag = '%s_%s_%s_s%d' % (pre, c, lrs_, k); short = tag.replace('_', ''); sb = seedbase + 1000 * n; n += 1
                tr = [copy.deepcopy(r) for r in P.grpo(tag, short, spec, pn, label, lr, sb) if r.get('done_file') == 'train_log.json']
                assert len(tr) == 12
                for i, r in enumerate(tr):
                    used.add(int(r['args'][r['args'].index('--seed') + 1]))
                    if r['probe'] == OLD: r['probe'] = NEW
                    r['args'] = [NEW if x == OLD else x for x in r['args']]
                    assert NEW in [r['probe']] + r['args'] and OLD not in [r['probe']] + r['args']
                    assert r['args'][r['args'].index('--lr') + 1] == lr and r['args'][r['args'].index('--epochs') + 1] == '2'
                    assert r['args'][r['args'].index('--std-floor') + 1] == '0.05' and '--learn-std' not in r['args']
                    if i == 0:
                        assert '--init-checkpoint' not in r['args'] and '--init-optimizer' not in r['args']
                    else:
                        j = r['args'].index('--init-checkpoint'); prev = r['args'][j + 1]
                        assert prev == E + '%s_p%02d/final.pt' % (tag, i - 1), prev
                        r['args'][j:j + 2] = ['--init-shift', E + '%s_p%02d/shift.pt' % (tag, i - 1)]
                        assert r['args'][r['args'].index('--init-optimizer') + 1] == E + '%s_p%02d/optimizer.pt' % (tag, i - 1)
                    r['args'] += ['--bias-coord', coord] + (['--save-full'] if i == 11 else [])
                ck = E + '%s_p11/final.pt' % tag
                chains.append(('srv_%s' % tag, tr + dev_evals(spec, pn, label, short, ck)))
                finals.append(('srv_bxfinal_%s' % tag, final_evals(c, tag, ck), ['runs/%s_p11/final.pt' % tag]))
                man['arms'].setdefault(coord, {}).setdefault('%s|%s' % (c, lr), []).append({'tag': tag, 'short': short})
fz = [r for r in json.load(open('srv_sel2k_frozen.json'))['runs'] if r['name'] == 'fmdet2k_munom_frozen'][0]
fz = dict(copy.deepcopy(fz), name='bx_fmdet2k_munom_frozen')
c0 = chains[0][1]; print('chains', len(chains), '|', chains[0][0], ' '.join([c0[0]['probe']] + c0[0]['args'])[:400])
print('   p01', ' '.join(c0[1]['args'])[-330:]); print('   p11', ' '.join(c0[11]['args'])[-200:])
cp = [x for x in chains if 'PCL' in x[0]][0][1]; print('   PCL p00', ' '.join([cp[0]['probe']] + cp[0]['args'])[:300])
print('   dev', [r['name'] for r in c0[12:]], '| finals e.g.', [r['name'] for r in finals[0][1]][:4])
print('evals: dev', 2 * len(chains) + 1, 'final', sum(len(f[1]) for f in finals), '| seeds from', seedbase, 'to', max(used))
if a.dry_run: print('dry-run'); raise SystemExit(0)
json.dump({'name': 'srv_bx_frozen_dev', 'purpose': 'output-shift coordinate control: frozen dev nominal baseline on this machine', 'runs': [fz]}, open('srv_bx_frozen_dev.json', 'x'), indent=1)
json.dump({'plan': 'reports/plans/srv_bx_frozen_dev.json'}, open(Q + 'pending/0547a00_srv_bx_frozen_dev.json', 'x'))
for j, (name, runs) in enumerate(chains):
    pf = name + '.json'; assert not os.path.exists(pf), pf
    json.dump({'name': name, 'purpose': 'output-shift coordinate control (2026-10-08)', 'runs': runs}, open(pf, 'w'), indent=1)
    json.dump({'plan': 'reports/plans/' + pf}, open(Q + 'pending/0547a%02d_%s.json' % (j + 1, name), 'w'))
os.makedirs(Q + 'waiting', exist_ok=True)
for j, (name, runs, req) in enumerate(finals):
    pf = name + '.json'; assert not os.path.exists(pf), pf
    json.dump({'name': name, 'purpose': 'output-shift coordinate control: final-protocol test evaluation, eval seed %d' % ES, 'runs': runs}, open(pf, 'w'), indent=1)
    json.dump({'plan': 'reports/plans/' + pf, 'requires': req}, open(Q + 'pending/0547b%02d_%s.json' % (j, name), 'w'))
json.dump(man, open(ROOT + 'temp/bx_manifest.json', 'w'), indent=1); print('written')
