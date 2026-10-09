#!/usr/bin/env python3
"""Matched-GRPO control (2026-10-07; plan: reports/补充实验锁定说明_设置匹配对照_1007.md). Run on the server in the repo root.
18 training chains, fixed budget:
  * fsr_<C>_<lr>_s<k>: residual GRPO with exploration fixed to max(sigma0, 0.05) (rho not trained) and 2 epochs
    (research_plugin_rl_fs.py --fixed-scale-floor 0.05 --epochs 2); cloned from the main residual GRPO chains of the same
    condition (srv_pluggrpo_shlockLF_lr1e3_s{k}.json / srv_pluggrpo_pcl1cm_lr1e3_s{k}.json): same 12 phases, objects,
    iterations, group size 8, minibatches 4, Adam state across phases; lr in {1e-3, 3e-3}; C in {LF, PCL}; k = 1..3.
  * fsf_<C>_s<k>: full-weight GRPO, unchanged main configuration (plangen.grpo, lr 1e-5), new training seeds.
Training keeps the original physics (enableRobotTableCollision=False). Fresh unused training seeds.
Per chain: development evals (fault 512 episodes fmdet_munom_<label>_<short>_64x8; nominal 2048 fmdet2k_munom_<short>; seed 777).
Final-protocol test evals (seed 4242, seen/unseen) for every chain: LF models -> LF, nom, LF25, LF75; PCL models -> pn10, nom.
Also re-measures the frozen planner's 2048-episode development nominal success on this machine (fs_fmdet2k_munom_frozen).
Writes temp/fs_manifest.json. --dry-run prints only."""
import argparse, copy, glob, json, os, sys
ROOT = '/root/autodl-tmp/DexGraspResearch/'; E = '/mnt/e/DexGraspResearch/runs/'; Q = ROOT + 'server_queue/'
sys.path.insert(0, ROOT + 'tools'); os.chdir(ROOT + 'reports/plans')
import plangen as P
ap = argparse.ArgumentParser(); ap.add_argument('--dry-run', action='store_true'); a = ap.parse_args()
ES = 4242; OLD, NEW = 'research_plugin_rl.py', 'research_plugin_rl_fs.py'
LRS = (('1e-3', '1em3'), ('3e-3', '3em3'))
# condition: shift spec, pcl noise, dev label, final fault name, residual template, residual template tag, extra final conditions
COND = {'LF': ('lock=LF', None, 'shlockLF', 'LF', 'srv_pluggrpo_shlockLF_lr1e3_s%d.json', 'pluggrpo_shlockLF_lr1e3_s%d',
               [('lock=LF|lockpos=0.25', None, 'LF25'), ('lock=LF|lockpos=0.75', None, 'LF75')]),
        'PCL': (None, '0.01', 'pn10', 'pn10', 'srv_pluggrpo_pcl1cm_lr1e3_s%d.json', 'pluggrpo_pcl1cm_lr1e3_s%d', [])}
used = P.used_seeds(); seedbase = max(used) // 1000 * 1000 + 2000
def ev(script, ol, eps, seed, pn, ckflag, ck, extra=()):
    return [script, '--mode', 'eval', '--object-list', ol, '--mus', 'none', '--episodes', str(eps), '--seed', str(seed),
            '--disable-gpu-jit-fusion', '--deterministic'] + (['--pcl-noise', pn] if pn else []) + [ckflag, ck] + list(extra)
def run(name, spec, args):
    return {'name': name, 'probe': 'research_shift_run.py', 'args': ['--shift=' + spec] + args} if spec else {'name': name, 'probe': args[0], 'args': args[1:]}
DEV = 'union_ycb_unidex/research_dfg_dev64_x2.yaml'
def dev_evals(spec, pn, label, short, script, ckflag, ck, extra=()):
    return [run('fmdet_munom_%s_%s_64x8' % (label, short), spec, ev(script, DEV, 4, 777, pn, ckflag, ck, extra)),
            run('fmdet2k_munom_%s' % short, None, ev(script, DEV, 16, 777, None, ckflag, ck, extra))]
def final_evals(c, tag, script, ckflag, ck, extra=()):
    spec, pn, _, cname, _, _, more = COND[c]
    out = []
    for (sp_, pn_, cn) in [(spec, pn, cname), (None, None, 'nom')] + more:
        for split in ('seen', 'unseen'):
            out.append(run('final_%s_%s_%s' % (split, cn, tag), sp_, ev(script, 'union_ycb_unidex/test_set_%s_cat.yaml' % split, 8, ES, pn_, ckflag, ck, extra)))
    return out
chains, finals, man, n = [], [], {'eval_seed': ES, 'fixed_scale_floor': 0.05, 'epochs_residual_matched': 2, 'fsr': {}, 'fsf': {}}, 0
for c, (spec, pn, label, cname, srcpat, oldpat, more) in COND.items():
    for lr, lrs in LRS:
        for k in (1, 2, 3):
            src, old = srcpat % k, oldpat % k
            tr = [copy.deepcopy(r) for r in json.load(open(src))['runs'] if isinstance(r, dict) and r.get('done_file') == 'train_log.json']
            assert len(tr) == 12, src
            tag = 'fsr_%s_%s_s%d' % (c, lrs, k); short = tag.replace('_', '')
            for i, r in enumerate(tr):
                assert r['name'] == '%s_p%02d' % (old, i), r['name']
                r['name'] = '%s_p%02d' % (tag, i)
                r['args'] = [x.replace('/runs/' + old + '_', '/runs/' + tag + '_') for x in r['args']]
                if r['probe'] == OLD: r['probe'] = NEW
                r['args'] = [NEW if x == OLD else x for x in r['args']]
                assert r['args'][r['args'].index('--algo') + 1] == 'grpo'
                j = r['args'].index('--lr'); assert r['args'][j + 1] == '1e-3'; r['args'][j + 1] = lr
                j = r['args'].index('--epochs'); assert r['args'][j + 1] == '5'; r['args'][j + 1] = '2'
                r['args'] += ['--fixed-scale-floor', '0.05']
                v = seedbase + 1000 * n + i; assert v not in used, v; used.add(v); r['args'][r['args'].index('--seed') + 1] = str(v)
                assert not any(old in x for x in r['args']) and NEW in [r['probe']] + r['args']
            ck = E + '%s_p11/head.pt' % tag; xf = ('--fixed-scale-floor', '0.05')
            chains.append(('srv_%s' % tag, tr + dev_evals(spec, pn, label, short, NEW, '--init-head', ck, xf)))
            finals.append(('srv_fsfinal_%s' % tag, final_evals(c, tag, NEW, '--init-head', ck, xf), ['runs/%s_p11/head.pt' % tag]))
            man['fsr'].setdefault('%s|%s' % (c, lr), []).append({'tag': tag, 'short': short}); n += 1
    for k in (1, 2, 3):
        tag = 'fsf_%s_s%d' % (c, k); short = tag.replace('_', ''); sb = seedbase + 1000 * n; n += 1
        tr = [r for r in P.grpo(tag, short, spec, pn, label, '1e-5', sb) if r.get('done_file') == 'train_log.json']
        assert len(tr) == 12
        for v in [int(r['args'][r['args'].index('--seed') + 1]) for r in tr]: used.add(v)
        ck = E + '%s_p11/final.pt' % tag
        chains.append(('srv_%s' % tag, tr + dev_evals(spec, pn, label, short, 'research_grpo_fm.py', '--init-checkpoint', ck)))
        finals.append(('srv_fsfinal_%s' % tag, final_evals(c, tag, 'research_grpo_fm.py', '--init-checkpoint', ck), ['runs/%s_p11/final.pt' % tag]))
        man['fsf'].setdefault(c, []).append({'tag': tag, 'short': short})
# frozen development nominal baseline re-measured on this machine (clone of srv_sel2k_frozen.json run fmdet2k_munom_frozen)
fz = [r for r in json.load(open('srv_sel2k_frozen.json'))['runs'] if r['name'] == 'fmdet2k_munom_frozen'][0]
fz = dict(copy.deepcopy(fz), name='fs_fmdet2k_munom_frozen')
# consistency check: generated full-weight chain vs the main full-weight GRPO chains (args equal apart from seed/paths)
def norm(args, tag):
    a = list(args); a[a.index('--seed') + 1] = 'SEED'
    return [x.split('/runs/')[0] + '/runs/X' if '/runs/' in x else x for x in a]
for c, mainplan in (('LF', 'srv_grpo_shlockLF_s1.json'), ('PCL', 'srv_grpo_pcl1cm_v3.json')):
    if os.path.exists(mainplan):
        mt = [r for r in json.load(open(mainplan))['runs'] if r.get('done_file') == 'train_log.json']
        gt = [r for r in dict(chains)['srv_fsf_%s_s1' % c] if r.get('done_file') == 'train_log.json']
        same = len(mt) == len(gt) and all(norm(x['args'], '') == norm(y['args'], '') and x['probe'] == y['probe'] for x, y in zip(mt, gt))
        print('full-weight template check vs', mainplan, ':', 'IDENTICAL apart from seed/paths' if same else 'DIFFERENT')
        if not same:
            for x, y in zip(mt, gt):
                if norm(x['args'], '') != norm(y['args'], ''): print('  main:', ' '.join(norm(x['args'], ''))[:400]); print('  gen :', ' '.join(norm(y['args'], ''))[:400]); break
    else:
        print('main plan not found for check:', mainplan)
c0 = chains[0][1]; print('chains', len(chains), '|', chains[0][0], ' '.join([c0[0]['probe']] + c0[0]['args'])[:420])
print('   last train', ' '.join(c0[11]['args'])[-300:]); print('   dev', [r['name'] for r in c0[12:]], ' '.join(c0[12]['args'])[-240:])
cf = dict(chains)['srv_fsf_PCL_s1']; print('   fw PCL', ' '.join([cf[0]['probe']] + cf[0]['args'])[:300])
print('   finals e.g.', [r['name'] for r in finals[0][1]], '|', ' '.join([finals[0][1][0]['probe']] + finals[0][1][0]['args'])[:300])
print('evals: dev', 2 * len(chains), 'final', sum(len(f[1]) for f in finals), '| seeds from', seedbase)
if a.dry_run: print('dry-run'); raise SystemExit(0)
json.dump({'name': 'srv_fs_frozen_dev', 'purpose': 'matched-GRPO control: frozen dev nominal baseline on this machine', 'runs': [fz]}, open('srv_fs_frozen_dev.json', 'x'), indent=1)
json.dump({'plan': 'reports/plans/srv_fs_frozen_dev.json'}, open(Q + 'pending/0545a00_srv_fs_frozen_dev.json', 'x'))
for j, (name, runs) in enumerate(chains):
    pf = name + '.json'; assert not os.path.exists(pf), pf
    json.dump({'name': name, 'purpose': 'matched-GRPO control (2026-10-07)', 'runs': runs}, open(pf, 'w'), indent=1)
    json.dump({'plan': 'reports/plans/' + pf}, open(Q + 'pending/0545a%02d_%s.json' % (j + 1, name), 'w'))
for j, (name, runs, req) in enumerate(finals):
    pf = name + '.json'; assert not os.path.exists(pf), pf
    json.dump({'name': name, 'purpose': 'matched-GRPO control: final-protocol test evaluation, eval seed %d' % ES, 'runs': runs}, open(pf, 'w'), indent=1)
    json.dump({'plan': 'reports/plans/' + pf, 'requires': req}, open(Q + 'waiting/0545b%02d_%s.json' % (j, name), 'w'))
json.dump(man, open(ROOT + 'temp/fs_manifest.json', 'w'), indent=1); print('written')
