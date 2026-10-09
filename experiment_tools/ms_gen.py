#!/usr/bin/env python3
"""Lock step 4 (2026-10-02; supplementary single-factor intervention + constant latent-offset diagnostic; lock document
section 11). Run on the server in the repository root. --dry-run prints only.
A) Mean-scale intervention: residual GRPO with S = D_c (block-RMS of sigma0 within wrist translation / wrist rotation / hand)
   instead of S = diag(sigma0) (research_plugin_rl_ms.py --mean-scale block_rms). 2 conditions (lock=LF, pcl-noise 0.01) x
   lr {1e-3, 3e-3} x s1-s3 = 12 chains, cloned from the existing S = diag(sigma0) chains with the same k (identical args,
   budget and phases) except script, --mean-scale and fresh training seeds. Per chain: development evals for the restricted
   selection (fault 512 fmdet_munom_<label>_<short>_64x8, nominal 2048 fmdet2k_munom_<short>, seed 777) and final-protocol
   test evals (seed 4242, seen/unseen, fault condition + nominal). Reference S = diag(sigma0): the existing s1-s3 models at the
   same two learning rates (their dev and test evals already exist).
B) Constant latent-offset diagnostic: for the residual GRPO models of the reference little-finger lock (lr 1e-3, s1-s3), the
   75%-lock retrained models (S2, s1-s3) and the ring+little lock (S5, s1-s5), heads whose delta is replaced by its mean over
   the fixed 2048 training-object states (runs/const_<tag>/head.pt, temp/const_heads.json) are evaluated with the original
   research_plugin_rl.py on the final protocol (own fault condition + nominal). Writes temp/ms_manifest.json."""
import argparse, copy, json, os, re, sys
ROOT = '/root/autodl-tmp/DexGraspResearch/'; E = '/mnt/e/DexGraspResearch/runs/'; Q = ROOT + 'server_queue/'
sys.path.insert(0, ROOT + 'tools'); os.chdir(ROOT + 'reports/plans')
import plangen as P
ap = argparse.ArgumentParser(); ap.add_argument('--dry-run', action='store_true'); a = ap.parse_args()
ES = 4242; OLD, NEW = 'research_plugin_rl.py', 'research_plugin_rl_ms.py'; FLAG = ['--mean-scale', 'block_rms']
COND = {'LF': ('lock=LF', None, 'shlockLF', 'LF', 'shlockLF'), 'PCL': (None, '0.01', 'pn10', 'pn10', 'pcl1cm')}
used = P.used_seeds(); seedbase = max(used) // 1000 * 1000 + 2000
def ev_args(ol, eps, seed, pn, script, ck, extra):
    return [script, '--mode', 'eval', '--object-list', ol, '--mus', 'none', '--episodes', str(eps), '--seed', str(seed),
            '--disable-gpu-jit-fusion', '--deterministic'] + (['--pcl-noise', pn] if pn else []) + ['--init-head', ck] + extra
def run(name, spec, args):
    return {'name': name, 'probe': 'research_shift_run.py', 'args': ['--shift=' + spec] + args} if spec else {'name': name, 'probe': args[0], 'args': args[1:]}
chains, finals, consts, man, n = [], [], [], {'eval_seed': ES, 'ms': {}, 'd0': {}, 'const': {}}, 0
for c, (spec, pn, label, cname, oldc) in COND.items():
    for lr, lrs in (('1e-3', '1e3'), ('3e-3', '3e3')):
        for k in (1, 2, 3):
            src = 'srv_pluggrpo_%s_lr%s_s%d.json' % (oldc, lrs, k); old = 'pluggrpo_%s_lr%s_s%d' % (oldc, lrs, k)
            tr = [copy.deepcopy(r) for r in json.load(open(src))['runs'] if isinstance(r, dict) and r.get('done_file') == 'train_log.json']
            assert len(tr) == 12, src
            tag = 'ms_%s_%s_s%d' % (c, lr.replace('e-', 'em'), k); short = tag.replace('_', '')
            for i, r in enumerate(tr):
                r['name'] = r['name'].replace(old, tag)
                r['args'] = [x.replace('/runs/' + old + '_', '/runs/' + tag + '_') for x in r['args']]
                if r['probe'] == OLD: r['probe'] = NEW
                r['args'] = [NEW if x == OLD else x for x in r['args']] + FLAG
                assert r['args'][r['args'].index('--lr') + 1] == lr and '--algo' in r['args'] and r['args'][r['args'].index('--algo') + 1] == 'grpo'
                v = seedbase + 1000 * n + i; assert v not in used, v; used.add(v); r['args'][r['args'].index('--seed') + 1] = str(v)
                assert not any(old in x for x in r['args']) and NEW in [r['probe']] + r['args']
            ck = E + '%s_p11/head.pt' % tag
            dev = [run('fmdet_munom_%s_%s_64x8' % (label, short), spec, ev_args('union_ycb_unidex/research_dfg_dev64_x2.yaml', 4, 777, pn, NEW, ck, FLAG)),
                   run('fmdet2k_munom_%s' % short, None, ev_args('union_ycb_unidex/research_dfg_dev64_x2.yaml', 16, 777, None, NEW, ck, FLAG))]
            fin = [run('final_%s_%s_%s' % (sp, cn, tag), sp_, ev_args('union_ycb_unidex/test_set_%s_cat.yaml' % sp, 8, ES, p_, NEW, ck, FLAG))
                   for (sp_, p_, cn) in ((spec, pn, cname), (None, None, 'nom')) for sp in ('seen', 'unseen')]
            chains.append(('srv_%s' % tag, tr + dev)); finals.append(('srv_msfinal_%s' % tag, fin, ['runs/%s_p11/head.pt' % tag])); n += 1
            man['ms'].setdefault('%s|%s' % (c, lr), []).append({'tag': tag, 'short': short})
            man['d0'].setdefault('%s|%s' % (c, lr), []).append({'tag': old, 'short': '%spluggrpolr%ss%d' % (oldc, lrs, k)})
CH = json.load(open(ROOT + 'temp/const_heads.json'))['models']
CC = {'pluggrpo_shlockLF_lr1e3': ('lock=LF', 'LF'), 'sx_LF75_rg_1em3': ('lock=LF|lockpos=0.75', 'LF75'), 'sx_RFLF_rg_1em3': ('lock=RF,LF', 'RFLF')}
for tag in sorted(CH):
    base = re.sub(r'_s\d+$', '', tag); spec, cname = CC[base]
    ck = E + 'const_%s/head.pt' % tag; assert os.path.exists(ROOT + 'runs/const_%s/head.pt' % tag), tag
    ct = 'const_' + tag
    consts.append(('srv_msconst_%s' % tag, [run('final_%s_%s_%s' % (sp, cn, ct), sp_, ev_args('union_ycb_unidex/test_set_%s_cat.yaml' % sp, 8, ES, None, OLD, ck, []))
                                           for (sp_, cn) in ((spec, cname), (None, 'nom')) for sp in ('seen', 'unseen')]))
    man['const'][tag] = {'const_tag': ct, 'cond': cname, 'head_sha256': CH[tag]['head_sha256']}
print('chains', len(chains), '| e.g.', chains[0][0], ' '.join([chains[0][1][0]['probe']] + chains[0][1][0]['args'])[:330])
print('   last train', ' '.join(chains[0][1][11]['args'])[-230:]); print('   dev', [r['name'] for r in chains[0][1][12:]])
print('   final', [r['name'] for r in finals[0][1]], '|', ' '.join([finals[0][1][0]['probe']] + finals[0][1][0]['args'])[:260])
print('const plans', len(consts), '| e.g.', [r['name'] for r in consts[0][1]], '|', ' '.join([consts[0][1][0]['probe']] + consts[0][1][0]['args'])[:260])
print('evals: dev', 2 * len(chains), 'final', 4 * len(finals), 'const', 4 * len(consts))
if a.dry_run: print('dry-run'); raise SystemExit(0)
for j, (name, runs) in enumerate(chains):
    pf = name + '.json'; assert not os.path.exists(pf), pf
    json.dump({'name': name, 'purpose': 'lock step 4: residual GRPO with block-RMS mean scale (single-factor intervention)', 'runs': runs}, open(pf, 'w'), indent=1)
    json.dump({'plan': 'reports/plans/' + pf}, open(Q + 'pending/0540a%02d_%s.json' % (j, name), 'w'))
for j, (name, runs, req) in enumerate(finals):
    pf = name + '.json'; assert not os.path.exists(pf), pf
    json.dump({'name': name, 'purpose': 'lock step 4: final-protocol test evaluation, eval seed %d' % ES, 'runs': runs}, open(pf, 'w'), indent=1)
    json.dump({'plan': 'reports/plans/' + pf, 'requires': req}, open(Q + 'waiting/0540b%03d_%s.json' % (j, name), 'w'))
for j, (name, runs) in enumerate(consts):
    pf = name + '.json'; assert not os.path.exists(pf), pf
    json.dump({'name': name, 'purpose': 'lock step 4: constant latent-offset diagnostic, final protocol', 'runs': runs}, open(pf, 'w'), indent=1)
    json.dump({'plan': 'reports/plans/' + pf}, open(Q + 'pending/0540c%02d_%s.json' % (j, name), 'w'))
json.dump(man, open(ROOT + 'temp/ms_manifest.json', 'w'), indent=1); print('written')
