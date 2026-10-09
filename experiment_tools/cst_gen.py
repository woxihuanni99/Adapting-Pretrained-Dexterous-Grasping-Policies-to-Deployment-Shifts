#!/usr/bin/env python3
"""Lock step 5 (2026-10-02; lock document section 12): directly optimised state-independent constant mean (exploratory
baseline, not part of the nine tests). Run on the server in the repository root. --dry-run prints only.
Model: research_plugin_rl_cst.py --const-mean: eps' ~ N(c, diag(e^{2 rho})), u = m0(s) + sigma0 * eps', c zero-initialised,
rho learnable; Adam groups: c at --lr in {3e-3, 1e-2, 3e-2}, rho (and all other head parameters) at --lr-rho 1e-3 (the main
residual GRPO setting). Conditions: reference little-finger lock (lock=LF) and ring+little lock (lock=RF,LF); 3 seeds each.
Training runs are cloned from the existing residual GRPO chains of the same condition (srv_pluggrpo_shlockLF_lr1e3_s{k},
srv_sx_RFLF_rg_1em3_s{k}): identical budget (12 phases), group size 8, epochs 5, minibatches 4, gradient clip, phases;
only the script, --lr, --const-mean --lr-rho 1e-3 and fresh training seeds change. Per chain: development evals (fault 512,
fmdet_munom_<label>_<short>_64x8; nominal 2048, fmdet2k_munom_<short>; seed 777) and final-protocol test evals for ALL 18
models (seed 4242; seen/unseen; own fault condition + nominal). Writes temp/cst_manifest.json."""
import argparse, copy, json, os, sys
ROOT = '/root/autodl-tmp/DexGraspResearch/'; E = '/mnt/e/DexGraspResearch/runs/'; Q = ROOT + 'server_queue/'
sys.path.insert(0, ROOT + 'tools'); os.chdir(ROOT + 'reports/plans')
import plangen as P
ap = argparse.ArgumentParser(); ap.add_argument('--dry-run', action='store_true'); a = ap.parse_args()
ES = 4242; OLD, NEW = 'research_plugin_rl.py', 'research_plugin_rl_cst.py'
LRS = (('3e-3', '3em3'), ('1e-2', '1em2'), ('3e-2', '3em2'))
COND = {'LF': ('lock=LF', 'shlockLF', 'LF', 'srv_pluggrpo_shlockLF_lr1e3_s%d.json', 'pluggrpo_shlockLF_lr1e3_s%d'),
        'RFLF': ('lock=RF,LF', 'shlockRFLF', 'RFLF', 'srv_sx_RFLF_rg_1em3_s%d.json', 'sx_RFLF_rg_1em3_s%d')}
used = P.used_seeds(); seedbase = max(used) // 1000 * 1000 + 2000
def ev_args(ol, eps, seed, ck):
    return [NEW, '--mode', 'eval', '--object-list', ol, '--mus', 'none', '--episodes', str(eps), '--seed', str(seed),
            '--disable-gpu-jit-fusion', '--deterministic', '--init-head', ck, '--const-mean']
def run(name, spec, args):
    return {'name': name, 'probe': 'research_shift_run.py', 'args': ['--shift=' + spec] + args} if spec else {'name': name, 'probe': args[0], 'args': args[1:]}
chains, finals, man, n = [], [], {'eval_seed': ES, 'lr_rho': '1e-3', 'cst': {}, 'refs': {}}, 0
for c, (spec, label, cname, srcpat, oldpat) in COND.items():
    man['refs'][c] = {'cond': cname, 'label': label}
    for lr, lrs in LRS:
        for k in (1, 2, 3):
            src, old = srcpat % k, oldpat % k
            tr = [copy.deepcopy(r) for r in json.load(open(src))['runs'] if isinstance(r, dict) and r.get('done_file') == 'train_log.json']
            assert len(tr) == 12, src
            tag = 'cst_%s_%s_s%d' % (c, lrs, k); short = tag.replace('_', '')
            for i, r in enumerate(tr):
                assert r['name'] == '%s_p%02d' % (old, i), r['name']
                r['name'] = '%s_p%02d' % (tag, i)
                r['args'] = [x.replace('/runs/' + old + '_', '/runs/' + tag + '_') for x in r['args']]
                if r['probe'] == OLD: r['probe'] = NEW
                r['args'] = [NEW if x == OLD else x for x in r['args']]
                j = r['args'].index('--lr'); assert r['args'][j + 1] == '1e-3'; r['args'][j + 1] = lr
                assert r['args'][r['args'].index('--algo') + 1] == 'grpo'
                r['args'] += ['--const-mean', '--lr-rho', '1e-3']
                v = seedbase + 1000 * n + i; assert v not in used, v; used.add(v); r['args'][r['args'].index('--seed') + 1] = str(v)
                assert not any(old in x for x in r['args']) and NEW in [r['probe']] + r['args']
            ck = E + '%s_p11/head.pt' % tag
            dev = [run('fmdet_munom_%s_%s_64x8' % (label, short), spec, ev_args('union_ycb_unidex/research_dfg_dev64_x2.yaml', 4, 777, ck)),
                   run('fmdet2k_munom_%s' % short, None, ev_args('union_ycb_unidex/research_dfg_dev64_x2.yaml', 16, 777, ck))]
            fin = [run('final_%s_%s_%s' % (sp, cn, tag), sp_, ev_args('union_ycb_unidex/test_set_%s_cat.yaml' % sp, 8, ES, ck))
                   for (sp_, cn) in ((spec, cname), (None, 'nom')) for sp in ('seen', 'unseen')]
            chains.append(('srv_%s' % tag, tr + dev)); finals.append(('srv_cstfinal_%s' % tag, fin, ['runs/%s_p11/head.pt' % tag])); n += 1
            man['cst'].setdefault('%s|%s' % (c, lr), []).append({'tag': tag, 'short': short})
man['refs']['LF'].update({'residual': ['pluggrpo_shlockLF_lr1e3_s%d' % k for k in (1, 2, 3)],
                          'posthoc_const': ['const_pluggrpo_shlockLF_lr1e3_s%d' % k for k in (1, 2, 3)]})
man['refs']['RFLF'].update({'residual': ['sx_RFLF_rg_1em3_s%d' % k for k in (1, 2, 3, 4, 5)],
                            'posthoc_const': ['const_sx_RFLF_rg_1em3_s%d' % k for k in (1, 2, 3, 4, 5)]})
print('chains', len(chains), '| e.g.', chains[0][0], ' '.join([chains[0][1][0]['probe']] + chains[0][1][0]['args'])[:360])
print('   last train', ' '.join(chains[0][1][11]['args'])[-260:]); print('   dev', [r['name'] for r in chains[0][1][12:]], ' '.join(chains[0][1][12]['args'])[-200:])
print('   RFLF e.g.', chains[9][0], ' '.join(chains[9][1][0]['args'])[:200])
print('   final', [r['name'] for r in finals[0][1]], '|', ' '.join([finals[0][1][0]['probe']] + finals[0][1][0]['args'])[:260])
print('evals: dev', 2 * len(chains), 'final', 4 * len(finals), '| seeds from', seedbase)
if a.dry_run: print('dry-run'); raise SystemExit(0)
for j, (name, runs) in enumerate(chains):
    pf = name + '.json'; assert not os.path.exists(pf), pf
    json.dump({'name': name, 'purpose': 'lock step 5: directly optimised state-independent constant mean (exploratory baseline)', 'runs': runs}, open(pf, 'w'), indent=1)
    json.dump({'plan': 'reports/plans/' + pf}, open(Q + 'pending/0541a%02d_%s.json' % (j, name), 'w'))
for j, (name, runs, req) in enumerate(finals):
    pf = name + '.json'; assert not os.path.exists(pf), pf
    json.dump({'name': name, 'purpose': 'lock step 5: final-protocol test evaluation, eval seed %d' % ES, 'runs': runs}, open(pf, 'w'), indent=1)
    json.dump({'plan': 'reports/plans/' + pf, 'requires': req}, open(Q + 'waiting/0541b%03d_%s.json' % (j, name), 'w'))
json.dump(man, open(ROOT + 'temp/cst_manifest.json', 'w'), indent=1); print('written')
