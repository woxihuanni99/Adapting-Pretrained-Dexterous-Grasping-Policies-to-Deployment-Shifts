#!/usr/bin/env python3
"""Supplementary KL-retention baseline (lock step 3, 2026-10-02; see claude/DexGrasp_锁定说明_草稿.md section 10).
Full-weight GRPO + beta * KL(pi_theta || pi_0) to the FROZEN original planner (research_grpo_finetune_v2_kl.py), same
budget/phases/hyper-parameters as the main full-weight GRPO chains except --kl-coef.
Grid (fixed before training): selection conditions LF (lock=LF) and PCL (pcl-noise 0.01) x lr {3e-5, 1e-4} x
beta {bA, bB} x seeds s1-s3 = 24 chains. beta = {0.1, 1} / KLbar_res, KLbar_res = mean over the six main residual GRPO
models (LF and PCL, lr 1e-3, s1-s3) of the mean nominal-state KL of their mean shift under the same fixed-std metric
(runs/diag_delta_phase00b/delta_diag.json, kl_floor_mean; research_delta_diag.py). Per chain: dev evals for selection
(fault 512 episodes fmdet_munom_<label>_<short>_64x8, nominal 2048 episodes fmdet2k_munom_<short>, seed 777) and
final-protocol test evals (seed 4242, seen/unseen, fault condition + nominal). The thumb-lock transfer of the selected
config is generated later by tools/kl_select.py. Writes temp/kl_manifest.json. --dry-run prints only."""
import argparse, json, os, sys
ROOT = '/root/autodl-tmp/DexGraspResearch/'; E = '/mnt/e/DexGraspResearch/runs/'; Q = ROOT + 'server_queue/'
sys.path.insert(0, ROOT + 'tools'); os.chdir(ROOT + 'reports/plans')
import plangen as P
ap = argparse.ArgumentParser(); ap.add_argument('--dry-run', action='store_true'); a = ap.parse_args()
ES = 4242
D = json.load(open(ROOT + 'runs/diag_delta_phase00b/delta_diag.json'))
anchor = ['rgLF_1', 'rgLF_2', 'rgLF_3', 'rgPCL_1', 'rgPCL_2', 'rgPCL_3']
klbar = sum(D['models'][k]['kl_floor_mean'] for k in anchor) / len(anchor)
def r3(x): return float('%.3g' % x)
BETA = {'bA': r3(0.1 / klbar), 'bB': r3(1.0 / klbar)}
COND = {'LF': ('lock=LF', None, 'shlockLF', 'LF'), 'PCL': (None, '0.01', 'pn10', 'pn10')}
OLD, NEW = 'research_grpo_finetune_v2.py', 'research_grpo_finetune_v2_kl.py'
used = P.used_seeds(); seedbase = max(used) // 1000 * 1000 + 2000
def dev_evals(shift, pn, label, short, ck):
    base = lambda eps, p: ['--mode', 'eval', '--object-list', 'union_ycb_unidex/research_dfg_dev64_x2.yaml', '--mus', 'none', '--episodes', str(eps),
                           '--seed', '777', '--disable-gpu-jit-fusion', '--deterministic'] + (['--pcl-noise', pn] if (p and pn) else []) + ['--init-checkpoint', ck]
    pr, aa = P.wrap(shift, 'research_grpo_fm.py', base(4, True))
    return [{'name': 'fmdet_munom_%s_%s_64x8' % (label, short), 'probe': pr, 'args': aa},
            {'name': 'fmdet2k_munom_%s' % short, 'probe': 'research_grpo_fm.py', 'args': base(16, False)}]
def final_eval(spec, pn, cname, split, tag, ck):
    args = ['--mode', 'eval', '--object-list', 'union_ycb_unidex/test_set_%s_cat.yaml' % split, '--mus', 'none', '--episodes', '8', '--seed', str(ES),
            '--disable-gpu-jit-fusion', '--deterministic'] + (['--pcl-noise', pn] if pn else []) + ['--init-checkpoint', ck]
    name = 'final_%s_%s_%s' % (split, cname, tag)
    return {'name': name, 'probe': 'research_shift_run.py', 'args': ['--shift=' + spec, 'research_grpo_fm.py'] + args} if spec else {'name': name, 'probe': 'research_grpo_fm.py', 'args': args}
chains, finals, man, n = [], [], {'eval_seed': ES, 'klbar_res': klbar, 'anchor_models': {k: D['models'][k]['path'] for k in anchor},
                                  'diag_sha256': D['script_sha256'], 'beta': BETA, 'configs': {}}, 0
for c, (shift, pn, label, cname) in COND.items():
    for lr in ('3e-5', '1e-4'):
        for bi, beta in BETA.items():
            for k in (1, 2, 3):
                tag = 'kl_%s_%s_%s_s%d' % (c, lr.replace('e-', 'em'), bi, k); short = tag.replace('_', ''); sb = seedbase + 1000 * n; n += 1
                tr = [r for r in P.grpo(tag, short, shift, pn, label, lr, sb) if r.get('done_file') == 'train_log.json']
                for r in tr:
                    if r['probe'] == OLD: r['probe'] = NEW
                    r['args'] = [NEW if x == OLD else x for x in r['args']] + ['--kl-coef', str(beta)]
                    assert NEW in [r['probe']] + r['args'] and OLD not in [r['probe']] + r['args']
                ck = E + '%s_p11/final.pt' % tag
                chains.append(('srv_%s' % tag, tr + dev_evals(shift, pn, label, short, ck)))
                finals.append(('srv_klfinal_%s' % tag, [final_eval(sp_, pn_, cn_, s, tag, ck) for (sp_, pn_, cn_) in ((shift, pn, cname), (None, None, 'nom')) for s in ('seen', 'unseen')],
                               ['runs/%s_p11/final.pt' % tag]))
                man['configs'].setdefault('%s|%s|%s' % (c, lr, bi), []).append({'tag': tag, 'short': short})
for name, runs in chains[:1] + chains[-1:]:
    tr = [r for r in runs if r.get('done_file') == 'train_log.json']
    print(name, len(tr), 'train', len(runs) - len(tr), 'dev evals |', ' '.join([tr[0]['probe']] + tr[0]['args'])[:330])
    print('   last train:', ' '.join(tr[-1]['args'])[-160:]); print('   evals:', [r['name'] for r in runs if 'done_file' not in r])
print('chains', len(chains), 'final eval plans', len(finals), 'final evals', sum(len(f[1]) for f in finals), '| klbar_res %.4f beta %s' % (klbar, BETA))
if a.dry_run: print('dry-run'); raise SystemExit(0)
for j, (name, runs) in enumerate(chains):
    pf = name + '.json'; assert not os.path.exists(pf), pf
    json.dump({'name': name, 'purpose': 'supplementary KL-retention baseline (lock step 3)', 'runs': runs}, open(pf, 'w'), indent=1)
    json.dump({'plan': 'reports/plans/' + pf}, open(Q + 'pending/0539a%02d_%s.json' % (j, name), 'w'))
for j, (name, runs, req) in enumerate(finals):
    pf = name + '.json'; assert not os.path.exists(pf), pf
    json.dump({'name': name, 'purpose': 'KL-retention baseline final-protocol test evaluation, eval seed %d' % ES, 'runs': runs}, open(pf, 'w'), indent=1)
    json.dump({'plan': 'reports/plans/' + pf, 'requires': req}, open(Q + 'waiting/0539b%03d_%s.json' % (j, name), 'w'))
json.dump(man, open(ROOT + 'temp/kl_manifest.json', 'w'), indent=1); print('written')
