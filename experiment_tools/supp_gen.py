#!/usr/bin/env python3
"""Supplementary runs queued AFTER the locked final evaluation and Allegro (post-lock; secondary / exploratory, never part of
the 9 primary tests; reported as such). Same final evaluation protocol (eval seed 4242, sealed seen/unseen, names final_*).
  S1 (05380) delta = 0.10 supplementary shared lr (lock file section 3: "0.10 supplementary, applied to all four methods"):
     the LF / PCL selection models at the d10 lr, evaluated on the same conditions as the main LF / PCL model sets.
  S4 (05381) fixed z bias 2 cm at the main shared lr (4 methods, s1-s3; identical existing chains reused) + frozen planner with
     constant execution compensation (1.5 cm, chosen earlier on post-training objects) and, for the random z bias, the constant
     compensation chosen by the compsearch_bzr3 search on post-training objects (c = 0 included).
  S2 (05382) lock-position diagnostic (post hoc, motivated by the H3b result): train the 4 methods at their main shared lr with
     the little-finger lock at 75 % of the joint range (s1-s3), evaluate LF75, LF, nominal.
  S3 (05383) learning-rate curve on the final test: remaining grid points (lr 1e-2 excluded: diverged on dev for both
     residual algorithms), LF models on LF + nominal, PCL models on pn10 + nominal.
Writes temp/supp_manifest.json. Run on the server. --dry-run prints only."""
import argparse, glob, json, os, re, sys
ROOT = '/root/autodl-tmp/DexGraspResearch/'; E = '/mnt/e/DexGraspResearch/runs/'; Q = ROOT + 'server_queue/'
sys.path.insert(0, ROOT + 'tools'); os.chdir(ROOT + 'reports/plans')
import plangen as P
ap = argparse.ArgumentParser(); ap.add_argument('--dry-run', action='store_true'); a = ap.parse_args()
ES = 4242
SEL = json.load(open(ROOT + 'temp/shared_lr_select.json')); TRIG = [tuple(x) for x in SEL['trigger_list']]
src = open(ROOT + 'tools/select_shared_lr.py').read(); ns = {}; exec(src[src.index('C = {'):src.index('TRIG =')], ns); C = ns['C']
MC = {'full_grpo': 'fg', 'full_ppo': 'fp', 'res_ppo': 'rp', 'res_grpo': 'rg'}
COND = {
 'nom': (None, None), 'LF': ('lock=LF', None), 'LF25': ('lock=LF|lockpos=0.25', None), 'LF75': ('lock=LF|lockpos=0.75', None),
 'LFpn': ('lock=LF', '0.01'), 'pn05': (None, '0.005'), 'pn10': (None, '0.01'), 'pn15': (None, '0.015'), 'pn20': (None, '0.02'),
 'bz2': ('exbias=0,0,0.02', None), 'bz2comp': ('exbias=0,0,0.02|excomp=0,0,0.015', None), 'nomcomp': ('excomp=0,0,0.015', None)}
LFSET = ['nom', 'LF', 'LF25', 'LF75', 'LFpn']; PCLSET = ['nom', 'pn10', 'pn05', 'pn15', 'pn20', 'LFpn']
plans = {}
for f in sorted(glob.glob('*.json')):
    try: q = json.load(open(f))
    except Exception: continue
    if isinstance(q, dict) and isinstance(q.get('runs'), list): plans[f] = q
names = {r['name'] for q in plans.values() for r in q['runs'] if isinstance(r, dict) and 'name' in r}
queued = {}
for d in ('pending', 'waiting', 'running', 'held'):
    for f in glob.glob(Q + d + '/*.json'):
        try: queued[json.load(open(f))['plan'].split('/')[-1]] = d + '/' + os.path.basename(f)
        except Exception: pass
def seeds(shorts, m, lr): return list(shorts) + [shorts[-1][:-1] + str(k) for k in (4, 5)] if (m, lr) in TRIG else list(shorts)
def model_of_short(short):
    for q in plans.values():
        for r in q['runs']:
            if r.get('name') == 'fmdet2k_munom_' + short:
                ck = [x for x in r['args'] if x.startswith(E)][0]; fl = r['args'][r['args'].index(ck) - 1]
                return {'tag': re.sub(r'_p\d+$', '', ck.split('/')[-2]), 'ck': ck, 'flag': fl,
                        'probe': r['probe'] if r['probe'] != 'research_shift_run.py' else r['args'][1]}
    raise SystemExit('no fmdet2k run for ' + short)
def norm(r):
    out = []; skip = False
    for x in r['args']:
        if skip: skip = False; out.append('SEED'); continue
        if x == '--seed': skip = True; out.append(x); continue
        out.append('RUN/' + x.split('/')[-1] if x.startswith(E) else x)
    return (r['probe'], tuple(out))
def sig(runs): return tuple(norm(r) for r in runs if r.get('done_file') == 'train_log.json')
sigs = {}
for fn, q in plans.items():
    s = sig(q['runs'])
    if s: sigs.setdefault(s, []).append(fn)
lrc = lambda lr: lr.replace('e-', 'em')
natkey = lambda f: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', f)]
used = P.used_seeds(); seedbase = max(used) // 1000 * 1000 + 2000; nnew = [0]
def gen_chain(m, lr, shift, pn, label, tag):
    sb = seedbase + 1000 * nnew[0]; nnew[0] += 1; short = tag.replace('_', '')
    if m == 'full_grpo': runs = P.grpo(tag, short, shift, pn, label, lr, sb)
    elif m == 'full_ppo': runs = P.ppo(tag, short, shift, pn, label, lr, sb)
    else:
        runs = P.plug(tag, short, m.split('_')[1], shift, pn, label, sb)
        for r in runs:
            if r.get('done_file') == 'train_log.json': r['args'][r['args'].index('--lr') + 1] = lr
    return [r for r in runs if r.get('done_file') == 'train_log.json']
def mdl(m, tag, fin, extra):
    res = m.startswith('res')
    d = {'tag': tag, 'ck': E + '%s/%s' % (fin, 'head.pt' if res else 'final.pt'), 'flag': '--init-head' if res else '--init-checkpoint',
         'probe': 'research_plugin_rl.py' if res else 'research_grpo_fm.py'}
    d.update(extra); return d
def ev(model, cond, split):
    spec, pn = COND[cond]; mid = model['tag'] if model else 'frozen'
    args = ['--mode', 'eval', '--object-list', 'union_ycb_unidex/test_set_%s_cat.yaml' % split, '--mus', 'none', '--episodes', '8', '--seed', str(ES),
            '--disable-gpu-jit-fusion', '--deterministic'] + (['--pcl-noise', pn] if pn else []) + ([model['flag'], model['ck']] if model else [])
    probe = model['probe'] if model else 'research_grpo_fm.py'; name = 'final_%s_%s_%s' % (split, cond, mid)
    return {'name': name, 'probe': 'research_shift_run.py', 'args': ['--shift=' + spec, probe] + args} if spec else {'name': name, 'probe': probe, 'args': args}
out, actions, problems, man = [], [], [], {'eval_seed': ES, 'S1_d10': {}, 'S2_LF75train': {}, 'S3_lrcurve': {}, 'S4_bz2': {}, 'frozen': {}}
def add_evals(model, conds, prio, req=None, suffix=''):
    runs = []
    for c in conds:
        for sp in ('seen', 'unseen'):
            r = ev(model, c, sp)
            if r['name'] in names: continue
            names.add(r['name']); runs.append(r)
    if runs: out.append(('srv_supp_%s%s' % (model['tag'] if model else 'frozen', suffix), runs, req, prio))
    return len(runs)
def chains(m, lr, shift, pn, label, cond_code, reuse):
    lst = []
    if reuse:
        runs0 = gen_chain(m, lr, shift, pn, label, 'probe_only'); nnew[0] -= 1
        for fn in [f for f in sorted(set(sigs.get(sig(runs0), [])), key=natkey) if not f.startswith('srv_topup')][:3]:
            tr = [r for r in plans[fn]['runs'] if r.get('done_file') == 'train_log.json']; fin = tr[-1]['name']
            d = mdl(m, re.sub(r'_p\d+$', '', fin), fin, {'reuse': fn})
            d['state'] = 'trained' if os.path.exists(d['ck'].replace(E, ROOT + 'runs/')) else queued.get(fn, 'NOT QUEUED')
            if d['state'] == 'NOT QUEUED' and any(os.path.isdir(ROOT + 'runs/' + r['name']) for r in tr): problems.append('partially run, not queued: ' + fn); continue
            if d['state'].startswith(('held/', 'pending/')) or d['state'] == 'NOT QUEUED': actions.append(('queue_existing', fn, d['state']))
            lst.append(d)
    for k in range(len(lst) + 1, 4):
        tag = 'sx_%s_%s_%s_s%d' % (cond_code, MC[m], lrc(lr), k)
        runs = gen_chain(m, lr, shift, pn, label, tag); fin = runs[-1]['name']
        lst.append(mdl(m, tag, fin, {'state': 'new'})); actions.append(('new_chain', 'srv_%s' % tag, runs))
    return lst
def req_of(d): return None if d.get('state') == 'trained' else ['runs/' + d['ck'].split('/runs/')[1]]
# S1: delta = 0.10
for m in C:
    lr0, lr10 = SEL['methods'][m]['d05']['shared_lr'], SEL['methods'][m]['d10']['shared_lr']
    if lr10 not in C[m]['LF'] or lr10 == lr0: man['S1_d10'][m] = {'lr': lr10, 'note': 'same as main or none'}; continue
    man['S1_d10'][m] = {'lr': lr10, 'LF': [], 'PCL': []}
    for c, conds in (('LF', LFSET), ('PCL', PCLSET)):
        for s in seeds(C[m][c][lr10], m, lr10):
            md = model_of_short(s); man['S1_d10'][m][c].append(md['tag']); add_evals(md, conds, '05380')
# S4: fixed z bias 2 cm at the main shared lr + compensation references on the frozen planner
for m in C:
    lr0 = SEL['methods'][m]['d05']['shared_lr']
    lst = chains(m, lr0, 'exbias=0,0,0.02', None, 'shbz2', 'BZ2', True); man['S4_bz2'][m] = {'lr': lr0, 'models': [d['tag'] for d in lst]}
    for d in lst: add_evals(d, ['bz2', 'nom'], '05381', req_of(d))
cs = {}
for r in plans.get('srv_compsearch_bzr3.json', {'runs': []})['runs']:
    mt = re.search(r'excomp=0,0,(-?[0-9.]+)', ' '.join(r['args'])); f = ROOT + 'runs/' + r['name'] + '/result.json'
    c = float(mt.group(1)) if mt else 0.0
    if os.path.exists(f): cs.setdefault(c, []).append(json.load(open(f))['totals']['success_rate'])
    else: cs.setdefault(c, []).append(None)
csm = {c: sum(v) / len(v) for c, v in cs.items() if v and None not in v}
if csm and len(csm) == len(cs):
    czr = max(csm, key=lambda c: (round(csm[c], 6), -abs(c)))
    COND['zr'] = ('exrand=0,0,0.03,1,%d' % ES, None)
    COND['zrcomp'] = ('exrand=0,0,0.03,1,%d|excomp=0,0,%g' % (ES, czr), None); COND['nomcompzr'] = ('excomp=0,0,%g' % czr, None)
    man['frozen']['zr_comp_c'] = czr; man['frozen']['compsearch_bzr3_means'] = csm
    fz = ['bz2', 'bz2comp', 'nomcomp'] + (['zrcomp', 'nomcompzr'] if czr != 0 else [])
else:
    problems.append('compsearch_bzr3 incomplete -> random-bias compensation evals skipped'); fz = ['bz2', 'bz2comp', 'nomcomp']
add_evals(None, fz, '05381')
# S2: lock-position diagnostic (train at 75 %)
for m in C:
    lr0 = SEL['methods'][m]['d05']['shared_lr']
    lst = chains(m, lr0, 'lock=LF|lockpos=0.75', None, 'shlockLF75', 'LF75', False); man['S2_LF75train'][m] = {'lr': lr0, 'models': [d['tag'] for d in lst]}
    for d in lst: add_evals(d, ['LF75', 'LF', 'nom'], '05382', req_of(d))
# S3: lr curve on the final test (remaining grid points, 1e-2 excluded)
for m in C:
    man['S3_lrcurve'][m] = {}
    for lr in C[m]['LF']:
        if lr == '1e-2': continue
        man['S3_lrcurve'][m][lr] = {}
        for c, conds in (('LF', ['LF', 'nom']), ('PCL', ['pn10', 'nom'])):
            tags = []
            for s in seeds(C[m][c][lr], m, lr):
                md = model_of_short(s); tags.append(md['tag']); add_evals(md, conds, '05383', suffix='_lrc')
            man['S3_lrcurve'][m][lr][c] = tags
nt = sum(1 for x in actions if x[0] == 'new_chain')
for x in actions: print('ACTION', x[0], x[1], x[2] if isinstance(x[2], str) else '%d train runs' % len(x[2]))
for p in ('05380', '05381', '05382', '05383'):
    print(p, 'eval plans', sum(1 for o in out if o[3] == p), 'evals', sum(len(o[1]) for o in out if o[3] == p))
print('new chains', nt, '| frozen extra', fz, '| zr comp', man['frozen'].get('zr_comp_c'), man['frozen'].get('compsearch_bzr3_means'))
print('PROBLEMS:', problems or 'none')
if a.dry_run: print('dry-run'); raise SystemExit(0)
idx = 0
for x in actions:
    if x[0] == 'new_chain':
        pf = x[1] + '.json'; assert not os.path.exists(pf), pf
        pr = '05381' if '_BZ2_' in x[1] else '05382'
        json.dump({'name': x[1], 'purpose': 'supplementary (post-lock) training', 'runs': x[2]}, open(pf, 'w'), indent=1)
        json.dump({'plan': 'reports/plans/' + pf}, open(Q + 'pending/%sa%02d_%s.json' % (pr, idx, x[1]), 'w')); idx += 1
    elif x[2].startswith(('held/', 'pending/')):
        os.rename(Q + x[2], Q + 'pending/05381a%02d_%s' % (idx, x[2].split('/')[-1].split('_', 1)[1])); idx += 1
    elif x[2] == 'NOT QUEUED':
        json.dump({'plan': 'reports/plans/' + x[1]}, open(Q + 'pending/05381a%02d_%s' % (idx, x[1]), 'w')); idx += 1
for j, (name, runs, req, pr) in enumerate(out):
    pf = name + '.json'; assert not os.path.exists(pf), pf
    json.dump({'name': name, 'purpose': 'supplementary (post-lock) final-protocol evaluation, eval seed %d' % ES, 'runs': runs}, open(pf, 'w'), indent=1)
    item = {'plan': 'reports/plans/' + pf}
    if req: item['requires'] = req
    json.dump(item, open(Q + ('waiting' if req else 'pending') + '/%sb%03d_%s.json' % (pr, j, name), 'w'))
json.dump(man, open(ROOT + 'temp/supp_manifest.json', 'w'), indent=1); print('written')
