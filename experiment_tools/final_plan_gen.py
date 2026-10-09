#!/usr/bin/env python3
"""Lock step 2 (protocol 2026-09-29): from temp/shared_lr_select.json build
  (1) the final model list per method (selected shared lr, d05 main): LF & PCL models = the selection runs (s1-s3, or s1-s5 if
      triggered); transfer models (TH lock, random z U(-3,3)cm, z2cm+1cm noise, TH+1cm noise) = seeds s1-s3 at the same lr,
      reusing an existing chain only if its training runs are identical to what would be generated (same script, shift, noise,
      lr and all other arguments except seeds/paths; first three matches by plan name), otherwise new chains (new unused seeds);
      H5 models = LF selection runs at the per-method unconstrained LF dev-best lr (switch_LF.unconstrained_dev_best);
  (2) final evaluation plans: NEW eval seed ES on the sealed seen/unseen test sets, conditions per model set as in the lock file,
      frozen planner on every condition;
  (3) temp/final_manifest.json (method -> lr, model tags per set; eval run names) for tools/final_stats.py.
Run on the server. --dry-run prints only."""
import argparse, glob, json, os, re, sys
ROOT = '/root/autodl-tmp/DexGraspResearch/'; E = '/mnt/e/DexGraspResearch/runs/'; Q = ROOT + 'server_queue/'
sys.path.insert(0, ROOT + 'tools'); os.chdir(ROOT + 'reports/plans')
import plangen as P
ap = argparse.ArgumentParser(); ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--select', default=ROOT + 'temp/shared_lr_select.json')
a = ap.parse_args()
ES = 4242   # final evaluation seed (dev/test used 777); also the exrand draw seed in final evaluations
SEL = json.load(open(a.select)); TRIG = [tuple(x) for x in SEL['trigger_list']]
src = open(ROOT + 'tools/select_shared_lr.py').read(); ns = {}; exec(src[src.index('C = {'):src.index('TRIG =')], ns); C = ns['C']
MC = {'full_grpo': 'fg', 'full_ppo': 'fp', 'res_ppo': 'rp', 'res_grpo': 'rg'}
COND = {  # name -> (shift spec or None, pcl noise or None)
 'nom': (None, None), 'LF': ('lock=LF', None), 'LF25': ('lock=LF|lockpos=0.25', None), 'LF75': ('lock=LF|lockpos=0.75', None),
 'LFpn': ('lock=LF', '0.01'), 'pn05': (None, '0.005'), 'pn10': (None, '0.01'), 'pn15': (None, '0.015'), 'pn20': (None, '0.02'),
 'TH': ('lock=TH', None), 'THzr': ('lock=TH|exrand=0,0,0.03,1,%d' % ES, None), 'zr': ('exrand=0,0,0.03,1,%d' % ES, None),
 'zp1': ('exbias=0,0,0.01', None), 'zp3': ('exbias=0,0,0.03', None), 'zm2': ('exbias=0,0,-0.02', None),
 'x2': ('exbias=0.02,0,0', None), 'y2': ('exbias=0,0.02,0', None), 'zp4': ('exbias=0,0,0.04', None), 'zm4': ('exbias=0,0,-0.04', None),
 'bz2pn': ('exbias=0,0,0.02', '0.01'), 'THpn': ('lock=TH', '0.01')}
SETS = {  # model set -> evaluation conditions (lock file sections 5, 6, 6b)
 'LF': ['nom', 'LF', 'LF25', 'LF75', 'LFpn'], 'PCL': ['nom', 'pn10', 'pn05', 'pn15', 'pn20', 'LFpn'],
 'TH': ['nom', 'TH', 'THzr'], 'ZR': ['nom', 'zr', 'zp1', 'zp3', 'zm2', 'x2', 'y2', 'zp4', 'zm4', 'THzr'],
 'BZ2PN': ['nom', 'bz2pn'], 'THPN': ['nom', 'THpn'], 'LF_H5': ['LF']}
XFER = {'TH': ('lock=TH', None, 'shlockTH'), 'ZR': ('exrand=0,0,0.03,8', None, 'shbzr3'),
        'BZ2PN': ('exbias=0,0,0.02', '0.01', 'shbz2pn10'), 'THPN': ('lock=TH', '0.01', 'shlockTHpn10')}
plans = {}
for f in sorted(glob.glob('*.json')):
    try: q = json.load(open(f))
    except Exception: continue
    if isinstance(q, dict) and isinstance(q.get('runs'), list): plans[f] = q
queued = {}
for d in ('pending', 'waiting', 'running', 'held'):
    for f in glob.glob(Q + d + '/*.json'):
        try: queued[json.load(open(f))['plan'].split('/')[-1]] = d + '/' + os.path.basename(f)
        except Exception: pass
def seeds(shorts, m, lr): return list(shorts) + [shorts[-1][:-1] + str(k) for k in (4, 5)] if (m, lr) in TRIG else list(shorts)
def model_of_short(short):
    for q in plans.values():
        for r in q['runs']:
            if r['name'] == 'fmdet2k_munom_' + short:
                ck = [x for x in r['args'] if x.startswith(E)][0]; fl = r['args'][r['args'].index(ck) - 1]
                return {'tag': re.sub(r'_p\d+$', '', ck.split('/')[-2]), 'short': short, 'ck': ck, 'flag': fl,
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
def gen_chain(m, lr, cond, k):
    shift, pn, label = XFER[cond]; tag = 'xf_%s_%s_%s_s%d' % (cond, MC[m], lrc(lr), k); short = 'xf%s%s%ss%d' % (cond, MC[m], lrc(lr), k)
    sb = seedbase + 1000 * nnew[0]; nnew[0] += 1
    if m == 'full_grpo': runs = P.grpo(tag, short, shift, pn, label, lr, sb)
    elif m == 'full_ppo': runs = P.ppo(tag, short, shift, pn, label, lr, sb)
    else:
        runs = P.plug(tag, short, m.split('_')[1], shift, pn, label, sb)
        for r in runs:
            if r.get('done_file') == 'train_log.json': r['args'][r['args'].index('--lr') + 1] = lr
    for r in runs:
        if r.get('done_file') != 'train_log.json': r['args'] = [x.replace('exrand=0,0,0.03,8', 'exrand=0,0,0.03,1') for x in r['args']]
    return tag, runs
def mdl(m, tag, fin, extra):
    res = m.startswith('res')
    d = {'tag': tag, 'ck': E + '%s/%s' % (fin, 'head.pt' if res else 'final.pt'), 'flag': '--init-head' if res else '--init-checkpoint',
         'probe': 'research_plugin_rl.py' if res else 'research_grpo_fm.py'}
    d.update(extra); return d
actions, problems = [], []
manifest = {'eval_seed': ES, 'selection': a.select, 'methods': {}, 'evals': {}, 'frozen': {}}
for m in C:
    res = SEL['methods'][m]; lr = res['d05']['shared_lr']
    mm = manifest['methods'][m] = {'lr_main': lr, 'lr_H5': res['switch_LF']['unconstrained_dev_best'], 'sets': {}}
    h5 = mm['lr_H5']; mm['sets']['LF_H5'] = [model_of_short(s) for s in seeds(C[m]['LF'][h5], m, h5)] if h5 != lr else []
    if lr not in C[m]['LF']:
        problems.append('%s: no feasible shared lr (%s) -> reported, no main-protocol models (H5 models only)' % (m, lr)); continue
    for c in ('LF', 'PCL'):
        mm['sets'][c] = [model_of_short(s) for s in seeds(C[m][c][lr], m, lr)]
    for cond in XFER:
        lst = []
        _, runs0 = gen_chain(m, lr, cond, 0)
        match = sorted(set(sigs.get(sig(runs0), [])), key=natkey)
        match = [fn for fn in match if not fn.startswith('srv_topup')][:3]
        for fn in match:
            tr = [r for r in plans[fn]['runs'] if r.get('done_file') == 'train_log.json']
            fin = tr[-1]['name']; d = mdl(m, re.sub(r'_p\d+$', '', fin), fin, {'reuse': fn})
            d['state'] = 'trained' if os.path.exists(d['ck'].replace(E, ROOT + 'runs/')) else queued.get(fn, 'NOT QUEUED')
            lst.append(d)
            if d['state'].startswith('held/') or d['state'] == 'NOT QUEUED' or (d['state'].startswith('pending/') and not d['state'].startswith('pending/0536')):
                actions.append(('queue_existing', fn, d['state']))
        for k in range(len(lst) + 1, 4):
            tag, runs = gen_chain(m, lr, cond, k)
            fin = [r for r in runs if r.get('done_file') == 'train_log.json'][-1]['name']
            lst.append(mdl(m, tag, fin, {'state': 'new'})); actions.append(('new_chain', 'srv_%s' % tag, runs))
        mm['sets'][cond] = lst
def ev(model, cond, split):
    spec, pn = COND[cond]; mid = model['tag'] if model else 'frozen'
    args = ['--mode', 'eval', '--object-list', 'union_ycb_unidex/test_set_%s_cat.yaml' % split, '--mus', 'none', '--episodes', '8', '--seed', str(ES),
            '--disable-gpu-jit-fusion', '--deterministic'] + (['--pcl-noise', pn] if pn else []) + ([model['flag'], model['ck']] if model else [])
    probe = model['probe'] if model else 'research_grpo_fm.py'; name = 'final_%s_%s_%s' % (split, cond, mid)
    r = {'name': name, 'probe': 'research_shift_run.py', 'args': ['--shift=' + spec, probe] + args} if spec else {'name': name, 'probe': probe, 'args': args}
    return r
evplans, seen_names = [], set()
for m, mm in manifest['methods'].items():
    for setname, models in mm['sets'].items():
        for md in models:
            runs = [ev(md, c, sp) for c in SETS[setname] for sp in ('seen', 'unseen')]
            runs = [r for r in runs if r['name'] not in seen_names]; seen_names |= {r['name'] for r in runs}
            if runs:
                req = None if md.get('state') in (None, 'trained') else ['runs/' + md['ck'].split('/runs/')[1]]
                evplans.append(('srv_final_' + md['tag'] + ('_h5' if setname == 'LF_H5' else ''), runs, req, setname))
allc = sorted({c for s in SETS.values() for c in s})
for i in range(0, len(allc), 3):
    evplans.append(('srv_final_frozen_%d' % (i // 3), [ev(None, c, sp) for c in allc[i:i + 3] for sp in ('seen', 'unseen')], None, 'frozen'))
for p in evplans:
    for r in p[1]:
        split, cond, mid = r['name'].split('_', 3)[1:]
        (manifest['frozen'] if mid == 'frozen' else manifest['evals'].setdefault(mid, {})).setdefault(cond, {})[split] = r['name']
print('selected (main d05, H5):', {m: (mm['lr_main'], mm['lr_H5']) for m, mm in manifest['methods'].items()})
for m, mm in manifest['methods'].items():
    for s, l in mm['sets'].items(): print(' ', m, s, [(x['tag'], x.get('state', 'sel')) for x in l])
for x in actions: print('ACTION', x[0], x[1], x[2] if isinstance(x[2], str) else '%d runs' % len(x[2]))
print('eval plans', len(evplans), 'eval runs', sum(len(p[1]) for p in evplans), '| new chains', sum(1 for x in actions if x[0] == 'new_chain'))
print('PROBLEMS:', problems or 'none')
if a.dry_run:
    json.dump(manifest, open(ROOT + 'temp/final_manifest.dryrun.json', 'w'), indent=1); print('dry-run: nothing queued'); raise SystemExit(0)
idx = 0
for x in actions:
    if x[0] == 'new_chain':
        pf = x[1] + '.json'; assert not os.path.exists(pf), pf
        json.dump({'name': x[1], 'purpose': 'final: transfer training at the selected shared lr', 'runs': x[2]}, open(pf, 'w'), indent=1)
        json.dump({'plan': 'reports/plans/' + pf}, open(Q + 'pending/0536%02d_%s.json' % (idx, x[1]), 'w')); idx += 1
    elif x[2].startswith('held/') or x[2].startswith('pending/'):
        os.rename(Q + x[2], Q + 'pending/0536%02d_%s' % (idx, x[2].split('/')[-1].split('_', 1)[1])); idx += 1
    elif x[2] == 'NOT QUEUED':
        json.dump({'plan': 'reports/plans/' + x[1]}, open(Q + 'pending/0536%02d_%s' % (idx, x[1]), 'w')); idx += 1
pri = {'frozen': '05350', 'LF': '05351', 'PCL': '05352', 'LF_H5': '05353', 'TH': '05354', 'ZR': '05355', 'BZ2PN': '05356', 'THPN': '05357'}
for j, (name, runs, req, setname) in enumerate(evplans):
    pf = name + '.json'; assert not os.path.exists(pf), pf
    json.dump({'name': name, 'purpose': 'final evaluation (lock step 2), eval seed %d' % ES, 'runs': runs}, open(pf, 'w'), indent=1)
    item = {'plan': 'reports/plans/' + pf}
    if req: item['requires'] = req
    json.dump(item, open(Q + ('waiting' if req else 'pending') + '/%s%03d_%s.json' % (pri[setname], j, name), 'w'))
json.dump(manifest, open(ROOT + 'temp/final_manifest.json', 'w'), indent=1)
print('written; manifest temp/final_manifest.json')
