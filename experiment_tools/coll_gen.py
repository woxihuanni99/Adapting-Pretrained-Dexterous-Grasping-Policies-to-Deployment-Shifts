#!/usr/bin/env python3
"""Full robot-table collision re-evaluation and reproduction check (2026-10-07; plan: reports/补充实验锁定说明_设置匹配对照_1007.md).
Evaluation only, existing locked models, eval seed 4242, same object lists. Each run is a CLONE of the archived final-protocol
run spec (found by result name in reports/plans/*.json); only the script is swapped for its flagged copy and renamed:
  research_plugin_rl.py -> research_plugin_rl_fs.py ; research_grpo_fm.py -> research_grpo_fm_coll.py (copies, flags off by default)
--stage repro : repro_<name>: copies WITHOUT flags (original setting) for the frozen planner (LF, nom) and LF s1 of residual GRPO and
                full-weight GRPO (seen/unseen), plus reproorig_final_seen_LF_frozen with the ORIGINAL script. Compared with the
                archived runs (per-env success flags and initial_state.npz) by coll_stats.py --repro.
--stage coll  : coll_<name>: copies WITH --robot-table-collision for the frozen planner (LF, pn10, nom) and all main LF and PCL
                models of the four schemes (own fault condition + nom).
--stage ref   : collref_<name>: same set as coll, copies WITHOUT flags (only if the reproduction check fails).
--dry-run prints only."""
import argparse, copy, glob, json, os, sys
ROOT = '/root/autodl-tmp/DexGraspResearch/'; Q = ROOT + 'server_queue/'
os.chdir(ROOT + 'reports/plans')
ap = argparse.ArgumentParser(); ap.add_argument('--stage', choices=('repro', 'coll', 'ref'), required=True); ap.add_argument('--dry-run', action='store_true')
a = ap.parse_args()
SWAP = {'research_plugin_rl.py': 'research_plugin_rl_fs.py', 'research_grpo_fm.py': 'research_grpo_fm_coll.py'}
idx = {}
for f in sorted(glob.glob('*.json')):
    if f.endswith('_ledger_srv.json'): continue
    try: q = json.load(open(f))
    except Exception: continue
    if not isinstance(q, dict) or not isinstance(q.get('runs'), list): continue
    for r in q['runs']:
        if isinstance(r, dict) and str(r.get('name', '')).startswith('final_'): idx.setdefault(r['name'], []).append((f, r))
man = json.load(open(ROOT + 'temp/final_manifest.json'))
def spec(name):
    c = idx.get(name); assert c, 'no archived spec for ' + name
    specs = {json.dumps(r, sort_keys=True) for _, r in c}; assert len(specs) == 1, ('ambiguous spec', name, [f for f, _ in c])
    assert os.path.exists(ROOT + 'runs/' + name + '/result.json'), 'archived result missing: ' + name
    return copy.deepcopy(c[0][1])
def clone(name, prefix, flag, swap=True):
    r = spec(name); r['name'] = prefix + name
    if swap:
        r['probe'] = SWAP.get(r['probe'], r['probe']); r['args'] = [SWAP.get(x, x) for x in r['args']]
        assert any(x in SWAP.values() for x in [r['probe']] + r['args']), name
    if flag: r['args'] = r['args'] + ['--robot-table-collision']
    return r
def names_for_set(cset, cond):
    out = []
    for m in ('res_ppo', 'res_grpo', 'full_ppo', 'full_grpo'):
        for e in man['methods'][m]['sets'][cset]:
            ev = man['evals'][e['tag']]
            for c in (cond, 'nom'):
                out += [ev[c]['seen'], ev[c]['unseen']]
    return out
runs = []
if a.stage == 'repro':
    tag_r = man['methods']['res_grpo']['sets']['LF'][0]['tag']; tag_f = man['methods']['full_grpo']['sets']['LF'][0]['tag']
    names = [man['frozen'][c][s] for c in ('LF', 'nom') for s in ('seen', 'unseen')]
    names += [man['evals'][t][c][s] for t in (tag_r, tag_f) for c in ('LF', 'nom') for s in ('seen', 'unseen')]
    runs = [clone(nm, 'repro_', False) for nm in names] + [clone(man['frozen']['LF']['seen'], 'reproorig_', False, swap=False)]
    plan = 'srv_coll_repro'; prio = 'pending/0544a00_'
else:
    names = [man['frozen'][c][s] for c in ('LF', 'pn10', 'nom') for s in ('seen', 'unseen')]
    names += names_for_set('LF', 'LF') + names_for_set('PCL', 'pn10')
    names = list(dict.fromkeys(names))
    pre = 'coll_' if a.stage == 'coll' else 'collref_'
    runs = [clone(nm, pre, a.stage == 'coll') for nm in names]
    plan = 'srv_coll_' + a.stage; prio = 'pending/0546a00_' if a.stage == 'coll' else 'pending/0546b00_'  # after the 0545 training chains
print(a.stage, 'runs', len(runs)); 
for r in runs[:3] + runs[-2:]: print('  ', r['name'], '|', ' '.join([r['probe']] + r['args'])[:330])
if a.dry_run: print('dry-run'); raise SystemExit(0)
# split into 4 plans so up to 4 can run in parallel
k = 4 if len(runs) > 8 else 1
for i in range(k):
    part = runs[i::k]; nm = plan + ('_%d' % i if k > 1 else '')
    json.dump({'name': nm, 'purpose': 'collision re-evaluation / reproduction check (2026-10-07), stage ' + a.stage, 'runs': part}, open(nm + '.json', 'x'), indent=1)
    json.dump({'plan': 'reports/plans/' + nm + '.json'}, open(Q + prio.replace('00_', '%02d_' % i) + nm + '.json', 'x'))
print('written')
