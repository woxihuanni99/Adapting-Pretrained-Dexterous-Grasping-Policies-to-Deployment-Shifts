"""Plan generator for DemoGrasp post-training experiments (full-weight GRPO / PPO, residual heads, eval-only).
All plans follow the conventions of existing srv_* plans (same probes, eval protocol fmdet/testdet, seed 777)."""
import json, glob
E = '/mnt/e/DexGraspResearch/runs/'
_used = None
def used_seeds():
    global _used
    if _used is None:
        _used = set()
        for f in glob.glob('*.json'):
            try: q = json.load(open(f))
            except Exception: continue
            if not isinstance(q, dict) or not isinstance(q.get('runs'), list): continue
            for r in q['runs']:
                if isinstance(r, dict) and '--seed' in r.get('args', []):
                    try: _used.add(int(r['args'][r['args'].index('--seed') + 1]))
                    except Exception: pass
    return _used
GTR = [r for r in json.load(open('srv_grpo_shlockLF_lr3s1.json'))['runs'] if r.get('done_file') == 'train_log.json']
PTR = [r for r in json.load(open('srv_ppo_shlockLF_s1.json'))['runs'] if r.get('done_file') == 'train_log.json']
UTR = [r for r in json.load(open('srv_plugppo_shlockLF_s1.json'))['runs'] if r.get('done_file') == 'train_log.json']
def wrap(shift, script, args):
    return ('research_shift_run.py', ['--shift=' + shift, script] + args) if shift else (script, args)
def evals(shift, pn, label, short, ckflag, ck, script='research_grpo_fm.py'):
    base = lambda ol, eps, p: ['--mode', 'eval', '--object-list', ol, '--mus', 'none', '--episodes', str(eps), '--seed', '777',
                               '--disable-gpu-jit-fusion', '--deterministic'] + (['--pcl-noise', pn] if (p and pn) else []) + [ckflag, ck]
    out = []
    pr, a = wrap(shift, script, base('union_ycb_unidex/research_dfg_dev64_x2.yaml', 4, True)); out.append({'name': 'fmdet_munom_%s_%s_64x8' % (label, short), 'probe': pr, 'args': a})
    out.append({'name': 'fmdet_munom_%s_64x8' % short, 'probe': script, 'args': base('union_ycb_unidex/research_dfg_dev64_x2.yaml', 4, False)})
    for sp in ('seen', 'unseen'):
        pr, a = wrap(shift, script, base('union_ycb_unidex/test_set_%s_cat.yaml' % sp, 8, True)); out.append({'name': 'testdet_%s_munom_%s_%s_x8' % (sp, label, short), 'probe': pr, 'args': a})
    return out
def _seed(a, v):
    assert v not in used_seeds(), v
    a[a.index('--seed') + 1] = str(v)
def grpo(tag, short, shift, pn, label, lr, seedbase, keep_opt=True):
    runs = []
    for k, r in enumerate(GTR):
        a = list(r['args'][2:]); _seed(a, seedbase + k)
        a[a.index('--lr') + 1] = lr
        for fl in ('--init-checkpoint', '--init-optimizer'):
            if fl in a: a[a.index(fl) + 1] = E + '%s_p%02d/%s' % (tag, k - 1, a[a.index(fl) + 1].split('/')[-1])
        if not keep_opt and '--init-optimizer' in a:
            i = a.index('--init-optimizer'); del a[i:i + 2]
        if pn: a += ['--pcl-noise', pn]
        pr, aa = wrap(shift, 'research_grpo_finetune_v2.py', a)
        runs.append({'name': '%s_p%02d' % (tag, k), 'probe': pr, 'done_file': 'train_log.json', 'args': aa})
    return runs + evals(shift, pn, label, short, '--init-checkpoint', E + '%s_p11/final.pt' % tag)
def ppo(tag, short, shift, pn, label, lr, seedbase):
    runs = []
    for k, r in enumerate(PTR):
        a = list(r['args'][2:]); _seed(a, seedbase + k)
        a[a.index('--lr') + 1] = lr
        if '--init-checkpoint' in a: a[a.index('--init-checkpoint') + 1] = E + '%s_p%d/final.pt' % (tag, k - 1)
        script = 'research_ppo_finetune.py'
        if pn: script = 'research_ppo_pcl.py'; a += ['--pcl-noise=' + pn]
        pr, aa = wrap(shift, script, a)
        runs.append({'name': '%s_p%d' % (tag, k), 'probe': pr, 'done_file': 'train_log.json', 'args': aa})
    return runs + evals(shift, pn, label, short, '--init-checkpoint', E + '%s_p2/final.pt' % tag)
def plug(tag, short, algo, shift, pn, label, seedbase, extra=()):
    runs = []
    for k, r in enumerate(UTR):
        a = list(r['args'][2:]); a[a.index('--algo') + 1] = algo; _seed(a, seedbase + k)
        for fl in ('--init-head', '--init-optimizer'):
            if fl in a: a[a.index(fl) + 1] = E + '%s_p%02d/%s' % (tag, k - 1, a[a.index(fl) + 1].split('/')[-1])
        if pn: a += ['--pcl-noise', pn]
        a += list(extra)
        pr, aa = wrap(shift, 'research_plugin_rl.py', a)
        runs.append({'name': '%s_p%02d' % (tag, k), 'probe': pr, 'done_file': 'train_log.json', 'args': aa})
    return runs + evals(shift, pn, label, short, '--init-head', E + '%s_p11/head.pt' % tag, script='research_plugin_rl.py')
def save(name, runs, purpose):
    json.dump({'name': name, 'purpose': purpose, 'runs': runs}, open(name + '.json', 'w'), indent=1)
    return name
