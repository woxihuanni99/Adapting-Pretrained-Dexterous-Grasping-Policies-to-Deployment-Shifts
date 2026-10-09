#!/usr/bin/env python3
"""One-shot seed top-up (s4, s5) for the shared-lr trigger list (protocol 2026-09-29). Run on the server.
For each triggered (method, lr), each selection condition (LF, PCL), each seed k in (4, 5):
  short_k = short_s3 with trailing '3' -> k   (same rule as select_shared_lr.seeds)
  needed results: fmdet2k_munom_<short_k> (2048-ep nominal dev) and FAULT[c] % short_k (512-ep fault dev)
  A. both exist                                  -> nothing
  B. training of tag_k finished                  -> eval-only plan for the missing evals (pending)
  C. a queue item already runs/holds tag_k       -> eval-only plan in waiting (requires the final checkpoint); held items reported
  D. a plan for tag_k exists but is not queued and no run dirs exist -> queue it (+ waiting eval-only plan for fmdet2k)
  E. nothing exists                              -> clone the s3 training plan (names s3->sk, fresh unused training seeds,
                                                    identical evals) and append the fmdet2k eval; queue it
Run dirs that exist without result.json are reported (daemon skips them) and nothing is queued for that item.
--dry-run prints the actions only; --candidates all ignores the trigger list (lookup test, always dry-run)."""
import argparse, copy, glob, json, os, re
ROOT = '/root/autodl-tmp/DexGraspResearch/'
PL = ROOT + 'reports/plans/'; Q = ROOT + 'server_queue/'; RUNS = ROOT + 'runs/'
ap = argparse.ArgumentParser(); ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--candidates', default='trigger')
ap.add_argument('--prefix', default='0537'); a = ap.parse_args()
src = open(ROOT + 'tools/select_shared_lr.py').read()
ns = {}; exec(src[src.index('C = {'):src.index('TRIG =')], ns); C, FAULT = ns['C'], ns['FAULT']
if a.candidates == 'all':
    cands = sorted({(m, lr) for m in C for c in C[m] for lr in C[m][c]}); a.dry_run = True
else:
    t = json.load(open(ROOT + 'temp/shared_lr_trigger.json'))
    cands = [tuple(x) for x in (t['trigger_list'] if isinstance(t, dict) else t)]
print('candidates:', cands)
plans = {}
for f in sorted(glob.glob(PL + '*.json')):
    try: q = json.load(open(f))
    except Exception: continue
    if isinstance(q, dict) and isinstance(q.get('runs'), list): plans[os.path.basename(f)] = q
used = set()
for q in plans.values():
    for r in q['runs']:
        if isinstance(r, dict) and '--seed' in r.get('args', []):
            try: used.add(int(r['args'][r['args'].index('--seed') + 1]))
            except Exception: pass
seedbase = max(used) // 1000 * 1000 + 2000
queued = {}
for d in ('pending', 'waiting', 'running', 'held'):
    for f in glob.glob(Q + d + '/*.json'):
        try: queued[json.load(open(f))['plan'].split('/')[-1]] = d + '/' + os.path.basename(f)
        except Exception: pass
def res(name): return os.path.exists(RUNS + name + '/result.json')
def dir_nores(name): return os.path.isdir(RUNS + name) and not res(name)
def ren(x, t3, tk, s3, sk):
    return x.replace(t3, tk).replace(s3, sk) if isinstance(x, str) else x
def ren_run(r, t3, tk, s3, sk):
    r = copy.deepcopy(r); r['name'] = ren(r['name'], t3, tk, s3, sk); r['args'] = [ren(x, t3, tk, s3, sk) for x in r['args']]; return r
actions, problems, nplan = [], [], 0
for m, lr in cands:
    for c in ('LF', 'PCL'):
        short3 = C[m][c][lr][-1]; assert short3.endswith('s3')
        sel = plans['srv_sel2k_%s_%s_%s.json' % (m, c, lr.replace('e-', 'em'))]
        r2k = [r for r in sel['runs'] if r['name'] == 'fmdet2k_munom_' + short3][0]
        ck = [x for x in r2k['args'] if x.startswith('/mnt/e/DexGraspResearch/runs/')][0]
        final3 = ck.split('/')[-2]; ckfile = ck.split('/')[-1]; tag3 = re.sub(r'_p\d+$', '', final3)
        fault3 = FAULT[c] % short3
        tp = [(any(r['name'] == fault3 for r in q['runs']), sum(1 for r in q['runs'] if r.get('done_file') == 'train_log.json'), fn) for fn, q in plans.items()
              if any(r['name'] == final3 and r.get('done_file') == 'train_log.json' for r in q['runs'])]
        fr = [r for q in plans.values() for r in q['runs'] if r['name'] == fault3]
        if not tp or not fr: problems.append('no s3 training plan / fault eval for %s %s %s (%s)' % (m, c, lr, final3)); continue
        tp.sort(reverse=True); tpn = tp[0][2]; tq = plans[tpn]; fr = fr[0]
        ntrain = tp[0][1]
        for k in (4, 5):
            sk = short3[:-1] + str(k); tk = tag3[:-1] + str(k) if tag3.endswith('3') else None
            if tk is None: problems.append('tag does not end with 3: ' + tag3); continue
            finalk = final3.replace(tag3, tk); ckk = 'runs/%s/%s' % (finalk, ckfile)
            need = [n for n in ('fmdet2k_munom_' + sk, FAULT[c] % sk) if not res(n)]
            key = '%s %s %s s%d [%s]' % (m, c, lr, k, tk)
            if not need: actions.append((key, 'A: done')); continue
            bad = [n for n in need if dir_nores(n)]
            if bad: problems.append('%s: run dir without result (daemon would skip): %s' % (key, bad)); continue
            ev = []
            for n in need:
                if n.startswith('fmdet2k_'): ev.append(ren_run(r2k, tag3, tk, short3, sk))
                else: ev.append(ren_run(fr, tag3, tk, short3, sk))
            kplans = [fn for fn, q in plans.items() if any(r['name'] == finalk and r.get('done_file') == 'train_log.json' for r in q['runs'])]
            kq = [queued[fn] for fn in kplans if fn in queued]
            if os.path.exists(ROOT + ckk):
                actions.append((key, 'B: trained -> eval-only pending %s' % [r['name'] for r in ev], ('pending', 'srv_topup_eval_' + sk, ev, None)))
            elif kq:
                inplan = {r['name'] for fn in kplans for r in plans[fn]['runs']}
                evw = [e for e in ev if e['name'] not in inplan]
                held = [x for x in kq if x.startswith('held/')]
                actions.append((key, 'C: queued as %s -> eval-only waiting %s on %s%s' % (kq, [e['name'] for e in evw], ckk, '  (release held -> pending)' if held else ''))
                               + tuple(('release', h, None, None) for h in held) + ((('waiting', 'srv_topup_eval_' + sk, evw, [ckk]),) if evw else ()))
            elif kplans:
                tr = [r['name'] for r in plans[kplans[0]]['runs'] if r.get('done_file') == 'train_log.json']
                if any(os.path.isdir(RUNS + n) for n in tr): problems.append('%s: plan %s partially run, not queued -> manual' % (key, kplans[0])); continue
                inplan = {r['name'] for r in plans[kplans[0]]['runs']}
                evw = [e for e in ev if e['name'] not in inplan]
                actions.append((key, 'D: existing plan %s not queued -> queue it + eval-only waiting %s' % (kplans[0], [e['name'] for e in evw]), ('queue', kplans[0], None, None))
                               + ((('waiting', 'srv_topup_eval_' + sk, evw, [ckk]),) if evw else ()))
            else:
                runs = []
                for r in tq['runs']:
                    rr = ren_run(r, tag3, tk, short3, sk)
                    if r.get('done_file') == 'train_log.json':
                        i = rr['args'].index('--seed'); v = seedbase + 1000 * nplan + len(runs)
                        assert v not in used, v; used.add(v); rr['args'][i + 1] = str(v)
                    if any(x.startswith('/mnt/e/') and tag3 in x for x in rr['args']): problems.append('%s: leftover tag3 in %s' % (key, rr['name']))
                    runs.append(rr)
                if FAULT[c] % sk not in [r['name'] for r in runs]: runs.append(ren_run(fr, tag3, tk, short3, sk))
                runs.append(ren_run(r2k, tag3, tk, short3, sk)); nplan += 1
                actions.append((key, 'E: clone %s (%d train phases, %d runs, seeds %s..) + fmdet2k' % (tpn, ntrain, len(runs),
                                [r['args'][r['args'].index('--seed') + 1] for r in runs if r.get('done_file') == 'train_log.json'][0]),
                                ('pending', 'srv_topup_' + tk, runs, None)))
for x in actions: print(x[0], '|', x[1])
print('PROBLEMS:', problems if problems else 'none')
if a.dry_run or problems:
    print('dry-run: nothing written' if a.dry_run else 'problems -> nothing written'); raise SystemExit(0)
idx = 0
for x in actions:
    for act in x[2:]:
        kind, name, runs, req = act
        if kind == 'release':
            dst = Q + 'pending/%s%02d_%s' % (a.prefix, idx, name.split('/')[-1].split('_', 1)[1]); os.rename(Q + name, dst); idx += 1; print('released', name, '->', dst); continue
        if kind == 'queue':
            item = Q + 'pending/%s%02d_%s' % (a.prefix, idx, name); json.dump({'plan': 'reports/plans/' + name}, open(item, 'w')); idx += 1; print('queued', item); continue
        pf = PL + name + '.json'; assert not os.path.exists(pf), pf
        json.dump({'name': name, 'purpose': 'one-shot seed top-up for shared-lr selection (%s)' % x[0], 'runs': runs}, open(pf, 'w'), indent=1)
        item = {'plan': 'reports/plans/' + name + '.json'}
        if req: item['requires'] = req
        qf = Q + '%s/%s%02d_%s.json' % (kind, a.prefix, idx, name); json.dump(item, open(qf, 'w')); idx += 1; print('wrote', pf, '->', qf)
