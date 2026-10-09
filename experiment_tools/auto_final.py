#!/usr/bin/env python3
"""CPU-only watcher: waits until every locked final evaluation (all final_* runs in srv_final_* / srv_al_* plans) has a
result, then runs tools/final_stats.py once (output temp/final_stats_output.txt, temp/final_stats.json) and writes a
backup archive of all small result files to temp/backup_final/. Polls every 5 min, gives up after 30 h. Log temp/auto_final.log."""
import glob, json, os, subprocess, time
ROOT = '/root/autodl-tmp/DexGraspResearch/'; PY = '/root/autodl-tmp/envs/dex/bin/python'
LOG = open(ROOT + 'temp/auto_final.log', 'a', buffering=1)
def log(*x): LOG.write(time.strftime('%Y-%m-%dT%H:%M:%SZ ', time.gmtime()) + ' '.join(map(str, x)) + '\n')
sh, al = set(), set()
for f in glob.glob(ROOT + 'reports/plans/srv_final_*.json') + glob.glob(ROOT + 'reports/plans/srv_al_*.json'):
    tgt = al if ('srv_al_' in f or 'alfrozen' in f) else sh
    try: runs = json.load(open(f))['runs']
    except Exception: continue
    for r in runs:
        if isinstance(r, dict) and str(r.get('name', '')).startswith('final_'): tgt.add(r['name'])
log('start; required Shadow', len(sh), 'Allegro', len(al))
t0 = time.time()
def wait(need, tag):
    while True:
        miss = [n for n in sorted(need) if not os.path.exists(ROOT + 'runs/' + n + '/result.json')]
        if not miss: return
        if time.time() - t0 > 30 * 3600: log('timeout', tag, 'missing', len(miss), miss[:10]); raise SystemExit(1)
        log(tag, 'missing', len(miss)); time.sleep(300)
def backup(d):
    os.makedirs(ROOT + d, exist_ok=True)
    subprocess.run('cd runs && find . -maxdepth 2 \\( -name result.json -o -name train_log.json -o -name diag.json \\) | tar czf ../%s/runs_small.tgz -T - && '
                   'cd .. && tar czf %s/meta.tgz temp/*.json temp/*.log temp/*.txt reports/plans server_queue/*.log tools' % (d, d), shell=True, cwd=ROOT)
wait(sh, 'shadow')
log('all Shadow final results present -> final_stats')
p = subprocess.run([PY, '-B', 'tools/final_stats.py'], cwd=ROOT, capture_output=True, text=True)
open(ROOT + 'temp/final_stats_output.txt', 'w').write(p.stdout + '\n' + p.stderr)
log('final_stats rc', p.returncode); backup('temp/backup_final_shadow'); log('shadow backup written')
wait(al, 'allegro')
backup('temp/backup_final_all')
log('backup written; done')
