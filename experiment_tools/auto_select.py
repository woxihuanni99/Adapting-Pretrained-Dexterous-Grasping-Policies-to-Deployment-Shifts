#!/usr/bin/env python3
"""Waits (CPU only, polls every 2 min, gives up after 10 h) until the one-shot top-up results (s4, s5 for every triggered
(method, lr) on LF and PCL: 2048-ep nominal + own-condition fault dev) exist, then runs select_shared_lr.py --stage select once.
Log: temp/auto_select.log (full selection printout; the decision itself follows the pre-fixed rule)."""
import json, os, subprocess, sys, time
ROOT = '/root/autodl-tmp/DexGraspResearch/'; PY = '/root/autodl-tmp/envs/dex/bin/python'
LOG = open(ROOT + 'temp/auto_select.log', 'a', buffering=1)
def log(*x): LOG.write(time.strftime('%Y-%m-%dT%H:%M:%SZ ', time.gmtime()) + ' '.join(map(str, x)) + '\n')
src = open(ROOT + 'tools/select_shared_lr.py').read(); ns = {}
exec(src[src.index('C = {'):src.index('TRIG =')], ns); C, FAULT = ns['C'], ns['FAULT']
t = json.load(open(ROOT + 'temp/shared_lr_trigger.json')); trig = t['trigger_list'] if isinstance(t, dict) else t
need = []
for m, lr in trig:
    for c in ('LF', 'PCL'):
        s3 = C[m][c][lr][-1]
        for k in (4, 5):
            sk = s3[:-1] + str(k); need += ['fmdet2k_munom_' + sk, FAULT[c] % sk]
log('start; trigger list', trig, '; required', len(need))
t0 = time.time()
while True:
    miss = [n for n in need if not os.path.exists(ROOT + 'runs/' + n + '/result.json')]
    if not miss: break
    if time.time() - t0 > 10 * 3600: log('timeout; still missing', miss); sys.exit(1)
    log('missing', len(miss), miss[:8]); time.sleep(120)
if os.path.exists(ROOT + 'temp/shared_lr_select.json'): log('select output already exists; abort'); sys.exit(1)
p = subprocess.run([PY, '-B', 'tools/select_shared_lr.py', '--stage', 'select'], cwd=ROOT, capture_output=True, text=True)
log('select rc', p.returncode); log(p.stdout[-8000:]); log(p.stderr[-3000:]); log('done')
