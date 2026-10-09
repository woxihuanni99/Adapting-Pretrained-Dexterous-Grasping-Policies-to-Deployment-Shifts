#!/usr/bin/env python3
"""Waits (CPU only, polls every 2 min, gives up after 10 h) until every s1-s3 selection result exists
(fmdet2k nominal 2048 + own-condition fault dev 512 for all 4 methods x 2 conditions x lr grid, plus frozen N0 and frozen fault),
then runs, exactly once: select_shared_lr.py --stage trigger ; topup_seeds.py (writes and queues the one-shot s4/s5 top-up).
Log: temp/auto_trigger_topup.log. Pre-fixed rules only; no discretion involved."""
import json, os, subprocess, sys, time
ROOT = '/root/autodl-tmp/DexGraspResearch/'; PY = '/root/autodl-tmp/envs/dex/bin/python'
LOG = open(ROOT + 'temp/auto_trigger_topup.log', 'a', buffering=1)
def log(*x): LOG.write(time.strftime('%Y-%m-%dT%H:%M:%SZ ', time.gmtime()) + ' '.join(map(str, x)) + '\n')
src = open(ROOT + 'tools/select_shared_lr.py').read(); ns = {}
exec(src[src.index('C = {'):src.index('TRIG =')], ns); C, FAULT, FF = ns['C'], ns['FAULT'], ns['FROZEN_FAULT']
need = ['fmdet2k_munom_frozen'] + list(FF.values())
for m in C:
    for c in C[m]:
        for lr, ss in C[m][c].items():
            need += ['fmdet2k_munom_' + s for s in ss] + [FAULT[c] % s for s in ss]
log('start; required results:', len(need))
t0 = time.time()
while True:
    miss = [n for n in need if not os.path.exists(ROOT + 'runs/' + n + '/result.json')]
    if not miss: break
    if time.time() - t0 > 10 * 3600: log('timeout; still missing', miss); sys.exit(1)
    log('missing', len(miss), miss[:6]); time.sleep(120)
log('all present -> trigger stage')
if os.path.exists(ROOT + 'temp/shared_lr_trigger.json'): log('trigger file already exists; abort'); sys.exit(1)
p = subprocess.run([PY, '-B', 'tools/select_shared_lr.py', '--stage', 'trigger'], cwd=ROOT, capture_output=True, text=True)
log('trigger rc', p.returncode); log(p.stdout[-6000:]); log(p.stderr[-3000:])
if p.returncode != 0 or not os.path.exists(ROOT + 'temp/shared_lr_trigger.json'): sys.exit(1)
trig = json.load(open(ROOT + 'temp/shared_lr_trigger.json')); log('TRIGGER LIST', trig)
if not trig: log('empty trigger list -> no top-up'); sys.exit(0)
p = subprocess.run([PY, '-B', 'tools/topup_seeds.py'], cwd=ROOT, capture_output=True, text=True)
log('topup rc', p.returncode); log(p.stdout[-8000:]); log(p.stderr[-3000:]); log('done')
