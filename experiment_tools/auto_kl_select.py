#!/usr/bin/env python3
"""CPU-only watcher (lock step 3): waits until all 48 development evaluations of the KL-baseline grid exist
(fmdet2k_munom_<short> and fmdet_munom_<label>_<short>_64x8 for the 24 chains in temp/kl_manifest.json), then runs
tools/kl_select.py once (real run: writes temp/kl_select.json and queues the thumb-lock transfer of the selected config).
Polls every 5 min, gives up after 20 h. Log temp/auto_kl_select.log. Does not read any test-set result."""
import json, os, subprocess, time
ROOT = '/root/autodl-tmp/DexGraspResearch/'; PY = '/root/autodl-tmp/envs/dex/bin/python'
LOG = open(ROOT + 'temp/auto_kl_select.log', 'a', buffering=1)
def log(*x): LOG.write(time.strftime('%Y-%m-%dT%H:%M:%SZ ', time.gmtime()) + ' '.join(map(str, x)) + '\n')
M = json.load(open(ROOT + 'temp/kl_manifest.json'))
LABEL = {'LF': 'shlockLF', 'PCL': 'pn10'}
need = []
for key, tags in M['configs'].items():
    c = key.split('|')[0]
    for t in tags: need += ['fmdet2k_munom_%s' % t['short'], 'fmdet_munom_%s_%s_64x8' % (LABEL[c], t['short'])]
log('start; required dev evals', len(need)); t0 = time.time()
while True:
    miss = [n for n in need if not os.path.exists(ROOT + 'runs/' + n + '/result.json')]
    if not miss: break
    if time.time() - t0 > 20 * 3600: log('timeout, missing', len(miss)); raise SystemExit(1)
    log('missing', len(miss)); time.sleep(300)
p = subprocess.run([PY, '-B', 'tools/kl_select.py'], cwd=ROOT, capture_output=True, text=True)
open(ROOT + 'temp/kl_select_output.txt', 'w').write(p.stdout + '\n' + p.stderr)
log('kl_select rc', p.returncode, '| done')
