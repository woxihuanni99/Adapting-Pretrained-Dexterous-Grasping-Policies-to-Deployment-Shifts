"""MULTI-GPU VARIANT (2026-10-08; copy of server_daemon.py, original unchanged): each started plan is pinned to one GPU via
CUDA_VISIBLE_DEVICES (the GPU with the fewest running plans, then most free memory); at most --per-gpu plans per GPU.

Plan scheduler for the rented Linux GPU server (runs there under nohup; never on the laptop).

Usage: python3 server_daemon.py [--max-parallel 5] [--min-free-mb 5000]
Queue: $ROOT/server_queue/pending/*.json  each = {"plan": "reports/plans/X.json", "requires": [files]} (name order = priority;
items whose required files do not exist yet are skipped until they do).
Up to --max-parallel plans run at once; a new plan starts only when the GPU has at least --min-free-mb free
and 45 s have passed since the last start (start-up memory is not visible immediately). Each plan's runs are
sequential: python <workspace>/<probe> --output $ROOT/runs/<name> <args>, log $ROOT/logs/<name>.log,
ledger reports/plans/<plan>_ledger_srv.json (completed runs are skipped -> resume-safe).
Stop: touch $ROOT/server_queue/STOP (no new plans; running ones finish). Heartbeat: server_queue/heartbeat.json.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path('/root/autodl-tmp/DexGraspResearch')
PY = '/root/autodl-tmp/envs/dex/bin/python'
WORKSPACE = {'research_dfg_grpo.py': 'DemoFunGrasp'}
Q = ROOT / 'server_queue'


def now():
    return datetime.now(timezone.utc).isoformat()


def gpu_free_mb(gpu=0):
    try:
        out = subprocess.run(['nvidia-smi', '--id=%d' % gpu, '--query-gpu=memory.free,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        free, used, util = [int(x) for x in out.split(',')]
        return free, used, util
    except Exception:
        return 0, -1, -1


def run_env(gpu=0):
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    c = ROOT / 'cache'
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', TMPDIR=str(ROOT / 'temp'), TMP=str(ROOT / 'temp'), TEMP=str(ROOT / 'temp'),
               XDG_CACHE_HOME=str(c), TORCH_HOME=str(c / 'torch'), TORCH_EXTENSIONS_DIR=str(c / 'torch_extensions'),
               CUDA_CACHE_PATH=str(c / 'cuda'), MPLCONFIGDIR=str(c / 'matplotlib'), WANDB_MODE='disabled',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               LD_LIBRARY_PATH='/root/miniconda3/lib:' + env.get('LD_LIBRARY_PATH', ''),
               PATH='/root/autodl-tmp/envs/dex/bin:/root/miniconda3/bin:' + env.get('PATH', '/usr/bin:/bin'))
    return env


def run_plan(plan_rel, log, gpu=0):
    plan_path = ROOT / plan_rel
    plan = json.loads(plan_path.read_text())
    ledger_path = plan_path.with_name(plan_path.stem + '_ledger_srv.json')
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        'plan': str(plan_path), 'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(), 'started_utc': now(), 'runs': {}}

    def save():
        t = ledger_path.with_name(ledger_path.name + '.tmp'); t.write_text(json.dumps(ledger, indent=1)); os.replace(t, ledger_path)
    fails = 0
    for run in plan['runs']:
        name, probe = run['name'], run['probe']
        out = ROOT / 'runs' / name
        done_file = run.get('done_file', 'result.json')
        row = ledger['runs'].setdefault(name, {})
        if (out / done_file).exists():
            row.setdefault('status', 'completed'); save(); continue
        if out.exists():
            row['status'] = 'failed_previously_keep_evidence'; fails += 1; save(); continue
        ws = ROOT / 'workspace' / WORKSPACE.get(probe, 'DemoGrasp')
        cmd = [PY, '-B', '-u', str(ws / probe), '--output', str(out)] + run['args']
        row.update(status='running', started_utc=now()); save()
        log.write('%s START %s gpu%d\n' % (now(), name, gpu)); log.flush()
        t0 = time.time()
        with open(ROOT / 'logs' / (name + '.log'), 'ab') as f:
            code = subprocess.run(cmd, cwd=str(ws), env=run_env(gpu), stdout=f, stderr=subprocess.STDOUT).returncode
        ok = code == 0 and (out / done_file).exists()
        row.update(status='completed' if ok else 'failed', exit_code=code, finished_utc=now(), wall_seconds=round(time.time() - t0, 1))
        if ok and done_file == 'result.json':
            try:
                res = json.loads((out / 'result.json').read_text())
                row['totals'] = res.get('totals')
            except Exception:
                pass
        fails += 0 if ok else 1
        save()
        log.write('%s %s %s %.0fs %s\n' % (now(), 'DONE' if ok else 'FAIL', name, time.time() - t0, json.dumps(row.get('totals'))))
        log.flush()
    ledger.update(status='completed' if not fails else 'completed_with_failures', failures=fails, finished_utc=now()); save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-parallel', type=int, default=5)
    ap.add_argument('--min-free-mb', type=int, default=5000)
    ap.add_argument('--gpus', default='0,1')
    ap.add_argument('--per-gpu', type=int, default=4)
    a = ap.parse_args()
    gpus = [int(x) for x in a.gpus.split(',')]
    a.max_parallel = min(a.max_parallel, a.per_gpu * len(gpus))
    for sub in ('pending', 'running', 'done'):
        (Q / sub).mkdir(parents=True, exist_ok=True)
    (ROOT / 'logs').mkdir(exist_ok=True)
    log = open(Q / 'daemon.log', 'a')
    threads, last_start, on_gpu = {}, 0.0, {}
    log.write('%s daemon start pid %d max_parallel %d per_gpu %d gpus %s min_free_mb %d\n' % (now(), os.getpid(), a.max_parallel, a.per_gpu, a.gpus, a.min_free_mb)); log.flush()
    while True:
        for k in [k for k, t in threads.items() if not t.is_alive()]:
            threads.pop(k); on_gpu.pop(k, None)
            os.replace(Q / 'running' / k, Q / 'done' / k)
            log.write('%s plan finished %s\n' % (now(), k)); log.flush()
        stats = {g: gpu_free_mb(g) for g in gpus}
        load = {g: sum(1 for v in on_gpu.values() if v == g) for g in gpus}
        cand = sorted([g for g in gpus if load[g] < a.per_gpu and stats[g][0] >= a.min_free_mb], key=lambda g: (load[g], -stats[g][0]))
        free = stats[cand[0]][0] if cand else 0
        (Q / 'heartbeat.json').write_text(json.dumps({'utc': now(), 'pid': os.getpid(), 'running': sorted(threads),
                                                      'plans_per_gpu': load, 'gpu': {str(g): {'free_mb': v[0], 'used_mb': v[1], 'util': v[2]} for g, v in stats.items()}}))
        stop = (Q / 'STOP').exists()
        if stop and not threads:
            (Q / 'STOP').unlink(); log.write('%s daemon stop\n' % now()); return
        pend = sorted((Q / 'pending').glob('*.json'))
        ready = []
        for q in pend:  # an item may declare prerequisites (files that must exist, e.g. a checkpoint from another plan)
            try:
                sq = json.loads(q.read_text())
            except Exception:
                continue
            if all((ROOT / r).exists() if not r.startswith('/') else Path(r).exists() for r in sq.get('requires', [])):
                ready.append(q)
        if ready and not stop and len(threads) < a.max_parallel and free >= a.min_free_mb and time.time() - last_start > 30 and cand:
            p = ready[0]
            spec = json.loads(p.read_text())
            os.replace(p, Q / 'running' / p.name)
            g = cand[0]
            t = threading.Thread(target=run_plan, args=(spec['plan'], log, g), daemon=True)
            t.start(); threads[p.name] = t; on_gpu[p.name] = g; last_start = time.time()
            log.write('%s plan started %s (%s) gpu%d free=%dMB\n' % (now(), p.name, spec['plan'], g, free)); log.flush()
        time.sleep(10)


if __name__ == '__main__':
    main()
