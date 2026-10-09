"""Run task processes with E-drive logs and process-local E-drive caches."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINUX = '/mnt/e/DexGraspResearch'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--base', action='store_true')
    parser.add_argument('--simulator', action='store_true',
                        help='Keep Isaac Gym internal cache/temp paths on E for this child process')
    parser.add_argument('--cwd', default=LINUX)
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.cwd.startswith(LINUX + '/') and args.cwd != LINUX:
        raise ValueError('Working directory must remain inside E research workspace')
    if Path(args.label).name != args.label or not args.label:
        raise ValueError('Use a simple log label')
    log = ROOT / 'logs' / (args.label + '.log')
    if log.exists():
        raise FileExistsError(log)
    for relative in ['temp', 'cache/pip', 'cache/torch_extensions', 'cache/cuda',
                     'cache/matplotlib', 'cache/huggingface', 'envs']:
        (ROOT / relative).mkdir(parents=True, exist_ok=True)
    env = {
        'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUTF8': '1',
        'TMPDIR': LINUX + '/temp', 'TMP': LINUX + '/temp', 'TEMP': LINUX + '/temp',
        'PIP_CACHE_DIR': LINUX + '/cache/pip', 'PIP_DISABLE_PIP_VERSION_CHECK': '1',
        'PIP_NO_COMPILE': '1',
        'PIP_CONFIG_FILE': '/dev/null', 'XDG_CACHE_HOME': LINUX + '/cache',
        'TORCH_HOME': LINUX + '/cache/torch',
        'TORCH_EXTENSIONS_DIR': LINUX + '/cache/torch_extensions',
        'CUDA_CACHE_PATH': LINUX + '/cache/cuda',
        'MPLCONFIGDIR': LINUX + '/cache/matplotlib',
        'HF_HOME': LINUX + '/cache/huggingface',
        'WANDB_MODE': 'disabled', 'WANDB_DIR': LINUX + '/cache/wandb',
        'WANDB_CACHE_DIR': LINUX + '/cache/wandb',
        'WANDB_CONFIG_DIR': LINUX + '/cache/wandb_config',
        'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
        'MAX_JOBS': '2', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'LD_LIBRARY_PATH': '/usr/lib/wsl/lib:/usr/local/cuda-11.3/lib64',
    }
    support_files = [Path(__file__).resolve()]
    if args.simulator:
        shim = ROOT / 'tools/libisaacgym_cache_redirect.so'
        if not shim.is_file():
            raise FileNotFoundError(shim)
        (ROOT / 'cache/isaacgym_home').mkdir(parents=True, exist_ok=True)
        env.update(LD_PRELOAD=LINUX + '/tools/libisaacgym_cache_redirect.so',
                   DEXGRASP_SIM_HOME=LINUX + '/cache/isaacgym_home',
                   DEXGRASP_SIM_TEMP=LINUX + '/temp')
        support_files += [shim, ROOT / 'tools/isaacgym_cache_redirect.cpp']
    support_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in support_files}
    python = '/usr/bin/python' if args.base else LINUX + '/envs/shadow/bin/python'
    if not args.base:
        env['VIRTUAL_ENV'] = LINUX + '/envs/shadow'
    trailing = args.args[1:] if args.args[:1] == ['--'] else args.args
    command = ['wsl.exe', '-d', 'Ubuntu-20.04', '--cd', args.cwd, '--', '/usr/bin/env']
    command += [key + '=' + value for key, value in env.items()]
    command += [python, '-B', '-u'] + trailing
    windows_env = dict(os.environ)
    windows_env.update(TMP=str(ROOT / 'temp'), TEMP=str(ROOT / 'temp'), PYTHONDONTWRITEBYTECODE='1')
    start = time.monotonic()
    with log.open('xb') as out:
        process = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, env=windows_env)
    record = {'label': args.label, 'command': command, 'returncode': process.returncode,
              'seconds': time.monotonic() - start, 'log': str(log),
              'support_files_sha256': support_hashes}
    (ROOT / 'reports' / (args.label + '.json')).write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps({key: record[key] for key in ['label', 'returncode', 'seconds', 'log']}, ensure_ascii=False))
    if process.returncode:
        print(log.read_text(encoding='utf-8', errors='replace')[-6000:])
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
