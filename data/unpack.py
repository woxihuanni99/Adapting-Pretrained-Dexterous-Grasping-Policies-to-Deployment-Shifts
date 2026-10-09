#!/usr/bin/env python3
"""Verify and unpack the released data archives into the layout expected by analysis/scripts.

    python data/unpack.py            # verify SHA-256, then extract
    python data/unpack.py --check    # verify only

Resulting layout (all under data/):
    runs/<name>/result.json ...           locked post-lock evaluations and training logs (29 Sep 2026 protocol)
    meta/temp/*.json                      manifests and locked statistics (committed unpacked)
    meta/reports/plans/*.json             every training/evaluation plan of the locked study
    supp_20261007/{runs,temp,plans}       setting-matched control and all-collision re-evaluation
    supp_20261008/{runs,temp,plans}       output-shift coordinate control (36 runs)
    output_analysis_20261008/             saved states, frozen and adapted action means of 72 checkpoints
"""
import argparse, hashlib, sys, tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCH = HERE / 'archives'
LOCKED_RUNS_SHA256 = '34d17fde03594b8d40271fc880431e3d98ced5f943e69857ee97b2103f2f5c9a'  # joined locked_runs.tar.gz
TARGETS = {'locked_runs.tar.gz': HERE, 'locked_plans.tar.gz': HERE / 'meta',
           'supp_20261007.tar.gz': HERE / 'supp_20261007', 'supp_20261008.tar.gz': HERE / 'supp_20261008',
           'output_analysis_20261008.tar.gz': HERE / 'output_analysis_20261008'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def safe_extract(tar_path, dest):
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            target = (dest / m.name).resolve()
            if not str(target).startswith(str(dest)) or m.issym() or m.islnk():
                raise RuntimeError('unsafe member in %s: %s' % (tar_path.name, m.name))
        tf.extractall(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    for line in (ARCH / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        got = sha256(ARCH / name.strip())
        if got != digest:
            sys.exit('checksum mismatch: %s' % name)
    print('archive checksums OK')
    joined = ARCH / 'locked_runs.tar.gz'
    with open(joined, 'wb') as out:
        for part in sorted(ARCH.glob('locked_runs.tar.gz.part*')):
            out.write(part.read_bytes())
    if sha256(joined) != LOCKED_RUNS_SHA256:
        sys.exit('joined locked_runs.tar.gz checksum mismatch')
    if a.check:
        joined.unlink()
        return
    for name, dest in TARGETS.items():
        safe_extract(ARCH / name, dest)
        print('extracted', name, '->', dest.relative_to(HERE.parent))
    joined.unlink()


if __name__ == '__main__':
    main()
