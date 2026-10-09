"""Shared helpers for the matched-GRPO control and collision re-evaluation analyses (2026-10-07). Run on the server.
Scores come only from runs/<name>/result.json; independent training runs are the units; Welch-Satterthwaite intervals."""
import hashlib, json, math, os, statistics
ROOT = '/root/autodl-tmp/DexGraspResearch/'
FAKE = None  # selftest: dict-like name -> score

def score(name, expected=None):
    if FAKE is not None:
        h = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return FAKE(name, h)
    r = json.load(open(ROOT + 'runs/%s/result.json' % name))
    assert r['status'] == 'completed', name
    n = r['totals']['episodes']
    if expected is not None: assert n == expected, (name, n, expected)
    return r['totals']['success_rate']

def Y(src):  # src = {'seen': name, 'unseen': name}
    return (score(src['seen'], 1128) + score(src['unseen'], 800)) / 2

def yname(cond, tag):
    return {'seen': 'final_seen_%s_%s' % (cond, tag), 'unseen': 'final_unseen_%s_%s' % (cond, tag)}

def welch(groups):
    """groups: list of (weight, values). Returns estimate, se, df, 95% CI (two-sided)."""
    from scipy.stats import t as T
    est = sum(w * statistics.mean(v) for w, v in groups)
    comps = [w * w * statistics.variance(v) / len(v) for w, v in groups]
    var = sum(comps); den = sum(c * c / (len(v) - 1) for c, (w, v) in zip(comps, groups))
    df = var * var / den if den > 0 else float('inf'); se = math.sqrt(var)
    q = float(T.ppf(0.975, df)) if math.isfinite(df) else 1.959963984540054
    return {'est': est, 'se': se, 'df': df, 'ci': [est - q * se, est + q * se], 'n': [len(v) for _, v in groups],
            'means': [statistics.mean(v) for _, v in groups]}

def classify(g_new, delta):
    """Pre-specified categories (plan section E): order of checks is part of the rule."""
    if g_new['ci'][1] < 0: return 'reversed'
    if delta['ci'][1] < 0: return 'reduced, remaining gap positive' if g_new['ci'][0] > 0 else 'reduced, remaining gap uncertain'
    if g_new['ci'][0] > 0: return 'maintained (no reduction detected)'
    return 'insufficient evidence'

def dump(path, obj):
    json.dump(obj, open(path, 'w'), indent=1)
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()
