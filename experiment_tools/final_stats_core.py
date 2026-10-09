"""Statistical core for the locked final analysis (DexGrasp, locked 2026-09-29/10-01). Formulas are fixed in
claude/DexGrasp_锁定说明_草稿.md; this file must not be changed after the final evaluation results are viewed.
Unit = independent training run; each run contributes one value Y (seen/unseen averaged inside the run).
contrast(): weighted linear contrast over independent groups with Welch-Satterthwaite df, one-sided p in the
pre-registered direction (+1: contrast > 0 predicted) and a two-sided 95% CI. holm(): Holm step-down adjusted p.
classify(): the four pre-specified outcome categories."""
import math
from scipy import stats


def _mv(x):
    n = len(x)
    m = sum(x) / n
    v = sum((xi - m) ** 2 for xi in x) / (n - 1) if n > 1 else float('nan')
    return m, v, n


def contrast(groups, weights, direction=+1, alpha=0.05):
    """groups: list of lists of per-run values; weights: same length. Returns dict with estimate, se, df, t,
    one-sided p (in `direction`), two-sided CI, and per-group summaries."""
    assert len(groups) == len(weights)
    est, V, den, summ = 0.0, 0.0, 0.0, []
    for g, w in zip(groups, weights):
        m, v, n = _mv(g)
        if n < 2:
            raise ValueError('each group needs >= 2 runs')
        est += w * m
        c = w * w * v / n
        V += c
        den += c * c / (n - 1)
        summ.append({'n': n, 'mean': m, 'sd': math.sqrt(v), 'values': list(g), 'weight': w})
    se = math.sqrt(V)
    df = V * V / den if den > 0 else float('inf')
    t = est / se if se > 0 else float('inf') * (1 if est > 0 else -1)
    p_one = stats.t.sf(direction * t, df)
    q = stats.t.ppf(1 - alpha / 2, df)
    return {'estimate': est, 'se': se, 'df': df, 't': t, 'p_one_sided': float(p_one),
            'ci95_two_sided': (float(est - q * se), float(est + q * se)), 'groups': summ}


def holm(pvals):
    """pvals: dict name -> raw p. Returns dict name -> Holm-adjusted p (monotone step-down)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def classify(estimate, direction, p_holm, p_opposite_holm=None, alpha=0.05):
    """Four categories: supported / consistent-insufficient / opposite-insufficient / evidence-for-opposite.
    p_opposite_holm is only used if a two-sided statement about the opposite direction is reported; by default
    the opposite category requires the two-sided CI to exclude 0 on the opposite side (checked by caller)."""
    if direction * estimate > 0:
        return 'direction supported (Holm)' if p_holm < alpha else 'mean direction consistent, evidence insufficient'
    if p_opposite_holm is not None and p_opposite_holm < alpha:
        return 'evidence for the opposite direction'
    return 'mean direction opposite, evidence insufficient'


if __name__ == '__main__':
    # self-test on synthetic numbers (no experimental data)
    r = contrast([[0.60, 0.62, 0.58], [0.61, 0.66, 0.59], [0.40, 0.45, 0.43], [0.30, 0.35, 0.33]], [0.5, 0.5, -0.5, -0.5])
    assert abs(r['estimate'] - 0.2333333) < 1e-6, r
    # equal-weight two-group case reduces to Welch
    a, b = [0.1, 0.2, 0.3], [0.0, 0.05, 0.2]
    r2 = contrast([a, b], [1, -1])
    w = stats.ttest_ind(a, b, equal_var=False)
    assert abs(r2['t'] - w.statistic) < 1e-9 and abs(2 * min(r2['p_one_sided'], 1 - r2['p_one_sided']) - w.pvalue) < 1e-9
    h = holm({'a': 0.01, 'b': 0.04, 'c': 0.03})
    assert abs(h['a'] - 0.03) < 1e-12 and abs(h['c'] - 0.06) < 1e-12 and abs(h['b'] - 0.06) < 1e-12, h
    print('self-test ok', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != 'groups'})
