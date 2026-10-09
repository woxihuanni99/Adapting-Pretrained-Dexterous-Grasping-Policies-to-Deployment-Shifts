"""Offline DemoFunGrasp joint metric from saved per-episode records (result.json 'episodes').
J_tau = (1/N) * #[success & style_match & afford_dist < tau], N = all evaluated trials (failures kept).
Also reports the original conditional metrics (SA = style acc on success, SAD = mean afford dist on success).
Read-only over runs/dfg*/result.json; writes temp/dfg_joint_metric.json."""
import glob, json, os
import numpy as np
R = '/root/autodl-tmp/DexGraspResearch'
TAUS = [0.02, 0.03, 0.04, 0.05, 0.06]
out = {}
for f in sorted(glob.glob(R + '/runs/dfg*_x8/result.json') + glob.glob(R + '/runs/dfgsel_*/result.json')):
    name = os.path.basename(os.path.dirname(f))
    try:
        d = json.load(open(f))
        eps = d['episodes']
        S = np.array([e['official_ppo_print_successes'] for e in eps], dtype=float).ravel()
        M = np.array([e['style_match'] for e in eps], dtype=float).ravel()
        A = np.array([e['afford_dist'] for e in eps], dtype=float).ravel()
    except Exception as ex:
        out[name] = {'error': repr(ex)}
        continue
    ok = S > 0
    rec = {'N': int(S.size), 'success': float(S.mean()),
           'SA_on_success': float(M[ok].mean()) if ok.any() else None,
           'SAD_on_success': float(A[ok].mean()) if ok.any() else None,
           'success_and_style': float((ok & (M > 0)).mean())}
    for t in TAUS:
        rec['J_%dcm' % round(t * 100)] = float((ok & (M > 0) & (A < t)).mean())
    out[name] = rec
json.dump(out, open(R + '/temp/dfg_joint_metric.json', 'w'), indent=1)
print('files', len(out), 'errors', sum(1 for v in out.values() if 'error' in v))
for k in sorted(out)[:3]: print(k, out[k])
