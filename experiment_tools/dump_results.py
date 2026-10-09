"""Dump every finished run from server_queue/daemon.log (name, wall seconds, totals) plus parameter counts of a
full-weight checkpoint and a residual head, for building result tables. Read-only; writes temp/results_dump.json."""
import glob, json, os
R = '/root/autodl-tmp/DexGraspResearch'
res = {}
for line in open(R + '/server_queue/daemon.log', errors='replace'):
    parts = line.rstrip('\n').split(' ', 4)
    if len(parts) >= 4 and parts[1] == 'DONE':
        tot = None
        if len(parts) == 5:
            try:
                tot = json.loads(parts[4])
            except Exception:
                tot = None
        res[parts[2]] = {'utc': parts[0], 'sec': float(parts[3].rstrip('s') or 0), 'totals': tot}
params = {}
try:
    import torch
    def count(obj):
        if torch.is_tensor(obj):
            return obj.numel() if obj.is_floating_point() else 0
        if isinstance(obj, dict):
            return sum(count(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(count(v) for v in obj)
        return 0
    for key, pat in (('full_final', 'runs/grpo_shlockLF_lr3s1_p11/final.pt'), ('residual_head', 'runs/pluggrpo_shlockLF_s1_p11/head.pt'),
                     ('ppo_final', 'runs/ppoft_shlockLF_s1_p2/final.pt')):
        p = os.path.join(R, pat)
        if os.path.exists(p):
            sd = torch.load(p, map_location='cpu')
            params[key] = {'path': pat, 'float_elems': count(sd), 'keys': list(sd.keys())[:8] if isinstance(sd, dict) else str(type(sd))}
except Exception as e:
    params['error'] = repr(e)
json.dump({'runs': res, 'params': params}, open(R + '/temp/results_dump.json', 'w'))
print('runs', len(res), 'params', json.dumps(params)[:600])
