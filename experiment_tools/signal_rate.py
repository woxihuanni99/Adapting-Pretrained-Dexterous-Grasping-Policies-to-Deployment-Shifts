"""Group-signal statistics from GRPO training logs (read-only).
For every logs/{grpo,dfg_grpo}_<tag>_pNN.log collect the per-iteration JSON lines (success, informative_groups,
all_zero/all_one when logged). Writes temp/signal_rate.json: {tag: {phase: [ {it, success, informative, all_zero, all_one, reward} ]}}."""
import glob, json, os, re
R = '/root/autodl-tmp/DexGraspResearch'
out = {}
for f in sorted(glob.glob(R + '/logs/grpo_*_p[0-9][0-9].log') + glob.glob(R + '/logs/dfg_grpo_*_p[0-9][0-9].log')):
    m = re.match(r'(.*)_p(\d\d)\.log$', os.path.basename(f))
    if not m:
        continue
    tag, ph = m.group(1), int(m.group(2))
    rows = []
    for line in open(f, errors='ignore'):
        line = line.strip()
        if not (line.startswith('{') and '"it"' in line and 'informative_groups' in line):
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        rows.append({k: d.get(k) for k in ('it', 'success', 'informative_groups', 'all_zero_groups', 'all_one_groups', 'reward')})
    if rows:
        out.setdefault(tag, {})[ph] = rows
json.dump(out, open(R + '/temp/signal_rate.json', 'w'))
print('tags', len(out), 'phases', sum(len(v) for v in out.values()))
