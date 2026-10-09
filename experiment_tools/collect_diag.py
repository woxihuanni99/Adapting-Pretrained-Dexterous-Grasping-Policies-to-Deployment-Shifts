"""Collect runs/diag_*/diag.json summaries -> temp/diag_summary.json (read-only)."""
import glob, json, os
R = '/root/autodl-tmp/DexGraspResearch'
out = {}
for f in sorted(glob.glob(R + '/runs/diag_*/diag.json')):
    out[os.path.basename(os.path.dirname(f))] = json.load(open(f))['summary']
json.dump(out, open(R + '/temp/diag_summary.json', 'w'), indent=1)
print(len(out))
