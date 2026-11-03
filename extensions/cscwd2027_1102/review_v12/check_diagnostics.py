"""Reconstruct the reported checkpoint-level diagnostics without simulation."""
from pathlib import Path
from statistics import mean
import csv,hashlib,json

R=Path(__file__).resolve().parent
SOURCE=R.parents[2]/'analysis/numbers.json'
N=json.loads(SOURCE.read_text(encoding='utf8'));A=N['supp_1008']['output_analysis']
def ranks(v):
 out=[0.0]*len(v);ix=sorted(range(len(v)),key=lambda i:v[i]);i=0
 while i<len(ix):
  j=i+1
  while j<len(ix) and v[ix[j]]==v[ix[i]]:j+=1
  for k in ix[i:j]:out[k]=(i+j+1)/2
  i=j
 return out
def spearman(x,y):
 x=ranks(x);y=ranks(y);mx=mean(x);my=mean(y)
 return sum((p-mx)*(q-my) for p,q in zip(x,y))/(sum((p-mx)**2 for p in x)*sum((q-my)**2 for q in y))**.5
rows=[]
for key,g in A['groups'].items():
 for i,run in enumerate(g['runs']):
  rows.append(dict(group=key,run=run,scheme=g['scheme'],rate=g['lr'],selected=g['selected'],RFJ3_rad=g['per_run']['dq_rfj3'][i],LF_success_pct=100*g['per_run']['fault'][i],whole_shift=g['per_run']['shift'][i]))
assert len(rows)==A['n_runs_rank']==51 and len({r['run'] for r in rows})==51
rho=spearman([r['RFJ3_rad'] for r in rows],[r['LF_success_pct'] for r in rows])
rho_shift=spearman([r['whole_shift'] for r in rows],[r['LF_success_pct'] for r in rows])
assert abs(rho-A['spearman_dq_rfj3_vs_LF'])<1e-12 and abs(rho_shift-A['spearman_shift_vs_LF'])<1e-12
extra=[dict(group=k,run=run,RFJ3_rad=g['per_run']['dq_rfj3'][i],success_pct=100*g['per_run']['fault'][i]) for k,g in A['extra_groups'].items() if k.endswith('LF75') for i,run in enumerate(g['runs'])]
assert len(extra)==12
report=dict(source='analysis/numbers.json',source_SHA256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),reference_runs=51,available_groups=len(A['groups']),group_coverage=[dict(group=k,scheme=g['scheme'],rate=g['lr'],n=len(g['runs'])) for k,g in A['groups'].items()],spearman_RFJ3_success=rho,spearman_whole_shift_success=rho_shift,correlation_scope='Pooled available diagnostic checkpoints, not every learning rate or a causal RFJ3 intervention.',retrained_75_runs=extra,sigma_RFJ3=A['sigma0'][17],constant_RFJ3_fraction=A['groups']['cstLF']['mean']['dq_rfj3']/A['groups']['cstLF']['mean']['sum_abs_dq'],new_training=0,new_simulation=0)
(R/'numbers.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
with (R/'reference_runs.csv').open('w',encoding='utf8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Validated',len(rows),'reference and',len(extra),'retrained checkpoints; rho',rho,'whole shift',rho_shift)
