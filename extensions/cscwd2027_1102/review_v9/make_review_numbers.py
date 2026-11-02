"""Descriptive retention checks of released same-server LF results."""
from pathlib import Path
import hashlib,json,math
R=Path(__file__).resolve().parent;B=R.parent
SP=R.parent/'review_v5/numbers.json'
TP=R.parents[2]/'manuscript/cscwd2027_v4/table_audit.json'
S=json.loads(SP.read_text(encoding='utf8'));A=json.loads(TP.read_text(encoding='utf8'))
labels={'fwppo':'Full PPO','fwgrpo':'Full GRPO','res_ppo':'Residual PPO','res_grpo':'Residual GRPO'}
rows=[]
for method,label in labels.items():
 r=next(v for v in S['same_runtime_selected'] if v['method']==method and v['condition']=='LF')
 rows.append(dict(method=method,label=label,success=r['success'],loss=r['loss'],mean_le_5=r['loss']['mean']<=5,upper_CI_le_5=r['loss']['CI95'][1]<=5))
gains=[]
for row in A['TABLE II. ORIGINAL SUCCESS (%) AND NOMINAL LOSS (pp)']['rows']:
 base=float(row[1]);full_grpo=float(row[3].split()[0]);gains.append(dict(condition=row[0],frozen_pct=base,full_grpo_pct=full_grpo,displayed_gain_pp=round(full_grpo-base,1)))
result=dict(sources=[dict(path=p.relative_to(R.parents[2]).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [SP,TP]],retention_checks=rows,
            threshold_pp=5.0,scope='Illustrative LF review of existing test results, not a replacement of development selection or a deployment certification.',
            original_displayed_gains=gains,
            bootstrap_mean_SE_ratio_n3=math.sqrt(2/3),bootstrap_ratio_scope='Conditional variance identity for a single arithmetic sample mean relative to the unbiased plug-in SE. No common multiplicative correction is established for the nonlinear shared-support curve statistic.',
            new_training=0,new_simulation=0)
(R/'numbers.json').write_bytes(json.dumps(result,ensure_ascii=False,indent=2).encode('utf8'))
print(json.dumps({'retention_checks':[(x['label'],x['mean_le_5'],x['upper_CI_le_5']) for x in rows],'gains':gains},ensure_ascii=False,indent=2))
