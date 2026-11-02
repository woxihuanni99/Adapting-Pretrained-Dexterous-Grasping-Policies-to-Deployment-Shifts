"""Existing-record checks for the v8 review; no experiment or primary-test changes."""
from pathlib import Path
import json,hashlib,urllib.request
import numpy as np
from scipy.stats import t

R=Path(__file__).resolve().parent; B=R.parent
paths=[R.parents[2]/'analysis/numbers.json',R.parent/'results/numbers.json',R.parent/'review_v5/numbers.json']
N,X,S=[json.loads(p.read_text(encoding='utf8')) for p in paths]
def old_curve(method,loss_pp):
    points=sorted((100*r['nom_loss'],100*r['fault']) for r in N['lrcurve']['LF'][method])
    assert points[0][0]<=loss_pp<=points[-1][0]
    return float(np.interp(loss_pp,[v[0] for v in points],[v[1] for v in points]))
def selected(method):return next(r for r in S['same_runtime_selected'] if r['method']==method and r['condition']=='LF')
scaled=next(r for r in S['constant_diagnostics'] if r['name'].lower().startswith('scaled'))
cards=[]
for method,label,changes,form in [('fwppo','Full PPO','Planner + scale','Weights'),('res_ppo','Residual PPO','2×256 MLP + scale','State-dependent correction')]:
    row=selected(method)
    cards.append(dict(label=label,source='A',changes=changes,review_form=form,postures='Ref., 25%, 75%',success=row['success'],loss=row['loss']))
cards.append(dict(label='Scaled shift',source='B',changes='24 mean edits',review_form='Fixed output vector',postures='Reference',success=scaled['success'],loss=scaled['loss']))
kl=next(r for r in N['kl']['grid']['LF'] if r['lr']==N['kl']['selected']['lr'] and r['beta_id']==N['kl']['selected']['beta_id'])
kl_loss=100*kl['nom_loss']
kl_comparison=dict(loss_pp=kl_loss,success_pct=100*kl['fault'],interpolated_residual_pct={m:old_curve(m,kl_loss) for m in ['res_ppo','res_grpo']},source='Original-protocol ledger; descriptive interpolation only.')
support=S['interpolation']['estimate']
curve_support={m:[100*r['nom_loss'] for r in rows if support['lower']<100*r['nom_loss']<support['upper']] for m,rows in N['lrcurve']['LF'].items()}
assert not curve_support['full_grpo'] and not curve_support['res_ppo']
cross_runtime=[]
for method,oldname in [('fwppo','full_ppo'),('fwgrpo','full_grpo')]:
    r=next(v for v in X['test_summary'] if v['method']==method and v['learning_rate']=='2e-5' and v['test_condition']=='LF')
    cross_runtime.append(dict(method=method,new_server_loss_pp=r['nominal_loss_mean_pp'],new_server_success_pct=r['success_mean_pct'],original_curve_prediction_pct=old_curve(oldname,r['nominal_loss_mean_pp'])))
c=next(v for v in X['contrasts'] if v['test_condition']=='LF75' and v['comparison']=='equal_optimizer_average_selected_residual_minus_expanded_selected_FW')
estimate=c['difference_pp'];df=c['df'];lo,hi=c['CI95_unadjusted_two_sided_pp'];se=(hi-lo)/(2*t.ppf(.975,df))
margin=3.0;ps=[float(t.sf((estimate+margin)/se,df)),float(t.cdf((estimate-margin)/se,df))]
tost=dict(margin_pp=margin,estimate_pp=estimate,df=df,se_pp=se,one_sided_p=ps,p=max(ps),scope='Post hoc ±3 pp margin; audited for reviewer reply only, not added to the paper or original nine-test family.')
refs=json.loads((R/'reference_metadata.json').read_text(encoding='utf8'))
assert refs[0]['page']=='220-229' and refs[1]['page']=='291-300'
result=dict(sources=[dict(path=p.relative_to(R.parents[2]).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],cards=cards,
            card_scope='A: same-server extension, Table III. B: separate 24-parameter shift-control series, Table V. Do not compute a cross-source performance contrast. The 24-parameter arm has no reported LF75/RFLF evaluation.',
            kl_interpolation=kl_comparison,observed_means_in_common_range_interior=curve_support,
            cross_runtime_points=cross_runtime,cross_runtime_point_interpretation='Different runtime from original curves; not a validation of interpolation bias.',
            optional_tost_audit=tost,references=refs,new_training=0,new_simulation=0)
(R/'numbers.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(kl=kl_comparison,tost=tost,cross_runtime=cross_runtime,reference_pages=[x['page'] for x in refs]),ensure_ascii=False,indent=2))
