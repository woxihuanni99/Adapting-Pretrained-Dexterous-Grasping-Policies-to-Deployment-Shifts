"""Post-review analysis of existing runs only; original records stay read-only."""
from pathlib import Path
import csv, hashlib, json, math, os, statistics
from collections import Counter,defaultdict
import numpy as np
from scipy.stats import t
R=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(R/'mplconfig')
OLD=R.parents[2]/'analysis/numbers.json'
NEW=R.parent/'results/numbers.json'
N=json.loads(OLD.read_text(encoding='utf8'));X=json.loads(NEW.read_text(encoding='utf8'))

def summary(values):
    v=[float(x) for x in values];n=len(v);m=statistics.mean(v);s=statistics.stdev(v)
    h=float(t.ppf(.975,n-1))*s/math.sqrt(n)
    return dict(n=n,mean=m,sd=s,CI95=[m-h,m+h],runs=v)

with (R.parent/'results/per_run.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
selected=[]
for method,role in [('fwppo','expanded_selected_fullweight'),('fwgrpo','expanded_selected_fullweight'),('res_ppo','original_selected_residual'),('res_grpo','original_selected_residual')]:
    for condition in ('LF','pn10'):
        q=[x for x in rows if x['method']==method and x['role']==role and x['test_condition']==condition]
        selected.append(dict(method=method,condition=condition,rate=q[0]['learning_rate'],success=summary([float(x['success_pct']) for x in q]),loss=summary([float(x['nominal_loss_pp']) for x in q])))

lf=N['const']['LF'];direct=N['cst']['grid_test']['LF'][N['cst']['select']['LF']['selected_lr']]
bias=N['supp_1008']['bias_control']['stats']['conditions']
constants=[]
for name,param,fault,nominal,baseline in [
    ('Residual network','state-dependent',lf['full']['fault'],lf['full']['nom'],lf['Y0_nom']),
    ('Post-training mean','constant',lf['const']['fault'],lf['const']['nom'],lf['Y0_nom']),
    ('Direct constant + scale','48',direct['fault'],direct['nom'],N['frozen']['nom']),
    ('Scaled shift','24',bias['LF|LF']['runs']['sigma|3e-2'],bias['LF|nom']['runs']['sigma|3e-2'],N['frozen']['nom']),
    ('Raw shift','24',bias['LF|LF']['runs']['raw|3e-3'],bias['LF|nom']['runs']['raw|3e-3'],N['frozen']['nom'])]:
    constants.append(dict(name=name,parameters=param,success=summary([100*x for x in fault]),loss=summary([100*(baseline-x) for x in nominal])))

# A descriptive functional of the original observed operating points. Each
# bootstrap draw resamples independent runs within each rate, keeping each
# checkpoint's fault and nominal scores together. No extrapolation is used.
curves=N['lrcurve']['LF'];keys=('full_ppo','full_grpo','res_ppo','res_grpo')
def evaluate_curve_set(points):
    pooled={}
    for key,values in points.items():
        ties=defaultdict(list)
        for x,y in values:ties[x].append(y)
        pooled[key]=[(x,statistics.mean(ys)) for x,ys in ties.items()]
    points=pooled
    lower=max(min(p[0] for p in v) for v in points.values())
    upper=min(max(p[0] for p in v) for v in points.values())
    if upper<=lower:return None
    knots=sorted({lower,upper}|{p[0] for values in points.values() for p in values if lower<p[0]<upper})
    vals={k:np.interp(knots,[p[0] for p in sorted(v)],[p[1] for p in sorted(v)]) for k,v in points.items()}
    delta=(vals['res_ppo']-vals['full_ppo']+vals['res_grpo']-vals['full_grpo'])/2
    area=float(sum((knots[i+1]-knots[i])*(delta[i+1]+delta[i])/2 for i in range(len(knots)-1))/(upper-lower))
    return dict(lower=lower,upper=upper,effect=area,knots=knots,difference=delta.tolist())
points={k:[(r['nom_loss']*100,r['fault']*100) for r in curves[k]] for k in keys}
point=evaluate_curve_set(points)
rng=np.random.default_rng(110205);boot=[];invalid=0
for _ in range(10000):
    sample={}
    for k in keys:
        sample[k]=[]
        for cell in curves[k]:
            ids=rng.integers(0,len(cell['fault_runs']),len(cell['fault_runs']))
            fault=np.asarray(cell['fault_runs'])[ids].mean()*100
            loss=(N['frozen']['nom']-np.asarray(cell['nom_runs'])[ids].mean())*100
            sample[k].append((float(loss),float(fault)))
    b=evaluate_curve_set(sample)
    if b is None:invalid+=1
    else:boot.append(b['effect'])
interpolation=dict(estimate=point,bootstrap_seed=110205,bootstrap_draws=10000,valid_draws=len(boot),no_overlap_draws=invalid,
                   percentile_CI95=np.quantile(boot,[.025,.975]).tolist(),
                   estimand='Equal-optimizer average separation of piecewise-linear test curves, uniformly averaged over their common observed nominal-loss support. Support is recomputed within each bootstrap draw.',
                   status='Post-review descriptive curve analysis; not a directly trained or selected matched-retention comparison.')
interpolation['tied_cost_policy']='If candidate means have exactly identical nominal costs, average their success means before interpolation.'

# Exact one-step optimizer-call budgets from phase iterations and minibatches.
updates=[dict(method='Full-weight PPO',phases=3,iterations=77,epochs=5,minibatches=2,adam_steps=770,attempts=19712),
         dict(method='Full-weight GRPO',phases=12,iterations=38,epochs=2,minibatches=4,adam_steps=304,attempts=19456),
         dict(method='Residual PPO',phases=12,iterations=37,epochs=5,minibatches=4,adam_steps=740,attempts=18944),
         dict(method='Residual GRPO',phases=12,iterations=37,epochs=5,minibatches=4,adam_steps=740,attempts=18944),
         dict(method='Matched residual GRPO',phases=12,iterations=37,epochs=2,minibatches=4,adam_steps=296,attempts=18944)]
for x in updates:assert x['adam_steps']==x['iterations']*x['epochs']*x['minibatches']
object_list=R.parent/'training/object_lists/union_ycb_unidex/test_set_unseen_cat.yaml'
slots=[v.strip() for v in object_list.read_text(encoding='utf8').splitlines() if v.strip()]
counts=Counter(slots)
assert len(slots)==100 and len(counts)==78 and Counter(counts.values())=={1:56,2:22}
out=dict(sources=[dict(path=p.relative_to(R.parents[2]).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (OLD,NEW)],
         same_runtime_selected=selected,constant_diagnostics=constants,interpolation=interpolation,optimizer_calls=updates,
         constant_minus_network_paired=lf['const_minus_full_fault'],new_training=0,new_physics=0,
         unseen_object_list=dict(slots=100,unique=78,duplicated_exactly_twice=22,sha256=hashlib.sha256(object_list.read_bytes()).hexdigest()),
         interval_scope='Two-sided 95% Student intervals across independent training runs, conditional on the fixed frozen reference; post-training mean versus source network uses paired run differences.')
(R/'numbers.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'curve_gap':point['effect'],'overlap':[point['lower'],point['upper']],'bootstrap_CI':interpolation['percentile_CI95'],'valid':len(boot),'selected_LF':[(x['method'],x['loss']['CI95']) for x in selected if x['condition']=='LF']},indent=2))
