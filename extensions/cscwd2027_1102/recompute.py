"""Rebuild the CSCWD extension from released per-attempt official flags."""
from pathlib import Path
from fractions import Fraction
from collections import defaultdict
import argparse,csv,gzip,hashlib,json,math,statistics
from scipy.stats import t
R=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,default=R/'reproduced')
args=parser.parse_args(); A=args.out; A.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf8'))
records=json.loads(gzip.decompress((R/'data/attempt_outcomes.json.gz').read_bytes()))
chains=read(R/'data/chains.json'); batches={k:k for k in chains}
def score(label,name):
    x=records[label+':'+name]; y=x['success_flags']
    assert len(y)==x['n'] and all(v in (0,1) for v in y)
    return Fraction(sum(y),len(y))
def average(x):return sum(x,Fraction())/len(x)
def test(label,name,condition):
    return average([score(label,f'{name}_test_{condition}_{s}') for s in ('seen','unseen')])
def dump(name,x):(A/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def csv_table(name,rows):
    with (A/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def linear_contrast(groups, coefficients):
    assert len(groups) == len(coefficients) and all(len(x) >= 2 for x in groups)
    effect = sum(c * statistics.mean(x) for x, c in zip(groups, coefficients))
    terms = [c*c*statistics.variance(x)/len(x) for x, c in zip(groups, coefficients)]
    variance = sum(terms)
    if variance == 0:
        return {"difference_pp": effect, "CI95_unadjusted_two_sided_pp": None, "df": None,
                "group_sizes": [len(x) for x in groups], "reason": "Zero estimated between-run variance"}
    df = variance**2 / sum(v*v/(len(x)-1) for x, v in zip(groups, terms))
    half = float(t.ppf(.975, df)) * math.sqrt(variance)
    return {"difference_pp": effect, "CI95_unadjusted_two_sided_pp": [effect-half, effect+half],
            "df": df, "group_sizes": [len(x) for x in groups]}


def development():
    baseline=score('main','preflight_frozen_dev_nom_r2')
    groups=defaultdict(lambda:defaultdict(list)); rows=[]
    for c in chains['main']:
        if c['family']=='frozen':continue
        if 'reference' in c:
            ref=c['reference'];method=ref['method'];rate=ref['learning_rate'];seed=ref['seed_index'];origin='archived_checkpoint_current_runtime'
        else:
            method=c['family'];rate='3e-4' if method=='matchedres' else '2e-5';seed=c['seed_index'];origin='new_training'
        cond=c['condition'];fault='LF' if cond=='LF' else 'pn10'
        s=score('main',c['name']+'_dev_'+fault);n=score('main',c['name']+'_dev_nom')
        loss=100*(baseline-n)
        groups[(method,rate)][cond].append((s,n))
        rows.append(dict(method=method,learning_rate=rate,condition=cond,seed_index=seed,source=origin,
                         fault_success_pct=float(100*s),nominal_success_pct=float(100*n),nominal_loss_pp=float(loss),loss_exact_fraction=str(loss)))
    grid=[]
    for (method,rate),conditions in groups.items():
        row=dict(method=method,learning_rate=rate);losses=[];successes=[]
        for cond,values in conditions.items():
            s=average([v[0] for v in values]);loss=100*(baseline-average([v[1] for v in values]));losses.append(loss);successes.append(s)
            row.update({cond+'_n':len(values),cond+'_success_pct':float(100*s),cond+'_loss_pp':float(loss),cond+'_loss_exact':str(loss)})
        obj=average(successes)
        row.update(feasible=all(v<=5 for v in losses),objective=float(obj),objective_exact=str(obj),worst_loss_pp=float(max(losses)),worst_loss_exact=str(max(losses)))
        grid.append(row)
    selected={}
    for method in ('fwppo','fwgrpo'):
        feasible=[x for x in grid if x['method']==method and x['feasible']]
        selected[method]=min(feasible,key=lambda x:(-Fraction(x['objective_exact']),Fraction(x['worst_loss_exact']),float(x['learning_rate'])))['learning_rate']
    expected=read(R/'data/selection_record.json')
    assert grid==expected['expanded_grid'] and selected==expected['selected_full_weight_learning_rates']
    assert str(baseline)==expected['frozen_dev_exact']
    result=dict(expected,expanded_grid=grid,selected_full_weight_learning_rates=selected)
    dump('development_selection.json',result);csv_table('development_per_run.csv',rows);csv_table('development_grid.csv',grid)
    old=read(R/'data/original_development.json');oldbase=Fraction(old['frozen_exact'])
    lookup={(x['method'],x['learning_rate'],x['condition'],x['seed_index']):x for x in old['candidates']}
    drift=[]
    for row in rows:
        if row['source']!='archived_checkpoint_current_runtime':continue
        prior=lookup[(row['method'],row['learning_rate'],row['condition'],row['seed_index'])]
        oldnom=Fraction(prior['nominal_exact']);oldloss=100*(oldbase-oldnom);newloss=Fraction(row['loss_exact_fraction'])
        drift.append(dict(method=row['method'],learning_rate=row['learning_rate'],condition=row['condition'],seed_index=row['seed_index'],
                          nominal_change_pp=float(100*((baseline-newloss/100)-oldnom)),paired_loss_change_pp=float(newloss-oldloss)))
    assert len(drift)==40
    csv_table('environment_drift_per_checkpoint.csv',drift)
    summary={key:{'mean':statistics.mean(x[key] for x in drift),'sd':statistics.stdev(x[key] for x in drift),'min':min(x[key] for x in drift),'max':max(x[key] for x in drift),'n':40} for key in ('nominal_change_pp','paired_loss_change_pp')}
    dump('environment_drift.json',dict(n_identical_checkpoints=40,summary=summary,frozen_change_pp=float(100*(baseline-oldbase))))
    return result

def main():
    selection=development()
    matched_feasible=next(x['feasible'] for x in selection['expanded_grid'] if x['method']=='matchedres' and x['learning_rate']=='3e-4')
    frozen_nominal = test("main", "cx1101_frozen", "nom")
    frozen = {c: float(100*test("main", "cx1101_frozen", c)) for c in ("LF", "LF25", "LF75", "pn10", "nom")}
    model_rows = []
    for label, batch in batches.items():
        queue = chains[label]
        for chain in queue:
            if chain["family"] == "frozen":
                continue
            family = chain["family"]
            if label == "main":
                if "train_interactions" not in chain:
                    continue
                role = "expanded_selected_fullweight" if family == "fwppo" else "matched_residual" if family == "matchedres" else "new_fullweight_unselected"
                method = family
                seed = chain["seed_index"]
                rate = "3e-4" if family == "matchedres" else "2e-5"
            elif label == "fixed":
                ref = chain["reference"]
                role = "original_selected_residual" if ref["role"] == "selected_residual" else "matched_fullweight_reference"
                method, seed = ref["method"], ref["seed"]
                rate = "3e-3" if method == "res_ppo" else "1e-3" if method == "res_grpo" else "1e-5"
            else:
                ref = chain["reference"]
                role, method, seed, rate = "expanded_selected_fullweight", ref["method"], ref["seed_index"], ref["learning_rate"]
            nominal = test(label, chain["name"], "nom")
            loss = 100*(frozen_nominal-nominal)
            for condition in (["LF", "LF25", "LF75", "nom"] if chain["condition"] == "LF" else ["pn10", "nom"]):
                value = test(label, chain["name"], condition)
                model_rows.append({"role": role, "method": method, "learning_rate": rate,
                                   "training_condition": chain["condition"], "seed_index": seed,
                                   "test_condition": condition, "source_batch": label, "source_chain": chain["name"],
                                   "success_pct": float(100*value), "success_exact": str(value),
                                   "nominal_loss_pp": float(loss), "nominal_loss_exact_pp": str(loss)})
    csv_table("per_run.csv", model_rows)
    grouped = defaultdict(list)
    for row in model_rows:
        grouped[(row["role"], row["method"], row["training_condition"], row["test_condition"])].append(row)
    summary = []
    for (role, method, train, condition), rows in grouped.items():
        summary.append({"role": role, "method": method, "learning_rate": rows[0]["learning_rate"],
                        "training_condition": train, "test_condition": condition, "n": len(rows),
                        "success_mean_pct": statistics.mean(x["success_pct"] for x in rows),
                        "success_sd_pp": statistics.stdev(x["success_pct"] for x in rows),
                        "nominal_loss_mean_pp": statistics.mean(x["nominal_loss_pp"] for x in rows)})
    csv_table("summary.csv", summary)
    contrasts = []
    for condition in ("LF", "LF25", "LF75", "pn10"):
        train = "PCL" if condition == "pn10" else "LF"
        all_groups = []
        for optimizer in ("ppo", "grpo"):
            residual = grouped[("original_selected_residual", "res_"+optimizer, train, condition)]
            fullweight = grouped[("expanded_selected_fullweight", "fw"+optimizer, train, condition)]
            assert len(residual) == (5 if optimizer == "ppo" else 3) and len(fullweight) == 3
            all_groups.extend([residual, fullweight])
            effect = linear_contrast([[r["success_pct"] for r in residual], [r["success_pct"] for r in fullweight]], [1, -1])
            contrasts.append(dict(effect, comparison="selected_residual_minus_expanded_selected_FW", optimizer=optimizer,
                                  test_condition=condition, nominal_loss_difference_pp=statistics.mean(r["nominal_loss_pp"] for r in residual)-statistics.mean(r["nominal_loss_pp"] for r in fullweight)))
        pooled = linear_contrast([[r["success_pct"] for r in g] for g in all_groups], [.5, -.5, .5, -.5])
        loss = linear_contrast([[r["nominal_loss_pp"] for r in g] for g in all_groups], [.5, -.5, .5, -.5])
        contrasts.append(dict(pooled, comparison="equal_optimizer_average_selected_residual_minus_expanded_selected_FW", test_condition=condition,
                              nominal_loss_difference_pp=loss["difference_pp"], nominal_loss_CI95_pp=loss["CI95_unadjusted_two_sided_pp"]))
        res = grouped[("matched_residual", "matchedres", train, condition)]
        fw = grouped[("matched_fullweight_reference", "fwgrpo_matched_reference", train, condition)]
        effect = linear_contrast([[r["success_pct"] for r in res], [r["success_pct"] for r in fw]], [1, -1])
        contrasts.append(dict(effect, comparison="matched_residual_3e-4_minus_matched_fullweight_1e-5", test_condition=condition,
                              development_feasible_residual=matched_feasible, interpretation="Fixed-configuration descriptive comparison; report joint development feasibility separately."))
    dump("contrasts.json", contrasts)

    expected=read(R/'results/numbers.json')
    assert frozen==expected['frozen_test_success_pct']
    assert summary==expected['test_summary']
    assert contrasts==expected['contrasts']
    rebuilt=dict(expected,frozen_test_success_pct=frozen,development_selection=selection,test_summary=summary,contrasts=contrasts)
    dump('numbers.json',rebuilt)
    assert sum(x['n'] for k,x in records.items() if k!='main:preflight_frozen_dev_nom_r2')==565072
    dump('verification.json',dict(passed=True,formal_evaluations=543,formal_attempts=565072,contrasts=len(contrasts),development_selection_recomputed=True,per_attempt_flags=True))
    print(f'Reproduced development selection, 40-checkpoint drift, {len(model_rows)} run-condition rows and {len(contrasts)} contrasts. Output: {A}')
if __name__=='__main__':main()
