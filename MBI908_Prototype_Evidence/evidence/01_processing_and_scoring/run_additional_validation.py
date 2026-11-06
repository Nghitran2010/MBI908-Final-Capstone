"""Reproduce baseline scoring and five controlled business-plausibility scenarios.
Runs against the final documented rules; no browser/UI actions are performed.
"""
from pathlib import Path
import sys, io, contextlib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / 'prototype'
SENS = HERE.parent / '02_sensitivity'
sys.path[:0] = [str(ROOT), str(SENS)]
import pipeline
with contextlib.redirect_stdout(io.StringIO()):
    import rerun_24  # contains a separately implemented scoring calculation

pre = pd.read_csv(HERE / 'scoring_validation_results.csv')
recalc = rerun_24.compute({}).set_index('account_id').sort_index()
original = pre.set_index('account_id').sort_index()
comparisons = [('Health score','health_new','health_score'),('Health band','health_band_new','health_band'),
               ('Priority score','priority_new','priority_score'),('Priority band','priority_band_new','priority_band')]
results = []
for name,a,b in comparisons:
    left,right = recalc[a],original[b]
    matches = np.isclose(left.to_numpy(dtype=float),right.to_numpy(dtype=float),rtol=0,atol=1e-6,equal_nan=True) if pd.api.types.is_numeric_dtype(left) else left.to_numpy()==right.to_numpy()
    results.append(dict(Check=name,**{'Compared accounts':len(matches),'Mismatches':int((~matches).sum()),'Status':'PASS' if matches.all() else 'FAIL'}))
ranking = rerun_24.compute({}).assign(recomputed_rank=lambda df:np.arange(1,len(df)+1)).set_index('account_id').sort_index()
rank_matches = ranking.recomputed_rank.eq(original.priority_rank)
results.append(dict(Check='Priority rank',**{'Compared accounts':len(rank_matches),'Mismatches':int((~rank_matches).sum()),'Status':'PASS' if rank_matches.all() else 'FAIL'}))
pd.DataFrame(results).to_csv(HERE/'independent_baseline_comparison.csv',index=False)

# Override only the relevant prescore fields on a known valid template row.
# This is a set of controlled examples, not observations from real customers.
template=pre.iloc[[0]].copy()
common=dict(tenure_days=180,has_valid_usage=True,tickets_90=0,high_urgent_tickets_90=0,
            escalated_tickets_90=0,avg_resolution_time_hours_90=np.nan,
            avg_satisfaction=np.nan,satisfaction_response_count=0,satisfaction_response_coverage=0)
cases=[
 ('High MRR, strong engagement, no support pressure',dict(common,days_since_last_usage=2,active_days_90_per_90_exposure=12,mrr_amount=10000),'Low','Scoreable'),
 ('Zero MRR, recent escalation',dict(common,days_since_last_usage=2,active_days_90_per_90_exposure=12,tickets_90=1,escalated_tickets_90=1,avg_resolution_time_hours_90=12,mrr_amount=0),'Medium','Scoreable'),
 ('Mature account, no valid usage',dict(common,has_valid_usage=False,days_since_last_usage=np.nan,active_days_90_per_90_exposure=0,mrr_amount=100),'Needs Data Check','Verify Data'),
 ('Low engagement, no urgency, zero MRR',dict(common,days_since_last_usage=35,active_days_90_per_90_exposure=5,mrr_amount=0),'Low','Scoreable'),
 ('New account, insufficient exposure',dict(common,tenure_days=12,days_since_last_usage=3,active_days_90_per_90_exposure=6,mrr_amount=100),'Needs Data Check','Insufficient Exposure'),
]
scenario_rows=[]
for i,(desc,inputs,expected,evidence) in enumerate(cases,1):
    x=template.copy();x['account_id']=f'TEST-{i:02d}';x['account_name']=f'Controlled scenario {i:02d}'
    for col,val in inputs.items():x[col]=val
    y=pipeline.score_prescore(x).iloc[0]
    label='Needs Data Check' if y.health_band=='Limited Evidence' else y.priority_band
    passed=label==expected and y.evidence_status==evidence
    scenario_rows.append({'Scenario':f'S{i:02d}','Description':desc,
                          'Input overrides':'; '.join(f'{k}={v}' for k,v in inputs.items()),
                          'Expected outcome':expected,'Actual outcome':label,
                          'Evidence status':y.evidence_status,
                          'Health score':y.health_score if y.evidence_status=='Scoreable' else 'Hidden',
                          'Priority score':y.priority_score,'Priority band':y.priority_band,
                          'Status':'PASS' if passed else 'FAIL'})
pd.DataFrame(scenario_rows).to_csv(HERE/'business_plausibility_tests.csv',index=False)
print(f'Independent scoring baseline: {sum(r["Status"]=="PASS" for r in results)}/{len(results)} checks PASS, 500 accounts')
print(f'Business plausibility: {sum(r["Status"]=="PASS" for r in scenario_rows)}/{len(scenario_rows)} scenarios PASS')
if not all(r['Status']=='PASS' for r in results+scenario_rows):raise SystemExit(1)
