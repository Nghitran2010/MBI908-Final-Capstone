import sys, pathlib, json, hashlib, io, traceback
import numpy as np
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[2]/'prototype'; sys.path.insert(0,str(ROOT))
import pipeline as p
out=pathlib.Path(__file__).resolve().parent; checks=[]
def check(name, condition, detail=''):
    checks.append(dict(check=name,status='PASS' if condition else 'FAIL',detail=str(detail)))
    print(('PASS' if condition else 'FAIL'),name,detail)
files=sorted((ROOT/'sample_raw_data').glob('*.csv'))
class Upload:
    def __init__(self,path): self.name=path.name;self.raw=path.read_bytes()
    def getvalue(self): return self.raw
frames, errors, warnings, digest = p.read_uploaded_files([Upload(f) for f in files]);check('five CSV schema accepted',not errors and len(frames)==5,errors)
result=p.process_raw_frames(frames)
check('complete processing no error',not result['errors'],result['errors'])
if result['errors']: raise SystemExit('pipeline failed')
x=result['scored']; ready=result['app_ready']; pre=result['prescore']; bundled=pd.read_csv(ROOT/'processed_demo_data.csv')
check('500 unique account rows',len(x)==500 and x.account_id.nunique()==500)
check('expected health bands',x.health_band.value_counts().to_dict()=={'Healthy':323,'Medium':120,'Low':35,'Limited Evidence':22},x.health_band.value_counts().to_dict())
check('expected priority bands',x.priority_band.value_counts().to_dict()=={'Low':321,'Medium':154,'High':25},x.priority_band.value_counts().to_dict())
check('support urgency tiers',x.support_urgency_score.value_counts().sort_index().to_dict()=={0.:308,30.:53,60.:112,85.:14,100.:13},x.support_urgency_score.value_counts().sort_index().to_dict())
check('500 exact ranks',list(x.priority_rank)==list(range(1,501)))
check('ranking descending and account id tie-break',x[['priority_score','account_id']].reset_index(drop=True).equals(x.sort_values(['priority_score','account_id'],ascending=[False,True])[['priority_score','account_id']].reset_index(drop=True)))
check('priority formula all 500',np.allclose(x.priority_score,((.50*x.health_concern_score+.25*x.support_urgency_score+.15*x.commercial_score+.1*x.evidence_concern_score).round(2))), 'max_abs_difference='+str((x.priority_score-(.50*x.health_concern_score+.25*x.support_urgency_score+.15*x.commercial_score+.1*x.evidence_concern_score).round(2)).abs().max()))
check('driver exactly maps to predefined advisory action',x.suggested_action.equals(x.leading_driver.map(p.ACTIONS)))
check('no commercial leading drivers',not x.leading_driver.str.contains('commercial',case=False).any())
check('escalation driver precedence',bool((x.loc[(x.escalated_tickets_90>0)&(x.evidence_status=='Scoreable'),'leading_driver']=='Recent escalation history').all()))
check('NDC masked on Priority List in app source', 'table.loc[table["health_band"] == "Limited Evidence", "health_score"] = np.nan' in (ROOT/'app.py').read_text())
check('same score and band app_ready vs scored', ready.set_index('account_id')['priority_score'].equals(x.set_index('account_id')['priority_score']) and ready.set_index('account_id')['priority_band'].equals(x.set_index('account_id')['priority_band']))
check('same main outputs bundled vs recomputed',all(pd.to_numeric(ready.set_index('account_id')[c],errors='coerce').round(2).sort_index().equals(pd.to_numeric(bundled.set_index('account_id')[c],errors='coerce').round(2).sort_index()) for c in ['priority_score','priority_rank']), 'bundled dataset output')
check('rank1 example',str(x.iloc[0].account_name)=='Company_176' and x.iloc[0].priority_score==70.34,(str(x.iloc[0].account_name),float(x.iloc[0].priority_score)))
check('outcome fields excluded from scoring function and UI output',not any(t in (ROOT/'pipeline.py').read_text().split('def score_prescore(')[1].split('def build_app_ready(')[0] for t in ['churn_flag','termination_flag','churn_date']) and not any(t in ready.columns for t in ['churn_flag','termination_flag','churn_date']))
# Negative upload tests
bad=frames['accounts'].drop(columns=['seats'])
check('invalid schema rejected',p.identify_table(bad) is None)
check('order-independent schema detection',p.identify_table(frames['accounts'][list(reversed(frames['accounts'].columns))])=='accounts')
missing={k:v for k,v in frames.items() if k!='subscriptions'}
check('missing required table rejected',bool(p.process_raw_frames(missing)['errors']))
check('duplicate table uploads rejected',bool(p.read_uploaded_files([Upload(f) for f in files]+[Upload(files[0])])[1]))
check('valid files identified correctly',set(frames)==set(p.EXPECTED_SCHEMAS))
# Dynamic page-equivalent logic for combinatorial filtered paging
for col in ['industry','priority_band','health_band','evidence_status']:
    groups=ready.groupby(col,dropna=False)
    ok=True
    for val,sub in groups:
        for page_size in [10,25,50,100]:
            chunks=[sub.iloc[i:i+page_size] for i in range(0,len(sub),page_size)]
            merged=pd.concat(chunks) if chunks else sub.iloc[:0]
            if merged.account_id.tolist()!=sub.account_id.tolist():ok=False
    check('offline group paging: '+col,ok,'not interactive Streamlit UI')
# Detailed source integrity and output reconciliation evidence
validation, validation_errors, validation_warnings = p.validate_frames(frames)
foreign_key_rows = validation.loc[validation["Check"].astype(str).str.contains("foreign key", case=False)]
check('foreign-key relationships: 4/4',len(foreign_key_rows)==4 and foreign_key_rows['Result'].eq('PASS').all(),f"{int(foreign_key_rows['Result'].eq('PASS').sum())}/4")
usage=frames['feature_usage']; usage_ids=usage.groupby('usage_id').size()
repeats=int((usage_ids>1).sum()); composite=int(usage.duplicated(['subscription_id','usage_id']).sum())
check('repeated usage identifiers and unique composite keys',repeats==21 and composite==0, f'{repeats} repeated identifiers; {composite} composite duplicates')
stats={
 'valid usage rows and accounts': ('valid_usage_rows','valid_usage_accounts',11802,486),
 'valid support rows and accounts': ('valid_support_rows','valid_support_accounts',923,374),
 'satisfaction observations and accounts': ('satisfaction_responses','satisfaction_accounts',551,306),
}
for label,(rows_key,accounts_key,expected_rows,expected_accounts) in stats.items():
 actual_rows=int(result['stats'][rows_key]); actual_accounts=int(result['stats'][accounts_key])
 check(label,actual_rows==expected_rows and actual_accounts==expected_accounts,f'{actual_rows} rows; {actual_accounts} accounts')
check('escalation precedence 13/13',int(((x.escalated_tickets_90>0)&(x.evidence_status=='Scoreable')).sum())==13 and (x.loc[(x.escalated_tickets_90>0)&(x.evidence_status=='Scoreable'),'leading_driver']=='Recent escalation history').all())
# Compare every field in the full 38-column account ready artefact (account_id is key).
left=ready.set_index('account_id').sort_index();right=bundled.set_index('account_id').sort_index()
recon=[]
check('full output 38 columns',len(ready.columns)==38 and set(ready.columns)==set(bundled.columns) and left.index.equals(right.index),'38 columns including account_id key')
recon.append(dict(column='account_id',mismatched_accounts=int(len(left.index.symmetric_difference(right.index))),status='PASS' if left.index.equals(right.index) else 'FAIL'))
for column in left.columns:
 a,b=left[column],right[column]
 if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
  ok=np.isclose(pd.to_numeric(a,errors='coerce').to_numpy(dtype=float),pd.to_numeric(b,errors='coerce').to_numpy(dtype=float),rtol=0,atol=1e-6,equal_nan=True)
 else:
  ok=((a.fillna('__MISSING__').astype(str)==b.fillna('__MISSING__').astype(str))).to_numpy()
 num=int((~ok).sum());recon.append(dict(column=column,mismatched_accounts=num,status='PASS' if num==0 else 'FAIL'))
pd.DataFrame(recon).to_csv(out/'full_38_column_reconciliation.csv',index=False)
check('full output reconciliation 38/38',len(recon)==38 and all(r['status']=='PASS' for r in recon),f"{sum(r['status']=='PASS' for r in recon)}/38")
claims=[]
def claim(test,num,den,method):claims.append(dict(test=test,numerator=num,denominator=den,result='PASS' if num==den else 'FAIL',method=method))
claim('Cross-package full output reconciliation',sum(r['status']=='PASS' for r in recon),38,'Account_id plus all 37 values; numeric tolerance 1e-6, exact categorical text/NA')
escal=x.loc[x.escalated_tickets_90>0];claim('Escalation driver precedence',int((escal.leading_driver=='Recent escalation history').sum()),len(escal),'Accounts with recent escalations')
ndc=x.loc[x.health_band=='Limited Evidence'];claim('Needs Data Check evidence statuses',len(ndc),22,'Account evidence classification; UI masking checked separately in source')
claim('Driver-action mapping',int(x.suggested_action.eq(x.leading_driver.map(p.ACTIONS)).sum()),len(x),'Frozen driver-action mapping')
for _,r in x.head(3).iterrows():
 reconstructed=round(.50*r.health_concern_score+.25*r.support_urgency_score+.15*r.commercial_score+.1*r.evidence_concern_score,2)
 claim('Top-3 Priority formula '+str(r.account_id),int(abs(reconstructed-r.priority_score)<1e-8),1,'Weighted formula from frozen scored component values')
pd.DataFrame(claims).to_csv(out/'report_claims_reconciliation.csv',index=False)
# Boundary checks test scalar thresholds by injecting controlled component values into score_prescore.
base=pre.iloc[[0]].copy()
def evaluate(**kw):
 test=base.copy()
 for k,v in kw.items():test[k]=v
 return p.score_prescore(test).iloc[0]
for days,expected in [(14,100),(15,65),(30,65),(31,30),(60,30),(61,0)]:
 actual=evaluate(days_since_last_usage=days).recency_score
 check('boundary recency '+str(days),actual==expected,f'{actual} expected {expected}')
for active,expected in [(3.99,25),(4,60),(7.99,60),(8,100)]:
 actual=evaluate(active_days_90_per_90_exposure=active).activity_score
 check('boundary activity '+str(active),actual==expected,f'{actual} expected {expected}')
# Boundary tests invoke the frozen scoring function. Expected bands are specification
# assertions, not copies of the classification predicates in the application source.
# A controlled, scoreable row is used so the Health=50 cut-off (not present in the
# observed portfolio) can be tested exactly.
controlled=dict(tenure_days=180,has_valid_usage=True,
                tickets_90=1,high_urgent_tickets_90=0,escalated_tickets_90=0,
                avg_resolution_time_hours_90=12.0,
                satisfaction_response_count=2,satisfaction_response_coverage=1.0)
for label,patch,expected_score,expected_band in [
 ('49.99',dict(days_since_last_usage=15,active_days_90_per_90_exposure=4,
               tickets_90=2,high_urgent_tickets_90=2,avg_satisfaction=4.438),49.99,'Low'),
 ('50.00',dict(days_since_last_usage=15,active_days_90_per_90_exposure=4,
               tickets_90=2,high_urgent_tickets_90=2,avg_satisfaction=4.44),50.00,'Medium'),
 ('69.99',dict(days_since_last_usage=14,active_days_90_per_90_exposure=3,
               avg_satisfaction=3.598),69.99,'Medium'),
 ('70.00',dict(days_since_last_usage=14,active_days_90_per_90_exposure=3,
               avg_satisfaction=3.6),70.00,'Healthy'),
]:
 row=evaluate(**(controlled|patch))
 check('health-band pipeline boundary '+label,
       abs(float(row.health_score)-expected_score)<1e-8 and row.health_band==expected_band,
       f'Health={float(row.health_score):.2f}; band={row.health_band}; expected={expected_band}')
# Exact 30.00/50.00 Priority examples from the actual dataset. Verify both the
# score and the classification after full processing, with named source IDs.
for target,expected_band,expected_ids in [
 (30.00,'Medium',['A-075038','A-781cc0','A-cd458b']),
 (50.00,'High',['A-39ddf4','A-9174e0']),
]:
 subset=x.loc[x.account_id.isin(expected_ids)]
 ok=(len(subset)==len(expected_ids) and
     (subset.priority_score==target).all() and
     (subset.priority_band==expected_band).all())
 check('priority-band real-account boundary '+f'{target:.2f}',ok,
       f'{len(subset)}/{len(expected_ids)} matching source accounts; band={expected_band}')
# Adjoining observed bands are checked without fabricating missing exact cases.
low=x.loc[x.priority_band=='Low','priority_score']
medium=x.loc[x.priority_band=='Medium','priority_score']
check('priority-band observed Low upper boundary',float(low.max())==29.90,
      f'max Low={float(low.max()):.2f}')
check('priority-band observed Medium lower boundary',float(medium.min())==30.00,
      f'min Medium={float(medium.min()):.2f}')
health70=x.loc[x.account_id.isin(['A-89ee83','A-396e5f','A-c16101'])]
check('health-band real-account boundary 70.00',
      len(health70)==3 and (health70.health_score==70.0).all() and (health70.health_band=='Healthy').all(),
      f'{len(health70)}/3 matching source accounts')
# Support Score precedence and resolution threshold: same function as the app.
support_base=dict(tenure_days=180,has_valid_usage=True,avg_satisfaction=np.nan,
                  satisfaction_response_count=0,satisfaction_response_coverage=0,
                  tickets_90=1,high_urgent_tickets_90=0,escalated_tickets_90=0,
                  avg_resolution_time_hours_90=12.0)
for name,overrides,expected in [
 ('zero recent tickets',dict(tickets_90=0),100.0),
 ('one ordinary ticket',{},75.0),
 ('one high/urgent ticket',dict(high_urgent_tickets_90=1),50.0),
 ('resolution exactly 51.25h',dict(avg_resolution_time_hours_90=51.25),75.0),
 ('resolution 51.26h',dict(avg_resolution_time_hours_90=51.26),50.0),
 ('two high/urgent tickets',dict(tickets_90=2,high_urgent_tickets_90=2),25.0),
 ('recent escalation',dict(escalated_tickets_90=1),25.0),
]:
 row=evaluate(**(support_base|overrides))
 check('support-score pipeline boundary '+name,float(row.support_score)==expected,
       f'actual={float(row.support_score):.2f}; expected={expected:.2f}')
missing=evaluate(**support_base)
check('missing satisfaction pipeline modifier zero',
      pd.isna(missing.avg_satisfaction) and float(missing.satisfaction_modifier)==0,
      f'avg_satisfaction={missing.avg_satisfaction}; modifier={float(missing.satisfaction_modifier):.2f}')
# Save files
pd.DataFrame(checks).to_csv(out/'pipeline_checks.csv',index=False)
x.to_csv(out/'scoring_validation_results.csv',index=False)
ready.to_csv(out/'app_ready_validation_results.csv',index=False)
(out/'summary.json').write_text(json.dumps({'pass':sum(c['status']=='PASS' for c in checks),'fail':sum(c['status']=='FAIL' for c in checks),'count':len(checks),'errors':result['errors'],'source_dir':'prototype/'},indent=2))
print('\nRESULT',sum(c['status']=='PASS' for c in checks),'/',len(checks),'PASS')
