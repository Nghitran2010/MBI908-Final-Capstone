import sys,pathlib,numpy as np,pandas as pd
BASE=pathlib.Path(__file__).resolve().parents[2]/'prototype';OUT=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(BASE));import pipeline as p
pre=pd.read_csv(OUT.parent/'01_processing_and_scoring/scoring_validation_results.csv')
original=pre.set_index('account_id'); stored=pd.read_csv(OUT/'sensitivity_baseline_results.csv')

def compute(config):
 x=pre.copy()
 rec=config.get('rec',(14,30,60)); act=config.get('act',(4,8)); resolution=config.get('res',p.RESOLUTION_P75_HOURS)
 v=x.days_since_last_usage.fillna(999999)
 x['recency_new']=np.select([v<=rec[0],v<=rec[1],v<=rec[2]],[100.,65.,30.],0.)
 v=x.active_days_90_per_90_exposure.fillna(0)
 x['activity_new']=np.select([v<act[0],v<act[1]],[25.,60.],100.)
 x['engagement_new']=(.6*x.recency_new+.4*x.activity_new).round(2)
 slow=x.avg_resolution_time_hours_90.gt(resolution)
 x['support_new']=np.select([x.tickets_90.le(0),x.escalated_tickets_90.ge(1)|x.high_urgent_tickets_90.ge(2),x.high_urgent_tickets_90.ge(1)|slow],[100.,25.,50.],75.)
 x['urgency_new']=np.select([x.escalated_tickets_90.ge(1),x.high_urgent_tickets_90.ge(2),x.high_urgent_tickets_90.ge(1)|slow,x.tickets_90.ge(1)],[100.,85.,60.,30.],0.)
 hw=config.get('hw',.6)
 x['health_new']=(hw*x.engagement_new+(1-hw)*x.support_new+x.satisfaction_modifier).clip(0,100).round(2)
 exposure=config.get('exposure',30)
 x['evidence_new']=np.where(x.tenure_days<exposure,'Insufficient Exposure',np.where(~x.has_valid_usage.astype(bool),'Verify Data','Scoreable'))
 x['health_band_new']=np.where(x.evidence_new!='Scoreable','Limited Evidence',np.select([x.health_new<config.get('hb',(50,70))[0],x.health_new<config.get('hb',(50,70))[1]],['Low','Medium'],'Healthy'))
 mrrquant=config.get('cq',(0.5,.75,.9));q=x.mrr_amount.quantile(list(mrrquant)).to_numpy();m=x.mrr_amount
 x['commercial_new']=np.select([m.isna()|m.le(0),m.le(q[0]),m.le(q[1]),m.le(q[2])],[0.,25.,50.,75.],100.)
 x['hc_new']=np.where(x.evidence_new=='Scoreable',100-x.health_new,50.).round(2)
 x['ec_new']=np.where(x.evidence_new=='Scoreable',0.,100.)
 w=np.array([.5,.25,.15,.1]);w_idx=config.get('w_idx');
 if w_idx is not None:
  w[w_idx]*=config.get('factor',1.)
  if config.get('renorm',False):w=w/w.sum()
 x['priority_new']=(w[0]*x.hc_new+w[1]*x.urgency_new+w[2]*x.commercial_new+w[3]*x.ec_new).round(2)
 lo,hi=config.get('pb',(30,50))
 x['priority_band_new']=np.select([x.priority_new>=hi,x.priority_new>=lo],['High','Medium'],'Low')
 x=x.sort_values(['priority_new','account_id'],ascending=[False,True])
 return x

# mapped 24 rows as in original CSV
configs=[]
for _,row in stored.iterrows():
 area,test,variant=row['area'],row['test'],row['variant'];d={}
 if area=='Health':
  if test=='Domain weight':d={'hw':float(variant.split()[1])}
  if test=='Recency thresholds':d={'rec':tuple(map(int,variant.strip('()').split(',')))}
  if test=='Activity thresholds':d={'act':tuple(map(int,variant.strip('()').split(',')))}
  if test=='Resolution threshold':d={'res':float(variant.split('=')[1].split()[0])}
  if test=='Health band boundaries':d={'hb':tuple(map(int,variant.strip('()').split(',')))}
  if test=='Exposure minimum':d={'exposure':14}
 else:
  if test=='Component weight':
   label=variant.split(' x')[0];d={'w_idx':{'Health concern':0,'Recent support urgency':1,'Commercial exposure':2,'Evidence concern':3}[label],'factor':float(variant.rsplit('x',1)[1])}
  if test=='Commercial thresholds':d={'cq':tuple(map(float,variant.strip('()').split(',')))}
  if test=='Priority band boundaries':d={'pb':tuple(map(int,variant.strip('()').split(',')))}
  if test=='Exposure minimum':d={'exposure':14}
 configs.append(d)

base_top=set(pre.nsmallest(10,'priority_rank').account_id)
rows=[]
for (_,s),d in zip(stored.iterrows(),configs):
 variants=[]
 for renorm in ([True, False] if 'w_idx' in d else [False]):
  y=compute({**d,'renorm':renorm})
  bfield='health_band' if s.area=='Health' else 'priority_band'
  retention=round(100*(y.set_index('account_id')[bfield+'_new'].sort_index()==original[bfield].sort_index()).mean(),1)
  overlap=len(base_top.intersection(y.head(10).account_id)) if s.area=='Priority' else np.nan
  variants.append((renorm,retention,overlap))
 nearest=variants[0]
 rows.append(dict(area=s.area,test=s.test,variant=s.variant,stored_retention=s.band_retention_pct,recalc_retention=nearest[1],stored_top10=s.top10_overlap,recalc_top10=nearest[2],retention_match=bool(abs(nearest[1]-s.band_retention_pct)<.001),top10_match=bool(pd.isna(s.top10_overlap) or nearest[2]==s.top10_overlap),renormalised=nearest[0],unadjusted_retention=next((v[1] for v in variants if not v[0]),np.nan),unadjusted_top10=next((v[2] for v in variants if not v[0]),np.nan)))
out=pd.DataFrame(rows);out.to_csv(OUT/'sensitivity_recalculation_comparison.csv',index=False)
print(out[['area','test','variant','stored_retention','recalc_retention','stored_top10','recalc_top10','retention_match','top10_match','renormalised','unadjusted_retention']].to_string(index=False));print('MATCH',out.retention_match.sum(),out.top10_match.sum(),'/',len(out))
