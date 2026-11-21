"""Run 003: paired stale-scaler deletion audit, CPU-bounded research battery.
No credentials are read, printed, or saved by this file. TABPFN_TOKEN is inherited.
"""
import os
os.environ.setdefault('TABPFN_NO_BROWSER', '1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import json, time, hashlib, datetime, platform, gc, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import scipy, sklearn, torch, tabpfn
from scipy import stats
from sklearn.datasets import load_iris, fetch_openml, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tabpfn import TabPFNClassifier

torch.set_num_threads(2)
OUT=Path(os.environ.get('AUDIT_OUTPUT','/content/run003')); OUT.mkdir(parents=True,exist_ok=True)
SEEDS=[42,123,7]; TRIALS=50; ENSEMBLE=4; MAX_ROWS=256

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def writej(name,obj):
 p=OUT/name; tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2)); tmp.replace(p)
def load_data(name):
 if name=='iris':
  d=load_iris(); X=d.data; y=d.target; features=d.feature_names; source='sklearn.datasets.load_iris'
 elif name=='adult_numeric':
  d=fetch_openml(name='adult',version=2,as_frame=True)
  frame=d.data.select_dtypes(include='number'); features=list(frame.columns)
  valid=frame.notna().all(axis=1)&d.target.notna(); X=frame.loc[valid].to_numpy(); y=LabelEncoder().fit_transform(d.target.loc[valid]); source=d.details
 elif name=='covertype':
  d=fetch_covtype(); X=d.data; y=LabelEncoder().fit_transform(d.target); features=list(d.feature_names); source='sklearn.datasets.fetch_covtype, shuffle=False'
 else: raise ValueError(name)
 return np.asarray(X,dtype=np.float64),np.asarray(y,dtype=np.int64),features,source

def setup(name,seed,data):
 X,y,features,source=data; ids=np.arange(len(y))
 if len(ids)>MAX_ROWS:
  # Stratified bounded sample with at least four examples per class so the
  # subsequent 75/25 split retains rare Covertype classes in both sets.
  rng_sample=np.random.default_rng(seed+9000); classes,counts=np.unique(y,return_counts=True)
  allocation=np.maximum(4,np.floor(counts/counts.sum()*MAX_ROWS).astype(int))
  while allocation.sum()>MAX_ROWS:
   j=int(np.argmax(allocation));allocation[j]-=1
  while allocation.sum()<MAX_ROWS:
   j=int(np.argmax(counts/counts.sum()*MAX_ROWS-allocation));allocation[j]+=1
  ids=np.concatenate([rng_sample.choice(np.flatnonzero(y==cl),size=int(n),replace=False) for cl,n in zip(classes,allocation)])
  rng_sample.shuffle(ids)
 tr,te=train_test_split(ids,test_size=.25,stratify=y[ids],random_state=seed)
 a,b=X[tr],X[te]; ya=y[tr]; stale=StandardScaler().fit(a)
 blind=np.random.default_rng(seed+1000).permutation(ya)
 rng=np.random.default_rng(seed); singles=rng.choice(len(tr),size=TRIALS,replace=False)
 k=max(1,int(np.ceil(.05*len(tr))))
 # Preserve every class in retained context, both true and shuffled labels.
 groups=[]; seen=set()
 while len(groups)<TRIALS:
  g=np.sort(rng.choice(len(tr),size=k,replace=False)); mask=np.ones(len(tr),bool); mask[g]=False
  if len(np.unique(ya[mask]))!=len(np.unique(ya)) or len(np.unique(blind[mask]))!=len(np.unique(blind)): continue
  key=tuple(g)
  if key in seen:continue
  seen.add(key); groups.append(g.tolist())
 manifest={'dataset':name,'seed':seed,'source':source,'features':features,'full_n':len(y),'sample_n':len(ids),'train_n':len(tr),'test_n':len(te),'class_counts_train':np.bincount(ya).tolist(),'sample_ids':ids.tolist(),'train_source_ids':tr.tolist(),'test_source_ids':te.tolist(),'sample_X_sha256':sha(X[ids]),'sample_y_sha256':sha(y[ids]),'train_X_sha256':sha(a),'train_y_sha256':sha(ya),'test_X_sha256':sha(b),'test_y_sha256':sha(y[te]),'shuffled_y_sha256':sha(blind),'single_deletions_local':singles.tolist(),'group_deletions_local':groups,'group_n':k,'group_actual_fraction':k/len(tr)}
 writej(f'{name}_{seed}_manifest.json',manifest)
 return dict(name=name,seed=seed,X=a,T=b,y=ya,yt=y[te],blind=blind,stale=stale,singles=singles,groups=groups,sourceids=tr)

def predict(X,y,T,seed):
 np.random.seed(seed); torch.manual_seed(seed)
 start=time.perf_counter()
 m=TabPFNClassifier(n_estimators=ENSEMBLE,device='cpu',random_state=seed,fit_mode='fit_preprocessors')
 m.fit(X,y); fit=time.perf_counter()-start
 p=m.predict_proba(T); total=time.perf_counter()-start
 del m; gc.collect()
 return np.asarray(p),fit,total

def metrics(p,q):
 return {'mad':float(np.max(np.abs(p-q))),'mean_l2':float(np.linalg.norm(p-q,axis=1).mean()),'mean_abs':float(np.abs(p-q).mean()),'label_disagreement':float(np.mean(p.argmax(axis=1)!=q.argmax(axis=1)))}

def trial(c,kind,i):
 d=[int(c['singles'][i])] if kind=='single' else c['groups'][i]
 mask=np.ones(len(c['y']),bool);mask[d]=False
 X=c['X'][mask]; start_scaler=time.perf_counter(); fresh=StandardScaler().fit(X); scaler_fit_s=time.perf_counter()-start_scaler
 inputs={}; transform_s={}
 for mode,sc in [('A',c['stale']),('B',fresh)]:
  t=time.perf_counter();inputs[mode]=(sc.transform(X),sc.transform(c['T']));transform_s[mode]=time.perf_counter()-t
 record={'dataset':c['name'],'seed':c['seed'],'kind':kind,'trial':i,'deleted_local':d,'deleted_source':c['sourceids'][d].tolist(),'deleted_n':len(d),'retained_n':int(mask.sum()),'timestamp_utc':now(),'scaler_mean_delta_max':float(np.max(np.abs(c['stale'].mean_-fresh.mean_)))}
 # Counterbalance execution order to reduce systematic timing order bias.
 order=['A','B'] if i%2==0 else ['B','A']
 for label,labels in [('real',c['y']),('blind',c['blind'])]:
  ps={}; times={}
  for mode in order:
   ps[mode],tf,tt=predict(inputs[mode][0],labels[mask],inputs[mode][1],c['seed']);times[mode]={'fit_s':tf,'fit_predict_s':tt,'pipeline_s':tt+transform_s[mode]+(scaler_fit_s if mode=='B' else 0.)}
  record[label]=dict(metrics(ps['A'],ps['B']),times=times,accuracy_A=float(np.mean(ps['A'].argmax(1)==c['yt'])),accuracy_B=float(np.mean(ps['B'].argmax(1)==c['yt'])))
  if label=='real' and i==0:
   pr,tf,tt=predict(inputs['B'][0],labels[mask],inputs['B'][1],c['seed'])
   record['identical_input_repeat']=dict(metrics(ps['B'],pr),fit_s=tf,fit_predict_s=tt)
 record['excess_mad_real_minus_blind']=record['real']['mad']-record['blind']['mad']
 return record

def ci(x,rng):
 x=np.asarray(x); n=len(x)
 if n<2:return [float(x.mean()),float(x.mean())]
 b=x[rng.integers(0,n,size=(3000,n))].mean(1)
 return np.quantile(b,[.025,.975]).tolist()
def summarize(rows):
 groups={}
 for r in rows:groups.setdefault((r['dataset'],r['seed'],r['kind']),[]).append(r)
 summaries=[]
 for (name,seed,kind),rr in sorted(groups.items()):
  rng=np.random.default_rng(seed+555)
  real=np.array([r['real']['mad'] for r in rr]);blind=np.array([r['blind']['mad'] for r in rr]);delta=real-blind
  nz=delta[delta!=0]
  if len(nz):
   w=stats.wilcoxon(delta,alternative='two-sided',zero_method='wilcox',method='auto'); ranks=stats.rankdata(abs(nz)); rb=float(np.sum(np.sign(nz)*ranks)/ranks.sum()); p=float(w.pvalue);stat=float(w.statistic)
  else:p=1.;stat=0.;rb=0.
  timing=np.array([r['real']['times']['B']['pipeline_s']-r['real']['times']['A']['pipeline_s'] for r in rr])
  summaries.append({'dataset':name,'seed':seed,'kind':kind,'n':len(rr),'complete':len(rr)==TRIALS,'real_mad_mean':float(real.mean()),'real_mad_ci95':ci(real,rng),'blind_mad_mean':float(blind.mean()),'blind_mad_ci95':ci(blind,rng),'excess_mad_mean':float(delta.mean()),'excess_mad_ci95':ci(delta,rng),'excess_mad_quantiles':np.quantile(delta,[0,.25,.5,.75,1]).tolist(),'real_mad_quantiles':np.quantile(real,[0,.25,.5,.75,1]).tolist(),'wilcoxon_two_sided_p':p,'wilcoxon_stat':stat,'paired_rank_biserial':rb,'paired_standardized_mean_difference':float(delta.mean()/delta.std(ddof=1)) if len(delta)>1 and delta.std(ddof=1)>0 else None,'clean_minus_stale_total_s_mean':float(timing.mean()),'clean_minus_stale_total_s_ci95':ci(timing,rng),'mean_l2':float(np.mean([r['real']['mean_l2'] for r in rr])),'mean_label_disagreement':float(np.mean([r['real']['label_disagreement'] for r in rr]))})
 # Holm across all 18 planned cells; unobserved hypotheses conservatively p=1.
 ix=sorted(range(len(summaries)),key=lambda j:summaries[j]['wilcoxon_two_sided_p']);running=0.
 for rank,j in enumerate(ix):
  running=max(running,min(1.,(18-rank)*summaries[j]['wilcoxon_two_sided_p']));summaries[j]['holm_p_18_planned']=running
 writej('summary.json',{'timestamp_utc':now(),'planned_cells':18,'planned_trials':900,'completed_trials':len(rows),'cells':summaries,'interpretation':['Shuffled-label baseline is a negative control, not a numerical-noise floor.','Excess MAD compares label-dependent sensitivity, not proof of privacy loss.','Bootstrap intervals and Wilcoxon tests are conditional on fixed train/test splits.','Deletion trials share training and test data, so population-level independence is not claimed.','Three seeds are separate replications, not 150 independent datasets.','Bounded 256-row subsamples; Adult uses numeric features only.','No internal memorization or certified unlearning claim.','Identical-input repeat checks exist for first trial in each cell only.']})
 return summaries

def main():
 meta={'run':'003','start_utc':now(),'python':platform.python_version(),'platform':platform.platform(),'versions':{m.__name__:m.__version__ for m in [np,pd,scipy,sklearn,torch,tabpfn]},'device':'cpu','cpu_threads':2,'n_estimators':ENSEMBLE,'fit_mode':'fit_preprocessors','seeds':SEEDS,'trials_per_cell':TRIALS,'maximum_dataset_rows':MAX_ROWS,'test_fraction':.25,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 writej('environment.json',meta)
 configs=[]
 for name in ['iris','adult_numeric','covertype']:
  print('LOADING',name,flush=True);data=load_data(name)
  for seed in SEEDS:configs.append(setup(name,seed,data))
  del data;gc.collect()
 for c in configs:
  print('WARMUP',c['name'],c['seed'],flush=True)
  predict(c['stale'].transform(c['X']),c['y'],c['stale'].transform(c['T']),c['seed'])
 rows=[]; path=OUT/'trials.jsonl'
 if path.exists():rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
 done={(r['dataset'],r['seed'],r['kind'],r['trial']) for r in rows}
 # Breadth-first blocks of ten, keeping single and group cells balanced.
 for block in range(0,TRIALS,10):
  for c in configs:
   for kind in ['single','group5pct']:
    for i in range(block,min(block+10,TRIALS)):
     key=(c['name'],c['seed'],kind,i)
     if key in done:continue
     print('START',key,now(),flush=True)
     r=trial(c,kind,i)
     with path.open('a') as f:f.write(json.dumps(r)+'\n');f.flush();os.fsync(f.fileno())
     rows.append(r);done.add(key);summarize(rows)
     print('DONE',len(rows),'of 900',key,'real',r['real']['mad'],'blind',r['blind']['mad'],flush=True)
 writej('COMPLETE.json',{'completed_utc':now(),'trials':len(rows),'expected':900})
 print('BATTERY_COMPLETE',len(rows),now(),flush=True)

if __name__=='__main__':
 try:main()
 except Exception:
  # Avoid environment dumping. Trace contains code locations, no credentials.
  traceback.print_exc();writej('FAILED.json',{'timestamp_utc':now(),'reason':'See run.log traceback'});raise
