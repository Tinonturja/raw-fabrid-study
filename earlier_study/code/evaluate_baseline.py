#!/usr/bin/env python3
"""Run an exploratory nearest-neighbor benchmark with grouped source images."""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors

ROOT=Path(__file__).resolve().parents[1]
FEATURES=ROOT/'results'/'features'
SEED=13017
TRAIN_PATCHES=1000
BOOTSTRAPS=500

def read_csv(p):
    with p.open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def metrics(labels,scores):
    labels=np.asarray(labels,dtype=int); scores=np.asarray(scores,float)
    if len(np.unique(labels))<2: return {'auroc':None,'average_precision':None,'n':len(labels)}
    return {'auroc':float(roc_auc_score(labels,scores)),'average_precision':float(average_precision_score(labels,scores)),
            'n':len(labels),'positive':int(labels.sum()),'negative':int((1-labels).sum())}

def source_bootstrap(labels,scores,seed=SEED):
    labels=np.asarray(labels,int); scores=np.asarray(scores,float); rng=np.random.default_rng(seed)
    pos=np.flatnonzero(labels==1); neg=np.flatnonzero(labels==0)
    if not len(pos) or not len(neg): return None
    auc=[]
    for _ in range(BOOTSTRAPS):
        idx=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
        auc.append(roc_auc_score(labels[idx],scores[idx]))
    return [float(x) for x in np.quantile(auc,[.025,.975])]

def main():
    global SEED, FEATURES
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=SEED); ap.add_argument('--output',type=Path,default=ROOT/'results'/'exploratory_baseline.json'); ap.add_argument('--features',type=Path,default=FEATURES); ap.add_argument('--skip-official',action='store_true',help='Skip patch labels, e.g. after reconstructing a new test grid'); args=ap.parse_args()
    SEED=args.seed
    FEATURES=args.features
    rows=json.loads((FEATURES/'feature_rows.json').read_text()); x=np.load(FEATURES/'features.npy'); valid=np.load(FEATURES/'valid.npy')
    if not valid.all(): raise SystemExit(f'Incomplete feature cache: {valid.sum()}/{len(valid)}')
    high=read_csv(ROOT/'data'/'remote_metadata'/'RAW_FABRID_HighRes_Metadata.csv')
    info={r['filename']:r for r in high}; roll={k:v['roll_id'] for k,v in info.items()}; defect={k:v['has_defect'].lower()=='true' for k,v in info.items()}
    x=x / np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    parents=np.array([r['parent_raw'] for r in rows]); types=np.array([r['type'] for r in rows]); rolls=np.array([roll[p] for p in parents])
    byparent=defaultdict(list)
    for i,p in enumerate(parents): byparent[p].append(i)
    rng=np.random.default_rng(SEED); all_clean=[p for p in byparent if not defect[p]]; holdout=[]
    for group in sorted(set(roll[p] for p in all_clean)):
        pool=sorted(p for p in all_clean if roll[p]==group); rng.shuffle(pool); holdout.extend(pool[:max(1,round(.2*len(pool)))])
    holdout=set(holdout)

    def run(name,train_parents,test_idx,seed):
        choices=np.flatnonzero(np.isin(parents,sorted(train_parents)) & (types=='train_good'))
        rng=np.random.default_rng(seed)
        if len(choices)<TRAIN_PATCHES: raise RuntimeError(f'{name}: too few training patches')
        train_idx=np.sort(rng.choice(choices,size=TRAIN_PATCHES,replace=False))
        model=NearestNeighbors(n_neighbors=1,metric='euclidean',algorithm='brute',n_jobs=1).fit(x[train_idx])
        scores=np.empty(len(test_idx),dtype=float)
        for st in range(0,len(test_idx),256):
            en=min(st+256,len(test_idx)); scores[st:en]=model.kneighbors(x[test_idx[st:en]],return_distance=True)[0][:,0]
        return scores,train_idx

    result={'status':'Exploratory; not a final research claim','seed':SEED,'training_patch_count_per_fold':TRAIN_PATCHES,
            'score':'nearest Euclidean distance of L2-normalized features',
            'feature_spec':json.loads((FEATURES/'feature_spec.json').read_text()),'folds':{}}
    if not args.skip_official:
        official_idx=np.flatnonzero(types!='train_good')
        official_scores,official_train_idx=run('official patch split',set(all_clean),official_idx,SEED)
        official_y=(types[official_idx]=='test_defect').astype(int)
        result['folds']['official_patch_split']={'metrics':metrics(official_y,official_scores),
           'test_unit':'patches from defective source images','train_source_count':len(all_clean),
           'test_source_count':len(set(parents[official_idx])),'train_patch_indices':official_train_idx.tolist(),
           'predictions':[{'patch':rows[int(i)]['patch_name'],'parent':parents[int(i)],'label':int(y),'score':float(s)} for i,y,s in zip(official_idx,official_y,official_scores)],
           'note':'Patch metric and clean-source image metric measure different populations and are not directly comparable.'}
        print('Official patch split',json.dumps(result['folds']['official_patch_split']['metrics']),flush=True)
    heldout_idx=np.flatnonzero(np.isin(parents,sorted(holdout)) | np.array([defect[p] for p in parents]))
    all_other=set(all_clean)-holdout
    scores,train_idx=run('clean holdout',all_other,heldout_idx,SEED)
    per=defaultdict(list)
    for i,s in zip(heldout_idx,scores): per[parents[i]].append(float(s))
    parent_names=sorted(per); y=np.array([int(defect[p]) for p in parent_names]); vals=np.array([max(per[p]) for p in parent_names])
    result['folds']['clean_image_holdout']={'metrics':metrics(y,vals),'auroc_95pct_source_bootstrap':source_bootstrap(y,vals,SEED),
       'train_source_count':len(all_other),'test_clean_sources':len(holdout),'test_defect_sources':int(sum(defect[p] for p in parent_names)),
       'aggregation':'maximum patch score per source image','split_seed':SEED,
       'train_patch_indices':train_idx.tolist(),'test_parents':parent_names,
       'predictions':[{'parent':p,'roll':roll[p],'label':int(y0),'score':float(s)} for p,y0,s in zip(parent_names,y,vals)]}
    for i,group in enumerate(sorted(set(roll.values()))):
        train=set(p for p in all_clean if roll[p]!=group); test_idx=np.flatnonzero(rolls==group)
        pred,trainidx=run(group,train,test_idx,SEED+i+1); by=defaultdict(list)
        for j,s in zip(test_idx,pred): by[parents[j]].append(float(s))
        names=sorted(by); yy=np.array([int(defect[p]) for p in names]); ss=np.array([max(by[p]) for p in names])
        result['folds'][f'held_out_{group}']={'metrics':metrics(yy,ss),'auroc_95pct_source_bootstrap':source_bootstrap(yy,ss,SEED+i+1),
          'train_source_count':len(train),'test_source_count':len(names),'aggregation':'maximum patch score per source image',
          'train_patch_indices':trainidx.tolist(),'test_parents':names,
          'predictions':[{'parent':p,'roll':roll[p],'label':int(y0),'score':float(s)} for p,y0,s in zip(names,yy,ss)]}
        print(group,json.dumps(result['folds'][f'held_out_{group}']['metrics']),flush=True)
    out=args.output; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+'\n'); print('Saved',out,flush=True)

if __name__=='__main__': main()
