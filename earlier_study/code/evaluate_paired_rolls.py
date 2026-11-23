#!/usr/bin/env python3
"""Compare within-roll and held-out-roll training on the same test sources."""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors

ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(); ap.add_argument('--features',type=Path,default=ROOT/'results'/'features'); ap.add_argument('--baseline-prefix',default='baseline_seed_'); ap.add_argument('--output',type=Path,default=ROOT/'results'/'paired_rolls_exploratory.json');args=ap.parse_args()
F=args.features
rows=json.loads((F/'feature_rows.json').read_text()); x=np.load(F/'features.npy'); valid=np.load(F/'valid.npy')
if not valid.all(): raise SystemExit('Feature cache is incomplete')
with (ROOT/'data'/'remote_metadata'/'RAW_FABRID_HighRes_Metadata.csv').open(newline='',encoding='utf-8-sig') as f:
    train_roll={r['filename']:r['roll_id'] for r in csv.DictReader(f)}
x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
parents=np.array([r['parent_raw'] for r in rows]); types=np.array([r['type'] for r in rows])
result={'purpose':'Exploratory comparison; two training pools evaluated on identical held-out source images.',
        'score':'Euclidean nearest neighbor distance between L2-normalized features',
        'train_patch_count':1000,'aggregation':'max patch score per source image',
        'limitations':'One fixed pretrained extractor and one random reference sample per seed. Source images from a roll can be temporally dependent. Pool composition differs in more ways than roll membership.',
        'seeds':{}}

def score(train_parents,test_idx,seed):
    pool=np.flatnonzero(np.isin(parents,sorted(train_parents)) & (types=='train_good'))
    idx=np.sort(np.random.default_rng(seed).choice(pool,1000,replace=False))
    model=NearestNeighbors(n_neighbors=1,metric='euclidean',algorithm='brute',n_jobs=1).fit(x[idx]); out=[]
    for k in range(0,len(test_idx),256): out.extend(model.kneighbors(x[test_idx[k:k+256]])[0][:,0])
    return np.asarray(out),idx

for seed in (13017,13018,13019):
    baseline=json.loads((ROOT/'results'/f'{args.baseline_prefix}{seed}.json').read_text())
    heldout=baseline['folds']['clean_image_holdout']['predictions']
    test_clean={r['parent'] for r in heldout if not r['label']}
    clean={p for p,t in zip(parents,types) if t=='train_good'}
    label={r['parent']:r['label'] for r in heldout}; roll={r['parent']:r['roll'] for r in heldout}
    seed_result={}
    for group in sorted(set(roll.values())):
        test_names=sorted(p for p in label if roll[p]==group); test_idx=np.flatnonzero(np.isin(parents,test_names))
        allowed=clean-test_clean; outside={p for p in allowed if train_roll[p]!=group}
        a,ai=score(allowed,test_idx,seed+37); b,bi=score(outside,test_idx,seed+73)
        ga=defaultdict(list);gb=defaultdict(list)
        for idx,va,vb in zip(test_idx,a,b): ga[parents[idx]].append(va); gb[parents[idx]].append(vb)
        y=np.array([label[p] for p in test_names]); sa=np.array([max(ga[p]) for p in test_names]); sb=np.array([max(gb[p]) for p in test_names])
        within=float(roc_auc_score(y,sa)); excluded=float(roc_auc_score(y,sb))
        seed_result[group]={'n':len(y),'positive':int(y.sum()),'negative':int((1-y).sum()),
          'within_pool_auroc':within,'heldout_roll_auroc':excluded,'difference_heldout_minus_within':excluded-within,
          'within_pool_average_precision':float(average_precision_score(y,sa)),'heldout_roll_average_precision':float(average_precision_score(y,sb)),
          'within_pool_train_patch_indices':ai.tolist(),'heldout_roll_train_patch_indices':bi.tolist(),
          'predictions':[{'parent':p,'label':int(t),'within_score':float(a0),'heldout_score':float(b0)} for p,t,a0,b0 in zip(test_names,y,sa,sb)]}
        print(seed,group,'AUROC within',round(within,4),'held-out',round(excluded,4),flush=True)
    result['seeds'][str(seed)]=seed_result
path=args.output;path.write_text(json.dumps(result,indent=2)+'\n');print('Saved',path)
