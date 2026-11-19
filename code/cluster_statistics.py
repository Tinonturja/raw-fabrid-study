"""Paired, whole-group resampling of primary-split effects, conditional on fitted detectors."""
import json
import math
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'results'


def read(name):
    return json.loads((RESULTS/name).read_text())


def bootstrap_design(n):
    weights, frequencies = [], []
    for draws in combinations_with_replacement(range(n), n):
        counts = np.bincount(draws, minlength=n)
        frequencies.append(math.factorial(n) // math.prod(math.factorial(k) for k in counts))
        weights.append(counts)
    assert sum(frequencies) == n**n
    return np.array(weights), np.array(frequencies)


def auc_concordance(scores, positives, names_by_group):
    """Entries count concordant positive/negative pairs, with half credit for ties."""
    n = len(names_by_group)
    concordant, pairs = np.zeros((n,n)), np.zeros((n,n))
    for i, group in enumerate(names_by_group):
        pos = np.array([scores[p] for p in group if p in positives])[:,None]
        for j, other in enumerate(names_by_group):
            neg = np.array([scores[p] for p in other if p not in positives])
            pairs[i,j] = len(pos)*len(neg)
            concordant[i,j] = (pos>neg).sum() + 0.5*(pos==neg).sum()
    assert (pairs.diagonal()>0).all()
    return concordant, pairs


def weighted_auc(weights, matrices):
    wins, pairs = [np.einsum('bi,ij,bj->b',weights,m,weights) for m in matrices]
    return wins/pairs


def weighted_ap(weights, histograms, skip_unseen=False):
    # Re-pool counts before calculating AP. Chunking bounds peak memory.
    nbin = max(len(h[0]) for h in histograms)
    counts = np.zeros((2,len(histograms),nbin), dtype=np.int64)
    for j,(total,fg) in enumerate(histograms):
        counts[:,j,:len(total)] = [total,fg]
    start = 1 if skip_unseen else 0
    out = []
    for i in range(0,len(weights),64):
        all_bins, fg_bins = weights[i:i+64] @ counts[:,:,start:]
        fg = fg_bins[:,::-1]
        tp = fg.cumsum(axis=1)
        n = all_bins[:,::-1].cumsum(axis=1)
        out.extend((fg*(tp/np.maximum(n,1))).sum(axis=1)/tp[:,-1])
    return np.array(out)


def analyze(study, guided):
    budget = read(f'budgeted_rescan_{study}.json')
    if study == 'raw':
        base = read('localization_study.json')
        global_run = read('crop_publication_scope.json')['runs']['13017']
        global_scores, records, metrics = global_run['scores'], base['frames'], base['metrics']
    else:
        base = read('aitex_external.json')
        global_scores, records, metrics = base['global_scores'], base['spatial_frames'], base['spatial_metrics']
    groups = sorted(guided)
    assert set(groups)==set(budget['group_histograms'])==set(budget['group'].values())
    names = [[p for p,g in budget['group'].items() if g==group] for group in groups]
    pos = set(budget['defective'])
    assert set(global_scores)==set(records)==set(budget['group'])
    n = len(groups)
    design, freq = bootstrap_design(n)
    w = np.vstack([np.ones(n, dtype=int), 1-np.eye(n,dtype=int), design])
    effects = {}
    for label,frames in [('global',global_scores),('spatial',records)]:
        scores = {}
        for policy in ('released','fixed'):
            values = {p:(r[policy] if label=='global' else r[policy]['frame_score']) for p,r in frames.items()}
            scores[policy] = weighted_auc(w,auc_concordance(values,pos,names))
            assert np.isclose(scores[policy][0],roc_auc_score([p in pos for p in values],list(values.values())))
        effects[label+'_auc'] = scores['released']-scores['fixed']
    effects['interaction_auc'] = effects['global_auc']-effects['spatial_auc']
    fixed = weighted_ap(w,[budget['group_histograms'][g]['fixed'] for g in groups])
    assert np.isclose(fixed[0],metrics['fixed']['pixel_ap_all'])
    for key,skip in [('observed_ap',True),('whole_ap',False)]:
        a = weighted_ap(w,[guided[g] for g in groups],skip)
        assert np.isclose(a[0],metrics['released']['pixel_ap_observed' if skip else 'pixel_ap_all'])
        effects[key] = a-fixed
    rescan = {}
    for k in ('targeted','even'):
        rescan[k]=weighted_ap(w,[budget['group_histograms'][g][k] for g in groups])
        assert np.isclose(rescan[k][0],budget['metrics'][k]['pixel_ap_all'])
    effects['rescan_ap']=rescan['targeted']-rescan['even']
    for i,g in enumerate(groups,1):
        assert np.isclose(effects['rescan_ap'][i],budget['omit_one_group_ap'][g]['difference'])
    output = {}
    for name,values in effects.items():
        distribution=values[n+1:]
        order=np.argsort(distribution)
        mass=np.cumsum(freq[order])
        cuts=[distribution[order[np.searchsorted(mass,q*mass[-1])]] for q in (.025,.975)]
        output[name]={'estimate':values[0], 'interval_95':cuts,
                      'omit_group':dict(zip(groups,values[1:n+1]))}
    reversal=(effects['observed_ap']>0)&(effects['whole_ap']<0)
    return {'groups':groups, 'positive_frames':len(pos),'clean_frames_for_auc':len(records)-len(pos),
            'ordered_draws':n**n, 'unique_count_vectors':len(design),
            'effects':output, 'reversal_omit_group':dict(zip(groups,map(bool,reversal[1:n+1]))),
            'reversal_weighted_fraction':float(freq[reversal[n+1:]].sum()/freq.sum())}


def main():
    guided=read('guided_group_histograms.json')
    out={key:analyze(key,guided[key]) for key in ('raw','aitex')}
    dest=RESULTS/'cluster_statistics.json'
    dest.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    for key,data in out.items():
        print(key, 'reversal', data['reversal_weighted_fraction'])
        for name,row in data['effects'].items():
            print(name,*(round(x,5) for x in [row['estimate'],*row['interval_95']]))


if __name__=='__main__':
    main()
