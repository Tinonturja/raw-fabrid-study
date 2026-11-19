"""Recompute both policy comparisons from saved roll counts, without image data."""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/unseen_roll'
POLICIES = ('released', 'fixed')
METRICS = ('observed_ap', 'whole_ap', 'frame_auroc')
METHODS = ('original_resnet18', 'resnet18', 'swin_t')


def load(path):
    return json.loads(path.read_text())


def expand(hist):
    bins = np.asarray(hist['bins'], dtype=int)
    pair = np.zeros((2, 1+bins.max()), dtype=np.int64)
    pair[:, bins] = [hist['total'], hist['foreground']]
    assert np.all(pair[0] >= pair[1]) and np.all(pair >= 0)
    return pair


def add(left, right):
    width = max(left.shape[1], right.shape[1])
    return sum(np.pad(pair, ((0, 0), (0, width-pair.shape[1]))) for pair in (left, right))


def ap(pair):
    total, fg = pair
    bins = np.flatnonzero(total)
    weight = np.r_[fg[bins], total[bins]-fg[bins]]
    scores = np.tile(bins, 2)
    labels = np.repeat([1, 0], len(bins))
    checked = average_precision_score(labels, scores, sample_weight=weight)
    desc = bins[::-1]
    manual = ((fg[desc].cumsum()/total[desc].cumsum()) @ fg[desc])/fg.sum()
    np.testing.assert_allclose(checked, manual, rtol=1e-12)
    return float(checked)


def measure(frames, masks, cohort):
    selected = {p:r for p,r in frames.items() if cohort=='all' or r['primary_frames']}
    per_policy = {}
    for policy, pair in masks.items():
        assert pair[0].sum() == 1792*1024*sum(cohort=='all' or r['primary_pixels'] for r in frames.values())
        assert pair[1].sum() == sum(r['foreground'] for r in frames.values())
        assert pair[1,0] == (sum(r['unseen_foreground'] for r in frames.values()) if policy=='released' else 0)
        if policy=='fixed':
            assert pair[0,0] == 0
        per_policy[policy] = {'observed_ap': ap(pair[:, 1:]), 'whole_ap': ap(pair),
                              'frame_auroc': float(roc_auc_score([r['defect'] for r in selected.values()],
                                                  [r['scores'][policy] for r in selected.values()]))}
    return with_delta(per_policy)


def with_delta(values):
    values['difference'] = {m: values['released'][m] - values['fixed'][m] for m in METRICS}
    return values


def roll_mean(rows):
    per_roll = list(rows.values())
    summary = {p:{m:float(np.mean([r[p][m] for r in per_roll])) for m in METRICS}
               for p in (*POLICIES, 'difference')}
    deltas = summary['difference']
    dual = lambda row: row['observed_ap'] > 0 and row['whole_ap'] < 0
    summary['mean_reversal'] = dual(deltas)
    summary['individual_reversals'] = [g for g, r in rows.items() if dual(r['difference'])]
    summary['delete_one_difference'] = {
        g:{m:float(np.mean([r['difference'][m] for k,r in rows.items() if k!=g])) for m in METRICS}
        for g in rows}
    return summary


def main():
    results = ROOT / 'results'
    old = load(results/'localization_study.json')
    guided = load(results/'guided_group_histograms.json')['raw']
    fixed = load(results/'budgeted_rescan_raw.json')['group_histograms']
    folds = load(results/'baseline_seed_13017.json')['folds']
    primary_pix = set(old['defective_frames']+old['evaluation_clean'])
    primary_frames = primary_pix | set(old['calibration_clean'])
    assert len(primary_pix) == 255 and len(primary_frames) == 303
    protocol = hashlib.sha256((ROOT/'reports/unseen_roll_protocol.md').read_bytes()).hexdigest()
    manifest = hashlib.sha256((results/'baseline_seed_13017.json').read_bytes()).hexdigest()
    rows = {c: {m:{} for m in METHODS if c=='primary' or m!='original_resnet18'} for c in ('primary','all')}
    metadata = {}
    for roll in sorted(guided):
        rec = {}
        for method in METHODS[1:]:
            data = load(OUT/f'{method}_{roll}.json')
            assert data['model'] == method and data['roll'] == roll and data['references']==1000
            assert data['protocol_sha256']==protocol and data['manifest_sha256']==manifest
            frames = data['frames']
            assert set(frames) == set(folds['held_out_'+roll]['test_parents'])
            assert set(frames).isdisjoint(data['train_parents'])
            assert {p for p,r in frames.items() if r['primary_pixels']} == primary_pix & set(frames)
            assert {p for p,r in frames.items() if r['primary_frames']} == primary_frames & set(frames)
            for p,r in frames.items():
                assert r['defect'] == (p in old['defective_frames'])
                if not r['defect']:
                    assert r['scores']['fixed'] == r['scores']['released']
            rec[method] = (data['train_parents'],data['ordered_tiles_sha256'],
                           {p:{k:v for k,v in r.items() if k!='scores'} for p,r in frames.items()})
            primary = {p:expand(data['histograms']['primary_pixels'][p]) for p in POLICIES}
            full = {p:add(primary[p], expand(data['histograms']['other_clean'][p])) for p in POLICIES}
            for c,hist in [('primary',primary),('all',full)]:
                rows[c][method][roll] = measure(frames,hist,c)
        assert rec['resnet18'] == rec['swin_t']
        reference_pairs = {'released':np.array(guided[roll]),'fixed':np.array(fixed[roll]['fixed'])}
        original_frames = {p:dict(r, scores={k:old['frames'][p][k]['frame_score'] for k in POLICIES})
                           if p in primary_frames else r for p,r in frames.items()}
        rows['primary']['original_resnet18'][roll] = measure(original_frames, reference_pairs,'primary')
        metadata[roll] = {'frames':len(frames),'defects':sum(r['defect'] for r in frames.values()),
                          'primary_pixel_frames':sum(r['primary_pixels'] for r in frames.values()),
                          'primary_auc_frames':sum(r['primary_frames'] for r in frames.values()),
                          'foreground':sum(r['foreground'] for r in frames.values()),
                          'unseen_foreground':sum(r['unseen_foreground'] for r in frames.values())}
    assert [sum(v[k] for v in metadata.values()) for k in next(iter(metadata.values()))] == [
        709,202,255,303,1904926,140701]
    assert len(set.union(*(set(folds['held_out_'+g]['test_parents']) for g in metadata)))==709
    summary = {c:{m:roll_mean(groups) for m,groups in models.items()} for c,models in rows.items()}
    output = {'protocol_sha256':protocol, 'roll_populations':metadata, 'summary':summary, 'per_roll':rows}
    (OUT/'summary.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    for c,table in summary.items():
        for m,s in table.items():
            print(c,m,s['difference'],len(s['individual_reversals']),s['mean_reversal'])


if __name__=='__main__':
    main()
