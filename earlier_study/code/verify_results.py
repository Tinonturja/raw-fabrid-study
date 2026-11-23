"""Recompute the paper's main tables from the committed data, with Python alone."""
import argparse
import hashlib
import json
import math
from itertools import groupby, zip_longest
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
DONORS = ('donor_1', 'donor_2', 'donor_3')
SEEDS = (13017, 13018, 13019)


def read(path):
    return json.loads(path.read_text())


def auc(labels_scores):
    """Rank-probability AUROC, awarding half credit to ties."""
    pairs = sorted(labels_scores, key=lambda x: x[1])
    negatives = positives = 0
    concordance = 0.0
    for _, block in groupby(pairs, key=lambda x: x[1]):
        labels = [x[0] for x in block]
        pos = sum(labels)
        neg = len(labels) - pos
        concordance += pos * (negatives + neg / 2)
        positives += pos
        negatives += neg
    return concordance / (positives * negatives)


def average_precision(counts, foreground):
    tp = n = 0
    weighted = 0.0
    for count, hits in zip(reversed(counts), reversed(foreground)):
        assert 0 <= hits <= count
        n += count
        tp += hits
        if hits:
            weighted += hits * tp / n
    return weighted / tp


def check_near(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def verify_external():
    result = read(RESULTS / 'aitex_external.json')
    pos, clean, cal, train = [set(result[k]) for k in ('defective','evaluation_clean','calibration_clean','train_clean')]
    names = set(result['global_scores'])
    assert names == pos | clean | cal and len(names) == len(pos)+len(clean)+len(cal)
    assert names.isdisjoint(train)
    for k,v in result['spatial_metrics'].items():
        scores = {p: r[k]['frame_score'] for p,r in result['spatial_frames'].items()}
        check_near(v['auroc'], auc((p in pos,s) for p,s in scores.items()))
        check_near(result['global_auroc'][k], auc((p in pos,r[k]) for p,r in result['global_scores'].items()))
        cut = sorted(scores[p] for p in cal)[math.ceil(.95*(len(cal)-1))]
        check_near(cut, v['cutoff'])
        assert v['defects_flagged'] == sum(scores[p] > cut for p in pos)
        assert v['false_alarms'] == sum(scores[p] > cut for p in clean)
        maxima = [x for p in pos for x in result['spatial_frames'][p][k]['region_maxima']]
        assert v['regions_hit'] == sum(x > cut for x in maxima) and v['regions'] == len(maxima)
        h = result['histograms'][k]
        assert sum(h['all']) == (len(pos)+len(clean))*256*4096
        check_near(v['pixel_ap_all'], average_precision(h['all'], h['foreground']))
        check_near(v['pixel_ap_observed'], average_precision(h['all'][1:], h['foreground'][1:]))
        print('AITEX',k,'global AUROC',round(result['global_auroc'][k],4),
              'spatial AUROC',round(v['auroc'],4),'whole-frame pixel AP',round(v['pixel_ap_all'],4))


def verify_rescan(dataset, height, width, original_file):
    result = read(RESULTS / f'budgeted_rescan_{dataset}.json')
    original = read(RESULTS / original_file)
    assert result['original_coreset_indices_match'] and result['seed'] == original['seed']
    assert result['protocol_sha256'] == hashlib.sha256((ROOT/'reports/rescan_protocol.md').read_bytes()).hexdigest()
    if 'bank_sha256' in original:
        assert result['bank_sha256'] == original['bank_sha256']
    fixed = {(y,x) for y in range(0,height,256) for x in range(0,width,256)}
    dense = {(y,x) for y in range(0,height-255,128) for x in range(0,width-255,128)}
    candidates = sorted(dense - fixed)
    k = len(fixed)//4
    assert result['windows'] == dict(fixed=len(fixed), targeted=len(fixed)+k,
                                     even=len(fixed)+k, overlap=len(dense))
    assert [tuple(p) for p in result['even_extra_yx']] == [
        candidates[math.floor((i+.5)*len(candidates)/k)] for i in range(k)]
    rows = result['frames']
    cal, clean, pos = (set(result[key]) for key in ('calibration_clean','evaluation_clean','defective'))
    assert cal == set(original['calibration_clean']) and clean == set(original['evaluation_clean'])
    assert pos == set(original.get('defective', original.get('defective_frames')))
    assert set(rows) == set(result['targeted_extra_yx']) == set(result['group']) == cal | clean | pos
    assert len(rows) == len(cal)+len(clean)+len(pos)
    for name, choices in result['targeted_extra_yx'].items():
        selected = {tuple(p) for p in choices}
        assert len(choices) == len(selected) == k and selected <= dense - fixed
        scores = {p: r['frame_score'] for p,r in rows[name].items()}
        for p in ('targeted','even'):
            assert scores['fixed'] <= scores[p] <= scores['overlap']
    groups = result['group_histograms']
    assert set(groups) == set(result['group'].values())
    for policy,m in result['metrics'].items():
        scores = {p: r[policy]['frame_score'] for p,r in rows.items()}
        check_near(m['auroc'], auc((p in pos, s) for p,s in scores.items()))
        cutoff = sorted(scores[p] for p in cal)[math.ceil(.95*(len(cal)-1))]
        check_near(m['cutoff'], cutoff)
        assert m['false_alarms'] == sum(scores[p] > cutoff for p in clean)
        assert m['defects_flagged'] == sum(scores[p] > cutoff for p in pos)
        regions = [s for p in pos for s in rows[p][policy]['region_maxima']]
        assert m['regions'] == len(regions) and m['regions_hit'] == sum(s > cutoff for s in regions)
        for omitted in [None, *groups]:
            if omitted is not None and policy not in ('targeted','even'):
                continue
            included = [g[policy] for name,g in groups.items() if name != omitted]
            n = max(len(total) for total,_ in included)
            total, fg = [list(map(sum, zip_longest(*(h[i] for h in included), fillvalue=0))) for i in (0,1)]
            assert len(fg) == len(total) == n and total[0] == fg[0] == 0
            assert sum(total) == height*width*sum(result['group'][p] != omitted for p in pos | clean)
            if omitted is None:
                assert sum(fg) == sum(original['histograms']['fixed']['foreground'])
                check_near(m['pixel_ap_all'], average_precision(total, fg))
            else:
                check_near(result['omit_one_group_ap'][omitted][policy], average_precision(total, fg))
    orig = original.get('metrics', original.get('spatial_metrics'))
    for policy in ('fixed','overlap'):
        check_near(orig[policy]['auroc'], result['metrics'][policy]['auroc'])
        assert abs(orig[policy]['pixel_ap_all']-result['metrics'][policy]['pixel_ap_all']) < 0.00005
    for row in result['omit_one_group_ap'].values():
        check_near(row['difference'], row['targeted']-row['even'])
    coverage = read(RESULTS / 'rescan_coverage.json')[dataset]
    assert coverage['rescan_scores_sha256'] == hashlib.sha256(
        (RESULTS / f'budgeted_rescan_{dataset}.json').read_bytes()).hexdigest()
    counts = coverage['frames']
    assert set(counts) == pos and coverage['positive_frames'] == len(pos)
    assert coverage['total_foreground'] == sum(r['foreground'] for r in counts.values())
    assert coverage['total_foreground'] == sum(original['histograms']['fixed']['foreground'])
    assert coverage['extra_windows_on_positive_frames'] == len(pos)*k
    for policy in ('targeted', 'even'):
        summary = coverage['summary'][policy]
        revisits = [r[policy]['marked_pixels_revisited'] for r in counts.values()]
        assert all(0 <= r[policy]['marked_pixels_revisited'] <= r['foreground'] for r in counts.values())
        assert all(0 <= r[policy]['windows_touching_foreground'] <= k for r in counts.values())
        assert sum(revisits) == summary['revisited_pixels']
        assert sum(n > 0 for n in revisits) == summary['frames_with_marked_pixel_revisited']
        assert sum(r[policy]['windows_touching_foreground'] for r in counts.values()) == summary['windows_touching_foreground']
        check_near(sum(revisits)/coverage['total_foreground'], summary['fraction_of_marked_pixels'])
        check_near(mean(r[policy]['marked_pixels_revisited']/r['foreground'] for r in counts.values()),
                   summary['mean_within_frame_fraction'])
    print(f"{dataset} rescan: verified {len(rows)} frames, {len(groups)} group deletions and all four policies.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external', action='store_true', help='Also recompute the AITEX results.')
    parser.add_argument('--rescan', action='store_true', help='Check the two equal-budget follow-up runs.')
    args = parser.parse_args()
    scope = read(RESULTS / 'crop_publication_scope.json')
    cover = read(RESULTS / 'release_coverage.json')
    positive = set(scope['geometry'])
    assert cover['checked_patches'] == 19852 and not cover['grayscale_pixel_mismatches']
    assert positive == set(cover['by_parent_and_threshold']) and len(positive) == 202
    native = [r['128'] for r in cover['by_parent_and_threshold'].values()]
    missing = sum(r['unseen_foreground_pixels'] for r in native)
    foreground = sum(r['foreground_pixels'] for r in native)
    assert (missing, foreground) == (140701, 1904926)
    assert sum(r['unseen_foreground_pixels'] > 0 for r in native) == 69
    assert sorted(a for r in native for a in r['fully_missed_component_areas']) == [2,498,516,715,802,1099]
    assert scope['coordinate_control']['parent_table'] == {
        'defective': {'flagged':197,'unflagged':5}, 'clean': {'flagged':0,'unflagged':507}}
    print(f'Archive coverage: {missing:,}/{foreground:,} source-mask pixels outside the released crops.')
    print('Seed   Detector  Released   Fixed    Donor mean   Overlap   (frame AUROC)')
    for seed in SEEDS:
        global_run = scope['runs'][str(seed)]
        glob = global_run['metrics']
        for k,m in glob.items():
            check_near(m['image_auroc_303'], auc((p in positive, v[k]) for p,v in global_run['scores'].items()))
        print(f"{seed}  global    {glob['released']['image_auroc_303']:.4f}   {glob['fixed']['image_auroc_303']:.4f}   "
              f"{mean(glob[d]['image_auroc_303'] for d in DONORS):.4f}       {glob['overlap']['image_auroc_303']:.4f}")
        file = 'localization_study.json' if seed == 13017 else f'localization_study_{seed}.json'
        spatial = read(RESULTS / file)
        assert set(spatial['defective_frames']) == positive
        cal, clean = set(spatial['calibration_clean']), set(spatial['evaluation_clean'])
        assert len(cal) == 48 and len(clean) == 53 and cal.isdisjoint(clean | positive) and clean.isdisjoint(positive)
        frames = spatial['frames']
        assert set(frames) == set(global_run['scores']) == positive | cal | clean
        m = spatial['metrics']
        for k, values in m.items():
            check_near(values['auroc'], auc((p in positive, v[k]['frame_score']) for p,v in frames.items()))
            cutoff = sorted(frames[p][k]['frame_score'] for p in cal)[math.ceil(.95*(len(cal)-1))]
            check_near(values['cutoff'], cutoff)
            assert values['defects_flagged'] == sum(frames[p][k]['frame_score'] > cutoff for p in positive)
            assert values['false_alarms'] == sum(frames[p][k]['frame_score'] > cutoff for p in clean)
            regions = [v for p in positive for v in frames[p][k]['region_maxima']]
            assert values['regions'] == len(regions) == 371 and values['regions_hit'] == sum(v > cutoff for v in regions)
            counts, fg = [spatial['histograms'][k][a] for a in ('all', 'foreground')]
            assert sum(counts) == 255*1792*1024 and sum(fg) == foreground
            assert fg[0] == sum(frames[p][k]['unseen_foreground'] for p in positive)
            check_near(values['pixel_ap_all'], average_precision(counts, fg))
            check_near(values['pixel_ap_observed'], average_precision(counts[1:], fg[1:]))
        print(f"{seed}  spatial   {m['released']['auroc']:.4f}   {m['fixed']['auroc']:.4f}   "
              f"{mean(m[d]['auroc'] for d in DONORS):.4f}       {m['overlap']['auroc']:.4f}")
        print('       spatial   AP(all, observed) and region hits',
              [(k, round(m[k]['pixel_ap_all'],4), round(m[k]['pixel_ap_observed'],4), m[k]['regions_hit'])
               for k in ('released','fixed','overlap')])
    if args.external:
        verify_external()
    if args.rescan:
        verify_rescan('raw', 1024, 1792, 'localization_study.json')
        verify_rescan('aitex', 256, 4096, 'aitex_external.json')
    print('All independent arithmetic and cross-file checks passed.')


if __name__ == '__main__':
    main()
