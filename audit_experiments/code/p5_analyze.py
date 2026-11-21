"""Protocol 5 analysis (Amendment 1, A5 and A6). NumPy only.

Reads results_p5/<dataset>/<stage>/<unit>.{npz,json}; writes results_p5/<dataset>/<stage>/analysis.json
and prints a summary. Primary scorer: patchcore. Primary endpoint: AP(R2) - AP(R1), unique pixels.
Usage: python p5_analyze.py --dataset aitex|visa [--stage full|smoke]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p5_core as C

HERE = Path(__file__).resolve().parents[1]
SCORERS = ('patchcore', 'gauss', 'dino')
REPS = 2000
MIN_EFFECT = 0.01


def load_unit(npz, js):
    z = np.load(npz)
    meta = json.loads(Path(js).read_text())
    hists = {}
    for key in {k.rsplit('|', 1)[0] for k in z.files}:
        b, t, p, off = (z[f'{key}|{x}'] for x in ('bins', 'tot', 'pos', 'off'))
        hists[tuple(key.split('|'))] = [(b[off[i]:off[i + 1]], t[off[i]:off[i + 1]], p[off[i]:off[i + 1]])
                                       for i in range(len(off) - 1)]
    return meta, hists


def matrices(hists, keys):
    """Dense matrices on one shared bin axis, so AP differences compare like with like."""
    nb = max(int(h[0].max()) for k in keys for h in hists[k] if len(h[0])) + 1
    return {k: C.dense(hists[k], nb) for k in keys}


def exact_ap(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    if y.sum() == 0:
        return float('nan')
    o = np.argsort(-s, kind='mergesort')
    y = y[o].astype(float)
    return float((y * np.cumsum(y) / np.arange(1, len(y) + 1)).sum() / y.sum())


def summarise(dist, point):
    lo, hi = C.percentile_ci(dist)
    return {'estimate': point, 'ci95': [lo, hi], 'excludes_zero': bool(lo > 0 or hi < 0),
            'meets_min_point_estimate': bool(abs(point) >= MIN_EFFECT),
            'whole_interval_beyond_0.01': bool(lo > MIN_EFFECT or hi < -MIN_EFFECT)}


def unit_tables(meta, hists, W):
    """Per unit: APs, paired differences (bootstrap with shared weights W), secondary metrics."""
    scorers = [s for s in SCORERS if (s, 'R2') in hists]
    out = {'n_images': len(meta['images']), 'n_defective': sum(r['defect'] for r in meta['images'])}
    keys = [(s, l) for s in scorers for l in ('R1', 'R2', 'R2occ', 'A_all', 'A_obs', 'C4') if (s, l) in hists
            and all(len(h[0]) for h in hists[(s, l)])]
    mats = matrices(hists, keys)
    ap = {f'{s}|{l}': C.ap_from_hist(mats[(s, l)][0].sum(0), mats[(s, l)][1].sum(0)) for s, l in keys}
    out['ap'] = ap
    diffs = {}

    def diff(name, a, b):
        if a in mats and b in mats:
            d = C.boot_ap_diff(*mats[a], *mats[b], W)
            diffs[name] = (d, ap[f'{a[0]}|{a[1]}'] - ap[f'{b[0]}|{b[1]}'])
    for s in scorers:
        diff(f'{s}: R2 - R1', (s, 'R2'), (s, 'R1'))
        diff(f'{s}: R2occ - R2 (occurrence vs unique)', (s, 'R2occ'), (s, 'R2'))
        diff(f'{s}: A_obs - A_all (observed vs whole frame)', (s, 'A_obs'), (s, 'A_all'))
        diff(f'{s}: A_all - R2 (practice A vs grid)', (s, 'A_all'), (s, 'R2'))
        diff(f'{s}: C4 - R1 (downscaled vs native)', (s, 'C4'), (s, 'R1'))
    for lay in ('R1', 'R2'):
        for i, a in enumerate(scorers):
            for b in scorers[i + 1:]:
                diff(f'ranking {lay}: {a} - {b}', (a, lay), (b, lay))
    # patch- and parent-level detection, in their own units (never subtracted from pixel AP)
    det = {}
    for s in scorers:
        tiles = [(t[s], t['defect_px']) for r in meta['images'] for t in r['tiles'] if s in t]
        sc = np.array([t[0] for t in tiles]); px = np.array([t[1] for t in tiles])
        det[s] = {'patch_auroc_1px': C.auroc(sc, px >= 1), 'patch_ap_1px': exact_ap(sc, px >= 1),
                  'patch_auroc_1pct': C.auroc(sc[(px == 0) | (px >= 655)], px[(px == 0) | (px >= 655)] >= 655),
                  'n_tiles': len(tiles), 'n_positive_tiles_1px': int((px >= 1).sum())}
        y = [r['defect'] for r in meta['images']]
        for lay in ('R1', 'R2', 'C4'):
            k = f'{s}|{lay}'
            if all(k in r['frame_max'] for r in meta['images']):
                det[s][f'parent_auroc_{lay}'] = C.auroc([r['frame_max'][k] for r in meta['images']], y)
    out['detection'] = det
    out['r1_failures'] = len(meta.get('r1_failures', []))
    return out, diffs


def merge(parts):
    """Combine output files of one unit written by different feature-family runs."""
    meta, hists = parts[0]
    ids = [r['id'] for r in meta['images']]
    for m2, h2 in parts[1:]:
        assert [r['id'] for r in m2['images']] == ids, 'family runs cover different images'
        for r, r2 in zip(meta['images'], m2['images']):
            r['frame_max'].update(r2['frame_max'])
            for t, t2 in zip(r['tiles'], r2['tiles']):
                assert (t['y'], t['x']) == (t2['y'], t2['x'])
                t.update({k: v for k, v in t2.items() if k not in ('y', 'x', 'defect_px')})
        meta['r1_failures'] = meta.get('r1_failures', []) + m2.get('r1_failures', [])
        meta['banks'] = {**meta.get('banks', {}), **m2.get('banks', {})}
        hists.update(h2)
    return meta, hists


def main(a):
    d = HERE / 'results_p5' / a.dataset / a.stage
    files = sorted(p for p in d.glob('*.json') if p.stem != 'analysis')
    groups = {}
    for p in files:
        groups.setdefault(p.stem.split('__')[0], []).append(p)
    units = sorted(groups)
    if not units:
        sys.exit(f'no results in {d}')
    report = {'dataset': a.dataset, 'stage': a.stage, 'units': {}, 'reps': REPS, 'seed': C.SEED}
    unit_diffs = {}
    for u in units:
        meta, hists = merge([load_unit(p.with_suffix('.npz'), p) for p in groups[u]])
        W = C.bootstrap_counts(len(meta['images']), reps=REPS, seed=C.SEED)   # shared by every comparison
        tab, diffs = unit_tables(meta, hists, W)
        tab['differences'] = {k: summarise(v[0], v[1]) for k, v in diffs.items()}
        report['units'][u] = tab
        unit_diffs[u] = diffs
    # dataset level: AITEX is one unit; VisA = macro mean over categories with per-category draws
    names = set.intersection(*[set(v) for v in unit_diffs.values()])
    report['dataset_level'] = {}
    for k in sorted(names):
        dist = np.nanmean(np.stack([unit_diffs[u][k][0] for u in units]), 0)
        point = float(np.mean([unit_diffs[u][k][1] for u in units]))
        report['dataset_level'][k] = summarise(dist, point)
    if a.dataset == 'rawfabrid':
        # Amendment 1, A5: primary interval = paired bootstrap of the 7 roll differences (equal weights),
        # reported with the exact sign-flip test; the image bootstrap above is secondary (conditional).
        report['roll_level'] = {}
        for k in sorted(names):
            rd = np.array([unit_diffs[u][k][1] for u in units])
            rb = C.roll_bootstrap(rd, reps=REPS, seed=C.SEED)
            rec = summarise(rb, float(rd.mean()))
            rec.update(per_roll=dict(zip(units, rd.tolist())), rolls_positive=int((rd > 0).sum()),
                       signflip_p=C.signflip_p(rd))
            report['roll_level'][k] = rec
        prim = report['roll_level'].get('patchcore: R2 - R1')
    else:
        prim = report['dataset_level'].get('patchcore: R2 - R1')
    report['primary'] = prim
    out = d / 'analysis.json'
    out.write_text(json.dumps(report, indent=1))
    print(f'{a.dataset} ({a.stage}), {len(units)} unit(s)')
    for k, v in report['dataset_level'].items():
        print(f'  {k:55s} {v["estimate"]:+.4f}  [{v["ci95"][0]:+.4f}, {v["ci95"][1]:+.4f}]')
    for u, t in report['units'].items():
        print(f'  {u}: AP', {k: round(v, 4) for k, v in t['ap'].items()})
    print('wrote', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--stage', default='full', choices=('full', 'smoke'))
    main(ap.parse_args())
