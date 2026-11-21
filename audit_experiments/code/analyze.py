"""Summarize audit runs: per-roll AP (whole, observed, denominator-matched), AUROC,
calibrated hits/false alarms, exact sign-flip tests over rolls, delete-one ranges.
Needs only numpy and scikit-learn."""
import itertools, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parents[1]
ORIG = HERE.parent / 'Project files' / 'results'
POL = ('released', 'fixed', 'overlap', 'gapfill')


def ap(h):
    b, t, f = (np.asarray(h[k]) for k in ('bins', 'total', 'foreground'))
    o = np.argsort(-b)
    t, f = t[o].astype(float), f[o].astype(float)
    return float(np.dot(f, np.cumsum(f) / np.cumsum(t)) / f.sum())


def ap_observed(h):
    keep = np.asarray(h['bins']) > 0
    return ap({k: np.asarray(v)[keep] for k, v in h.items()})


def signflip(d):
    d = np.asarray(d); obs = abs(d.mean())
    m = [abs((d * s).mean()) for s in itertools.product((1, -1), repeat=len(d))]
    return float(np.mean(np.asarray(m) >= obs - 1e-15))


def roll_metrics(r, loc):
    cal, ev = set(loc['calibration_clean']), set(loc['evaluation_clean'])
    fr = r['frames']; H = r['histograms']
    m = {}
    for p in POL:
        m[f'AP_whole_{p}'] = ap(H[p]['all'])
        m[f'AP_S_{p}'] = ap(H[p]['S'])
        names = sorted(fr)
        m[f'AUROC_{p}'] = roc_auc_score([fr[n]['defect'] for n in names], [fr[n]['frame_score'][p] for n in names])
        thr = float(np.quantile([fr[n]['frame_score'][p] for n in names if n in cal], .95, method='higher'))
        regs = [v for n in names if fr[n]['defect'] for v in fr[n]['region_max'][p]]
        m[f'hits_{p}'] = sum(v is not None and v > thr for v in regs)
        m['regions'] = len(regs)
        m[f'FA_{p}'] = sum(fr[n]['frame_score'][p] > thr for n in names if n in ev)
        m['clean_eval'] = sum(n in ev for n in names)
        m[f'windows_{p}'] = float(np.mean([fr[n]['windows'][p] for n in names if fr[n]['defect']]))
    m['AP_obs_released'] = ap_observed(H['released']['all'])
    return m


def main(models):
    loc = json.loads((ORIG / 'localization_study.json').read_text())
    report = {}
    for model in models:
        files = sorted(p for p in (HERE / 'results' / model).glob('Rollo*.json') if 'smoke' not in p.name)
        if not files:
            continue
        rolls = {f.stem: roll_metrics(json.loads(f.read_text()), loc) for f in files}
        diffs = {
            'd_whole (rel-fix)': [m['AP_whole_released'] - m['AP_whole_fixed'] for m in rolls.values()],
            'd_obs_vs_fixed (rel obs - fix whole)': [m['AP_obs_released'] - m['AP_whole_fixed'] for m in rolls.values()],
            'd_placement_S (rel-fix on S)': [m['AP_S_released'] - m['AP_S_fixed'] for m in rolls.values()],
            'd_coverage (fix on S - fix whole)': [m['AP_S_fixed'] - m['AP_whole_fixed'] for m in rolls.values()],
            'd_whole (overlap-fix)': [m['AP_whole_overlap'] - m['AP_whole_fixed'] for m in rolls.values()],
            'd_whole (gapfill-fix)': [m['AP_whole_gapfill'] - m['AP_whole_fixed'] for m in rolls.values()],
            'd_whole (gapfill-rel)': [m['AP_whole_gapfill'] - m['AP_whole_released'] for m in rolls.values()],
            'd_AUROC (rel-fix)': [m['AUROC_released'] - m['AUROC_fixed'] for m in rolls.values()],
            'd_AUROC (overlap-fix)': [m['AUROC_overlap'] - m['AUROC_fixed'] for m in rolls.values()],
        }
        summ = {}
        for k, d in diffs.items():
            d = np.asarray(d)
            loo = [np.delete(d, i).mean() for i in range(len(d))] if len(d) > 1 else [d.mean()]
            summ[k] = {'mean': float(d.mean()), 'per_roll': [round(float(x), 5) for x in d],
                       'positive_rolls': int((d > 0).sum()), 'n': len(d),
                       'signflip_p': signflip(d) if len(d) > 1 else None,
                       'delete_one': [float(min(loo)), float(max(loo))]}
        means = {k: float(np.mean([m[k] for m in rolls.values()])) for k in next(iter(rolls.values()))
                 if not k.startswith(('hits', 'FA', 'regions', 'clean'))}
        totals = {k: int(sum(m[k] for m in rolls.values())) for k in next(iter(rolls.values()))
                  if k.startswith(('hits', 'FA', 'regions', 'clean'))}
        report[model] = {'rolls': sorted(rolls), 'means': means, 'totals': totals, 'differences': summ, 'per_roll': rolls}
        print(f'\n=== {model}: {len(rolls)} rolls')
        for k in ('AP_whole_released', 'AP_whole_fixed', 'AP_whole_overlap', 'AP_whole_gapfill', 'AP_obs_released',
                  'AP_S_released', 'AP_S_fixed', 'AUROC_released', 'AUROC_fixed', 'AUROC_overlap', 'AUROC_gapfill',
                  'windows_released', 'windows_gapfill', 'windows_overlap'):
            print(f'  {k:22s} {means[k]:.4f}')
        print('  totals', totals)
        for k, s in summ.items():
            print(f"  {k:40s} mean {s['mean']:+.5f}  pos {s['positive_rolls']}/{s['n']}  p={s['signflip_p']}  LOO [{s['delete_one'][0]:+.5f},{s['delete_one'][1]:+.5f}]")
    # harness check against the author's saved unseen-roll frame scores
    if (HERE / 'results' / 'orig_r18').exists():
        worst = 0.0
        for f in (HERE / 'results' / 'orig_r18').glob('Rollo*.json'):
            if 'smoke' in f.name: continue
            mine = json.loads(f.read_text())['frames']
            theirs = json.loads((ORIG / 'unseen_roll' / f'resnet18_{f.stem}.json').read_text())['frames']
            for n, r in mine.items():
                for p in ('released', 'fixed'):
                    worst = max(worst, abs(r['frame_score'][p] - theirs[n]['scores'][p]))
        print(f'\nHarness check: max |frame score difference| vs author unseen-roll ResNet-18 = {worst:.3g}')
        report['harness_max_abs_frame_diff'] = worst
    (HERE / 'results' / 'summary.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:] or ['orig_r18', 'r18_native', 'wrn50'])
