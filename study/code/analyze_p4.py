"""Protocol 4 analysis: label-free targeted vs random extra windows, vs fixed; AP and frame AUROC."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_p2 import dense, ap, signflip

HERE = Path(__file__).resolve().parents[1]
DRAWS, BUDGETS = 5, (2, 7)


def roll_metrics(r):
    H = r['histograms']
    size = 1 + max(max(h['bins'] or [0]) for h in H.values())
    A = {p: ap(*dense(H[p], size)) for p in H}
    fr = r['frames']; names = sorted(fr); y = [fr[n]['defect'] for n in names]
    auc = lambda p: roc_auc_score(y, [fr[n]['frame_score'][p] for n in names])
    m = {p: A[p] for p in ('fixed', 'overlap91', 'released', 'gapfill')}
    m.update({'auc_' + p: auc(p) for p in ('fixed', 'overlap91')})
    for k in BUDGETS:
        m[f'targeted{k}'] = A[f'targeted{k}']; m[f'auc_targeted{k}'] = auc(f'targeted{k}')
        m[f'random{k}'] = float(np.mean([A[f'random{k}_{d}'] for d in range(DRAWS)]))
        m[f'auc_random{k}'] = float(np.mean([auc(f'random{k}_{d}') for d in range(DRAWS)]))
    return m


C = [('PRIMARY targeted+2 - random+2', 'targeted2', 'random2'), ('targeted+2 - fixed', 'targeted2', 'fixed'),
     ('targeted+7 - random+7', 'targeted7', 'random7'), ('targeted+7 - fixed', 'targeted7', 'fixed'),
     ('overlap91 - targeted+7', 'overlap91', 'targeted7'), ('AUROC targeted+2 - fixed', 'auc_targeted2', 'auc_fixed'),
     ('AUROC targeted+7 - fixed', 'auc_targeted7', 'auc_fixed'), ('AUROC targeted+2 - random+2', 'auc_targeted2', 'auc_random2')]


def main(models):
    out = {}
    for model in models:
        files = sorted(p for p in (HERE / 'results_p4' / model).glob('Rollo*.json') if 'smoke' not in p.name)
        if not files:
            continue
        rolls = {f.stem: roll_metrics(json.loads(f.read_text())) for f in files}
        keys = ['fixed', 'random2', 'targeted2', 'random7', 'targeted7', 'overlap91', 'released', 'gapfill']
        print(f'\n=== {model}: {len(rolls)} rolls')
        print('  AP means: ' + '  '.join(f'{k} {np.mean([m[k] for m in rolls.values()]):.4f}' for k in keys))
        res = {}
        for name, a, b in C:
            d = np.array([m[a] - m[b] for m in rolls.values()])
            res[name] = {'mean': float(d.mean()), 'per_roll': d.round(5).tolist(), 'positive': int((d > 0).sum()),
                         'p': signflip(d) if len(d) > 1 else None}
            print(f"  {name:32s} {d.mean():+.5f}  pos {(d > 0).sum()}/{len(d)}  p={res[name]['p']}")
        out[model] = {'per_roll': rolls, 'contrasts': res}
    if out:
        (HERE / 'results_p4' / 'summary.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:] or ['r18_native', 'wrn50'])
