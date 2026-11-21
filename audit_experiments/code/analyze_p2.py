"""Protocol 2 analysis: gap-fill vs budget-matched controls, coverage-only hybrid, sign-flip tests."""
import itertools, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parents[1]
DRAWS = 5


def dense(h, size):
    t = np.zeros(size, np.int64); f = np.zeros(size, np.int64)
    t[h['bins']] = h['total']; f[h['bins']] = h['foreground']
    return t, f


def ap(t, f):
    t, f = t[::-1].astype(float), f[::-1].astype(float)
    keep = t > 0
    t, f = t[keep], f[keep]
    return float(np.dot(f, np.cumsum(f) / np.cumsum(t)) / f.sum())


def signflip(d):
    d = np.asarray(d); obs = abs(d.mean())
    return float(np.mean([abs((d * s).mean()) >= obs - 1e-15 for s in itertools.product((1, -1), repeat=len(d))]))


def roll_metrics(r):
    H = r['histograms']
    size = 1 + max(max(h['bins'] or [0]) for p in H.values() for h in p.values())
    A = {p: ap(*dense(H[p]['all'], size)) for p in H}
    S = {p: ap(*dense(H[p]['S'], size)) for p in H}
    rs, gs, ga = dense(H['released']['S'], size), dense(H['gapfill']['S'], size), dense(H['gapfill']['all'], size)
    hyb_t, hyb_f = rs[0] + (ga[0] - gs[0]), rs[1] + (ga[1] - gs[1])
    assert (hyb_t >= 0).all() and (hyb_f >= 0).all()
    m = {'released': A['released'], 'fixed': A['fixed'], 'gapfill': A['gapfill'], 'hybrid_cov_only': ap(hyb_t, hyb_f),
         'S_released': S['released'], 'S_gapfill': S['gapfill']}
    for c in ('C1', 'C2', 'C3'):
        m[c] = float(np.mean([A[f'{c}_{d}'] for d in range(DRAWS)]))
        m['S_' + c] = float(np.mean([S[f'{c}_{d}'] for d in range(DRAWS)]))
    fr = r['frames']; names = sorted(fr)
    y = [fr[n]['defect'] for n in names]
    for p in ('released', 'fixed', 'gapfill'):
        m['auc_' + p] = roc_auc_score(y, [fr[n]['frame_score'][p] for n in names])
    for c in ('C1', 'C3'):
        m['auc_' + c] = float(np.mean([roc_auc_score(y, [fr[n]['frame_score'][f'{c}_{d}'] for n in names]) for d in range(DRAWS)]))
    dk = [fr[n] for n in names if fr[n]['defect']]
    m['mean_k'] = float(np.mean([x['k'] for x in dk]))
    m['frames_k_pos'] = sum(x['k'] > 0 for x in dk)
    m['frames_C1_short'] = sum(x['near_shortfall'] > 0 for x in dk)
    return m


CONTRASTS = [('PRIMARY gapfill - C1 (nearby extra looks)', 'gapfill', 'C1'),
             ('gapfill - C2 (random extra looks)', 'gapfill', 'C2'),
             ('gapfill - C3 (label-free fixed+k)', 'gapfill', 'C3'),
             ('C1 - released (extra looks only)', 'C1', 'released'),
             ('hybrid cov-only - released (coverage only)', 'hybrid_cov_only', 'released'),
             ('C3 - fixed (label-free extra windows)', 'C3', 'fixed'),
             ('gapfill - released (total)', 'gapfill', 'released'),
             ('S: gapfill - released (seen pixels)', 'S_gapfill', 'S_released'),
             ('S: C1 - released (seen pixels)', 'S_C1', 'S_released'),
             ('AUROC gapfill - C1', 'auc_gapfill', 'auc_C1')]


def main(models):
    out = {}
    for model in models:
        files = sorted(p for p in (HERE / 'results_p2' / model).glob('Rollo*.json') if 'smoke' not in p.name)
        if not files:
            continue
        rolls = {f.stem: roll_metrics(json.loads(f.read_text())) for f in files}
        print(f'\n=== {model}: {len(rolls)} rolls')
        keys = ['released', 'fixed', 'gapfill', 'C1', 'C2', 'C3', 'hybrid_cov_only', 'mean_k']
        print('  means: ' + '  '.join(f'{k} {np.mean([m[k] for m in rolls.values()]):.4f}' for k in keys))
        print('  frames with k>0:', sum(m['frames_k_pos'] for m in rolls.values()),
              ' frames where C1 had too few candidates:', sum(m['frames_C1_short'] for m in rolls.values()))
        res = {}
        for name, a, b in CONTRASTS:
            d = np.array([m[a] - m[b] for m in rolls.values()])
            res[name] = {'mean': float(d.mean()), 'per_roll': d.round(5).tolist(), 'positive': int((d > 0).sum()),
                         'p': signflip(d) if len(d) > 1 else None}
            print(f"  {name:45s} {d.mean():+.5f}  pos {(d > 0).sum()}/{len(d)}  p={res[name]['p']}")
        out[model] = {'per_roll': rolls, 'contrasts': res}
    if out:
        (HERE / 'results_p2' / 'summary.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:] or ['orig_r18', 'r18_native', 'wrn50'])
