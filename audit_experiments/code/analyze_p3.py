"""Protocol 3 analysis: gap-fill vs proximity-matched extra looks (C4)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_p2 import dense, ap, signflip

HERE = Path(__file__).resolve().parents[1]
DRAWS = 5


def roll_metrics(r, p1):
    H = r['histograms']
    size = 1 + max(max(h['bins'] or [0]) for p in H.values() for h in p.values())
    A = {p: ap(*dense(H[p]['all'], size)) for p in H}
    S = {p: ap(*dense(H[p]['S'], size)) for p in H}
    dk = [f for f in r['frames'].values() if f['defect']]
    gap_fg = sum(f['gap_seen_fg'] for f in dk)
    c4_fg = np.mean([sum(f['c4_seen_fg'][d] for f in dk) for d in range(DRAWS)])
    return {'released': A['released'], 'gapfill': A['gapfill'], 'C4': float(np.mean([A[f'C4_{d}'] for d in range(DRAWS)])),
            'S_released': S['released'], 'S_gapfill': S['gapfill'], 'S_C4': float(np.mean([S[f'C4_{d}'] for d in range(DRAWS)])),
            'C1': p1['C1'], 'match_ratio': float(c4_fg / gap_fg) if gap_fg else 1.0}


def main(models):
    p2 = json.loads((HERE / 'results_p2' / 'summary.json').read_text())
    out = {}
    for model in models:
        files = sorted(p for p in (HERE / 'results_p3' / model).glob('Rollo*.json') if 'smoke' not in p.name)
        if not files:
            continue
        rolls = {f.stem: roll_metrics(json.loads(f.read_text()), p2[model]['per_roll'][f.stem]) for f in files}
        print(f'\n=== {model}: {len(rolls)} rolls; matched seen-defect pixels C4/gap-fill = '
              f"{np.mean([m['match_ratio'] for m in rolls.values()]):.3f} (min {min(m['match_ratio'] for m in rolls.values()):.3f})")
        res = {}
        for name, a, b in [('PRIMARY gapfill - C4', 'gapfill', 'C4'), ('C4 - released', 'C4', 'released'),
                           ('C4 - C1', 'C4', 'C1'), ('S: gapfill - C4', 'S_gapfill', 'S_C4'),
                           ('S: C4 - released', 'S_C4', 'S_released')]:
            d = np.array([m[a] - m[b] for m in rolls.values()])
            res[name] = {'mean': float(d.mean()), 'per_roll': d.round(5).tolist(), 'positive': int((d > 0).sum()),
                         'p': signflip(d) if len(d) > 1 else None}
            print(f"  {name:25s} {d.mean():+.5f}  pos {(d > 0).sum()}/{len(d)}  p={res[name]['p']}")
        out[model] = {'per_roll': rolls, 'contrasts': res}
    if out:
        (HERE / 'results_p3' / 'summary.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:] or ['r18_native', 'wrn50'])
