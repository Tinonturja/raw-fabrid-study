"""Descriptive E2 detector comparisons (DINOv2 minus WRN-50) under matched and P5-style banks, per dataset."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p5_core as C
d = json.loads((Path(__file__).resolve().parents[1] / 'results_p5/e2_bank_layout/full/analysis.json').read_text())
S = (13017, 13027, 13037)
pairs = {'native, matched bank (Bfull/R1)': ('Bfull', 'R1'), 'tiled, matched bank (Btile/R2)': ('Btile', 'R2'),
         'native, tile bank (Btile/R1, P5 pairing)': ('Btile', 'R1')}
out = {}
for ds, r in d['datasets'].items():
    for lab, (b, l) in pairs.items():
        v = np.array([np.mean([r['ap'][u][f'D_{b}_{l}_s{s}'] - r['ap'][u][f'W_{b}_{l}_s{s}'] for s in S]) for u in r['units']])
        lo, hi = C.percentile_ci(C.roll_bootstrap(v))
        out[f'{ds} {lab}'] = {'estimate': float(v.mean()), 'ci95_unit_bootstrap': [lo, hi], 'units_positive': int((v > 0).sum()),
                              'n_units': len(v), 'signflip_p': C.signflip_p(v) if len(v) <= 16 else None}
        print(f'{ds:9s} DINO - WRN, {lab:42s} {v.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  {(v > 0).sum()}/{len(v)}')
(Path(__file__).resolve().parents[1] / 'results_p5/e2_bank_layout/full/rankings.json').write_text(json.dumps(out, indent=1))
