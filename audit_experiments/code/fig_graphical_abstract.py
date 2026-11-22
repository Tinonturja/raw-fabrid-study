"""Graphical abstract (Elsevier: landscape, at least 1328 x 531 px; drawn at 13 x 5 cm, 600 dpi), shared style,
drawn only from saved results. Usage: python3 fig_graphical_abstract.py"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as S

S.apply()
HERE = Path(__file__).resolve().parents[1]
R = HERE / 'results_p5'
z = np.load(R / 'qualitative' / 'coverage_examples.npz')
meta = json.loads((R / 'qualitative' / 'coverage_examples.json').read_text())
j = lambda p: json.loads((R / p).read_text())
D = {'RAW-FABRID': j('rawfabrid/full/analysis.json')['roll_level'],
     'TSFabrics': j('tsfabrics/full/analysis_from_kaggle_log.json')['roll_level'],
     'VisA': j('visa/full/analysis.json')['dataset_level'],
     'AITEX': j('aitex/full/analysis.json')['dataset_level']}
rf = j('rawfabrid/full/analysis.json')

fig = plt.figure(figsize=(13 / 2.54 * 1.6, 5 / 2.54 * 1.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 0.8], wspace=0.38, left=0.02, right=0.98, top=0.72, bottom=0.14)
fig.text(0.02, 0.97, 'How a benchmark cuts high-resolution images changes what it measures', fontsize=10.5, fontweight='bold')

# 1. coverage gap
ax = fig.add_subplot(gs[0])
k = 'maxunseen'
img, mask, uns = z[f'{k}|image'], z[f'{k}|mask'], z[f'{k}|unseen']
s = meta[k]['scale']
ax.imshow(img)
ax.imshow(np.ma.masked_where(~mask, mask), cmap=ListedColormap([S.JADE]), alpha=0.6, interpolation='nearest')
ax.imshow(np.ma.masked_where(~uns, uns), cmap=ListedColormap([S.COPPER]), alpha=0.85, interpolation='nearest')
grid = set(tuple(c) for c in z[f'{k}|grid'])
for y, x in z[f'{k}|released']:
    if (y, x) not in grid:
        ax.add_patch(Rectangle((x * s, y * s), 256 * s, 256 * s, fill=False, ec=S.COPPER, lw=1.0))
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
TITLES = []
TITLES.append((ax, '1  Released defect-centred crops miss\n    7.4% of marked defect pixels (RAW-FABRID)'))
ax.text(0.0, -0.06, f"example frame: {100 * meta[k]['unseen_px'] / meta[k]['defect_px']:.0f}% of its defect pixels (copper) unseen",
        transform=ax.transAxes, fontsize=6, color=S.INK2, va='top')

# 2. layout effect, primary detector
ax = fig.add_subplot(gs[1])
names = list(D)
ax.axvspan(-0.01, 0.01, color=S.BAND, lw=0)
ax.axvline(0, color=S.INK2, lw=0.6)
for i, n in enumerate(names):
    v = D[n]['patchcore: R2 - R1']
    ax.plot(v['ci95'], [i, i], color=S.COBALT, lw=1.3)
    ax.plot(v['estimate'], i, 'o', color=S.COBALT, ms=4.5)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6.5)
ax.invert_yaxis()
ax.set_xlim(-0.05, 0.02)
ax.set_xlabel('Tiled − native, pixel AP', fontsize=6.5)
ax.tick_params(axis='x', labelsize=6)
ax.tick_params(axis='y', length=0)
ax.grid(axis='x')
TITLES.append((ax, '2  Tiling costs a CNN detector about\n    0.01 to 0.03 AP on 3 of 4 datasets'))

# 3. detector order flips with the layout
ax = fig.add_subplot(gs[2])
for fam in ('patchcore', 'dino'):
    a1 = np.mean([u['ap'][f'{fam}|R1'] for u in rf['units'].values()])
    a2 = np.mean([u['ap'][f'{fam}|R2'] for u in rf['units'].values()])
    ax.plot([0, 1], [a1, a2], '-' + {'patchcore': 'o', 'dino': 's'}[fam], color=S.DET[fam], lw=1.5, ms=4.5)
    ax.text(1.08, a2, 'PatchCore-\nstyle' if fam == 'patchcore' else 'DINOv2', color=S.DET[fam], fontsize=6.5, va='center')
ax.set_xticks([0, 1]); ax.set_xticklabels(['Native', 'Tiled'], fontsize=6.5)
ax.set_xlim(-0.25, 1.6)
ax.set_ylim(0, 0.5)   # axis from zero so the crossing is not exaggerated
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.tick_params(axis='y', labelsize=6)
ax.set_ylabel('Pixel AP', fontsize=6.5)
ax.grid(axis='y')
TITLES.append((ax, '3  ...and can reverse which\n    detector looks better (RAW-FABRID)'))
fig.canvas.draw()
for a, t in TITLES:
    x0 = a.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).x0 if a is not TITLES[0][0] else a.get_position().x0
    fig.text(x0, 0.86, t, fontsize=7.5, fontweight='bold', va='top', ha='left')

out = HERE / 'manuscript_v2' / 'figures'
fig.savefig(out / 'graphical_abstract.png', dpi=600)
fig.savefig(out / 'graphical_abstract.pdf')
print('ok')
