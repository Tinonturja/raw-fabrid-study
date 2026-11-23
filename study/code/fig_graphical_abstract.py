"""Graphical abstract (Elsevier: landscape, at least 1328 x 531 px; drawn at 13 x 5.8 cm, 600 dpi), drawn only from
saved results. Usage: python3 fig_graphical_abstract.py"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from scipy.ndimage import binary_dilation

HERE = Path(__file__).resolve().parents[1]
R = HERE / 'results_p5'
j = lambda p: json.loads(p.read_text())
coverage = j(HERE / 'results_cut' / 'cut_audit.json')['RAW-FABRID']['A_released_defect_aware']
raw = j(R / 'rawfabrid' / 'full' / 'analysis.json')['roll_level']
e2 = j(R / 'e2_bank_layout' / 'full' / 'analysis.json')['datasets']
z = np.load(R / 'qualitative' / 'coverage_examples.npz')

contrasts = [raw[f'{m}: R2 - R1'] for m in ['patchcore', 'dino']]
bank = {ds: (e2[ds]['cells']['D_Btile_R1']['ap'], e2[ds]['cells']['D_Bfull_R1']['ap'],
             e2[ds]['estimands']['D E_bank,full = Bfull/R1 - Btile/R1']) for ds in ['rawfabrid', 'visa']}
assert [round(c['estimate'], 3) for c in contrasts] == [-0.028, 0.207]
assert [round(bank[d][2]['estimate'], 3) for d in ['rawfabrid', 'visa']] == [0.131, 0.223]

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .5,
    'xtick.major.width': .5, 'ytick.major.width': .5})
ink, grey, pale, copper, light = '#172126', '#596168', '#D7DCDF', '#BD571C', '#B5BDC2'
fig = plt.figure(figsize=(13 / 2.54, 5.8 / 2.54), facecolor='white')
text = lambda x, y, s, **kw: fig.text(x, y, s, color=kw.pop('color', ink), **kw)

text(.025, .955, 'Hidden evaluation choices', fontsize=11.8, weight='bold', va='top')
text(.025, .845, 'Crop coverage, inference layout and memory banks change anomaly-localization scores',
     fontsize=6.9, va='top')
for x in [.341, .667]:
    fig.add_artist(Line2D([x, x], [.165, .765], transform=fig.transFigure, color=pale, lw=.7))
for x, label in [(.025, '1  CROP COVERAGE'), (.367, '2  INFERENCE LAYOUT'), (.693, '3  MEMORY BANK')]:
    text(x, .745, label, fontsize=7.1, weight='bold')

# 1. Coverage: the frame with the most uncovered defect pixels; unseen pixels dilated so they stay visible.
text(.025, .618, f"{coverage['unseen_pct']:.2f}%", fontsize=20, weight='bold', color=copper)
text(.025, .552, 'of marked defect pixels missed', fontsize=6.5)
text(.025, .498, 'by released RAW-FABRID crops', fontsize=6.5)
ax = fig.add_axes([.025, .218, .29, .247])
ax.imshow(z['maxunseen|image'])
unseen = binary_dilation(z['maxunseen|unseen'].astype(bool), iterations=4)
ax.imshow(np.ma.masked_where(~unseen, unseen), cmap=ListedColormap([copper]), interpolation='nearest')
ax.axis('off')
text(.025, .171, 'Orange: defects outside every crop', fontsize=6.0)

# 2. Layout: paired R2 - R1 contrasts on RAW-FABRID, same tile-built bank per detector.
text(.367, .666, 'RAW-FABRID; 7 held-out rolls', fontsize=6.5)
text(.367, .613, 'Same tile-built bank per detector', fontsize=6.0, color=grey)
ax = fig.add_axes([.452, .285, .195, .246])
ax.axvline(0, color=grey, lw=.6, ls=(0, (3, 3)))
for i, (c, marker) in enumerate(zip(contrasts, ['o', 's'])):
    y = 1 - i
    (lo, hi), est = c['ci95'], c['estimate']
    ax.plot([lo, hi], [y, y], color=ink, lw=1.1)
    ax.scatter(est, y, marker=marker, s=15, color=ink, zorder=3)
    ax.text(.345, y + .22, f'{est:+.3f}', ha='right', va='bottom', fontsize=7, weight='bold')
ax.set_xlim(-.065, .345); ax.set_ylim(-.5, 1.65)
ax.set_yticks([1, 0]); ax.set_yticklabels(['WRN-50', 'DINOv2'], fontsize=6.5)
ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0, pad=3)
ax.set_xticks([0, .15, .30]); ax.set_xticklabels(['0', '.15', '.30'], fontsize=6)
ax.tick_params(axis='x', length=2, pad=1.5)
text(.367, .160, 'Tiled minus native pixel AP', fontsize=6.4)
text(.367, .112, 'Points: means; lines: 95% intervals', fontsize=5.8)

# 3. Bank: exploratory E2, DINOv2 native-frame AP with a tile-built or frame-built bank (3 seeds averaged).
text(.693, .666, 'DINOv2; native-frame inference', fontsize=6.4)
text(.693, .613, 'Exploratory; paired banks, 3 seeds', fontsize=6.0, color=grey)
ax = fig.add_axes([.725, .285, .245, .27])
for g, ds in enumerate(['rawfabrid', 'visa']):
    tile, full, eff = bank[ds]
    ax.bar([g - .19, g + .19], [tile, full], width=.34, color=[light, ink], zorder=3)
    ax.text(g, max(tile, full) + .035, f"{eff['estimate']:+.2f}", ha='center', fontsize=7, weight='bold')
ax.set_xlim(-.6, 1.6); ax.set_ylim(0, .55)
ax.set_xticks([0, 1]); ax.set_xticklabels(['RAW-FABRID', 'VisA'], fontsize=6)
ax.set_yticks([0, .5]); ax.set_yticklabels(['0', '.5'], fontsize=6)
ax.tick_params(length=2, pad=1.5)
ax.set_ylabel('Pixel AP', fontsize=6, labelpad=1)
fig.text(.693, .160, '■', color=light, fontsize=7)
text(.712, .160, 'Tile-built', fontsize=6.2)
fig.text(.805, .160, '■', color=ink, fontsize=7)
text(.824, .160, 'Frame-built bank', fontsize=6.2)
text(.693, .112, 'Context not separated from position', fontsize=5.8)

fig.add_artist(Line2D([.025, .975], [.085, .085], transform=fig.transFigure, color=pale, lw=.7))
text(.025, .030, 'Report crop coverage, scoring unit, inference layout and memory-bank construction.',
     fontsize=7.0, weight='bold')

out = HERE / 'figures'
fig.savefig(out / 'graphical_abstract.png', dpi=600, facecolor='white')
fig.savefig(out / 'graphical_abstract.pdf', facecolor='white', metadata={'Creator': None, 'Producer': None})
print('graphical abstract written')
