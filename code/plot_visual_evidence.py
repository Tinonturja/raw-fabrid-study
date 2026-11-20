"""Plot reproducible source-mask examples and per-roll mask coverage."""
import io
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'manuscript/figures'
population = json.loads((ROOT/'results/unseen_roll/summary.json').read_text())['roll_populations']
records = {}
for roll in ('Rollo1A', 'Rollo4A', 'Rollo7A'):
    data = json.loads((ROOT/f'results/unseen_roll/resnet18_{roll}.json').read_text())
    candidates = sorted((r['foreground'], name) for name, r in data['frames'].items() if r['defect'])
    records[roll] = candidates[len(candidates)//2][1]

fig, axes = plt.subplots(3, 2, figsize=(6.5, 8.1))
with zipfile.ZipFile(ROOT/'data/RAW_FABRID.zip') as archive:
    for row, (roll, name) in enumerate(records.items()):
        def load(folder):
            with Image.open(io.BytesIO(archive.read(f'RAW_FABRID/{folder}/{name}'))) as im:
                return np.asarray(im.convert('L'))
        image, mask = load('images'), load('masks') >= 128
        yy, xx = np.nonzero(mask)
        center_y, center_x = int(np.median(yy)), int(np.median(xx))
        y0 = min(max(center_y-128, 0), image.shape[0]-256)
        x0 = min(max(center_x-128, 0), image.shape[1]-256)
        tile = image[y0:y0+256, x0:x0+256]
        selected_mask = mask[y0:y0+256, x0:x0+256]
        for column in range(2):
            ax = axes[row, column]
            ax.imshow(tile, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            if column:
                ax.contour(selected_mask, levels=[.5], colors=['#e8b52c'], linewidths=1.1)
            ax.set_axis_off()
        axes[row, 0].set_ylabel(roll.replace('Rollo','Roll '), fontsize=9)
        axes[row, 0].text(-.05, .5, roll.replace('Rollo','Roll '), transform=axes[row, 0].transAxes,
                          va='center', ha='right', fontsize=10, rotation=90)
        print(roll, name, 'crop y,x', y0, x0)
axes[0, 0].set_title('Source crop', fontsize=11, weight='bold')
axes[0, 1].set_title('Source mask outline', fontsize=11, weight='bold')
fig.subplots_adjust(left=.12, right=.98, top=.95, bottom=.02, hspace=.05, wspace=.07)
for ext in ('pdf','png'):
    fig.savefig(OUT/f'defect_examples.{ext}', dpi=300, bbox_inches='tight', pad_inches=.06)
plt.close(fig)

rolls = list(population)
all_mass = np.array([population[r]['foreground'] for r in rolls])
missed = np.array([population[r]['unseen_foreground'] for r in rolls])
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))
colors = '#225c78'
axes[0].barh(range(len(rolls)), all_mass/1000, color='#d4dce1', label='Marked pixels')
axes[0].barh(range(len(rolls)), missed/1000, color=colors, label='Outside released windows')
axes[0].set_yticks(range(len(rolls)), [r.replace('Rollo', '') for r in rolls])
axes[0].invert_yaxis()
axes[0].set_xlabel('Native marked pixels (thousands)')
axes[0].set_title('A   Marked area', loc='left', fontsize=10, weight='bold')
axes[0].legend(loc='upper right', fontsize=7, frameon=True, facecolor='white', framealpha=1, edgecolor='white')
pct = 100*missed/all_mass
axes[1].barh(range(len(rolls)), pct, color=colors)
axes[1].axvline(100*missed.sum()/all_mass.sum(), color='#9b4e18', ls='--', lw=1)
axes[1].set_yticks(range(len(rolls)), [])
axes[1].invert_yaxis()
axes[1].set_xlabel('Marked pixels outside windows (%)')
axes[1].set_title('B   Fraction omitted', loc='left', fontsize=10, weight='bold')
for i, value in enumerate(pct):
    axes[1].text(value+.1, i, f'{value:.1f}%', va='center', fontsize=8, bbox=dict(facecolor='white', edgecolor='none', pad=.6))
axes[1].set_xlim(0, max(pct)*1.25)
for ax in axes:
    ax.spines[['right','top']].set_visible(False)
    ax.tick_params(labelsize=8)
fig.subplots_adjust(left=.08, right=.98, top=.87, bottom=.23, wspace=.2)
for ext in ('pdf','png'):
    fig.savefig(OUT/f'roll_coverage.{ext}', dpi=300, bbox_inches='tight', pad_inches=.06)
plt.close(fig)
