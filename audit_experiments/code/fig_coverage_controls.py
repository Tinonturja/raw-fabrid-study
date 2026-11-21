"""Figure: per-roll paired whole-frame AP differences on RAW-FABRID (Protocols 1 and 3). Reads saved summaries only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]
p1 = json.loads((HERE / 'results/summary.json').read_text())
p3 = json.loads((HERE / 'results_p3/summary.json').read_text())
INK, MUTED, GRID, BLUE = '#1f1f1e', '#6b6a63', '#e4e3dc', '#2a78d6'
rows = [('Gap-fill − released', lambda r1, r3: r3['gapfill'] - r3['released']),
        ('Gap-fill − C4 (coverage, second looks matched)', lambda r1, r3: r3['gapfill'] - r3['C4']),
        ('C4 − released (second looks, no coverage)', lambda r1, r3: r3['C4'] - r3['released']),
        ('Released − fixed grid', lambda r1, r3: r1['AP_whole_released'] - r1['AP_whole_fixed'])]
names = {'r18_native': 'ResNet-18, native 256 px', 'wrn50': 'Wide-ResNet-50'}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, m in zip(axes, ('r18_native', 'wrn50')):
    rolls = p1[m]['rolls']
    for i, (label, f) in enumerate(rows):
        vals = np.array([f(p1[m]['per_roll'][r], p3[m]['per_roll'][r]) for r in rolls])
        y = len(rows) - 1 - i
        ax.scatter(vals, np.full(len(vals), y) + np.linspace(-0.12, 0.12, len(vals)), s=22, color=MUTED,
                   edgecolor='white', linewidth=0.8, zorder=3)
        ax.scatter([vals.mean()], [y], s=90, marker='D', color=BLUE, edgecolor='white', linewidth=1.5, zorder=4)
        ax.text(vals.mean(), y + 0.3, f'{vals.mean():+.3f}', ha='center', va='bottom', fontsize=8.5, color=INK)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9, color=INK)
    ax.set_title(names[m], fontsize=10, color=INK, loc='left')
    ax.set_xlabel('Difference in whole-frame pixel AP', fontsize=9, color=MUTED)
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.set_ylim(-0.6, len(rows) - 0.3)
axes[1].scatter([], [], s=22, color=MUTED, label='one held-out roll')
axes[1].scatter([], [], s=60, marker='D', color=BLUE, label='mean of 7 rolls')
axes[1].legend(frameon=False, fontsize=8.5, loc='lower right')
fig.tight_layout()
out = HERE / 'manuscript_v2/figures/fig_coverage_controls.png'
fig.savefig(out, dpi=300)
print('wrote', out)
