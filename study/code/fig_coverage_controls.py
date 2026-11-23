"""Figure: per-roll paired whole-frame AP differences on RAW-FABRID (Protocols 1 and 3). Reads saved summaries only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as S
S.apply()
import numpy as np

HERE = Path(__file__).resolve().parents[1]
p1 = json.loads((HERE / 'results/summary.json').read_text())
p3 = json.loads((HERE / 'results_p3/summary.json').read_text())
INK, MUTED, GRID, BLUE, ORANGE = S.INK, S.MUTED, S.GRID, S.COBALT, S.COPPER
rows = [('Gap-fill − released', lambda r1, r3: r3['gapfill'] - r3['released']),
        ('Gap-fill − M (coverage, second looks matched)', lambda r1, r3: r3['gapfill'] - r3['C4']),
        ('M − released (second looks, no coverage)', lambda r1, r3: r3['C4'] - r3['released']),
        ('Released − fixed grid', lambda r1, r3: r1['AP_whole_released'] - r1['AP_whole_fixed'])]
names = {'r18_native': 'ResNet-18, native 256 px', 'wrn50': 'Wide-ResNet-50'}
fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, 2.9), sharey=True)
for k, (ax, m) in enumerate(zip(axes, ('r18_native', 'wrn50'))):
    rolls = p1[m]['rolls']
    for i, (label, f) in enumerate(rows):
        vals = np.array([f(p1[m]['per_roll'][r], p3[m]['per_roll'][r]) for r in rolls])
        y = len(rows) - 1 - i
        ax.scatter(vals, np.full(len(vals), y) + np.linspace(-0.12, 0.12, len(vals)), s=14, color=S.MUTED,
                   edgecolor='white', linewidth=0.8, zorder=3)
        ax.scatter([vals.mean()], [y], s=40, marker='D', color=BLUE, edgecolor='white', linewidth=0.8, zorder=4)
        ax.text(vals.mean(), y + 0.3, f'{vals.mean():+.3f}'.replace('-', '\u2212'), ha='center', va='bottom', fontsize=7, color=INK)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7.5)
    ax.set_title(f"({'ab'[k]})  {names[m]}")
    ax.set_xlabel('Difference in whole-frame pixel AP', fontsize=8)
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=S.INK2, labelsize=7, length=0)
    ax.set_ylim(-0.6, len(rows) - 0.3)
axes[1].scatter([], [], s=14, color=MUTED, label='one held-out roll')
axes[1].scatter([], [], s=60, marker='D', color=BLUE, label='mean of 7 rolls')
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0))
lay = lambda: fig.tight_layout(rect=(0, 0, 1, 0.93))
lay()
S.save(fig, HERE / 'figures/fig06_coverage_controls', relayout=lay)
