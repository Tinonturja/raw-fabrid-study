"""Figure: label-only audit of blind-grid patch relabelling (practice B) across datasets. Values from RESULTS_CUT_AUDIT.md."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
INK, MUTED, GRID, BLUE, ORANGE = '#1f1f1e', '#6b6a63', '#e4e3dc', '#2a78d6', '#eb6834'
data = [  # dataset, regions cut with no whole view (B, A), defect patches < 1% defect (B, A)
    ('RAW-FABRID', 41.0, 19.4, 24.6, 22.1),
    ('AITEX', 18.5, 10.9, 52.2, 48.4),
    ('VisA', 23.0, 5.8, 34.0, 15.5),
    ('MVTec AD 2 Fabric', 28.1, 6.2, 44.6, 33.9),
]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.0), sharey=True)
titles = ['Defect regions cut so no patch shows them whole (%)', '"Defect" patches with under 1% defect (%)']
for ax, (j, title) in zip(axes, enumerate(titles)):
    for i, row in enumerate(data):
        y = len(data) - 1 - i
        b, a = row[1 + 2 * j], row[2 + 2 * j]
        ax.plot([a, b], [y, y], color=GRID, lw=2, zorder=1)
        ax.scatter([b], [y], s=70, color=BLUE, edgecolor='white', linewidth=1.5, zorder=3, label='B: blind grid' if i == 0 else None)
        ax.scatter([a], [y], s=70, color=ORANGE, marker='s', edgecolor='white', linewidth=1.5, zorder=3, label='A: defect-aware crops' if i == 0 else None)
        ax.text(b, y + 0.22, f'{b:.0f}', ha='center', fontsize=8.5, color=INK)
        ax.text(a, y - 0.42, f'{a:.0f}', ha='center', fontsize=8.5, color=INK)
    ax.set_title(title, fontsize=9.5, color=INK, loc='left')
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([d[0] for d in data][::-1], fontsize=9, color=INK)
    ax.set_xlim(0, 60)
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.set_ylim(-0.7, len(data) - 0.4)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=8.5, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.92))
out = HERE / 'manuscript_v2/figures/fig_cut_audit.png'
fig.savefig(out, dpi=300)
print('wrote', out)
