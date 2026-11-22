"""Figure: label-only audit of blind-grid patch relabelling (practice B) across datasets. Values from RESULTS_CUT_AUDIT.md; AITEX from results_cut/cut_audit_aitex_p5.json (frozen cohort, >=128, original floor rule)."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as S
S.apply()

HERE = Path(__file__).resolve().parents[1]
INK, MUTED, GRID, BLUE, ORANGE = S.INK, S.MUTED, S.GRID, S.COBALT, S.COPPER
data = [  # dataset, regions cut with no whole view (B, A), defect patches < 1% defect (B, A)
    ('RAW-FABRID', 41.0, 19.4, 24.6, 22.1),
    ('AITEX', 12.0, 7.4, 53.0, 49.2),
    ('VisA', 23.0, 5.8, 34.0, 15.5),
    ('MVTec AD 2 Fabric', 28.1, 6.2, 44.6, 33.9),
]
fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, 2.5), sharey=True)
titles = ['Regions no single patch shows whole (%)', 'Defect patches under 1% defect (%)']
for ax, (j, title) in zip(axes, enumerate(titles)):
    for i, row in enumerate(data):
        y = len(data) - 1 - i
        b, a = row[1 + 2 * j], row[2 + 2 * j]
        ax.plot([a, b], [y, y], color=GRID, lw=2, zorder=1)
        ax.scatter([b], [y], s=34, color=BLUE, edgecolor='white', linewidth=0.8, zorder=3, label='B: blind grid' if i == 0 else None)
        ax.scatter([a], [y], s=34, color=ORANGE, marker='s', edgecolor='white', linewidth=0.8, zorder=3, label='A: defect-aware crops' if i == 0 else None)
        ax.text(b, y + 0.22, f'{b:.0f}', ha='center', fontsize=7, color=INK)
        ax.text(a, y - 0.42, f'{a:.0f}', ha='center', fontsize=7, color=INK)
    ax.set_title(f"({'ab'[j]})  {title}")
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([d[0] for d in data][::-1], fontsize=7.5)
    ax.set_xlim(0, 60)
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=S.INK2, labelsize=7, length=0)
    ax.set_ylim(-0.7, len(data) - 0.4)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.92))
out = HERE / 'manuscript_v2/figures/fig03_cut_audit.png'
fig.savefig(out)
fig.savefig(out.with_suffix('.pdf'))
print('wrote', out)
