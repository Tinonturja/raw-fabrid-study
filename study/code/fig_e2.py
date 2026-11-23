"""Figure 13: exploratory analysis E2, memory-bank origin crossed with inference layout (paired 2 x 2), on RAW-FABRID
(7 rolls) and VisA (12 categories), from results_p5/e2_bank_layout/full/analysis.json. AP is averaged over three bank
seeds; large markers are unit means, small dots are units. Usage: python3 fig_e2.py"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as S

S.apply()
HERE = Path(__file__).resolve().parents[1]
d = json.loads((HERE / 'results_p5' / 'e2_bank_layout' / 'full' / 'analysis.json').read_text())['datasets']
SEEDS = (13017, 13027, 13037)
FAM = {'D': ('DINOv2-token kNN', S.JADE, 's', '#9fd3bf'), 'W': ('WRN-50 PatchCore-style', S.COBALT, 'o', '#a9b8de')}
DS = {'rawfabrid': ('RAW-FABRID', '7 rolls'), 'visa': ('VisA', '12 categories')}

fig, axes = plt.subplots(1, 4, figsize=(S.DOUBLE, 2.5), sharey=True, gridspec_kw={'wspace': 0.1})
rng = np.random.default_rng(0)          # horizontal jitter only
for ax, (ds, fk), letter in zip(axes, [(a, b) for a in DS for b in FAM], 'abcd'):
    r = d[ds]
    name, col, mk, light = FAM[fk]
    for bank, ls, off in (('Btile', (0, (3, 2)), -0.07), ('Bfull', '-', 0.07)):
        means = []
        for x, lay in enumerate(('R2', 'R1')):
            v = [np.mean([r['ap'][u][f'{fk}_{bank}_{lay}_s{s}'] for s in SEEDS]) for u in r['units']]
            ax.scatter(np.full(len(v), x + off) + rng.uniform(-0.03, 0.03, len(v)), v, s=7, color=light, lw=0, zorder=2)
            means.append(np.mean(v))
        ax.plot([off, 1 + off], means, ls=ls, color=col, lw=1.4, zorder=3)
        ax.plot([off, 1 + off], means, mk, ms=5, mec=col, mew=1.0, mfc='white' if bank == 'Btile' else col, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Tiled\n(R2)', 'Native\n(R1)'])
    ax.set_xlim(-0.45, 1.45)
    ax.tick_params(axis='x', length=0)
    ax.set_title(f'({letter})  {DS[ds][0]}, {"DINOv2" if fk == "D" else "WRN-50"}', fontsize=7.5)
    ax.grid(axis='y')
axes[0].set_ylim(0, 0.8)
axes[0].set_ylabel('Pixel AP (mean of 3 bank seeds)')
handles = [Line2D([], [], color=S.INK2, ls=(0, (3, 2)), marker='o', mfc='white', mec=S.INK2, ms=4.5, lw=1.3,
                  label='bank from 256-px tiles'),
           Line2D([], [], color=S.INK2, ls='-', marker='o', ms=4.5, lw=1.3, label='bank from full frames (same windows)'),
           Line2D([], [], color=S.JADE, ls='none', marker='s', ms=4.5, label='DINOv2-token kNN'),
           Line2D([], [], color=S.COBALT, ls='none', marker='o', ms=4.5, label='WRN-50 PatchCore-style')]
fig.legend(handles=handles, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.02), handlelength=2.2, columnspacing=1.0)
fig.subplots_adjust(left=0.08, right=0.99, top=0.8, bottom=0.17)
S.save(fig, HERE / 'figures' / 'fig13_e2_bank_layout')
