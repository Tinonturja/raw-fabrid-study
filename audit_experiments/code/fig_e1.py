"""Figure 12: post-hoc analysis E1 (DINOv2 input size and bank context) on six VisA categories, from
results_p5/e1_dino_size/full/analysis.json. Macro AP per arm (large markers) with each category (small dots).
Usage: python3 fig_e1.py"""
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
d = json.loads((HERE / 'results_p5' / 'e1_dino_size' / 'full' / 'analysis.json').read_text())
M, P, cats = d['macro_ap'], d['per_category_ap'], d['categories']
LIGHT = '#9fd3bf'   # light jade for per-category dots

fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, 2.6), sharey=True, gridspec_kw={'width_ratios': [1.9, 1], 'wspace': 0.08})

# (a) DINOv2: inference window size, with the bank built from 256 tiles (mismatched) or from the same window (matched)
ax = axes[0]
xs = [0, 1, 2, 3]
mis = ['D_b256_t256', 'D_b256_t512', 'D_b256_t768', 'D_b256_full']
mat = ['D_b256_t256', 'D_b512_t512', 'D_b768_t768', 'D_bfull_full']
rng = np.random.default_rng(0)            # horizontal jitter only, for legibility
for keys, ls, mk, off in ((mis, (0, (3, 2)), 'o', -0.13), (mat, '-', 's', 0.13)):
    px = [0] + [x + off for x in xs[1:]]          # the 256/256 arm is shared by both sweeps: drawn once at x = 0
    for i, k in enumerate(keys):
        if i == 0 and mk == 's':
            continue
        v = [P[c][k] for c in cats]
        ax.scatter(np.full(len(v), px[i]) + rng.uniform(-0.035, 0.035, len(v)), v, s=8, color=LIGHT, zorder=2, lw=0)
    ax.plot(px, [M[k] for k in keys], ls=ls, color=S.JADE, lw=1.4, zorder=3)
    ax.plot(px, [M[k] for k in keys], mk, ms=5, zorder=4, mec=S.JADE, mew=1.0,
            mfc='white' if mk == 'o' else S.JADE)
# training-resolution arm
v = [P[c]['D_b518_s518'] for c in cats]
ax.scatter(np.full(len(v), 4.0) + rng.uniform(-0.03, 0.03, len(v)), v, s=8, color=LIGHT, zorder=2, lw=0)
ax.plot(4.0, M['D_b518_s518'], 'D', color=S.JADE, ms=5, mec='white', mew=0.6, zorder=4)
ax.axvline(3.5, color=S.SAND, lw=0.6, ls=(0, (2, 2)))
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xticklabels(['256', '512', '768', 'full\nframe', 'frame resized\nto 518'])
ax.set_xlabel('Inference window (pixels)')
ax.set_ylabel('Macro pixel AP, six VisA categories')
ax.set_title('(a)  DINOv2-token kNN')
ax.grid(axis='y')
ax.tick_params(axis='x', length=0)
handles = [Line2D([], [], color=S.JADE, ls=(0, (3, 2)), marker='o', mfc='white', mec=S.JADE, ms=4.5, lw=1.4,
                  label='bank from 256-px tiles'),
           Line2D([], [], color=S.JADE, ls='-', marker='s', ms=4.5, lw=1.4, label='bank from the same window size'),
           Line2D([], [], color=S.JADE, ls='none', marker='D', ms=4.5, label='bank and input at 518 px'),
           Line2D([], [], color=LIGHT, ls='none', marker='o', ms=3.5, label='one category')]
fig.legend(handles=handles, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0), handlelength=2.2)

# (b) WRN-50 PatchCore control
ax = axes[1]
arms = [('W_b256_t256', '256 tiles,\nbank 256'), ('W_b256_full', 'full frame,\nbank 256'), ('W_bfull_full', 'full frame,\nbank full')]
for i, (k, _) in enumerate(arms):
    v = [P[c][k] for c in cats]
    ax.scatter(np.full(len(v), i) + rng.uniform(-0.05, 0.05, len(v)), v, s=8, color='#a9b8de', zorder=2, lw=0)
    ax.plot(i, M[k], 'o', color=S.COBALT, ms=5, mec='white', mew=0.6, zorder=4)
ax.plot(range(3), [M[k] for k, _ in arms], color=S.COBALT, lw=1.2, zorder=3)
ax.set_xticks(range(3))
ax.set_xticklabels([l for _, l in arms])
ax.set_xlim(-0.5, 2.5)
ax.set_title('(b)  WRN-50 PatchCore control')
ax.grid(axis='y')
ax.tick_params(axis='x', length=0)
axes[0].set_ylim(0, 0.65)
fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.2)
S.save(fig, HERE / 'manuscript_v2' / 'figures' / 'fig12_e1_bank_context')
